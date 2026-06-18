"""
Fact-Check Agent
-----------------
Upload a PDF -> extract factual claims -> verify each claim against
live web search results -> flag as Verified / Inaccurate / False.
"""

import os
import json
import time
import hashlib
from datetime import date
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
import pdfplumber
import requests
import pandas as pd
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# MUST be first Streamlit command - before anything else
st.set_page_config(page_title="Fact-Check Agent", page_icon="🔍", layout="wide")

# --------------------------------------------------------------------------
# Config / API keys
# --------------------------------------------------------------------------

def get_secret(key: str) -> str:
    try:
        return st.secrets.get(key, "") or os.environ.get(key, "")
    except Exception:
        return os.environ.get(key, "")

GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
TAVILY_API_KEY = get_secret("TAVILY_API_KEY")

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "") or get_secret("GEMINI_MODEL") or "gemini-2.5-flash-lite"
MAX_CLAIMS = 12
MAX_WORKERS = 3
MAX_OUTPUT_TOKENS = 4096
TODAY = date.today().isoformat()


def build_thinking_config():
    if GEMINI_MODEL.startswith("gemini-3"):
        return types.ThinkingConfig(thinking_level="low")
    else:
        return types.ThinkingConfig(thinking_budget=512)


_client = None

def get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def file_hash(uploaded_file) -> str:
    uploaded_file.seek(0)
    digest = hashlib.sha256(uploaded_file.read()).hexdigest()
    uploaded_file.seek(0)
    return digest


def extract_text_from_pdf(uploaded_file) -> str:
    text_parts = []
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
    return "\n".join(text_parts)


class QuotaExceededError(RuntimeError):
    pass


def call_gemini_json(prompt: str, retries: int = 2) -> dict:
    client = get_client()
    last_err = None
    for attempt in range(retries + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                    thinking_config=build_thinking_config(),
                ),
            )
            return json.loads(response.text)
        except json.JSONDecodeError as e:
            last_err = e
            time.sleep(1.0)
        except Exception as e:
            last_err = e
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                raise QuotaExceededError(
                    f"Gemini daily quota exceeded. Wait for reset at midnight "
                    f"Pacific time, or switch to a different model. Error: {e}"
                ) from e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Gemini call failed after retries: {last_err}")


def extract_claims(document_text: str) -> list:
    prompt = f"""Today's date is {TODAY}. You are a fact-checking assistant. Read the
document text below and extract a list of SPECIFIC, CHECKABLE factual claims:
statistics, percentages, dates, financial figures, technical specs, etc.

Rules:
- Skip vague marketing language with no number/date attached.
- Each claim must include the specific number/date/figure AND context to verify it.
- For each claim, write a short 3-8 word search query to find live verification.
- Return JSON in this exact shape:

{{
  "claims": [
    {{"id": 1, "claim": "...", "type": "statistic|date|financial|technical", "search_query": "..."}}
  ]
}}

Limit to the {MAX_CLAIMS} most checkable claims.

DOCUMENT TEXT:
\"\"\"
{document_text[:15000]}
\"\"\"
"""
    try:
        data = call_gemini_json(prompt)
        return data.get("claims", [])[:MAX_CLAIMS]
    except QuotaExceededError:
        raise
    except Exception as e:
        print(f"[extract_claims] Failed: {e}")
        return []


def web_search(query: str, max_results: int = 5) -> list:
    if not TAVILY_API_KEY:
        return []
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "search_depth": "advanced",
                "max_results": max_results,
                "include_answer": True,
            },
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if data.get("answer"):
            results.insert(0, {"title": "Tavily answer", "url": "", "content": data["answer"]})
        return results
    except Exception:
        return []


def verify_claim(claim: dict) -> dict:
    claim_text = claim["claim"]
    search_query = claim.get("search_query") or claim_text
    search_results = web_search(search_query)

    if not search_results:
        return {**claim, "status": "False",
                "explanation": "No live web evidence found.",
                "correct_fact": "N/A", "sources": []}

    evidence_block = "\n\n".join(
        f"[{i+1}] {r.get('title','')}\nURL: {r.get('url','')}\n{r.get('content','')[:800]}"
        for i, r in enumerate(search_results)
    )

    prompt = f"""Today's date is {TODAY}. You are a strict fact-checker.

CLAIM: "{claim_text}"

EVIDENCE:
{evidence_block}

Classify the claim:
- "Verified": evidence confirms the claim is currently accurate.
- "Inaccurate": claim was once true but is now outdated — give the current correct figure.
- "False": evidence contradicts the claim, or the number is wrong even slightly.

Be skeptical. Only mark Verified if evidence clearly supports it.
For correct_fact, always state the actual number/figure explicitly.

Return JSON:
{{
  "status": "Verified|Inaccurate|False",
  "explanation": "1-2 sentences citing evidence",
  "correct_fact": "explicit correct figure, or N/A if Verified"
}}
"""
    try:
        result = call_gemini_json(prompt)
    except Exception:
        result = {"status": "False", "explanation": "Verification failed.", "correct_fact": "N/A"}

    result["sources"] = [r.get("url") for r in search_results if r.get("url")][:3]
    return {**claim, **result}


def run_pipeline(doc_text: str, progress_callback=None) -> list:
    claims = extract_claims(doc_text)
    if not claims:
        return []

    results_by_id = {}
    done_count = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(verify_claim, c): c["id"] for c in claims}
        for future in as_completed(futures):
            claim_id = futures[future]
            try:
                results_by_id[claim_id] = future.result()
            except Exception as e:
                base = next((c for c in claims if c["id"] == claim_id), {})
                results_by_id[claim_id] = {**base, "status": "False",
                    "explanation": f"Error: {e}", "correct_fact": "N/A", "sources": []}
            done_count += 1
            if progress_callback:
                progress_callback(done_count, len(claims))

    return [results_by_id[c["id"]] for c in claims if c["id"] in results_by_id]


STATUS_COLORS = {"Verified": "#1a7f37", "Inaccurate": "#b08900", "False": "#cf222e"}
STATUS_ICONS  = {"Verified": "✅", "Inaccurate": "⚠️", "False": "❌"}

# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

st.title("🔍 Fact-Check Agent")
st.caption("Upload a PDF — the agent extracts claims, checks them against live web data, and flags inaccuracies.")

if not GEMINI_API_KEY or not TAVILY_API_KEY:
    st.warning("Missing API keys. Add GEMINI_API_KEY and TAVILY_API_KEY to Streamlit secrets.", icon="⚠️")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if "report_cache" not in st.session_state:
    st.session_state.report_cache = {}

if uploaded_file is not None and GEMINI_API_KEY:
    current_hash = file_hash(uploaded_file)

    if st.button("🔄 Re-analyze (ignore cache)"):
        st.session_state.report_cache.pop(current_hash, None)

    if current_hash not in st.session_state.report_cache:
        with st.spinner("Reading PDF..."):
            doc_text = extract_text_from_pdf(uploaded_file)

        if not doc_text.strip():
            st.error("No text found in this PDF. Try a text-based (non-scanned) PDF.")
            st.stop()

        with st.spinner("Extracting claims with Gemini..."):
            try:
                claims_preview = extract_claims(doc_text)
            except QuotaExceededError as e:
                st.error(str(e))
                st.stop()

        if not claims_preview:
            st.info("No specific checkable claims found in this document.")
            st.stop()

        st.success(f"Found {len(claims_preview)} claim(s). Verifying against live web data...")
        progress = st.progress(0)

        def update_progress(done, total):
            progress.progress(done / total)

        results = run_pipeline(doc_text, progress_callback=update_progress)
        progress.empty()
        st.session_state.report_cache[current_hash] = results
    else:
        results = st.session_state.report_cache[current_hash]
        st.info("Cached results — click Re-analyze to rerun.")

    counts = {"Verified": 0, "Inaccurate": 0, "False": 0}
    for r in results:
        counts[r.get("status", "False")] += 1

    col1, col2, col3 = st.columns(3)
    col1.metric("✅ Verified",   counts["Verified"])
    col2.metric("⚠️ Inaccurate", counts["Inaccurate"])
    col3.metric("❌ False",      counts["False"])

    df = pd.DataFrame([{
        "Claim":        r.get("claim"),
        "Type":         r.get("type"),
        "Status":       r.get("status"),
        "Explanation":  r.get("explanation"),
        "Correct Fact": r.get("correct_fact"),
        "Sources":      "; ".join(r.get("sources", [])),
    } for r in results])

    st.download_button("⬇️ Download report as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="fact_check_report.csv", mime="text/csv")

    st.divider()

    for r in results:
        status = r.get("status", "False")
        color  = STATUS_COLORS.get(status, "#666")
        icon   = STATUS_ICONS.get(status, "❔")
        with st.container(border=True):
            st.markdown(
                f"<span style='background:{color};color:white;padding:3px 10px;"
                f"border-radius:12px;font-weight:600;font-size:.85em'>{icon} {status}</span>",
                unsafe_allow_html=True)
            st.markdown(f"**Claim:** {r.get('claim')}")
            st.markdown(f"**Reasoning:** {r.get('explanation')}")
            if status != "Verified":
                st.markdown(f"**Correct fact:** {r.get('correct_fact')}")
            sources = r.get("sources", [])
            if sources:
                with st.expander("Sources"):
                    for s in sources:
                        st.markdown(f"- [{s}]({s})")

elif uploaded_file is not None and not GEMINI_API_KEY:
    st.error("Add your GEMINI_API_KEY to Streamlit secrets first.")

st.sidebar.header("How it works")
st.sidebar.markdown("""
1. **Extract** — Gemini reads the PDF and pulls out checkable claims with optimized search queries.
2. **Verify** — Each claim is searched live via Tavily, results fed back to Gemini with today's date.
3. **Report** — Claims flagged as ✅ Verified, ⚠️ Inaccurate (corrected), or ❌ False.

Results cached per file — use Re-analyze to force a fresh run.
""")
st.sidebar.caption("Built with Streamlit + Gemini + Tavily")
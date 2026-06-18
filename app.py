"""
Fact-Check Agent
-----------------
Upload a PDF -> extract factual claims -> verify each claim against
live web search results -> flag as Verified / Inaccurate / False.

Built for: CogCulture Assessment - Part 2 ("The Fact-Check Agent")
Stack: Streamlit (UI) + Gemini (extraction & reasoning) + Tavily (live web search)

v3 fixes (see notes at bottom of file for the "why"):
- Switched from the deprecated `google-generativeai` SDK to the current
  `google-genai` SDK. The old SDK is deprecated and doesn't talk to
  gemini-3.x models reliably -- this was the main cause of hangs/slowness.
- JSON is now requested via response_mime_type="application/json" (server-side
  enforced) instead of asking nicely in the prompt and regex-stripping fences.
- thinking_level is explicitly set to "low" for both extraction and
  verification calls, since these are short structured-output tasks that
  don't need deep reasoning -- this is the main lever for the new SDK's speed.
- Same session-state caching, parallel verification, and CSV export as v2.
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
load_dotenv()  # reads .env into os.environ for local runs

# --------------------------------------------------------------------------
# Config / API keys
# --------------------------------------------------------------------------

def get_secret(key: str) -> str:
    # Merely *accessing* st.secrets (even in a try/except) registers an
    # internal Streamlit element that breaks set_page_config() when no
    # secrets.toml exists at all. So only touch st.secrets if a secrets
    # file is actually present on disk.
    secrets_paths = [
        os.path.join(os.getcwd(), ".streamlit", "secrets.toml"),
        os.path.join(os.path.expanduser("~"), ".streamlit", "secrets.toml"),
    ]
    if any(os.path.exists(p) for p in secrets_paths):
        try:
            if key in st.secrets:
                return st.secrets[key]
        except Exception:
            pass
    return os.environ.get(key, "")


GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
TAVILY_API_KEY = get_secret("TAVILY_API_KEY")

# gemini-3.5-flash is a newer/preview-tier model with a much tighter free-tier
# daily quota (20 RPD at time of writing) than the established gemini-2.5-flash
# (250 RPD). Defaulting to 2.5-flash here avoids burning through the daily cap
# during development; set GEMINI_MODEL=gemini-3.5-flash in your .env once you
# have billing enabled or are ready to use the newer model's daily allowance.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
MAX_CLAIMS = 12                    # cap to control latency / free-tier usage
MAX_WORKERS = 3                    # parallel verification workers (keep modest for free-tier rate limits)
MAX_OUTPUT_TOKENS = 4096           # thinking tokens count against this budget even at low effort --
                                    # too small a value here causes truncated/empty JSON responses
TODAY = date.today().isoformat()


def build_thinking_config():
    """Build the correct thinking config for whichever GEMINI_MODEL is active.

    Gemini 3.x models use `thinking_level` (a semantic low/medium/high knob).
    Gemini 2.5.x models use `thinking_budget` (an explicit token count) and
    reject `thinking_level` outright with a 400 error -- the two parameters
    are mutually exclusive across model generations, not interchangeable.
    This keeps the model swappable via GEMINI_MODEL without re-breaking the
    thinking config every time, since hardcoding either parameter only works
    until someone changes the model name.
    """
    if GEMINI_MODEL.startswith("gemini-3"):
        return types.ThinkingConfig(thinking_level="low")
    else:
        # gemini-2.5-flash accepts a 0-24576 token budget; a small explicit
        # value here mirrors the "low effort, don't overthink it" intent
        # without falling back to dynamic (-1) allocation, which can behave
        # closer to the model's default depth than we actually want for
        # these short structured-extraction tasks.
        return types.ThinkingConfig(thinking_budget=512)

st.set_page_config(page_title="Fact-Check Agent", page_icon="🔍", layout="wide")

# A single client, created once GEMINI_API_KEY is known to be present.
# (See UI section below -- we only build this after confirming the key exists,
# so importing this module never crashes when the key is missing.)
_client = None


def get_client() -> "genai.Client":
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
    """Raised when Gemini returns 429 RESOURCE_EXHAUSTED on a daily quota.

    Distinct from a generic RuntimeError so callers (and the UI) can show
    the person something more useful than "no claims found" -- a quota
    exhaustion is a fundamentally different, non-retriable-within-this-run
    situation compared to a transient API error or a parse failure.
    """
    pass


def call_gemini_json(prompt: str, retries: int = 2) -> dict:
    """Call Gemini with JSON mode + low thinking enforced, return parsed dict.

    Using response_mime_type="application/json" makes the model emit JSON
    directly (no markdown fences, no preamble to strip), and thinking_level
    keeps latency down for these short structured tasks. max_output_tokens
    is set generously because thinking tokens are counted against this same
    budget even at low thinking levels -- too small a value here silently
    truncates the JSON before it's ever written, leaving response.text empty
    or cut off mid-object.
    """
    client = get_client()
    last_err = None
    last_raw_text = None
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
            last_raw_text = response.text
            return json.loads(response.text)
        except json.JSONDecodeError as e:
            last_err = e
            # Print what actually came back so truncation/empty-response
            # issues are visible in the terminal instead of silently
            # producing an empty claims list further up the call chain.
            print(f"[call_gemini_json] JSON parse failed on attempt {attempt + 1}: {e}")
            print(f"[call_gemini_json] Raw response.text was: {last_raw_text!r}")
            time.sleep(1.0)
        except Exception as e:  # noqa: BLE001
            last_err = e
            error_text = str(e)
            print(f"[call_gemini_json] Gemini call raised on attempt {attempt + 1}: {e}")
            # RESOURCE_EXHAUSTED (429) on the *daily* quota won't be fixed by
            # retrying within the same run -- a 12s/15s/17s wait does nothing
            # against a quota that resets at midnight Pacific. Fail fast with
            # a clear message instead of burning the remaining retry budget
            # (and adding more latency) on a call that can't succeed.
            if "RESOURCE_EXHAUSTED" in error_text or "429" in error_text:
                raise QuotaExceededError(
                    "Gemini daily quota exceeded for this model/project. "
                    "Either wait for the daily reset (midnight Pacific time), "
                    "switch GEMINI_MODEL to one with a higher free-tier quota "
                    "(e.g. gemini-2.5-flash), or enable billing on the project. "
                    f"Original error: {e}"
                ) from e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Gemini call failed after retries: {last_err}")


def extract_claims(document_text: str) -> list:
    """Ask Gemini to pull out specific, checkable factual claims."""
    prompt = f"""Today's date is {TODAY}. You are a fact-checking assistant. Read the
document text below and extract a list of SPECIFIC, CHECKABLE factual claims:
statistics, percentages, dates, financial figures, technical specs, named
records/achievements, etc.

Rules:
- Skip vague marketing language ("industry-leading", "best-in-class") with no number/date attached.
- Each claim must be a single, self-contained, checkable statement that includes the
  specific number/date/figure AND enough context to verify it (what it refers to, who/what, when).
- For each claim, also write a short, search-engine-optimized query (3-8 words,
  just the key entity + number/date) that would surface the most relevant live
  results — NOT the full sentence.
- Return JSON in this exact shape:
{{
  "claims": [
    {{"id": 1, "claim": "...", "type": "statistic|date|financial|technical", "search_query": "..."}}
  ]
}}

Limit to the {MAX_CLAIMS} most checkable/important claims if there are more.

DOCUMENT TEXT:
\"\"\"
{document_text[:15000]}
\"\"\"
"""
    try:
        data = call_gemini_json(prompt)
        return data.get("claims", [])[:MAX_CLAIMS]
    except QuotaExceededError:
        # Let this propagate -- the UI shows a specific message for quota
        # exhaustion rather than the generic "no claims found" fallback.
        raise
    except (RuntimeError, json.JSONDecodeError) as e:
        print(f"[extract_claims] Falling back to empty claims list due to: {e}")
        return []


def web_search(query: str, max_results: int = 5) -> list:
    """Live web search via Tavily API."""
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
            results.insert(0, {
                "title": "Tavily synthesized answer",
                "url": "",
                "content": data["answer"],
            })
        return results
    except Exception:  # noqa: BLE001
        return []


def verify_claim(claim: dict) -> dict:
    """Search the web for the claim, then ask Gemini to judge it."""
    claim_text = claim["claim"]
    search_query = claim.get("search_query") or claim_text
    search_results = web_search(search_query)

    if not search_results:
        return {
            **claim,
            "status": "False",
            "explanation": "No live web evidence could be found to support this claim.",
            "correct_fact": "N/A",
            "sources": [],
        }

    evidence_block = "\n\n".join(
        f"[{i+1}] {r.get('title','')}\nURL: {r.get('url','')}\n{r.get('content','')[:800]}"
        for i, r in enumerate(search_results)
    )

    prompt = f"""Today's date is {TODAY}. You are a strict fact-checker reviewing a
document that may contain deliberately fabricated statistics, invented dates, or
numbers that have been altered from the real figure -- not just outdated information.
A document makes the following claim:

CLAIM: "{claim_text}"

Here is live web search evidence gathered to verify it:

{evidence_block}

Decide the claim's status:
- "Verified": the evidence confirms the claim's figures/dates are accurate as of today,
  with no meaningful discrepancy from the real number.
- "Inaccurate": the claim was CORRECT at some point in the past, but the evidence shows
  the real-world figure has since changed (a genuinely outdated stat, not a fabrication) --
  state the correct CURRENT figure.
- "False": the evidence shows a different number/date than the claim states, even if the
  claim's number is plausible-sounding or only moderately different from the real one.
  A claim that is simply wrong -- whether wildly off or just slightly altered from the
  true figure -- is "False", not "Inaccurate". Reserve "Inaccurate" only for claims that
  were once true and have since become outdated due to real-world change over time.
  If no credible evidence supports the claim at all, that is also "False".

Be skeptical by default — only mark "Verified" if the evidence clearly and
specifically supports the claim as currently accurate. Do not give the claim the
benefit of the doubt just because its number is in a realistic range; check it
against what the evidence actually says.

CRITICAL for "correct_fact": when the claim concerns a specific number, statistic,
date, or figure, correct_fact MUST state that same kind of figure explicitly (e.g.
"45 million users" not "the actual number is different" or "lower than stated").
A vague correction without the real number is not useful and should be avoided
whenever the evidence contains the actual figure.

Return JSON in this exact shape:
{{
  "status": "Verified|Inaccurate|False",
  "explanation": "1-2 sentence reasoning citing what the evidence shows",
  "correct_fact": "the correct, current real-world fact/figure stated explicitly (or 'N/A' if Verified and nothing to correct)"
}}
"""
    try:
        result = call_gemini_json(prompt)
    except QuotaExceededError as e:
        # This was previously caught by the generic except below and
        # mislabeled "Could not parse verification reasoning" -- that message
        # is actively misleading for a rate-limit failure, since it implies
        # a parsing problem when the call never even completed. Surfacing
        # this distinctly matters because a 429 here means this claim was
        # NEVER actually checked, which is very different from a real
        # "False" verdict for grading purposes.
        result = {
            "status": "Unverified",
            "explanation": (
                "Verification could not run because Gemini's rate limit was "
                f"hit for this claim. This is NOT a real fact-check result. {e}"
            ),
            "correct_fact": "N/A",
        }
    except (RuntimeError, json.JSONDecodeError) as e:
        result = {
            "status": "Unverified",
            "explanation": f"Verification failed due to an unexpected error: {e}",
            "correct_fact": "N/A",
        }

    result["sources"] = [r.get("url") for r in search_results if r.get("url")][:3]
    return {**claim, **result}


def run_pipeline(doc_text: str, progress_callback=None) -> list:
    """Extract claims, then verify them in parallel. Returns ordered results."""
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
            except Exception as e:  # noqa: BLE001
                base = next((c for c in claims if c["id"] == claim_id), {})
                results_by_id[claim_id] = {
                    **base,
                    "status": "Unverified",
                    "explanation": f"Verification could not complete due to an unexpected error: {e}",
                    "correct_fact": "N/A",
                    "sources": [],
                }
            done_count += 1
            if progress_callback:
                progress_callback(done_count, len(claims))

    # Preserve original extraction order
    return [results_by_id[c["id"]] for c in claims if c["id"] in results_by_id]


STATUS_COLORS = {"Verified": "#1a7f37", "Inaccurate": "#b08900", "False": "#cf222e", "Unverified": "#6e7781"}
STATUS_ICONS = {"Verified": "✅", "Inaccurate": "⚠️", "False": "❌", "Unverified": "❔"}

# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

st.title("🔍 Fact-Check Agent")
st.caption("Upload a PDF. The agent extracts factual claims, checks them against live web data, and flags inaccuracies.")

if not GEMINI_API_KEY or not TAVILY_API_KEY:
    st.warning(
        "Missing API keys. Add `GEMINI_API_KEY` and `TAVILY_API_KEY` to your "
        "Streamlit secrets (or a local `.env`) before running a real check.",
        icon="⚠️",
    )

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if "report_cache" not in st.session_state:
    st.session_state.report_cache = {}  # {file_hash: results}

if uploaded_file is not None and GEMINI_API_KEY:
    current_hash = file_hash(uploaded_file)
    force_rerun = st.button("🔄 Re-analyze this PDF (ignore cache)")

    if force_rerun:
        st.session_state.report_cache.pop(current_hash, None)

    if current_hash not in st.session_state.report_cache:
        with st.spinner("Reading PDF..."):
            doc_text = extract_text_from_pdf(uploaded_file)

        if not doc_text.strip():
            st.error("Couldn't extract any text from this PDF (it may be a scanned image). Try a text-based PDF.")
            st.stop()

        with st.spinner("Extracting claims with Gemini..."):
            try:
                claims_preview = extract_claims(doc_text)
            except QuotaExceededError as e:
                st.error(
                    f"Gemini's daily free-tier quota for `{GEMINI_MODEL}` has been "
                    "exhausted. This resets at midnight Pacific time, or you can "
                    "set `GEMINI_MODEL` in your `.env` to a model with a higher "
                    "free-tier quota (e.g. `gemini-2.5-flash`) and restart the app."
                )
                st.caption(f"Details: {e}")
                st.stop()

        if not claims_preview:
            st.info("No specific checkable claims (stats/dates/figures) were found in this document.")
            st.stop()

        st.success(f"Found {len(claims_preview)} checkable claim(s). Verifying against live web data (in parallel)...")
        progress = st.progress(0)

        def update_progress(done, total):
            progress.progress(done / total)

        # Re-run full pipeline (extraction is cheap relative to verification;
        # simplest correct approach without re-architecting around partial state)
        results = run_pipeline(doc_text, progress_callback=update_progress)
        progress.empty()
        st.session_state.report_cache[current_hash] = results
    else:
        results = st.session_state.report_cache[current_hash]
        st.info("Showing cached results for this file. Click 'Re-analyze' above to force a fresh run.")

    # Summary counts
    counts = {"Verified": 0, "Inaccurate": 0, "False": 0, "Unverified": 0}
    for r in results:
        status = r.get("status", "Unverified")
        counts[status] = counts.get(status, 0) + 1

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("✅ Verified", counts["Verified"])
    col2.metric("⚠️ Inaccurate", counts["Inaccurate"])
    col3.metric("❌ False", counts["False"])
    col4.metric("❔ Unverified", counts["Unverified"])
    if counts["Unverified"] > 0:
        st.caption(
            f"{counts['Unverified']} claim(s) could not be verified due to rate limits or "
            "errors -- these are NOT confirmed-false results. Re-run to attempt them again."
        )

    # CSV export
    df = pd.DataFrame([
        {
            "Claim": r.get("claim"),
            "Type": r.get("type"),
            "Status": r.get("status"),
            "Explanation": r.get("explanation"),
            "Correct Fact": r.get("correct_fact"),
            "Sources": "; ".join(r.get("sources", [])),
        }
        for r in results
    ])
    st.download_button(
        "⬇️ Download report as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="fact_check_report.csv",
        mime="text/csv",
    )

    st.divider()

    for r in results:
        status = r.get("status", "Unverified")
        color = STATUS_COLORS.get(status, "#666")
        icon = STATUS_ICONS.get(status, "❔")
        with st.container(border=True):
            st.markdown(
                f"<span style='background-color:{color};color:white;padding:3px 10px;"
                f"border-radius:12px;font-weight:600;font-size:0.85em'>{icon} {status}</span>",
                unsafe_allow_html=True,
            )
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
    st.error("Add your GEMINI_API_KEY first (see sidebar / secrets).")

st.sidebar.header("How it works")
st.sidebar.markdown(
    """
1. **Extract** — PDF text is parsed and sent to Gemini, which pulls out specific,
   checkable claims (stats, dates, financial/technical figures) plus an optimized
   search query for each.
2. **Verify** — claims are searched live on the web via Tavily **in parallel**, and
   results are handed back to Gemini (with today's date) as evidence.
3. **Report** — Gemini judges each claim as **Verified**, **Inaccurate** (outdated/slightly
   off — corrected figure shown), or **False** (no evidence / contradicted).

Results are cached per file for this session, so toggling UI elements won't
re-trigger API calls. Use **Re-analyze** to force a fresh run.
"""
)
st.sidebar.divider()
st.sidebar.caption("Built with Streamlit + Gemini + Tavily")

# --------------------------------------------------------------------------
# Notes on the v2 -> v3 changes (why things were slow/broken)
# --------------------------------------------------------------------------
# 1. SDK: `import google.generativeai as genai` is the OLD, deprecated SDK
#    (package: google-generativeai). It's frozen and wasn't built for
#    gemini-3.x models -- running gemini-3.5-flash through it is the kind
#    of version mismatch that causes hangs or silent failures rather than
#    clean errors. The fix is `from google import genai` (package:
#    google-genai), which uses `client = genai.Client()` and
#    `client.models.generate_content(...)`.
#
# 2. JSON mode: previously the prompt just asked nicely for "STRICT JSON
#    only" and a regex (`clean_json_block`) stripped markdown fences after
#    the fact. The new SDK supports `response_mime_type="application/json"`
#    in GenerateContentConfig, which is enforced server-side -- no fences to
#    strip, no risk of the model adding commentary before/after the JSON.
#
# 3. thinking_level: gemini-3.5-flash defaults to "medium" thinking, which
#    is wasted depth for short, structured extraction/verification tasks.
#    Setting `thinking_level="low"` via ThinkingConfig cuts latency per call
#    noticeably, and since verify_claim runs MAX_CLAIMS times across a
#    ThreadPoolExecutor, that saving compounds across the whole pipeline.
#
# If you still see slowness after this fix, the next most likely bottleneck
# is Tavily's `search_depth="advanced"` (slower, more thorough than "basic")
# combined with MAX_WORKERS=3 -- try search_depth="basic" or bumping
# MAX_WORKERS if your Tavily plan's rate limits allow it.
#
# 4. Empty claims list despite a claim-rich document: gemini-3.x models keep
#    thinking enabled even at thinking_level="low" (thought signatures are
#    still produced), and those thinking tokens count against the SAME
#    max_output_tokens budget as the actual JSON response. Without an
#    explicit, generous max_output_tokens, a long prompt (e.g. extract_claims
#    embedding up to 15k chars of document text) can exhaust the budget on
#    thinking alone, leaving response.text empty or truncated mid-JSON --
#    which silently became an empty claims list before this fix, since the
#    failure was caught and swallowed without logging. Fixed by setting
#    MAX_OUTPUT_TOKENS=4096 explicitly and printing the raw response text
#    whenever JSON parsing fails, so this is debuggable from the terminal
#    instead of just showing "no checkable claims found" in the UI.
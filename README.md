# 🔍 Fact-Check Agent

An AI agent that reads a PDF, extracts specific factual claims (stats, dates,
financial/technical figures), verifies each one against **live web search**,
and reports whether it's **Verified**, **Inaccurate** (outdated/slightly off —
with the corrected figure), or **False** (no evidence / contradicted).

Built for the CogCulture "Fact-Check Agent" assessment.

## How it works

```
PDF Upload
   │
   ▼
Extract text (pdfplumber)
   │
   ▼
Gemini Pass 1 — Claim Extraction
   (pulls out checkable claims as structured JSON)
   │
   ▼
For each claim → Tavily live web search
   │
   ▼
Gemini Pass 2 — Verification
   (compares claim vs search evidence → Verified / Inaccurate / False
    + correct fact + sources)
   │
   ▼
Streamlit UI — color-coded report with sources
```

## Tech stack
- **Frontend:** Streamlit
- **PDF parsing:** pdfplumber
- **Claim extraction & reasoning:** Google Gemini (`gemini-2.0-flash`)
- **Live web verification:** Tavily Search API

## Local setup

1. Clone the repo and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Get free API keys:
   - **Gemini:** https://aistudio.google.com/app/apikey (free tier)
   - **Tavily:** https://tavily.com (free tier, no card required)

3. Create a `.env` file in the project root:
   ```
   GEMINI_API_KEY=your_key_here
   TAVILY_API_KEY=your_key_here
   ```
   Or export them as environment variables before running.

4. Run locally:
   ```bash
   streamlit run app.py
   ```

## Deploying to Streamlit Community Cloud (free)

1. Push this repo to GitHub.
2. Go to https://share.streamlit.io → "New app" → connect your GitHub repo →
   select `app.py` as the entry point.
3. In the app's **Settings → Secrets**, add:
   ```toml
   GEMINI_API_KEY = "your_key_here"
   TAVILY_API_KEY = "your_key_here"
   ```
4. Deploy. You'll get a public URL like `https://your-app.streamlit.app`.

## Notes / design decisions
- Claim extraction is capped at the 12 most checkable claims per document to
  keep latency and free-tier API usage reasonable on larger PDFs.
- Each claim's search query is the claim text itself, and Gemini reasons over
  the raw search snippets — this keeps the verification grounded in real
  retrieved evidence rather than the model's own (possibly stale) knowledge.
- If no web evidence is found for a claim at all, it's conservatively flagged
  as **False** rather than assumed true.

## Repo structure
```
.
├── app.py              # Streamlit app (extraction + verification + UI)
├── requirements.txt
├── .streamlit/
│   └── secrets.toml.example
└── README.md
```

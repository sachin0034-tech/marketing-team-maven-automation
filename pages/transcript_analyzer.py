import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_client() -> OpenAI:
    key = st.session_state.get("openai_api_key", "")
    if not key:
        st.error("Add your OpenAI API key in the sidebar first.")
        st.stop()
    return OpenAI(api_key=key)


def scrape_url(url: str) -> str:
    """Return visible text from a webpage."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        # Trim to ~8 000 chars to stay within token limits
        return text[:8000]
    except Exception as e:
        st.error(f"Could not scrape URL: {e}")
        return ""


def analyze_transcript(client: OpenAI, transcript: str) -> dict:
    """Extract marketing intelligence from a sales call transcript."""
    system_prompt = """You are a senior marketing strategist. A marketing team at an online course business
will use your analysis to improve their ads, emails, landing pages, and targeting.

Analyze this sales call transcript and return a JSON object with these exact keys:

{
  "call_snapshot": {
    "outcome": "Purchased | Not purchased | Follow-up needed",
    "fit_score": <1-10, how well does this person match the ideal buyer>,
    "awareness_level": "Problem-aware | Solution-aware | Product-aware",
    "what_brought_them_here": "<what ad / email / content / referral brought them to book this call — infer from transcript>",
    "one_line_summary": "<single sentence: who this person is and what they need>"
  },

  "voice_of_customer": {
    "pain_quotes": ["<exact quote from prospect describing their pain>", ...],
    "desire_quotes": ["<exact quote describing what outcome they want>", ...],
    "fear_quotes": ["<exact quote expressing a fear or risk they see>", ...],
    "language_patterns": ["<a specific phrase or word they repeated that marketing should use>", ...]
  },

  "objection_intelligence": [
    {
      "objection": "<exactly what they said>",
      "real_meaning": "<what this objection actually signals underneath>",
      "funnel_fix": "<what should be added to the landing page / email sequence to pre-handle this before the call>"
    },
    ...
  ],

  "buying_signals": [
    {
      "moment": "<what was being discussed when they showed interest>",
      "exact_quote": "<what the prospect said>",
      "marketing_use": "<how to use this in ad copy or email>"
    },
    ...
  ],

  "content_gaps": [
    {
      "gap": "<something the prospect didn't know or was confused about>",
      "should_have_been_covered_in": "Landing page | Email sequence | Ad copy | FAQ",
      "suggested_fix": "<specific content piece or copy change to add>"
    },
    ...
  ],

  "icp_signals": {
    "role_situation": "<job title, life stage, or situation inferred>",
    "goal": "<what they are ultimately trying to achieve>",
    "timeline": "<how urgently they need a solution>",
    "budget_signal": "<any signal about willingness or ability to pay>",
    "channels_they_trust": "<any mention of where they consume content, who they follow, what they read>"
  },

  "copy_bank": [
    {
      "type": "Email subject line | Ad headline | Landing page headline | CTA",
      "copy": "<ready-to-use copy derived from the prospect's own language>",
      "why_it_works": "<1 sentence>"
    },
    ...
  ],

  "marketing_action_items": [
    "<specific, actionable thing the marketing team should change or create based on this call>",
    ...
  ]
}

Rules:
- Use EXACT quotes from the transcript wherever possible — never paraphrase voice of customer
- copy_bank must have at least 5 entries across different types
- marketing_action_items must be concrete (not "improve messaging" — say WHAT to change WHERE)
- Never give generic marketing advice — everything must be grounded in this specific transcript"""

    resp = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Transcript:\n\n{transcript}"},
        ],
        temperature=0.3,
    )
    import json
    return json.loads(resp.choices[0].message.content)


def generate_demo_transcript(client: OpenAI, product: str, industry: str, pain_point: str) -> dict:
    """Generate a complete annotated 15-min demo sales call transcript."""
    system_prompt = """You are an elite sales coach. Generate a complete, realistic 15-minute B2B sales call transcript.

Return a JSON object with this structure:
{
  "call_title": "<short title for this demo call>",
  "context": "<2-sentence setup: who is calling whom, what product, what pain point>",
  "timeline": [
    {
      "timestamp": "0:00",
      "phase": "Opening",
      "speaker": "Rep" or "Prospect",
      "line": "<exactly what they say — natural, conversational>",
      "annotation": "<coaching note on WHY this line works or what technique is being used — only add to Rep lines>"
    },
    ...
  ],
  "key_techniques": [
    {"technique": "<name>", "description": "<what it is and when it appeared in the call>"},
    ...
  ],
  "what_made_this_call_work": "<3-4 sentence summary of the winning behaviours in this call>"
}

Rules:
- The call must be ~15 minutes so include 30-40 turns total
- Timestamps must progress realistically (0:00 to 15:00)
- Phases: Opening (0-2 min) → Discovery (2-6 min) → Pitch (6-10 min) → Objection Handling (10-13 min) → Close (13-15 min)
- Prospect must push back at least 3 times with realistic objections
- Rep must use discovery questions, not just pitch
- annotations only on Rep lines — keep them short (1 sentence)
- Make it feel like a real call, not a training video script"""

    resp = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Product/Service: {product}\n"
                    f"Industry: {industry}\n"
                    f"Main pain point to address: {pain_point}"
                ),
            },
        ],
        temperature=0.6,
    )
    import json
    return json.loads(resp.choices[0].message.content)


def generate_pitch(client: OpenAI, page_content: str, transcript: str = "") -> str:
    """Generate an inbound closer script for warm booked calls — goal is to get them to purchase."""
    context = f"Course/product page content:\n{page_content}"
    if transcript:
        context += f"\n\nAdditional context from a previous call:\n{transcript[:2000]}"

    system_prompt = """You are a world-class inbound sales closer.

IMPORTANT CONTEXT:
- The prospect already BOOKED this call — they are warm and interested
- They are NOT being cold-called. They came to us.
- The goal of this 15-minute call is to:
  1. Find out what's confusing them or holding them back from buying
  2. Clear that specific confusion or objection live on the call
  3. Get them to purchase the course before the call ends

Write a 15-minute inbound closer script. Format it exactly like this:

## 🎯 Inbound Closer Script — [Course/Product Name]
*(For booked calls where the prospect already showed interest)*

---

### 0:00 – 1:30 | Warm Open — Make them feel heard
[script — acknowledge they booked, make it feel like a conversation not a pitch]
*Why this works: ...*

### 1:30 – 4:00 | Discovery — Find the real hesitation
[script — 2-3 targeted questions to uncover exactly what's stopping them]
*Why this works: ...*

### 4:00 – 6:00 | Confirm the Gap
[script — mirror back what they said, make them feel understood before you say anything about the product]
*Why this works: ...*

### 6:00 – 10:00 | The Bridge — Connect their problem to the course outcome
[script — now pitch, but only the parts that directly solve what THEY said]
*Why this works: ...*

### 10:00 – 13:00 | Handle the Real Objection
[script — address the #1 objection that comes up for this type of buyer, with exact language]
*Why this works: ...*

### 13:00 – 15:00 | The Close — Get the decision now
[script — low-pressure, decision-forcing close that feels like a natural next step]
*Why this works: ...*

---

## 🚨 Most Common Objections & Exact Responses
| Objection | What they mean | How to respond |
|-----------|---------------|----------------|
| ...       | ...           | ...            |

---

## 💡 Personalisation Tips
[4-5 bullets on how to adapt this script based on what the prospect says in discovery]

Be specific to this course. Every line must be usable as-is. No filler."""

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context},
        ],
        temperature=0.4,
    )
    return resp.choices[0].message.content


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def render():
    st.title("📞 Sales Call Analyzer")
    st.caption("Upload a transcript for coaching feedback, or paste a URL to generate a 15-min pitch.")

    tab1, tab2, tab3 = st.tabs(["🔍 Analyze Transcript", "🚀 Inbound Closer Script", "🎓 Demo Call Transcript"])

    # ------------------------------------------------------------------
    # TAB 1 — Marketing Intelligence from Transcript
    # ------------------------------------------------------------------
    with tab1:
        st.markdown("### Upload a sales call transcript")
        st.caption("Get marketing intelligence: buyer language, objection fixes, copy ideas, and funnel gaps — all grounded in this specific call.")

        uploaded = st.file_uploader("Supports .txt files", type=["txt"], key="transcript_file")
        transcript_text = st.text_area(
            "Or paste the transcript directly",
            height=220,
            placeholder="Paste your call transcript here...",
            key="transcript_paste",
        )

        transcript = ""
        if uploaded:
            transcript = uploaded.read().decode("utf-8")
        elif transcript_text.strip():
            transcript = transcript_text.strip()

        if st.button("🔍 Analyze for Marketing Insights", use_container_width=True, disabled=not transcript):
            client = get_client()
            with st.spinner("Extracting marketing intelligence..."):
                st.session_state["analysis_result"] = analyze_transcript(client, transcript)
                st.session_state["last_transcript"] = transcript

        result = st.session_state.get("analysis_result")

        if result:
            # ── 1. CALL SNAPSHOT ─────────────────────────────────────────
            snap = result.get("call_snapshot", {})
            outcome = snap.get("outcome", "—")
            fit = snap.get("fit_score", "—")
            awareness = snap.get("awareness_level", "—")
            source = snap.get("what_brought_them_here", "—")
            summary = snap.get("one_line_summary", "—")

            outcome_color = "#2ecc71" if "Purchased" in outcome else "#e74c3c" if "Not" in outcome else "#f39c12"
            fit_color = "#2ecc71" if isinstance(fit, int) and fit >= 7 else "#f39c12" if isinstance(fit, int) and fit >= 5 else "#e74c3c"

            st.markdown(
                f"""<div style="background:#111a22;border:1px solid #2c3e50;border-radius:12px;
                padding:18px 22px;margin:16px 0;">
                <div style="font-size:13px;color:#7f8c8d;margin-bottom:10px;letter-spacing:1px;">CALL SNAPSHOT</div>
                <div style="font-size:16px;color:#ecf0f1;margin-bottom:14px;">{summary}</div>
                <div style="display:flex;gap:30px;flex-wrap:wrap;">
                  <div><span style="color:#7f8c8d;font-size:12px;">OUTCOME</span><br>
                    <span style="color:{outcome_color};font-weight:700;">{outcome}</span></div>
                  <div><span style="color:#7f8c8d;font-size:12px;">BUYER FIT</span><br>
                    <span style="color:{fit_color};font-weight:700;">{fit}/10</span></div>
                  <div><span style="color:#7f8c8d;font-size:12px;">AWARENESS LEVEL</span><br>
                    <span style="color:#3498db;font-weight:700;">{awareness}</span></div>
                  <div><span style="color:#7f8c8d;font-size:12px;">WHAT BROUGHT THEM</span><br>
                    <span style="color:#ecf0f1;">{source}</span></div>
                </div></div>""",
                unsafe_allow_html=True,
            )

            # ── 2. VOICE OF CUSTOMER ──────────────────────────────────────
            voc = result.get("voice_of_customer", {})
            with st.expander("🗣️ Voice of Customer — Use these words in your copy", expanded=True):
                st.caption("These are the prospect's exact words. Use them verbatim in ads, emails, and landing pages.")

                voc_cols = st.columns(2)
                categories = [
                    ("🔥 Pain — what's hurting them", "pain_quotes",    "#e74c3c", "#2a0d0d"),
                    ("✨ Desire — what they want",     "desire_quotes",  "#2ecc71", "#0d2a18"),
                    ("😨 Fear — what's holding them back", "fear_quotes","#f39c12", "#2a1e08"),
                    ("💬 Their language patterns",    "language_patterns","#3498db","#0d1e30"),
                ]
                for i, (label, key, color, bg) in enumerate(categories):
                    with voc_cols[i % 2]:
                        st.markdown(f"**{label}**")
                        for q in voc.get(key, []):
                            st.markdown(
                                f"""<div style="background:{bg};border-left:3px solid {color};
                                border-radius:6px;padding:10px 14px;margin:4px 0;
                                color:#ecf0f1;font-style:italic;">"{q}"</div>""",
                                unsafe_allow_html=True,
                            )
                        st.markdown("")

            # ── 3. OBJECTION INTELLIGENCE ─────────────────────────────────
            objections = result.get("objection_intelligence", [])
            with st.expander("🚧 Objection Intelligence — What to fix in your funnel", expanded=True):
                st.caption("Each objection on this call = a gap in your landing page, emails, or ads. Fix these upstream.")
                for obj in objections:
                    col1, col2, col3 = st.columns([2, 2, 3])
                    with col1:
                        st.markdown(
                            f"""<div style="background:#2a0d0d;border-left:4px solid #e74c3c;
                            border-radius:6px;padding:10px 12px;">
                            <div style="color:#e74c3c;font-size:11px;margin-bottom:4px;">THEY SAID</div>
                            <div style="color:#ecf0f1;font-size:13px;font-style:italic;">"{obj.get('objection','')}"</div>
                            </div>""",
                            unsafe_allow_html=True,
                        )
                    with col2:
                        st.markdown(
                            f"""<div style="background:#2a1e08;border-left:4px solid #f39c12;
                            border-radius:6px;padding:10px 12px;">
                            <div style="color:#f39c12;font-size:11px;margin-bottom:4px;">REAL MEANING</div>
                            <div style="color:#ecf0f1;font-size:13px;">{obj.get('real_meaning','')}</div>
                            </div>""",
                            unsafe_allow_html=True,
                        )
                    with col3:
                        st.markdown(
                            f"""<div style="background:#0d2a18;border-left:4px solid #2ecc71;
                            border-radius:6px;padding:10px 12px;">
                            <div style="color:#2ecc71;font-size:11px;margin-bottom:4px;">FUNNEL FIX</div>
                            <div style="color:#ecf0f1;font-size:13px;">{obj.get('funnel_fix','')}</div>
                            </div>""",
                            unsafe_allow_html=True,
                        )
                    st.markdown("")

            # ── 4. BUYING SIGNALS ─────────────────────────────────────────
            signals = result.get("buying_signals", [])
            with st.expander("📈 Buying Signals — What made them lean in", expanded=False):
                st.caption("These are the moments the prospect showed interest. The exact phrases belong in your ads and emails.")
                for sig in signals:
                    st.markdown(
                        f"""<div style="background:#0d1e30;border:1px solid #3498db;
                        border-radius:8px;padding:12px 16px;margin:6px 0;">
                        <div style="color:#7f8c8d;font-size:11px;">TRIGGER MOMENT</div>
                        <div style="color:#bdc3c7;font-size:13px;margin-bottom:8px;">{sig.get('moment','')}</div>
                        <div style="color:#3498db;font-size:15px;font-style:italic;">"{sig.get('exact_quote','')}"</div>
                        <div style="margin-top:8px;background:#1a2535;border-radius:4px;padding:6px 10px;">
                        <span style="color:#2ecc71;font-size:11px;">USE IN COPY → </span>
                        <span style="color:#ecf0f1;font-size:13px;">{sig.get('marketing_use','')}</span>
                        </div></div>""",
                        unsafe_allow_html=True,
                    )

            # ── 5. CONTENT GAPS ───────────────────────────────────────────
            gaps = result.get("content_gaps", [])
            with st.expander("🕳️ Content Gaps — What your funnel is missing", expanded=False):
                st.caption("Things the prospect was confused about that should have been addressed BEFORE this call.")
                for gap in gaps:
                    where = gap.get("should_have_been_covered_in", "")
                    badge_color = {
                        "Landing page": "#9b59b6",
                        "Email sequence": "#3498db",
                        "Ad copy": "#e74c3c",
                        "FAQ": "#f39c12",
                    }.get(where, "#7f8c8d")
                    st.markdown(
                        f"""<div style="background:#111a22;border:1px solid #2c3e50;
                        border-radius:8px;padding:12px 16px;margin:6px 0;">
                        <span style="background:{badge_color};color:#fff;font-size:11px;
                        padding:2px 8px;border-radius:10px;margin-right:8px;">{where}</span>
                        <span style="color:#ecf0f1;font-size:14px;">{gap.get('gap','')}</span>
                        <div style="margin-top:8px;color:#2ecc71;font-size:13px;">
                        → {gap.get('suggested_fix','')}</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )

            # ── 6. ICP SIGNALS ────────────────────────────────────────────
            icp = result.get("icp_signals", {})
            with st.expander("🎯 ICP Signals — What this call tells you about your buyer", expanded=False):
                icp_items = [
                    ("Role / Situation",   icp.get("role_situation", "—")),
                    ("Their Goal",         icp.get("goal", "—")),
                    ("Timeline / Urgency", icp.get("timeline", "—")),
                    ("Budget Signal",      icp.get("budget_signal", "—")),
                    ("Channels They Trust",icp.get("channels_they_trust", "—")),
                ]
                for label, val in icp_items:
                    st.markdown(
                        f"""<div style="display:flex;gap:16px;padding:10px 0;
                        border-bottom:1px solid #1c2833;">
                        <div style="color:#7f8c8d;font-size:12px;min-width:160px;">{label}</div>
                        <div style="color:#ecf0f1;">{val}</div></div>""",
                        unsafe_allow_html=True,
                    )

            # ── 7. COPY BANK ──────────────────────────────────────────────
            copy_bank = result.get("copy_bank", [])
            with st.expander("✍️ Copy Bank — Ready-to-use lines from this call", expanded=True):
                st.caption("Derived from the prospect's own language. Drop these into your next campaign.")
                for item in copy_bank:
                    ctype = item.get("type", "")
                    copy  = item.get("copy", "")
                    why   = item.get("why_it_works", "")
                    type_color = {
                        "Email subject line":      "#3498db",
                        "Ad headline":             "#e74c3c",
                        "Landing page headline":   "#9b59b6",
                        "CTA":                     "#2ecc71",
                    }.get(ctype, "#7f8c8d")
                    st.markdown(
                        f"""<div style="background:#111a22;border:1px solid #2c3e50;
                        border-radius:8px;padding:12px 16px;margin:6px 0;">
                        <span style="background:{type_color};color:#fff;font-size:11px;
                        padding:2px 8px;border-radius:10px;">{ctype}</span>
                        <div style="color:#ffffff;font-size:16px;font-weight:600;margin:8px 0;">
                        {copy}</div>
                        <div style="color:#7f8c8d;font-size:12px;">{why}</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )

            # ── 8. ACTION ITEMS ───────────────────────────────────────────
            actions = result.get("marketing_action_items", [])
            if actions:
                st.markdown("---")
                st.markdown("### ✅ Marketing Action Items from this Call")
                for i, action in enumerate(actions, 1):
                    st.markdown(
                        f"""<div style="background:#0d2a18;border-left:4px solid #2ecc71;
                        border-radius:6px;padding:10px 16px;margin:5px 0;color:#ecf0f1;">
                        <strong style="color:#2ecc71;">{i}.</strong> {action}</div>""",
                        unsafe_allow_html=True,
                    )

    # ------------------------------------------------------------------
    # TAB 2 — Inbound Closer Script
    # ------------------------------------------------------------------
    with tab2:
        # Flow banner
        st.markdown(
            """
            <div style="background:#1a2535;border:1px solid #3498db;border-radius:10px;padding:14px 18px;margin-bottom:18px;">
            <strong style="color:#3498db;font-size:15px;">📌 How this works</strong><br><br>
            <span style="color:#ecf0f1;">
            <b style="color:#2ecc71;">1. They found your course page</b> &nbsp;→&nbsp;
            <b style="color:#f39c12;">2. They booked a call themselves</b> &nbsp;→&nbsp;
            <b style="color:#e74c3c;">3. You run THIS script on the call</b>
            </span><br><br>
            <span style="color:#bdc3c7;font-size:13px;">
            ⚠️ <b>You are NOT calling them.</b> They came to you.
            Your only job on this call: find out what's confusing or holding them back — then clear it and close the purchase.
            </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Build your Inbound Closer Script")

        input_mode = st.radio(
            "How do you want to describe your course?",
            options=["🔗 Paste a page URL (auto-scrape)", "✏️ Describe it manually"],
            horizontal=True,
            key="pitch_input_mode",
        )

        page_text_manual = ""
        url = ""

        if input_mode == "🔗 Paste a page URL (auto-scrape)":
            url = st.text_input(
                "Course / product page URL",
                placeholder="https://yourcourse.com/enroll",
            )
        else:
            page_text_manual = st.text_area(
                "Describe your course — what it is, who it's for, what outcome it delivers",
                placeholder=(
                    "e.g. A 6-week online program for first-time founders who want to build "
                    "their first $10k MRR SaaS. Covers validation, pricing, outreach, and closing."
                ),
                height=150,
                key="pitch_manual_desc",
            )

        use_transcript = st.checkbox(
            "Also use transcript from Tab 1 as context (if analyzed)",
            value=bool(st.session_state.get("last_transcript")),
        )

        btn_ready = bool(url.strip()) if input_mode == "🔗 Paste a page URL (auto-scrape)" else bool(page_text_manual.strip())

        if st.button("🚀 Generate Closer Script", use_container_width=True, disabled=not btn_ready):
            client = get_client()

            if input_mode == "🔗 Paste a page URL (auto-scrape)":
                with st.spinner("Scraping page..."):
                    page_text = scrape_url(url)
                if not page_text:
                    st.stop()
            else:
                page_text = page_text_manual.strip()

            transcript_ctx = st.session_state.get("last_transcript", "") if use_transcript else ""

            with st.spinner("Writing your closer script..."):
                pitch = generate_pitch(client, page_text, transcript_ctx)

            st.session_state["generated_pitch"] = pitch

        if st.session_state.get("generated_pitch"):
            st.divider()

            view_tab, edit_tab = st.tabs(["📖 Preview", "✏️ Edit"])

            with view_tab:
                st.markdown(
                    """
                    <style>
                    .script-wrapper {
                        background: #0f1923;
                        border: 1px solid #2c3e50;
                        border-radius: 12px;
                        padding: 2rem 2.5rem;
                        line-height: 1.8;
                    }
                    .script-wrapper h2 {
                        color: #3498db;
                        border-bottom: 2px solid #2c3e50;
                        padding-bottom: 8px;
                        margin-bottom: 1.2rem;
                    }
                    .script-wrapper h3 {
                        color: #2ecc71;
                        margin-top: 1.6rem;
                        margin-bottom: 0.5rem;
                    }
                    .script-wrapper p { color: #ecf0f1; }
                    .script-wrapper em { color: #f39c12; font-style: normal; }
                    .script-wrapper strong { color: #ffffff; }
                    .script-wrapper ul li { color: #bdc3c7; margin-bottom: 4px; }
                    .script-wrapper table {
                        width: 100%;
                        border-collapse: collapse;
                        margin-top: 1rem;
                    }
                    .script-wrapper th {
                        background: #1a2535;
                        color: #3498db;
                        padding: 10px 14px;
                        text-align: left;
                        border: 1px solid #2c3e50;
                    }
                    .script-wrapper td {
                        padding: 9px 14px;
                        border: 1px solid #2c3e50;
                        color: #ecf0f1;
                        vertical-align: top;
                    }
                    .script-wrapper tr:nth-child(even) td { background: #141e28; }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )

                # Render section by section with coloured phase cards
                import re
                raw = st.session_state["generated_pitch"]
                sections = re.split(r"(?=\n### )", raw)

                for section in sections:
                    section = section.strip()
                    if not section:
                        continue

                    # Top-level heading (## …) — blue banner
                    if section.startswith("## "):
                        lines = section.split("\n", 1)
                        heading = lines[0].replace("## ", "").strip()
                        body = lines[1].strip() if len(lines) > 1 else ""
                        st.markdown(
                            f"""<div style="background:#1a2535;border-left:5px solid #3498db;
                            border-radius:8px;padding:14px 18px;margin:10px 0 4px 0;">
                            <span style="color:#3498db;font-size:20px;font-weight:700;">{heading}</span>
                            </div>""",
                            unsafe_allow_html=True,
                        )
                        if body:
                            st.markdown(body)

                    # Time-block section (### 0:00 – …)
                    elif section.startswith("### "):
                        lines = section.split("\n", 1)
                        heading = lines[0].replace("### ", "").strip()
                        body = lines[1].strip() if len(lines) > 1 else ""

                        # Colour by phase keyword
                        if any(k in heading for k in ["Open", "Warm"]):
                            bdr, bg = "#3498db", "#0d1e30"
                        elif any(k in heading for k in ["Discovery", "Find"]):
                            bdr, bg = "#1abc9c", "#0d2520"
                        elif any(k in heading for k in ["Gap", "Confirm"]):
                            bdr, bg = "#9b59b6", "#1a1030"
                        elif any(k in heading for k in ["Bridge", "Pitch", "Value"]):
                            bdr, bg = "#f39c12", "#2a1e08"
                        elif any(k in heading for k in ["Objection", "Handle"]):
                            bdr, bg = "#e74c3c", "#2a0d0d"
                        elif any(k in heading for k in ["Close", "Next"]):
                            bdr, bg = "#2ecc71", "#0d2a18"
                        else:
                            bdr, bg = "#7f8c8d", "#1c2833"

                        st.markdown(
                            f"""<div style="background:{bg};border-left:5px solid {bdr};
                            border-radius:8px;padding:12px 18px;margin:14px 0 6px 0;">
                            <span style="color:{bdr};font-size:16px;font-weight:700;">{heading}</span>
                            </div>""",
                            unsafe_allow_html=True,
                        )

                        # Split script body from *Why this works* line
                        if "*Why this works" in body:
                            parts = body.split("*Why this works", 1)
                            script_body = parts[0].strip()
                            why = "*Why this works" + parts[1]
                            st.markdown(
                                f"""<div style="background:#111a22;border-radius:6px;
                                padding:14px 18px;color:#ecf0f1;line-height:1.8;
                                font-size:15px;">{script_body}</div>""",
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                f"""<div style="background:#2c2c1a;border-left:3px solid #f39c12;
                                border-radius:6px;padding:10px 14px;margin-top:4px;
                                color:#f0c040;font-size:13px;">{why}</div>""",
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                f"""<div style="background:#111a22;border-radius:6px;
                                padding:14px 18px;color:#ecf0f1;line-height:1.8;
                                font-size:15px;">{body}</div>""",
                                unsafe_allow_html=True,
                            )
                    else:
                        # Everything else (objection table, tips) — render as markdown
                        st.markdown(section)

                st.divider()
                st.download_button(
                    "⬇️ Download Script (.txt)",
                    data=st.session_state["generated_pitch"],
                    file_name="closer_script.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

            with edit_tab:
                edited = st.text_area(
                    "Edit the script — changes will reflect in the preview after you rerun",
                    value=st.session_state["generated_pitch"],
                    height=700,
                    key="pitch_editor",
                )
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("💾 Save edits to preview", use_container_width=True):
                        st.session_state["generated_pitch"] = edited
                        st.rerun()
                with col2:
                    st.download_button(
                        "⬇️ Download edited script",
                        data=edited,
                        file_name="closer_script.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )

    # ------------------------------------------------------------------
    # TAB 3 — Demo Call Transcript
    # ------------------------------------------------------------------
    with tab3:
        st.markdown("### 🎓 Generate a Demo 15-Min Sales Call")
        st.caption(
            "Tell us what you sell and who you sell to. "
            "We'll generate a complete, annotated call your team can study, copy, and download."
        )

        col_a, col_b = st.columns(2)
        with col_a:
            product = st.text_input(
                "What is your product / service?",
                placeholder="e.g. AI-powered CRM for SMBs",
                key="demo_product",
            )
            industry = st.text_input(
                "Target industry / buyer",
                placeholder="e.g. SaaS companies, 10-50 employees",
                key="demo_industry",
            )
        with col_b:
            pain_point = st.text_area(
                "Main pain point you solve",
                placeholder="e.g. Sales reps waste 3 hours/day on manual data entry and miss follow-ups",
                height=120,
                key="demo_pain",
            )

        ready = bool(product.strip() and industry.strip() and pain_point.strip())

        if st.button("🎬 Generate Demo Transcript", use_container_width=True, disabled=not ready):
            client = get_client()
            with st.spinner("Writing your demo call — this takes ~20 seconds..."):
                st.session_state["demo_transcript"] = generate_demo_transcript(
                    client, product.strip(), industry.strip(), pain_point.strip()
                )

        demo_data = st.session_state.get("demo_transcript")

        if demo_data:
            st.divider()

            # Header
            st.markdown(f"## 📞 {demo_data.get('call_title', 'Demo Sales Call')}")
            st.info(demo_data.get("context", ""))

            # Phase colour map
            phase_colors = {
                "Opening":             "#1a3a5c",
                "Discovery":           "#1a3a3a",
                "Pitch":               "#2a2a1a",
                "Objection Handling":  "#3a1a1a",
                "Close":               "#1a2a1a",
            }
            phase_border = {
                "Opening":             "#3498db",
                "Discovery":           "#1abc9c",
                "Pitch":               "#f39c12",
                "Objection Handling":  "#e74c3c",
                "Close":               "#2ecc71",
            }

            current_phase = ""
            for turn in demo_data.get("timeline", []):
                ts        = turn.get("timestamp", "")
                phase     = turn.get("phase", "")
                speaker   = turn.get("speaker", "")
                line      = turn.get("line", "")
                note      = turn.get("annotation", "")

                # Phase header when it changes
                if phase != current_phase:
                    current_phase = phase
                    bg    = phase_colors.get(phase, "#1c2833")
                    bdr   = phase_border.get(phase, "#7f8c8d")
                    st.markdown(
                        f"""<div style="background:{bg};border:1px solid {bdr};
                        border-radius:8px;padding:8px 14px;margin:18px 0 6px 0;">
                        <strong style="color:{bdr};font-size:13px;letter-spacing:1px;">
                        ▶ {phase.upper()}</strong></div>""",
                        unsafe_allow_html=True,
                    )

                if speaker == "Rep":
                    bubble_bg  = "#1a3525"
                    border_col = "#2ecc71"
                    label      = f"🎤 Rep  <span style='color:#7f8c8d;font-size:12px;'>{ts}</span>"
                else:
                    bubble_bg  = "#1a2535"
                    border_col = "#3498db"
                    label      = f"🙋 Prospect  <span style='color:#7f8c8d;font-size:12px;'>{ts}</span>"

                st.markdown(
                    f"""<div style="background:{bubble_bg};border-left:4px solid {border_col};
                    padding:10px 14px;border-radius:6px;margin:4px 0 2px 0;">
                    <strong style="color:{border_col};">{label}</strong><br>
                    <span style="color:#ecf0f1;font-size:15px;">{line}</span>
                    </div>""",
                    unsafe_allow_html=True,
                )

                if note:
                    st.markdown(
                        f"""<div style="background:#2c2c1a;border-left:3px solid #f39c12;
                        padding:5px 12px;border-radius:0 0 6px 6px;margin:0 0 8px 4px;">
                        <span style="color:#f39c12;font-size:12px;">💡 {note}</span>
                        </div>""",
                        unsafe_allow_html=True,
                    )

            # Key techniques
            techniques = demo_data.get("key_techniques", [])
            if techniques:
                st.markdown("---")
                st.markdown("### 🧠 Key Techniques Used in This Call")
                for t in techniques:
                    with st.expander(f"**{t.get('technique', '')}**"):
                        st.write(t.get("description", ""))

            # What made it work
            summary = demo_data.get("what_made_this_call_work", "")
            if summary:
                st.markdown("---")
                st.success(f"**What made this call work:**\n\n{summary}")

            # Download full transcript as plain text
            st.markdown("---")
            plain_lines = []
            plain_lines.append(demo_data.get("call_title", "Demo Sales Call"))
            plain_lines.append("=" * 60)
            plain_lines.append(demo_data.get("context", ""))
            plain_lines.append("")
            current_phase = ""
            for turn in demo_data.get("timeline", []):
                if turn.get("phase") != current_phase:
                    current_phase = turn.get("phase", "")
                    plain_lines.append(f"\n--- {current_phase.upper()} ---")
                plain_lines.append(
                    f"[{turn.get('timestamp','')}] {turn.get('speaker','')}: {turn.get('line','')}"
                )
                if turn.get("annotation"):
                    plain_lines.append(f"  >> COACH NOTE: {turn.get('annotation')}")
            plain_lines.append("\n\nKEY TECHNIQUES")
            plain_lines.append("-" * 40)
            for t in demo_data.get("key_techniques", []):
                plain_lines.append(f"• {t.get('technique')}: {t.get('description')}")
            plain_lines.append(f"\n\nWHAT MADE THIS CALL WORK\n{demo_data.get('what_made_this_call_work','')}")

            st.download_button(
                "⬇️ Download Full Transcript (.txt)",
                data="\n".join(plain_lines),
                file_name="demo_sales_call.txt",
                mime="text/plain",
                use_container_width=True,
            )

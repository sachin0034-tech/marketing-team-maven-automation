import streamlit as st
import json
from openai import OpenAI


# ---------------------------------------------------------------------------
# Layer definitions — used for display and prompt construction
# ---------------------------------------------------------------------------

LAYERS = [
    {
        "id": 1,
        "name": "Hook & First Line",
        "icon": "🎯",
        "checkpoints": [
            (1,  "Creates curiosity gap or tension in the first line"),
            (2,  "First line is under 150 chars (no 'see more' cutoff on mobile)"),
            (3,  "Does NOT start with 'I'"),
            (4,  "Contains a bold claim, contrarian take, or surprising stat"),
            (5,  "Would stop the scroll — compelling enough to read on"),
        ],
    },
    {
        "id": 2,
        "name": "Structure & Readability",
        "icon": "📐",
        "checkpoints": [
            (6,  "Sentences are short (avg under 15 words)"),
            (7,  "White space between every paragraph"),
            (8,  "Clear narrative arc: Hook → Body → Insight → CTA"),
            (9,  "Optimal length: 900–1300 characters for a text post"),
            (10, "No wall-of-text blocks (max 3–4 lines per paragraph)"),
        ],
    },
    {
        "id": 3,
        "name": "Content Quality",
        "icon": "💎",
        "checkpoints": [
            (11, "Contains a specific insight — not generic advice"),
            (12, "Includes a personal story or concrete example"),
            (13, "Teaches, inspires, or entertains — not just an announcement"),
            (14, "Claim is backed by data, experience, or a real case"),
            (15, "Original thought — not a rephrased platitude"),
        ],
    },
    {
        "id": 4,
        "name": "Engagement Signals",
        "icon": "🔥",
        "checkpoints": [
            (16, "CTA asks a question or invites the audience's opinion"),
            (17, "Has a controversial or polarising angle (safe-controversy)"),
            (18, "Tags or references people/brands meaningfully (if applicable)"),
            (19, "Hashtags are relevant and under 5 (not spam)"),
            (20, "Ends with momentum — NOT 'thanks for reading'"),
        ],
    },
    {
        "id": 5,
        "name": "Tone & Brand Voice",
        "icon": "🎙️",
        "checkpoints": [
            (21, "Matches the author's defined voice (formal / casual / bold)"),
            (22, "No corporate jargon or buzzword overload"),
            (23, "Reads like a human — not a template or AI fill-in"),
            (24, "Tone is consistent throughout (no jarring shifts)"),
        ],
    },
    {
        "id": 6,
        "name": "Algorithm Fit",
        "icon": "⚡",
        "checkpoints": [
            (25, "No external links in the post body (kills reach)"),
            (26, "Format matches LinkedIn algorithm preference (text-first)"),
            (27, "No competitor mentions that could suppress distribution"),
            (28, "Post is self-contained — no 'link in first comment' hooks"),
        ],
    },
]

# Flat lookup: checkpoint_id → label
CHECKPOINT_LABELS = {cp_id: label for layer in LAYERS for cp_id, label in layer["checkpoints"]}
TOTAL_CHECKPOINTS = len(CHECKPOINT_LABELS)  # 28


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

def _build_system_prompt() -> str:
    layers_text = ""
    for layer in LAYERS:
        layers_text += f"\n### Layer {layer['id']} — {layer['name']}\n"
        for cp_id, label in layer["checkpoints"]:
            layers_text += f"  - Checkpoint {cp_id}: {label}\n"

    return f"""You are an expert LinkedIn content strategist and growth coach.

Evaluate the provided LinkedIn post against EVERY checkpoint in the 6-layer framework below. Be strict and specific — a checkpoint only PASSES if it is clearly and unambiguously satisfied.

{layers_text}

For EACH checkpoint return:
- "id": the checkpoint number
- "status": "PASS" or "FAIL"
- "reason": one sentence explaining the verdict, citing specific text from the post where possible
- "suggestion": (FAIL only) a specific, actionable fix — not generic advice. Name exactly what to add, change, or remove.

Also return:
- "overall_score": integer (number of checkpoints that PASS, out of {TOTAL_CHECKPOINTS})
- "best_posting_time": advisory string — best day + time slot to post based on content type (e.g. "Tuesday 8–9 AM — professional insight performs well mid-week")
- "improvement_ideas": list of 3–5 prioritised ideas to elevate this post. Each idea should:
    - name the exact thing to add/change
    - explain WHICH checkpoint it will flip from FAIL to PASS
    - be specific to this post's content (not generic)

Return ONLY a valid JSON object with this exact structure:
{{
  "layers": [
    {{
      "layer_id": 1,
      "layer_name": "Hook & First Line",
      "checkpoints": [
        {{
          "id": 1,
          "status": "PASS",
          "reason": "...",
          "suggestion": ""
        }},
        ...
      ]
    }},
    ...
  ],
  "overall_score": <int>,
  "best_posting_time": "...",
  "improvement_ideas": [
    {{
      "priority": 1,
      "idea": "...",
      "fixes_checkpoint_id": <int>,
      "fixes_checkpoint_label": "..."
    }},
    ...
  ]
}}"""


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def get_client() -> OpenAI:
    key = st.session_state.get("openai_api_key", "").strip()
    if not key:
        st.error("Add your OpenAI API key in the sidebar first.")
        st.stop()
    return OpenAI(api_key=key)


def review_linkedin_post(client: OpenAI, post: str) -> dict:
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _build_system_prompt()},
            {"role": "user", "content": f"LinkedIn post to review:\n\n{post}"},
        ],
        temperature=0.2,
    )
    return json.loads(resp.choices[0].message.content)


def generate_fixed_post(client: OpenAI, original_post: str, failing: list) -> str:
    """Rewrite the post so it passes every failing checkpoint."""
    failures_text = "\n".join(
        f"- Checkpoint #{cp.get('id')} ({CHECKPOINT_LABELS.get(cp.get('id'), '')}): "
        f"{cp.get('reason', '')} | Fix needed: {cp.get('suggestion', '')}"
        for cp in failing
    )
    system_prompt = (
        "You are an expert LinkedIn ghostwriter. "
        "Rewrite the provided LinkedIn post so it passes EVERY failing checkpoint listed below. "
        "Keep the author's core idea and voice intact — do not change the topic or story. "
        "The rewritten post must:\n"
        "- Start with a hook that is NOT 'I', is under 150 chars, and creates curiosity or tension\n"
        "- Use short sentences (avg under 15 words) with white space between every paragraph\n"
        "- Be 900-1300 characters total\n"
        "- End with a question CTA that invites comments (not 'thanks for reading')\n"
        "- Contain no external links\n"
        "- Include 3-5 relevant hashtags at the end\n\n"
        "FAILING CHECKPOINTS TO FIX:\n"
        f"{failures_text}\n\n"
        "Return ONLY the rewritten post text — no intro, no commentary, no quotes around it."
    )
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Original post:\n\n{original_post}"},
        ],
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Score colour helper
# ---------------------------------------------------------------------------

def score_color(score: int, total: int = TOTAL_CHECKPOINTS) -> str:
    pct = score / total
    if pct >= 0.8:
        return "#2ecc71"
    if pct >= 0.6:
        return "#f39c12"
    return "#e74c3c"


# ---------------------------------------------------------------------------
# UI components
# ---------------------------------------------------------------------------

def _checkpoint_card(cp: dict) -> None:
    cp_id      = cp.get("id", "")
    label      = CHECKPOINT_LABELS.get(cp_id, cp.get("label", f"Checkpoint {cp_id}"))
    status     = cp.get("status", "FAIL")
    reason     = cp.get("reason", "")
    suggestion = cp.get("suggestion", "")
    is_pass    = status == "PASS"

    if is_pass:
        st.success(f"**{label}**  \n{reason}", icon="✅")
    else:
        st.error(f"**{label}**  \n{reason}", icon="❌")
        if suggestion:
            st.info(f"**Suggestion:** {suggestion}", icon="💡")


def _layer_section(layer_result: dict) -> None:
    layer_id    = layer_result.get("layer_id", "")
    layer_name  = layer_result.get("layer_name", "")
    checkpoints = layer_result.get("checkpoints", [])

    passing   = [c for c in checkpoints if c.get("status") == "PASS"]
    failing   = [c for c in checkpoints if c.get("status") == "FAIL"]
    total     = len(checkpoints)
    pct       = len(passing) / total if total else 0
    bar_color = "#2ecc71" if pct >= 0.8 else "#f39c12" if pct >= 0.6 else "#e74c3c"
    icon      = next((l["icon"] for l in LAYERS if l["id"] == layer_id), "📌")

    # Header row using columns — no nested HTML
    col_title, col_stats = st.columns([3, 1])
    with col_title:
        st.markdown(
            f"<p style='margin:0;font-size:16px;font-weight:700;color:#ecf0f1;'>"
            f"{icon} Layer {layer_id} — {layer_name}</p>",
            unsafe_allow_html=True,
        )
    with col_stats:
        st.markdown(
            f"<p style='margin:0;text-align:right;font-size:14px;font-weight:700;color:{bar_color};'>"
            f"{len(passing)}/{total}</p>",
            unsafe_allow_html=True,
        )

    # Progress bar
    st.progress(pct)

    with st.expander(f"View all {total} checkpoints", expanded=(len(failing) > 0)):
        for cp in checkpoints:
            _checkpoint_card(cp)

    st.divider()


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render():
    st.title("📋 LinkedIn Post Review Agent")
    st.caption(
        "Paste your LinkedIn draft. The agent evaluates it across **6 layers / 28 checkpoints** — "
        "flags what's missing, scores each layer, and gives you prioritised suggestions to fix it."
    )

    st.divider()

    if True:

        # ── REFERENCE PANELS ─────────────────────────────────────────
        ref_col, sample_col = st.columns(2)

        with ref_col:
            with st.expander("📋 View all 28 checkpoints by layer", expanded=False):
                for layer in LAYERS:
                    st.markdown(f"**{layer['icon']} Layer {layer['id']} — {layer['name']}**")
                    for cp_id, label in layer["checkpoints"]:
                        st.markdown(f"- `#{cp_id}` {label}")

        with sample_col:
            with st.expander("💡 See a sample post that passes all 28 checkpoints", expanded=True):
                SAMPLE_POST = """Most AI agents fail in production. Not because the model is weak.

Because nobody defined what "done" looks like.

I've shipped agents across AWS Bedrock and Google Cloud, and consulted with 40+ enterprise teams on broken deployments.

The failure pattern is identical every time:

→ Agent runs. Output looks reasonable. Team ships.
→ Three weeks later, edge cases stack up.
→ Nobody can explain specific decisions. Trust collapses.

The fix isn't better prompts.

It's evaluation-first design.

Before writing a single line of agent code, lock down:
• What does PASS look like? (measurable — not vibes)
• What does FAIL look like? (named failure modes)
• Who reviews edge cases? (human-in-the-loop at <5% of volume)

Teams that ran structured evals before deployment cut production incidents by 60%. We saw this repeatedly at Bedrock.

Most teams treat eval as the final step.
The best teams treat it as step one.

Uncomfortable truth: if you can't describe failure, you're not ready to ship.

What's your biggest blocker when deploying AI agents in production?
Drop it below — I read every reply.

#AIAgents #ProductManagement #GenerativeAI #MachineLearning"""

                st.code(SAMPLE_POST, language=None)

                st.markdown("**Why it passes — layer by layer:**")

                SAMPLE_ANNOTATIONS = [
                    ("🎯", "Layer 1 - Hook",
                     'Starts with bold claim "Most AI agents fail..." - not "I", under 65 chars, '
                     "contrarian follow-up creates tension. Passes #1 #2 #3 #4 #5."),
                    ("📐", "Layer 2 - Structure",
                     "Short sentences, blank line between every block, Hook -> Body -> Insight -> CTA arc, "
                     "~1,080 chars, max 3 lines per block. Passes #6 #7 #8 #9 #10."),
                    ("💎", "Layer 3 - Content",
                     '"Evaluation-first design" with a 3-point framework (specific insight), '
                     "AWS/GCP experience named (story), 60% stat (backed claim), "
                     '"describe failure before you ship" (original). Passes #11 #12 #13 #14 #15.'),
                    ("🔥", "Layer 4 - Engagement",
                     'Ends with direct question + "I read every reply" (momentum CTA), '
                     '"fix isn\'t better prompts" is polarising, 4 relevant hashtags. '
                     "Passes #16 #17 #19 #20."),
                    ("🎙️", "Layer 5 - Voice",
                     "Direct, no buzzwords, reads like a practitioner, consistent tone throughout. "
                     "Passes #21 #22 #23 #24."),
                    ("⚡", "Layer 6 - Algorithm",
                     "Zero links in body, pure text post, no competitor names, fully self-contained. "
                     "Passes #25 #26 #27 #28."),
                ]

                for icon, title, note in SAMPLE_ANNOTATIONS:
                    st.info(f"**{icon} {title}**  \n{note}")

                if st.button("Use this as my starting point", key="load_sample_post"):
                    st.session_state["linkedin_post_input"] = SAMPLE_POST
                    st.rerun()

        st.divider()

        post_text = st.text_area(
            "Paste your LinkedIn post here",
            height=300,
            placeholder="Start typing or paste your draft post…",
            key="linkedin_post_input",
        )

        col_btn, col_clear = st.columns([4, 1])
        with col_btn:
            analyze = st.button(
                "🔍 Run 6-Layer Review",
                use_container_width=True,
                disabled=not post_text.strip(),
            )
        with col_clear:
            if st.button("Clear", use_container_width=True):
                st.session_state.pop("linkedin_review_result", None)
                st.session_state.pop("fixed_post", None)
                st.rerun()

        if analyze and post_text.strip():
            client = get_client()
            with st.spinner("Running all 28 checkpoints across 6 layers..."):
                try:
                    result = review_linkedin_post(client, post_text.strip())
                    st.session_state["linkedin_review_result"] = result
                except Exception as e:
                    err = str(e)
                    if "insufficient_quota" in err or "429" in err:
                        st.error("OpenAI quota exceeded. Please check your billing at platform.openai.com and top up your account.")
                    elif "Illegal header" in err or "LocalProtocolError" in err:
                        st.error("Invalid API key (possible trailing spaces). Please re-paste your key in the sidebar.")
                    else:
                        st.error(f"API error: {err}")

        result = st.session_state.get("linkedin_review_result")

        if result:
            layers_data    = result.get("layers", [])
            score          = result.get("overall_score", 0)
            best_time      = result.get("best_posting_time", "—")
            ideas          = result.get("improvement_ideas", [])

            # Flatten all checkpoints for global pass/fail counts
            all_cps  = [cp for layer in layers_data for cp in layer.get("checkpoints", [])]
            passing  = [c for c in all_cps if c.get("status") == "PASS"]
            failing  = [c for c in all_cps if c.get("status") == "FAIL"]
            total    = len(all_cps) or TOTAL_CHECKPOINTS
            s_color  = score_color(score, total)

            # ── SCORE BANNER ──────────────────────────────────────────────
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Overall Score", f"{score}/{total}")
            m2.metric("Passing", len(passing))
            m3.metric("Need Work", len(failing))
            m4.metric("Best Time to Post", best_time, help="Suggested posting window based on content type")
            st.progress(score / total)

            # ── LAYER-BY-LAYER BREAKDOWN ──────────────────────────────────
            st.markdown("### Layer-by-Layer Breakdown")

            for layer_result in layers_data:
                _layer_section(layer_result)

            # ── WHAT'S MISSING — quick reference ─────────────────────────
            if failing:
                st.markdown("---")
                st.markdown(f"#### ❌ {len(failing)} Checkpoints Need Work")
                for cp in failing:
                    cp_id = cp.get("id", "")
                    label = CHECKPOINT_LABELS.get(cp_id, f"Checkpoint {cp_id}")
                    sugg  = cp.get("suggestion", "")
                    st.error(f"**#{cp_id} — {label}**", icon="❌")
                    if sugg:
                        st.info(f"**Suggestion:** {sugg}", icon="💡")

                # ── GENERATE FIXED POST ───────────────────────────────────
                st.markdown("---")
                st.markdown("#### ✍️ Get a Ready-to-Post LinkedIn Version")
                st.caption(
                    "AI will rewrite your post, fixing every failing checkpoint — "
                    "keeps your idea, your story, your voice. Just paste and post."
                )

                if st.button("Generate Fixed Post", use_container_width=True, key="gen_fixed"):
                    client = get_client()
                    with st.spinner("Rewriting your post to pass all checkpoints..."):
                        try:
                            fixed = generate_fixed_post(client, post_text.strip(), failing)
                            st.session_state["fixed_post"] = fixed
                        except Exception as e:
                            err = str(e)
                            if "insufficient_quota" in err or "429" in err:
                                st.error("OpenAI quota exceeded. Please top up at platform.openai.com.")
                            else:
                                st.error(f"Error: {err}")

                fixed_post = st.session_state.get("fixed_post")
                if fixed_post:
                    st.success("**Fixed post — ready to copy and paste into LinkedIn:**")
                    st.code(fixed_post, language=None)
                    st.caption(f"Character count: {len(fixed_post)}")

            # ── IMPROVEMENT IDEAS ─────────────────────────────────────────
            if ideas:
                st.markdown("---")
                st.markdown("#### 💡 Prioritised Improvement Ideas")
                st.caption("Fix these in order — each one flips a failing checkpoint to passing.")

                for idea_obj in ideas:
                    priority    = idea_obj.get("priority", "")
                    idea        = idea_obj.get("idea", "")
                    fixes_id    = idea_obj.get("fixes_checkpoint_id", "")
                    fixes_label = idea_obj.get("fixes_checkpoint_label", "")
                    tag = f" — Fixes #{fixes_id}: {fixes_label}" if fixes_id else ""
                    st.warning(f"**{priority}.** {idea}  \n`{tag}`" if tag else f"**{priority}.** {idea}", icon="💡")


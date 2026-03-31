"""
panels/ai_radiologist_panel.py
==============================
AI Radiologist Assistant — Clinical QC reporting panel.

Supports multiple LLM providers:
  • Groq AI (LLaMA 4 Scout) — faster, free tier available
  • Google Gemini 2.0 Flash — advanced reasoning, multimodal

Use cases:
  • MRI Image Analysis (visual QC + metrics context)
  • MRIQC Metric Interpretation (quantitative QC)
  • Clinical follow-up Q&A based on generated reports
"""

from __future__ import annotations

import io
import re
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from loaders.nifti_loader import load_nifti, normalize_slice, extract_slice
from services.ai_api import (
    analyze_mri_slice,
    explain_iqm,
    followup_chat,
    slice_to_png_bytes,
    get_ai_provider,
    set_ai_provider,
)

_AXIS_NAMES = {0: "Sagittal (X)", 1: "Coronal (Y)", 2: "Axial (Z)"}
_AVAILABLE_PROVIDERS = ["groq", "gemini"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_verdict(text: str) -> str:
    """Extract PASS / WARN / FAIL from structured response."""
    m = re.search(r"\*\*QC VERDICT:\*\*\s*(PASS|WARN|FAIL)", text, re.IGNORECASE)
    return m.group(1).upper() if m else "UNKNOWN"


def _verdict_badge(verdict: str) -> str:
    colors = {
        "PASS":    ("#3ddc84", "#0d2a1a", "#1e5c38", "✅"),
        "WARN":    ("#ffa726", "#2a1e0d", "#5c3e1e", "⚠️"),
        "FAIL":    ("#ff6b6b", "#2a0d0d", "#5c1e1e", "🚫"),
        "UNKNOWN": ("#8fa3bc", "#111827", "#1e2d45", "❓"),
    }
    fg, bg, border, icon = colors.get(verdict, colors["UNKNOWN"])
    return (
        f'<span style="display:inline-block;padding:6px 20px;border-radius:4px;'
        f'background:{bg};color:{fg};border:1px solid {border};'
        f'font-family:JetBrains Mono,monospace;font-size:1.05rem;font-weight:700;'
        f'letter-spacing:2px;">{icon} {verdict}</span>'
    )


def _render_clinical_report(response: str, title: str = "Clinical QC Report") -> None:
    """Display structured AI response as a professional clinical report card."""
    verdict = _parse_verdict(response)
    provider = get_ai_provider().upper()

    # Remove the raw **QC VERDICT:** line — we display it as badge
    body = re.sub(r"\*\*QC VERDICT:\*\*.*\n?", "", response).strip()

    st.markdown(
        f"""
        <div style="background:#0d1629;border:1px solid #1e2d45;border-radius:8px;
                    padding:20px 24px;margin-top:12px;box-shadow:0 2px 8px rgba(0,0,0,0.3);">
          <div style="display:flex;align-items:center;justify-content:space-between;
                      margin-bottom:16px;border-bottom:1px solid #1e2d45;padding-bottom:12px;">
            <span style="font-family:Sora,sans-serif;font-size:0.95rem;font-weight:600;
                         color:#e0ecff;">🩻 {title}</span>
            <span style="font-family:JetBrains Mono,monospace;font-size:0.68rem;color:#4d9de0;">
              AI Provider: {provider}
            </span>
          </div>
          <div style="margin-bottom:16px;">
            <div style="font-size:0.7rem;color:#5a7a99;font-family:JetBrains Mono,monospace;
                        letter-spacing:1px;text-transform:uppercase;margin-bottom:6px;">
              Quality Verdict
            </div>
            {_verdict_badge(verdict)}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Render body as markdown (preserves **bold**, bullets, numbered lists)
    with st.container():
        st.markdown(body)


def _render_slice_preview(
    slice_arr: np.ndarray, axis_name: str, idx: int
) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 3.5), facecolor="#0d1221")
    ax.imshow(slice_arr, cmap="gray", vmin=0, vmax=1, aspect="equal",
              interpolation="bilinear")
    ax.set_title(f"{axis_name}  ·  slice {idx}",
                 color="#8fa3bc", fontsize=8, fontfamily="monospace", pad=3)
    ax.axis("off")
    plt.tight_layout(pad=0.1)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="#0d1221")
    plt.close(fig)
    buf.seek(0)
    st.image(buf, width=260)


def _chat_bubble(role: str, text: str) -> None:
    if role == "user":
        st.markdown(
            f"""<div style="background:#1a2d4a;border:1px solid #2a4a72;
              border-radius:6px;padding:10px 14px;margin:6px 0 6px 40px;
              font-size:0.85rem;color:#c8d6e5;font-family:Sora,sans-serif;">
              <span style="font-size:0.68rem;color:#4d9de0;
                font-family:JetBrains Mono,monospace;">🧑‍⚕️ CLINICIAN</span><br/>{text}
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""<div style="background:#0d1f1a;border:1px solid #1e4a38;
              border-radius:6px;padding:10px 14px;margin:6px 40px 6px 0;
              font-size:0.85rem;color:#c8d6e5;font-family:Sora,sans-serif;">
              <span style="font-size:0.68rem;color:#3ddc84;
                font-family:JetBrains Mono,monospace;">🤖 AI RADIOLOGIST</span><br/>{text}
            </div>""",
            unsafe_allow_html=True,
        )


# ── Tab A: Image Analysis ─────────────────────────────────────────────────────

def _render_image_analysis_tab(mri_file: Optional[object]) -> None:
    st.markdown(
        """
        <div style="background:#0d1629;border:1px solid #1e3a5c;border-radius:6px;
                    padding:12px 16px;margin-bottom:16px;">
          <span style="color:#4d9de0;font-size:0.8rem;font-family:JetBrains Mono,monospace;">
            � HOW TO USE
          </span>
          <p style="color:#8fa3bc;font-size:0.82rem;margin:6px 0 0 0;font-family:Sora,sans-serif;">
            1. Select an MRI slice using the controls below<br/>
            2. Optionally include MRIQC metrics for richer clinical context<br/>
            3. Click <strong style="color:#e0ecff;">Analyze Slice</strong> to generate QC report<br/>
            4. Review findings and ask follow-up questions if needed
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if mri_file is None:
        st.info("📂 Upload a NIfTI MRI file via the sidebar to begin image analysis.")
        return

    try:
        vol = load_nifti(mri_file.getvalue(), mri_file.name)
    except ValueError as exc:
        st.error(f"❌ Cannot load MRI: {exc}")
        return

    data = vol.data if vol.data.ndim == 3 else vol.data[..., 0]
    shape = data.shape

    # ── Controls ──────────────────────────────────────────────────────────
    st.markdown("#### 🎛️ Slice Selection")
    ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 2])

    with ctrl1:
        axis_label = st.selectbox("Anatomical Plane", list(_AXIS_NAMES.values()), key="mg_axis")
        axis_idx = {v: k for k, v in _AXIS_NAMES.items()}[axis_label]

    with ctrl2:
        max_slice = shape[axis_idx] - 1
        slice_idx = st.slider("Slice Index", 0, max_slice, max_slice // 2, key="mg_slice_idx")

    raw_slice  = extract_slice(data, axis_idx, slice_idx)
    norm_slice = normalize_slice(raw_slice)

    with ctrl3:
        _render_slice_preview(norm_slice, axis_label, slice_idx)

    # ── IQM context toggle ────────────────────────────────────────────────
    st.markdown("#### 📊 Optional: MRIQC Metrics Context")
    include_iqm = st.toggle(
        "Include MRIQC metrics for enhanced analysis",
        value=True, key="mg_include_iqm",
    )

    iqm_context: dict = {}
    if include_iqm and st.session_state.get("iqm_df") is not None:
        df  = st.session_state.iqm_df
        row = df.iloc[0] if len(df) >= 1 else None
        if row is not None:
            iqm_context = {k: v for k, v in row.items()
                           if isinstance(v, float) and not np.isnan(v)}
        if iqm_context:
            st.caption(f"✅ {len(iqm_context)} quality metrics loaded and will be included in analysis")
        else:
            st.caption("ℹ️ No numeric metrics found in loaded TSV")
    elif include_iqm:
        st.caption("ℹ️ No IQM TSV loaded — upload via sidebar for enriched analysis")

    # ── Action buttons ────────────────────────────────────────────────────
    st.markdown("---")
    col_btn, col_clear, col_spacer = st.columns([2, 1.5, 3])

    with col_btn:
        run_analysis = st.button(
            "🔬  Analyze Slice",
            key="mg_run_image_analysis",
            use_container_width=True,
            type="primary",
        )
    with col_clear:
        if st.button("🗑️  Clear Report", key="mg_clear_img_history", use_container_width=True):
            st.session_state["mg_img_report"] = None
            st.session_state["medgemma_img_history"] = []
            st.rerun()

    # ── Main report ───────────────────────────────────────────────────────
    if run_analysis:
        png_bytes = slice_to_png_bytes(norm_slice)
        with st.spinner("🧠 AI Radiologist is analyzing the MRI slice…"):
            try:
                response = analyze_mri_slice(
                    slice_png_bytes=png_bytes,
                    metrics_dict=iqm_context if iqm_context else None,
                )
                st.session_state["mg_img_report"] = {
                    "response": response,
                    "axis": axis_label,
                    "slice": slice_idx,
                }
                # Reset follow-up history on new analysis
                st.session_state["medgemma_img_history"] = []
            except RuntimeError as exc:
                st.error(f"⚠️ AI Error: {exc}")

    # ── Display main report ───────────────────────────────────────────────
    report = st.session_state.get("mg_img_report")
    if report:
        st.markdown(
            f"<div style='font-size:0.75rem;color:#5a7a99;font-family:JetBrains Mono,"
            f"monospace;margin-bottom:4px;'>📍 Report for: {report['axis']} · Slice {report['slice']}</div>",
            unsafe_allow_html=True,
        )
        _render_clinical_report(report["response"], "MRI Image QC Analysis")

        # ── Follow-up Q&A (text-only, uses report as context) ─────────────
        st.markdown("---")
        st.markdown(
            "<span style='font-size:0.8rem;color:#5a7a99;font-family:JetBrains Mono,monospace;'>"
            "💬 Clinical Follow-up Questions</span>",
            unsafe_allow_html=True,
        )

        followup_history = st.session_state.get("medgemma_img_history", [])
        for msg in followup_history:
            _chat_bubble(msg["role"], msg["content"])

        col_q, col_send = st.columns([4, 1])
        with col_q:
            followup_img = st.text_input(
                "Your question",
                placeholder="e.g. Should this scan be excluded due to artifacts?",
                key="mg_img_followup",
                label_visibility="collapsed",
            )
        with col_send:
            if st.button("Send", key="mg_img_send", use_container_width=True):
                if followup_img.strip():
                    followup_history.append({"role": "user", "content": followup_img.strip()})
                    with st.spinner("Thinking…"):
                        try:
                            resp = followup_chat(
                                question=followup_img.strip(),
                                previous_report=report["response"],
                            )
                            followup_history.append({"role": "assistant", "content": resp})
                        except RuntimeError as exc:
                            st.error(str(exc))
                    st.session_state["medgemma_img_history"] = followup_history
                    st.rerun()


# ── Tab B: IQM Report ─────────────────────────────────────────────────────────

def _render_iqm_explanation_tab(iqm_file: Optional[object]) -> None:
    st.markdown(
        """
        <div style="background:#0d1629;border:1px solid #1e3a5c;border-radius:6px;
                    padding:12px 16px;margin-bottom:16px;">
          <span style="color:#4d9de0;font-size:0.8rem;font-family:JetBrains Mono,monospace;">
            � HOW TO USE
          </span>
          <p style="color:#8fa3bc;font-size:0.82rem;margin:6px 0 0 0;font-family:Sora,sans-serif;">
            1. Select a subject/scan from the loaded MRIQC data<br/>
            2. Review the metrics preview to understand quality parameters<br/>
            3. Optionally ask a specific clinical question<br/>
            4. Click <strong style="color:#e0ecff;">Generate QC Report</strong> for AI-powered analysis
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if iqm_file is None:
        st.info("📂 Upload an MRIQC TSV file via the sidebar to generate quantitative QC reports.")
        return

    from loaders.iqm_loader import load_iqm
    try:
        dataset = load_iqm(iqm_file.getvalue(), iqm_file.name)
    except ValueError as exc:
        st.error(f"❌ Cannot load IQM file: {exc}")
        return

    df       = dataset.df
    iqm_cols = dataset.iqm_columns

    # ── Subject selector ──────────────────────────────────────────────────
    st.markdown("#### 👤 Subject/Scan Selection")
    if "subject_id" in df.columns:
        subject_options = df["subject_id"].astype(str).tolist()
        selected_subject = st.selectbox("Select Subject", subject_options, key="mg_iqm_row")
        row_idx = subject_options.index(selected_subject)
    else:
        row_idx = st.number_input(
            "Row Index", min_value=0, max_value=len(df) - 1, value=0, key="mg_iqm_row_idx"
        )

    row = df.iloc[row_idx]
    metrics = {
        col: float(row[col])
        for col in iqm_cols
        if col in row.index
        and not (isinstance(row[col], float) and np.isnan(row[col]))
    }

    # ── Metrics preview ───────────────────────────────────────────────────
    with st.expander("📋 Preview Metrics Being Analyzed", expanded=False):
        import pandas as pd

        # Flag abnormal values
        thresholds = {
            "snr": ("low", 8), "cnr": ("low", 1), "efc": ("high", 0.7),
            "cjv": ("high", 0.8), "tsnr": ("low", 25), "fd_mean": ("high", 0.5),
        }

        rows_list = []
        for k, v in metrics.items():
            key_lower = k.lower()
            status = "—"
            for tkey, (direction, threshold) in thresholds.items():
                if tkey in key_lower:
                    if direction == "low" and v < threshold:
                        status = "⚠️ Below threshold"
                    elif direction == "high" and v > threshold:
                        status = "⚠️ Above threshold"
                    else:
                        status = "✅ Normal range"
                    break
            rows_list.append({"Metric": k.upper(), "Value": f"{v:.4f}", "Status": status})

        preview_df = pd.DataFrame(rows_list)
        st.dataframe(preview_df, use_container_width=True, hide_index=True)

    # ── Clinical question ────────────────────────────────────────────────
    st.markdown("#### ❓ Optional: Specific Clinical Question")
    followup = st.text_input(
        "Your question",
        placeholder="e.g. Is this scan suitable for cortical thickness analysis?",
        key="mg_iqm_followup",
        label_visibility="collapsed",
    )

    # ── Action buttons ────────────────────────────────────────────────────
    st.markdown("---")
    col_btn, col_clear, col_spacer = st.columns([2, 1.5, 3])

    with col_btn:
        run_explain = st.button(
            "📄  Generate QC Report",
            key="mg_run_iqm_explain",
            use_container_width=True,
            type="primary",
        )
    with col_clear:
        if st.button("🗑️  Clear Report", key="mg_clear_iqm_history", use_container_width=True):
            st.session_state["mg_iqm_report"] = None
            st.session_state["medgemma_iqm_history"] = []
            st.rerun()

    # ── Generate report ───────────────────────────────────────────────────
    if run_explain:
        with st.spinner("📊 AI Radiologist is reviewing the metrics…"):
            try:
                payload = dict(metrics)
                if followup.strip():
                    payload["_question"] = followup.strip()
                response = explain_iqm(payload)
                subject_label = (
                    selected_subject if "subject_id" in df.columns else f"row {row_idx}"
                )
                st.session_state["mg_iqm_report"] = {
                    "response": response,
                    "subject": subject_label,
                }
                # Reset follow-up history on new report
                st.session_state["medgemma_iqm_history"] = []
            except RuntimeError as exc:
                st.error(f"⚠️ AI Error: {exc}")

    # ── Display report ────────────────────────────────────────────────────
    report = st.session_state.get("mg_iqm_report")
    if report:
        st.markdown(
            f"<div style='font-size:0.75rem;color:#5a7a99;font-family:JetBrains Mono,"
            f"monospace;margin-bottom:4px;'>📍 Report for: {report['subject']}</div>",
            unsafe_allow_html=True,
        )
        _render_clinical_report(report["response"], "MRIQC Metrics QC Analysis")

        # ── Follow-up Q&A ─────────────────────────────────────────────────
        st.markdown("---")
        st.markdown(
            "<span style='font-size:0.8rem;color:#5a7a99;font-family:JetBrains Mono,monospace;'>"
            "💬 Clinical Follow-up Questions</span>",
            unsafe_allow_html=True,
        )

        iqm_history = st.session_state.get("medgemma_iqm_history", [])
        for msg in iqm_history:
            _chat_bubble(msg["role"], msg["content"])

        col_q2, col_send2 = st.columns([4, 1])
        with col_q2:
            followup_iqm = st.text_input(
                "Your question",
                placeholder="e.g. Should this subject be excluded from the study?",
                key="mg_iqm_followup2",
                label_visibility="collapsed",
            )
        with col_send2:
            if st.button("Send", key="mg_iqm_send2", use_container_width=True):
                if followup_iqm.strip():
                    iqm_history.append({"role": "user", "content": followup_iqm.strip()})
                    with st.spinner("Thinking…"):
                        try:
                            resp = followup_chat(
                                question=followup_iqm.strip(),
                                previous_report=report["response"],
                            )
                            iqm_history.append({"role": "assistant", "content": resp})
                        except RuntimeError as exc:
                            st.error(str(exc))
                    st.session_state["medgemma_iqm_history"] = iqm_history
                    st.rerun()


# ── Main render ───────────────────────────────────────────────────────────────

def render_medgemma_panel(
    mri_file: Optional[object],
    iqm_file: Optional[object],
) -> None:
    """Render the AI Radiologist Assistant panel with provider selection."""

    # ── Header with provider selector ─────────────────────────────────────
    header_col1, header_col2, header_col3 = st.columns([2, 2, 1])
    
    with header_col1:
        st.markdown(
            """
            <div>
              <h3 style="font-family:Sora,sans-serif;font-weight:700;font-size:1.3rem;
                         color:#e0ecff;margin:0;">🤖 AI Radiologist Assistant</h3>
              <p style="font-size:0.78rem;color:#5a7a99;font-family:JetBrains Mono,monospace;
                        margin:4px 0 0 0;">
                Intelligent MRI Quality Control & Clinical Interpretation
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with header_col2:
        st.markdown("")
    
    with header_col3:
        provider = st.selectbox(
            "AI Provider",
            _AVAILABLE_PROVIDERS,
            index=0 if get_ai_provider() == "groq" else 1,
            key="ai_provider_selector",
        )
        if provider != get_ai_provider():
            set_ai_provider(provider)
        
        st.markdown(
            f"""<div style="display:inline-block;padding:4px 10px;border-radius:3px;
                         background:#1a2d4a;border:1px solid #2a4a72;
                         font-family:JetBrains Mono,monospace;font-size:0.65rem;color:#4d9de0;
                         text-align:center;width:100%;margin-top:4px;">
              {provider.upper()}
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Disclaimer ────────────────────────────────────────────────────────
    st.warning(
        "⚕️ **Clinical Disclaimer** — AI-generated reports are for **research and educational purposes only**. "
        "They do not constitute medical advice and must not replace the judgment of a qualified radiologist or physician.",
        icon=None,
    )

    # ── Capability overview ───────────────────────────────────────────────
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(
            """<div style="background:#0d1629;border:1px solid #1e2d45;border-radius:6px;
                          padding:12px;text-align:center;">
              <div style="font-size:1.4rem;">🖼️</div>
              <div style="font-size:0.78rem;color:#e0ecff;font-family:Sora,sans-serif;
                          font-weight:600;margin-top:4px;">Visual QC</div>
              <div style="font-size:0.7rem;color:#5a7a99;margin-top:2px;">
                MRI slice analysis
              </div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            """<div style="background:#0d1629;border:1px solid #1e2d45;border-radius:6px;
                          padding:12px;text-align:center;">
              <div style="font-size:1.4rem;">📊</div>
              <div style="font-size:0.78rem;color:#e0ecff;font-family:Sora,sans-serif;
                          font-weight:600;margin-top:4px;">Quantitative QC</div>
              <div style="font-size:0.7rem;color:#5a7a99;margin-top:2px;">
                MRIQC interpretation
              </div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col_c:
        st.markdown(
            """<div style="background:#0d1629;border:1px solid #1e2d45;border-radius:6px;
                          padding:12px;text-align:center;">
              <div style="font-size:1.4rem;">💬</div>
              <div style="font-size:0.78rem;color:#e0ecff;font-family:Sora,sans-serif;
                          font-weight:600;margin-top:4px;">Clinical Chatbot</div>
              <div style="font-size:0.7rem;color:#5a7a99;margin-top:2px;">
                Follow-up Q&A
              </div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br/>", unsafe_allow_html=True)

    # ── Main tabs ─────────────────────────────────────────────────────────
    tab_img, tab_iqm = st.tabs([
        "🖼️  MRI Image Analysis",
        "📊  MRIQC Metric Analysis",
    ])

    with tab_img:
        _render_image_analysis_tab(mri_file)

    with tab_iqm:
        _render_iqm_explanation_tab(iqm_file)

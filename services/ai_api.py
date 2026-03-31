"""
services/ai_api.py
==================
Unified AI Radiologist backend supporting multiple LLM providers:
  - Groq AI (LLaMA 4 Scout Vision) — faster, production-ready
  - Google Gemini 2.0 Flash — multimodal advanced reasoning

Public interface:
  analyze_mri_slice(slice_png_bytes, metrics_dict, provider="groq") -> str
  explain_iqm(metrics_dict, provider="groq")                        -> str
  followup_chat(question, previous_report, provider="groq")         -> str
  slice_to_png_bytes(slice_array)                                   -> bytes
  set_ai_provider(provider: str)                                    -> None
  get_ai_provider()                                                 -> str

Setup:
  1. pip install groq google-generativeai python-dotenv pillow
  2. Add GROQ_API_KEY=gsk_... to .env
  3. Add GEMINI_API_KEY=AIza... to .env (for Gemini support)
  4. Select provider via streamlit UI or set_ai_provider()
"""

from __future__ import annotations

import base64
import io
import os
import sys
from typing import Optional, Literal

import numpy as np

# ── Load .env ─────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Provider configuration ────────────────────────────────────────────────────
_GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "").strip() or None
_GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "").strip() or None
_ACTIVE_PROVIDER = os.getenv("AI_PROVIDER", "groq").lower()  # groq | gemini

# ── Model configs ─────────────────────────────────────────────────────────────
_CONFIG = {
    "groq": {
        "vision_model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "text_model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "max_tokens": 1024,
        "temperature": 0.3,
    },
    "gemini": {
        "model": "gemini-2.0-flash",
        "max_tokens": 1024,
        "temperature": 0.3,
    }
}

# ── Status logging ────────────────────────────────────────────────────────────
if _GROQ_API_KEY:
    print(f"[AI-API] ✓ Groq API Key: {_GROQ_API_KEY[:12]}...{_GROQ_API_KEY[-4:]}", file=sys.stderr)
else:
    print("[AI-API] ⚠ Groq API Key not configured (.env)", file=sys.stderr)

if _GEMINI_API_KEY:
    print(f"[AI-API] ✓ Gemini API Key: {_GEMINI_API_KEY[:12]}...{_GEMINI_API_KEY[-4:]}", file=sys.stderr)
else:
    print("[AI-API] ⚠ Gemini API Key not configured (.env)", file=sys.stderr)

print(f"[AI-API] Active provider: {_ACTIVE_PROVIDER.upper()}", file=sys.stderr)

# ── Clinical system context ──────────────────────────────────────────────────
_SYSTEM_CONTEXT = """You are an expert neuroradiologist and MRI quality control specialist 
with deep knowledge of MRIQC metrics, neuroimaging artifacts, and clinical workflow.

Core expertise:
  • MRIQC quality metrics (SNR, CNR, EFC, CJV, FBER, tSNR, FD, DVARS)
  • MRI acquisition artifacts (motion, ghosting, signal dropout, spike artifacts)
  • Clinical QC decision-making for neuroimaging research and diagnosis
  • Nipoppy/BIDS data collection and validation standards

Your responsibilities:
  1. Provide evidence-based, actionable quality assessments
  2. Structure reports clearly for radiologists and researchers
  3. Explain metrics in accessible clinical language
  4. Recommend next steps (accept, re-scan, exclude, pending review)
  5. Always maintain professional, conservative judgment

Quality interpretation guidelines:
  • SNR (Signal-to-Noise Ratio): higher is better; <8 = poor
  • CNR (Contrast-to-Noise Ratio): higher is better; <1 = poor
  • EFC (Entropy Focus Criterion): lower is better; >0.7 = ghosting artifact
  • CJV (Coefficient of Joint Variation): lower is better; >0.8 = poor GM/WM separation
  • FBER (Foreground-Background Energy Ratio): higher is better
  • tSNR (temporal SNR, fMRI): higher is better; <25 = poor baseline noise
  • FD (Framewise Displacement, fMRI): mean >0.5mm = excessive head motion
  • DVARS (Delta Variance): signal variability between volumes — lower is better

IMPORTANT: Always respond in structured clinical format. Be concise, accurate, and actionable.
Reports must follow radiological reporting standards."""

_IMAGE_PROMPT = """{context}

### Task: Analyze MRI brain slice for quality control

Examine this MRI brain scan slice carefully. Structure your response EXACTLY as follows:

**QC VERDICT:** [PASS / WARN / FAIL]

**SIGNAL & CONTRAST FINDINGS:**
[2–3 sentences analyzing: signal quality, contrast between gray/white matter, 
visible artifacts (motion, ghosting, dropout), noise level, coverage]

**ARTIFACT ASSESSMENT:**
[Identify any specific artifacts: motion blur, ghosting, spike artifacts, 
signal dropout, geometric distortions, aliasing, susceptibility artifacts]

**CLINICAL IMPACT:**
[1–2 sentences: How these findings affect downstream analysis (fMRI, structural 
segmentation, DTI) and clinical utility]

**RECOMMENDATIONS:**
[2–3 numbered actionable steps: Accept for analysis, Request re-acquisition, 
Exclude from study, Flag for radiologist review, etc.]

**CONFIDENCE LEVEL:** [HIGH / MODERATE / LOW]"""

_IQM_PROMPT = """You are a clinical specialist reviewing quantitative MRI quality metrics.

{metrics_block}

{question_block}

Structure your response EXACTLY as follows:

**QC VERDICT:** [PASS / WARN / FAIL]

**METRIC ANALYSIS:**
[Bullet list for each key metric: evaluation with clinical meaning and reference ranges]

**OVERALL QUALITY ASSESSMENT:**
[2–3 sentences: aggregate quality judgment, primary concern areas, impact on 
research use vs. clinical use]

**RECOMMENDATIONS:**
[2–3 numbered actionable recommendations for next steps in workflow]

**FOLLOW-UP ACTIONS:**
[If applicable: specific investigations, repeat imaging, or exclusion criteria met]

**CONFIDENCE LEVEL:** [HIGH / MODERATE / LOW]"""

_FOLLOWUP_CONTEXT = """You are answering a clinician's follow-up question about a previously 
generated QC report. Answer based ONLY on the report context and your medical expertise.
Do not re-analyze or make additional claims beyond what the report states.
Keep response concise and clinically relevant."""


# ── Provider management ──────────────────────────────────────────────────────

def set_ai_provider(provider: Literal["groq", "gemini"]) -> None:
    """Switch active AI provider at runtime."""
    global _ACTIVE_PROVIDER
    provider_lower = provider.lower()
    if provider_lower not in ("groq", "gemini"):
        raise ValueError(f"Unknown provider: {provider}. Use 'groq' or 'gemini'.")
    _ACTIVE_PROVIDER = provider_lower
    print(f"[AI-API] Switched to {provider_lower} provider", file=sys.stderr)

def get_ai_provider() -> str:
    """Return currently active provider."""
    return _ACTIVE_PROVIDER

# ── Groq client ───────────────────────────────────────────────────────────────

def _get_groq_client():
    """Return a configured Groq client."""
    try:
        from groq import Groq
    except ImportError:
        raise RuntimeError(
            "groq package not installed.\n"
            "Run: pip install groq"
        )
    if not _GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set.\n"
            "Add GROQ_API_KEY=gsk_... to your .env file.\n"
            "Get a free key at: https://console.groq.com/keys"
        )
    return Groq(api_key=_GROQ_API_KEY)

# ── Gemini client ─────────────────────────────────────────────────────────────

def _get_gemini_client():
    """Return a configured Gemini client."""
    try:
        from google import genai
    except ImportError:
        raise RuntimeError(
            "google-genai not installed.\n"
            "Run: pip install google-generativeai"
        )
    if not _GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set.\n"
            "Add GEMINI_API_KEY=AIza... to your .env file.\n"
            "Get a key at: https://aistudio.google.com/apikey"
        )
    return genai.Client(api_key=_GEMINI_API_KEY)


# ── Public API ────────────────────────────────────────────────────────────────

def analyze_mri_slice(
    slice_png_bytes: bytes,
    metrics_dict: Optional[dict] = None,
    provider: Optional[str] = None,
) -> str:
    """
    Multimodal: send MRI slice image + optional IQM context to AI provider.
    
    Args:
        slice_png_bytes: PNG image bytes of MRI slice
        metrics_dict: Optional dict of MRIQC metrics to include as context
        provider: "groq" | "gemini" | None (uses current active provider)
    
    Returns:
        Structured clinical QC report
    """
    try:
        import PIL.Image
    except ImportError:
        raise RuntimeError("Run: pip install Pillow")

    active_provider = (provider or _ACTIVE_PROVIDER).lower()
    
    # Build metrics context
    context_lines = []
    if metrics_dict:
        for k, v in metrics_dict.items():
            try:
                fv = float(v)
                if fv == fv:  # not NaN
                    context_lines.append(f"  • {k.upper()}: {fv:.4f}")
            except (TypeError, ValueError):
                pass

    context_block = (
        "MRIQC Quality Metrics Context:\n" + "\n".join(context_lines)
        if context_lines
        else "No MRIQC metrics provided — visual analysis only."
    )

    prompt = _IMAGE_PROMPT.format(context=context_block)

    if active_provider == "groq":
        return _analyze_mri_slice_groq(slice_png_bytes, prompt)
    elif active_provider == "gemini":
        return _analyze_mri_slice_gemini(slice_png_bytes, prompt)
    else:
        raise ValueError(f"Unknown provider: {active_provider}")

def _analyze_mri_slice_groq(slice_png_bytes: bytes, prompt: str) -> str:
    """Groq vision API call for MRI slice analysis."""
    b64_image = base64.b64encode(slice_png_bytes).decode("utf-8")
    image_url = f"data:image/png;base64,{b64_image}"

    client = _get_groq_client()
    cfg = _CONFIG["groq"]
    try:
        response = client.chat.completions.create(
            model=cfg["vision_model"],
            messages=[
                {"role": "system", "content": _SYSTEM_CONTEXT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            max_tokens=cfg["max_tokens"],
            temperature=cfg["temperature"],
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        raise RuntimeError(f"Groq API error: {exc}") from exc

def _analyze_mri_slice_gemini(slice_png_bytes: bytes, prompt: str) -> str:
    """Gemini multimodal API call for MRI slice analysis."""
    try:
        import io
        from PIL import Image
        pil_image = Image.open(io.BytesIO(slice_png_bytes)).convert("RGB")
    except Exception as exc:
        raise RuntimeError(f"Image processing error: {exc}")

    client = _get_gemini_client()
    cfg = _CONFIG["gemini"]
    try:
        from google.genai import types
        response = client.models.generate_content(
            model=cfg["model"],
            contents=[pil_image, prompt],
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_CONTEXT,
                max_output_tokens=cfg["max_tokens"],
                temperature=cfg["temperature"],
            ),
        )
        return response.text.strip()
    except Exception as exc:
        raise RuntimeError(f"Gemini API error: {exc}") from exc

def explain_iqm(
    metrics_dict: dict,
    provider: Optional[str] = None,
) -> str:
    """
    Text-only: send IQM metrics to AI provider for clinical explanation.
    
    Args:
        metrics_dict: Dict of MRIQC metrics. Use "_question" key for follow-up.
        provider: "groq" | "gemini" | None (uses current active provider)
    
    Returns:
        Structured clinical QC report
    """
    lines = []
    followup = metrics_dict.get("_question", "")

    for k, v in metrics_dict.items():
        if k == "_question":
            continue
        try:
            fv = float(v)
            if fv == fv:  # not NaN
                lines.append(f"  • {k.upper()}: {fv:.4f}")
        except (TypeError, ValueError):
            pass

    metrics_block = (
        "MRIQC Quality Metrics:\n" + "\n".join(lines)
        if lines else "  (no metrics provided)"
    )
    question_block = (
        f"Clinician follow-up question: {followup}"
        if followup
        else "Provide a comprehensive quality control assessment and clinical recommendations."
    )

    prompt = _IQM_PROMPT.format(
        metrics_block=metrics_block,
        question_block=question_block,
    )

    active_provider = (provider or _ACTIVE_PROVIDER).lower()

    if active_provider == "groq":
        return _explain_iqm_groq(prompt)
    elif active_provider == "gemini":
        return _explain_iqm_gemini(prompt)
    else:
        raise ValueError(f"Unknown provider: {active_provider}")

def _explain_iqm_groq(prompt: str) -> str:
    """Groq text API for IQM explanation."""
    client = _get_groq_client()
    cfg = _CONFIG["groq"]
    try:
        response = client.chat.completions.create(
            model=cfg["text_model"],
            messages=[
                {"role": "system", "content": _SYSTEM_CONTEXT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=cfg["max_tokens"],
            temperature=cfg["temperature"],
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        raise RuntimeError(f"Groq API error: {exc}") from exc

def _explain_iqm_gemini(prompt: str) -> str:
    """Gemini text API for IQM explanation."""
    client = _get_gemini_client()
    cfg = _CONFIG["gemini"]
    try:
        from google.genai import types
        response = client.models.generate_content(
            model=cfg["model"],
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_CONTEXT,
                max_output_tokens=cfg["max_tokens"],
                temperature=cfg["temperature"],
            ),
        )
        return response.text.strip()
    except Exception as exc:
        raise RuntimeError(f"Gemini API error: {exc}") from exc

def followup_chat(
    question: str,
    previous_report: str,
    provider: Optional[str] = None,
) -> str:
    """
    Text-only follow-up using previous report as context.
    
    Args:
        question: Clinician's follow-up question
        previous_report: Previous QC report for context
        provider: "groq" | "gemini" | None (uses current active provider)
    
    Returns:
        Follow-up response from AI provider
    """
    prompt = (
        f"## Previous QC Report (for context only):\n\n{previous_report}\n\n"
        f"---\n\n"
        f"## Clinician Follow-up Question:\n{question}\n\n"
        f"Respond concisely based ONLY on this report and your clinical expertise."
    )

    active_provider = (provider or _ACTIVE_PROVIDER).lower()

    if active_provider == "groq":
        return _followup_chat_groq(prompt)
    elif active_provider == "gemini":
        return _followup_chat_gemini(prompt)
    else:
        raise ValueError(f"Unknown provider: {active_provider}")

def _followup_chat_groq(prompt: str) -> str:
    """Groq text API for follow-up chat."""
    client = _get_groq_client()
    cfg = _CONFIG["groq"]
    try:
        response = client.chat.completions.create(
            model=cfg["text_model"],
            messages=[
                {"role": "system", "content": _FOLLOWUP_CONTEXT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=512,
            temperature=cfg["temperature"],
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        raise RuntimeError(f"Groq API error: {exc}") from exc

def _followup_chat_gemini(prompt: str) -> str:
    """Gemini text API for follow-up chat."""
    client = _get_gemini_client()
    cfg = _CONFIG["gemini"]
    try:
        from google.genai import types
        response = client.models.generate_content(
            model=cfg["model"],
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=_FOLLOWUP_CONTEXT,
                max_output_tokens=512,
                temperature=cfg["temperature"],
            ),
        )
        return response.text.strip()
    except Exception as exc:
        raise RuntimeError(f"Gemini API error: {exc}") from exc


def slice_to_png_bytes(slice_array: np.ndarray) -> bytes:
    """Convert a 2-D normalized [0,1] array to PNG bytes."""
    try:
        from PIL import Image
        uint8 = (np.clip(slice_array, 0, 1) * 255).astype(np.uint8)
        img = Image.fromarray(uint8, mode="L")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except ImportError as exc:
        raise RuntimeError("Run: pip install Pillow") from exc

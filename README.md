# 🧠 QC-Studio: AI-Powered MRI Quality Control Platform

**QC-Studio** is a production-ready, brain-specialist Streamlit application for semi-automated neuroimaging quality control within the **Nipoppy** neuroinformatics framework. It integrates multiple data formats, interactive dashboards, and AI-powered QC annotations to streamline clinical decision-making.

---

## ✨ Key Features

### 📊 **Multi-Format Data Support**
- **NIfTI (.nii, .nii.gz)** — Volumetric MRI data with interactive 3D navigation
- **SVG & PNG** — Pipeline output visualizations (segmentations, networks, surfaces)
- **TSV** — MRIQC quality metrics (SNR, CNR, EFC, CJV, tSNR, FD, DVARS, etc.)
- **MGZ** — FreeSurfer volumetric format *(planned enhancement)*
- **GII** — GIFTI surface data *(planned enhancement)*

### 🔎 **Interactive Visualization**
- **NiiVue-powered multiplanar viewing** — Sagittal, Coronal, Axial views with synchronization
- **Real-time slice navigation** — Smooth interaction with normalized intensity scaling
- **SVG exploration** — View pipeline masks, atlases, and derived surfaces
- **Overlay support** — Combine multiple modalities for clinical review

### 🤖 **AI Radiologist Assistant**
- **Dual LLM backends:**
  - **Groq AI (LLaMA 4 Scout)** — Ultra-fast, free tier, optimized for speed
  - **Google Gemini 2.0 Flash** — Advanced multimodal reasoning, better for complex cases
- **Switch providers on-the-fly** via intuitive UI dropdown
- **Visual QC Analysis** — Multimodal image analysis with optional metrics context
- **Quantitative QC** — Clinical interpretation of MRIQC metrics
- **Clinical Follow-up Chat** — Context-aware Q&A based on generated reports

### 📋 **Clinical Dashboard**
- **Metrics visualization** — Interactive tables and statistical summaries
- **Quality flagging** — Automatic detection of abnormal MRIQC values with clinical thresholds
- **Subject/scan comparison** — Side-by-side metric review across cohorts
- **Export-ready reporting** — Generate QC decisions for downstream pipelines

### 💾 **Data Export**
- **Structured QC decisions** — PASS / WARN / FAIL verdicts with clinical rationale
- **Report archiving** — Save AI-generated assessments for audit trails
- **CSV/JSON export** — Integration with study databases and BIDS validators

---

## 🚀 Quick Start

### Installation

```bash
# Clone or download the repository
cd d:/competition/gsoc2026/qc_studio

# Create Python virtual environment (Python 3.9+)
python -m venv venv
venv\Scripts\activate.ps1  # Windows PowerShell
# or: source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Install optional LLM packages (choose one or both)
pip install groq                    # For Groq AI backend (recommended)
pip install google-generativeai     # For Google Gemini backend
```

### Configuration

Create a `.env` file in the project root:

```env
# Groq API (free tier: https://console.groq.com/keys)
GROQ_API_KEY=gsk_your_key_here

# Google Gemini (free API key: https://aistudio.google.com/apikey)
GEMINI_API_KEY=AIza_your_key_here

# Default AI provider (groq | gemini)
AI_PROVIDER=groq
```

### Launch the Application

```bash
# Run Streamlit app
streamlit run app.py

# App will open at: http://localhost:8501
```

---

## 📖 User Guide

### 1. **NiiVue Viewer Tab** 🧠
Upload a NIfTI MRI file to visualize 3D brain anatomy.

- **Multiplanar Navigation:** View sagittal, coronal, and axial slices simultaneously
- **Slice Sync:** Navigate one plane; others update in real-time
- **Intensity Scaling:** Auto-normalized to [0,1] range for optimal visualization
- **Crosshair Navigation:** Click to focus on regions of interest

### 2. **SVG Outputs Tab** ◈
Visualize pipeline-generated SVG graphics (masks, atlases, statistical maps).

- **Load multiple SVG files** from preprocessing pipelines
- **Interactive pan/zoom** for detailed inspection
- **Supports FreeSurfer, FSL, and AFNI outputs**

### 3. **IQM Dashboard Tab** 📊
Quantitative quality metrics from MRIQC.

- **Upload TSV files** with columns: subject_id, snr, cnr, efc, cjv, fber, tsnr, fd_mean, dvars, etc.
- **Automatic flagging:**
  - SNR < 8 → ⚠️ Below threshold
  - CNR < 1 → ⚠️ Poor gray/white matter contrast
  - EFC > 0.7 → ⚠️ Ghosting artifact detected
  - CJV > 0.8 → ⚠️ Poor tissue separation
  - FD mean > 0.5mm → ⚠️ Excessive head motion

- **Interactive metrics preview** with status indicators
- **Subject-level drill-down** for detailed review

### 4. **AI Radiologist Tab** 🤖

#### 🖼️ MRI Image Analysis
1. Select anatomical plane (Sagittal/Coronal/Axial)
2. Use slider to choose slice index
3. Toggle MRIQC metrics context (optional)
4. Click **Analyze Slice** to generate QC report

**Output:** Structured clinical report with:
- QC Verdict (PASS/WARN/FAIL)
- Signal & Contrast Findings
- Artifact Assessment
- Clinical Impact
- Recommendations
- Confidence Level

#### 📊 MRIQC Metric Analysis
1. Select subject from loaded TSV
2. Review metrics preview (optional)
3. Ask a clinical question (optional)
4. Click **Generate QC Report** for AI analysis

**Output:** Comprehensive assessment including:
- Per-metric clinical interpretation
- Overall quality judgment
- Recommendations for study inclusion/exclusion
- Follow-up actions if needed

#### 💬 Clinical Follow-up Chat
Ask AI follow-up questions based on generated reports:
- "Should this scan be excluded from the cohort?"
- "Is this suitable for surface-based analysis?"
- "What steps should we take to improve quality?"

**Provider Selection:** Switch between Groq (fast) and Gemini (reasoning) using the dropdown in the AI Radiologist header.

### 5. **Export Tab** ↓
Download QC decisions and reports.

- **Subject-level QC Summary:** Pass/Fail with reasoning
- **Full Reports:** AI-generated clinical assessments
- **CSV Export:** Batch subject reviews for study review
- **JSON Archive:** Complete analysis history

---

## 📁 Project Structure

```
qc_studio/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── .env                            # Configuration (APIKEY management)
│
├── components/
│   ├── sidebar.py                  # File upload & study selector UI
│   └── __init__.py
│
├── panels/
│   ├── medgemma_panel.py           # AI Radiologist Assistant Tab
│   ├── niivue_panel.py             # NiiVue 3D Viewer Tab
│   ├── svg_panel.py                # SVG Visualization Tab
│   ├── iqm_panel.py                # IQM Dashboard Tab
│   └── __init__.py
│
├── services/
│   ├── ai_api.py                   # Unified AI backend (Groq + Gemini)
│   └── __init__.py
│
├── loaders/
│   ├── nifti_loader.py             # NIfTI file I/O
│   ├── iqm_loader.py               # MRIQC TSV parser
│   └── __init__.py
│
├── utils/
│   ├── session.py                  # Streamlit session state management
│   ├── export.py                   # Report & data export
│   └── __init__.py
│
├── data/
│   ├── generate_sample_data.py      # Demo dataset generation
│   ├── sample_group_T1w.tsv         # Example MRIQC metrics
│   └── sample/                      # Sample NIfTI files
│
└── README.md                        # This file
```

---

## 🔧 Configuration & API Setup

### Groq AI Setup (Recommended)

1. **Get free API key:**
   - Visit: https://console.groq.com/keys
   - Create account (no credit card required)
   - Copy your API key

2. **Add to `.env`:**
   ```env
   GROQ_API_KEY=gsk_your_key_here
   ```

3. **Verify in console:**
   ```bash
   streamlit run app.py
   # Look for: "[AI-API] ✓ Groq API Key: gsk_xxxx...xxxx"
   ```

### Gemini API Setup

1. **Get free API key:**
   - Visit: https://aistudio.google.com/apikey
   - Enable Gemini API in Google Cloud
   - Copy your API key

2. **Add to `.env`:**
   ```env
   GEMINI_API_KEY=AIza_your_key_here
   ```

3. **Switch provider in UI:**
   - Open QC-Studio → AI Radiologist tab
   - Select "gemini" from dropdown at top-right

---

## 🎓 Example Workflows

### Workflow 1: Screen Incoming MRI Dataset
```
1. Upload NIfTI file → NiiVue Viewer Tab
2. Visually inspect for obvious artifacts
3. Switch to AI Radiologist → MRI Image Analysis
4. Select slice with most pathology
5. Analyze slice → Review AI QC verdict
6. Log decision in Export Tab
```

### Workflow 2: Quantitative QC Review
```
1. Upload MRIQC TSV → IQM Dashboard Tab
2. Review metrics preview (auto-flagged abnormal values)
3. Ask: "Is this suitable for cortical thickness analysis?"
4. Generate AI Report → Read clinical interpretation
5. Ask follow-up: "Should we exclude this subject?"
6. Export final QC decision
```

### Workflow 3: Multi-Modal Review
```
1. Load NIfTI + SVG + MRIQC TSV
2. NiiVue Viewer: Inspect anatomy
3. SVG Viewer: Check segmentation/pipeline outputs
4. IQM Dashboard: Confirm metrics
5. AI Radiologist: Request multimodal analysis
6. Generate unified export report
```

---

## 🏥 Clinical Best Practices

### QC Verdict Interpretation

| Verdict | Meaning | Action |
|---------|---------|--------|
| **PASS** | Image quality acceptable for research/clinical use | Include in analysis |
| **WARN** | Quality concerns; recommend secondary review | Flag for radiologist review |
| **FAIL** | Severe artifacts or quality issues | Exclude from analysis; consider re-scan |

### Key Quality Thresholds

**T1w/T2w Structural:**
- SNR ≥ 12 (excellent), 8–12 (acceptable), <8 (poor)
- CNR ≥ 1.0 (adequate), <1 (poor tissue contrast)
- EFC < 0.7 (clean), >0.7 (ghosting)
- CJV < 0.8 (good GM/WM separation), >0.8 (poor)

**fMRI (func):**
- tSNR ≥ 30 (excellent), 25–30 (acceptable), <25 (poor)
- FD mean < 0.3mm (minimal motion), 0.3–0.5mm (moderate), >0.5mm (excessive)
- DVARS (normalized) < 1.5 std (minimal signal drift)

### Recommended Review Steps

1. **Automated Screening:** Use AI QC to flag potential issues
2. **Visual Inspection:** Confirm with NiiVue multiplanar viewer
3. **Quantitative Review:** Cross-check with MRIQC metrics
4. **Clinical Judgment:** Apply study-specific exclusion criteria
5. **Audit Trail:** Export decisions for study documentation

---

## 🐛 Troubleshooting

### "GROQ_API_KEY not set"
- **Solution:** Add to `.env` file or set environment variable
- Verify key is copied correctly (no spaces)

### "Image fails to load" in NiiVue
- **Check:** File is valid NIfTI (.nii or .nii.gz)
- **Verify:** File size < 200MB (memory constraints)
- **Try:** Redownload file; may be corrupted

### "AI response times > 30 seconds"
- **Cause:** Gemini may be slower; Groq is faster
- **Solution:** Switch to Groq provider, or wait for rate limit reset
- **Note:** Free tier has usage limits (Groq: 30 req/minute, Gemini: 60 req/minute)

### "No metrics appear in IQM Dashboard"
- **Check:** TSV file contains numeric columns (SNR, CNR, etc.)
- **Verify:** Column names match expected format (lowercase + underscore)
- **Example valid columns:** snr, cnr, efc, cjv, fber, tsnr, fd_mean, dvars

### "SVG files not rendering"
- **Verify:** Files are valid SVG (XML format)
- **Check:** File > 1KB and < 50MB
- **Try:** Open in browser first to confirm validity

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| streamlit | ≥1.40 | Web application framework |
| nibabel | ≥5.0 | NIfTI file I/O |
| numpy | ≥1.24 | Numerical computing |
| pandas | ≥2.0 | Data manipulation |
| matplotlib | ≥3.8 | Image rendering |
| pillow | ≥10.0 | PNG encoding |
| groq | ≥0.5 | Groq API client (LLaMA models) |
| google-generativeai | ≥0.3 | Gemini API client |
| python-dotenv | ≥1.0 | Environment variable management |

Install all with:
```bash
pip install -r requirements.txt
```

---

## 🔐 Privacy & Security

- **No data uploaded to external servers** (except AI API calls)
- API calls to Groq/Gemini include:
  - MRI slice image (PNG, ~50KB typical)
  - MRIQC metrics (text, <1KB)
  - No personally identifiable information (PII) included
- **Local processing:** All rendering, storage, and export done locally
- **API keys:** Never committed to version control; use `.env` file
- Add `.env` to `.gitignore` before pushing to repository

---

## 🚀 Performance Optimization

### For Large Datasets
1. **Batch mode:** Process multiple subjects sequentially
2. **Session caching:** Streamlit retains session state across reruns
3. **Lazy loading:** SVG/IQM data loaded on-demand
4. **Memory management:** NIfTI volumes downsampled if >200MB

### API Rate Limits
- **Groq:** 30 requests/minute (free tier)
- **Gemini:** 60 requests/minute (free tier)
- Cache AI responses using Streamlit session state to minimize API calls

---

## 📚 References

- **MRIQC:** https://mriqc.readthedocs.io/ — Metrics definitions
- **Nipoppy:** https://nipoppy.readthedocs.io/ — Neuroinformatics framework
- **NiiVue:** https://niivue.gitlab.io/ — 3D medical imaging viewer
- **Streamlit:** https://streamlit.io/ — Web app framework
- **BIDS:** https://bids-standard.github.io/ — Brain data standards

---

## 📝 Clinical Disclaimer

⚠️ **AI-generated QC reports are for research and educational purposes only.**  
They do not constitute medical advice and must not replace the judgment of a qualified radiologist or physician. Always have AI assessments reviewed by qualified personnel before making clinical decisions.

---

## 👥 Contributing

Contributions welcome! Areas for enhancement:
- [ ] Surface data support (GII, GIFTI)
- [ ] Multi-site study management
- [ ] Real-time batch processing
- [ ] Custom AI prompt templates
- [ ] Integration with XNAT/BIDS validators
- [ ] Mobile-responsive UI

---

## 📄 License

[Specify your license here - e.g., MIT, Apache 2.0, BSD]

---

## 📬 Support

For issues, feature requests, or questions:
- 🐛 **Bug Reports:** Open an issue on GitHub
- 💡 **Feature Requests:** Discuss ideas in Discussions
- 📧 **Direct Contact:** [Your email/organization]

---

**Happy quality control! 🧠🩻**

> ⚠️ MedGemma outputs are for **research use only** and must not be used
> for clinical diagnostic decisions.

---

## Compatible NiPreps Pipelines

- **fMRIPrep** — structural + functional preprocessing
- **QSIPrep** — diffusion MRI preprocessing
- **MRIQC** — automated IQM computation (primary IQM source)
- **FreeSurfer** — cortical surface reconstruction (mgz support planned)

---

## Nipoppy Integration

QC-Studio is designed as a modular QC layer within the
[Nipoppy](https://nipoppy.readthedocs.io/) framework:

```
dataset/
└── derivatives/
    ├── mriqc/
    │   └── group_T1w.tsv        ← IQM Dashboard input
    └── fmriprep/
        └── sub-001/figures/     ← SVG Outputs input
            ├── sub-001_brainmask.svg
            └── sub-001_carpetplot.svg
```

QC decisions exported as TSV follow the BIDS participants.tsv convention
and can be directly consumed by downstream Nipoppy pipeline steps.

---

## License

Apache 2.0 — see `LICENSE` for details.

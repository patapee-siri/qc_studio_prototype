"""
data/generate_sample_data.py
----------------------------
Generate a realistic sample MRIQC group_T1w.tsv for testing QC-Studio
without requiring a real neuroimaging dataset.

Run: python data/generate_sample_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
N = 12  # number of subjects

subjects = [f"sub-{i:03d}" for i in range(1, N + 1)]

# Simulate realistic T1w IQM distributions (based on MRIQC normative data)
data = {
    "subject_id": subjects,
    "snr":         np.random.normal(18.0, 4.5, N).clip(5, 40),
    "snr_gm":      np.random.normal(14.0, 3.5, N).clip(3, 35),
    "snr_wm":      np.random.normal(20.0, 5.0, N).clip(3, 35),
    "snr_csf":     np.random.normal(8.0,  2.5, N).clip(1, 20),
    "cnr":         np.random.normal(3.2,  0.8, N).clip(0.5, 8),
    "efc":         np.random.normal(0.50, 0.08, N).clip(0.3, 0.9),
    "fber":        np.random.normal(800,  200,  N).clip(50, 2000),
    "wm2max":      np.random.normal(0.75, 0.08, N).clip(0, 1),
    "qi_1":        np.random.exponential(0.03, N).clip(0, 0.2),
    "cjv":         np.random.normal(0.42, 0.12, N).clip(0.1, 1.5),
    "icvs_gm":     np.random.normal(0.44, 0.04, N).clip(0.3, 0.6),
    "icvs_wm":     np.random.normal(0.35, 0.04, N).clip(0.2, 0.5),
}

# Inject two "bad" subjects for testing FAIL flags
data["snr"][2]  = 6.2   # low SNR
data["efc"][7]  = 0.78  # high EFC (ghosting)
data["cjv"][10] = 0.92  # high CJV (poor GM/WM)

df = pd.DataFrame(data)
df = df.round(4)

out_path = Path(__file__).parent / "sample_group_T1w.tsv"
df.to_csv(out_path, sep="\t", index=False)
print(f"Saved: {out_path}  ({N} subjects)")
print(df.head(3).to_string())

"""
loaders/nifti_loader.py
-----------------------
Load NIfTI-1 / NIfTI-2 volumes from Streamlit UploadedFile objects.

Uses a temporary file to satisfy nibabel's path-based I/O, then cleans
up immediately. Returns both the voxel array and a structured metadata
dict used by the viewer and export panels.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from typing import Optional

import nibabel as nib
import numpy as np
import streamlit as st


@dataclass
class NiftiVolume:
    """Container for a loaded NIfTI volume and its metadata."""

    data: np.ndarray
    """Voxel data array, shape (X, Y, Z) or (X, Y, Z, T)."""

    affine: np.ndarray
    """4×4 world-space affine matrix."""

    shape: tuple
    """Voxel dimensions."""

    zooms: tuple
    """Voxel sizes in mm (and TR in seconds for 4-D)."""

    dtype: str
    """Numpy dtype string."""

    file_name: str
    """Original uploaded file name."""

    orientation: str = ""
    """RAS / LAS / etc. orientation string."""

    is_4d: bool = False
    """True if the volume has a temporal dimension."""

    extra_meta: dict = field(default_factory=dict)
    """Any additional header fields (intent code, descrip, etc.)."""


@st.cache_data(show_spinner=False)
def load_nifti(file_bytes: bytes, file_name: str) -> NiftiVolume:
    """Load a NIfTI volume from raw bytes.

    Parameters
    ----------
    file_bytes:
        Raw bytes from ``uploaded_file.getvalue()``.
    file_name:
        Original file name (used to infer compression suffix).

    Returns
    -------
    NiftiVolume
        Loaded volume with metadata.

    Raises
    ------
    ValueError
        If nibabel cannot parse the file.
    """
    suffix = ".nii.gz" if file_name.lower().endswith(".gz") else ".nii"
    tmp_path: Optional[str] = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        img = nib.load(tmp_path)
        data: np.ndarray = img.get_fdata(dtype=np.float32)
        header = img.header

        # Orientation string (e.g. "RAS+")
        try:
            orientation = "".join(nib.aff2axcodes(img.affine))
        except Exception:
            orientation = "unknown"

        # Extra header fields
        extra_meta: dict = {}
        try:
            extra_meta["intent_code"] = int(header.get_intent()[0])
            extra_meta["intent_name"] = str(header.get_intent()[2])
        except Exception:
            pass
        try:
            extra_meta["descrip"] = header["descrip"].tobytes().decode("utf-8").strip("\x00")
        except Exception:
            pass

        return NiftiVolume(
            data=data,
            affine=img.affine,
            shape=tuple(data.shape),
            zooms=tuple(float(z) for z in header.get_zooms()),
            dtype=str(data.dtype),
            file_name=file_name,
            orientation=orientation,
            is_4d=data.ndim == 4,
            extra_meta=extra_meta,
        )

    except Exception as exc:
        raise ValueError(f"Failed to load NIfTI '{file_name}': {exc}") from exc

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def normalize_slice(arr: np.ndarray, percentile_clip: float = 99.5) -> np.ndarray:
    """Normalize a 2-D slice to [0, 1] with percentile-based clipping.

    Parameters
    ----------
    arr:
        2-D float array (single MRI slice).
    percentile_clip:
        Upper percentile used for clipping before normalization.
        Reduces the influence of bright outlier voxels.

    Returns
    -------
    np.ndarray
        Float32 array in [0, 1].
    """
    p_low = float(np.percentile(arr, 100.0 - percentile_clip))
    p_high = float(np.percentile(arr, percentile_clip))
    if p_high <= p_low:
        return np.zeros_like(arr, dtype=np.float32)
    clipped = np.clip(arr, p_low, p_high)
    return ((clipped - p_low) / (p_high - p_low)).astype(np.float32)


def extract_slice(
    data: np.ndarray,
    axis: int,
    index: int,
    volume_index: int = 0,
) -> np.ndarray:
    """Extract a 2-D slice from a 3-D or 4-D volume.

    Parameters
    ----------
    data:
        3-D (X, Y, Z) or 4-D (X, Y, Z, T) array.
    axis:
        0 = sagittal, 1 = coronal, 2 = axial.
    index:
        Slice index along the chosen axis.
    volume_index:
        Time-point index for 4-D volumes.

    Returns
    -------
    np.ndarray
        2-D slice array.
    """
    vol = data[..., volume_index] if data.ndim == 4 else data
    slices: tuple = (slice(None), slice(None), slice(None))
    slices = list(slices)  # type: ignore[assignment]
    slices[axis] = index  # type: ignore[index]
    return np.rot90(vol[tuple(slices)])  # type: ignore[index]

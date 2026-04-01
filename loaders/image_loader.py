"""
loaders/image_loader.py
-----------------------
Load 2-D image files (PNG, JPG, SVG) into numpy arrays and stack
them into a 3-D NIfTI volume for use with the NiiVue viewer.
"""

from __future__ import annotations

import io
import os
import tempfile
from typing import List, Optional, Tuple

import nibabel as nib
import numpy as np
from PIL import Image


def load_2d_image(file_bytes: bytes, filename: str) -> np.ndarray:
    """Load a PNG or JPG file from raw bytes as a grayscale float32 array.

    Parameters
    ----------
    file_bytes : bytes
        Raw bytes from UploadedFile.getvalue().
    filename : str
        Original filename (used only for error messages).

    Returns
    -------
    np.ndarray
        2-D float32 array, shape (H, W), values in [0.0, 1.0].

    Raises
    ------
    ValueError
        If PIL cannot decode the image.
    """
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("L")  # force grayscale
        return np.array(img, dtype=np.float32) / 255.0
    except Exception as exc:
        raise ValueError(f"Cannot load image '{filename}': {exc}") from exc


def svg_to_array(svg_bytes: bytes) -> np.ndarray:
    """Rasterize an SVG to a grayscale float32 array.

    Tries svglib+reportlab first (pure-Python, no system libs needed),
    then cairosvg as fallback. Raises ValueError with install instructions
    if neither is available.

    Parameters
    ----------
    svg_bytes : bytes
        Raw SVG file bytes.

    Returns
    -------
    np.ndarray
        2-D float32 array, shape (H, W), values in [0.0, 1.0].

    Raises
    ------
    ValueError
        If no SVG rasterization backend is available, or rasterization fails.
    """
    # ── Attempt 1: svglib + reportlab ─────────────────────────────────────────
    try:
        from svglib.svglib import svg2rlg  # type: ignore
        from reportlab.graphics import renderPM  # type: ignore

        with tempfile.NamedTemporaryFile(delete=False, suffix=".svg") as tmp:
            tmp.write(svg_bytes)
            tmp_path = tmp.name
        try:
            drawing = svg2rlg(tmp_path)
            if drawing is None:
                raise ValueError("svglib returned None — SVG may be malformed.")
            png_buf = io.BytesIO()
            renderPM.drawToFile(drawing, png_buf, fmt="PNG")
            png_buf.seek(0)
            return load_2d_image(png_buf.read(), "svg_rasterized.png")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    except ImportError:
        pass  # svglib not installed, try next

    # ── Attempt 2: cairosvg ───────────────────────────────────────────────────
    try:
        import cairosvg  # type: ignore

        png_bytes = cairosvg.svg2png(bytestring=svg_bytes)
        return load_2d_image(png_bytes, "svg_rasterized.png")

    except ImportError:
        pass  # cairosvg not installed

    # ── Neither available ─────────────────────────────────────────────────────
    raise ValueError(
        "No SVG rasterization backend found. "
        "Install one of: 'pip install svglib reportlab'  or  'pip install cairosvg'"
    )


def stack_slices_to_nifti(
    arrays: List[np.ndarray],
    axis: int = 2,
    voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> bytes:
    """Stack 2-D grayscale arrays into a 3-D NIfTI volume and return raw bytes.

    Parameters
    ----------
    arrays : List[np.ndarray]
        Ordered list of 2-D float32 arrays (H, W). All must have the same
        shape; if not, each is resized to match arrays[0] via PIL LANCZOS.
    axis : int
        Axis along which to stack slices.
        0 = sagittal (X), 1 = coronal (Y), 2 = axial (Z).
    voxel_size : Tuple[float, float, float]
        Voxel dimensions (dx, dy, dz) in mm for the NIfTI affine/header.

    Returns
    -------
    bytes
        Raw NIfTI-1 file bytes (.nii, uncompressed).

    Raises
    ------
    ValueError
        If arrays is empty.
    """
    if not arrays:
        raise ValueError("No slices provided to stack.")

    target_shape = arrays[0].shape  # (H, W)

    # Resize any mismatched slices to match the first slice
    resized: List[np.ndarray] = []
    for arr in arrays:
        if arr.shape != target_shape:
            pil_img = Image.fromarray(
                (np.clip(arr, 0, 1) * 255).astype(np.uint8), mode="L"
            )
            # PIL size is (W, H), numpy shape is (H, W)
            pil_img = pil_img.resize(
                (target_shape[1], target_shape[0]), Image.LANCZOS
            )
            arr = np.array(pil_img, dtype=np.float32) / 255.0
        resized.append(arr)

    # Stack along the requested axis into a 3-D volume
    volume = np.stack(resized, axis=axis)

    # Build affine: diagonal voxel-size matrix (no rotation)
    dx, dy, dz = voxel_size
    affine = np.diag([dx, dy, dz, 1.0]).astype(np.float64)

    img = nib.Nifti1Image(volume.astype(np.float32), affine=affine)
    img.header.set_zooms(voxel_size)

    # Write to tempfile and read bytes back (mirrors nifti_loader.py pattern)
    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as tmp:
            tmp_path = tmp.name
        nib.save(img, tmp_path)
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

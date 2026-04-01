"""
panels/niivue_panel.py
----------------------
NiiVue multiplanar viewer panel.

Embeds a full NiiVue.js instance (v0.57) inside a Streamlit iframe
and passes the uploaded NIfTI volume as a base64-encoded blob URL.
Provides view-mode, colormap, and crosshair controls above the viewer.

Also renders a matplotlib fallback viewer (axial / coronal / sagittal)
for quick slice navigation that doesn't depend on WebGL.

2D Slice Reconstruction
-----------------------
When PNG / JPG / SVG slices are uploaded (instead of a NIfTI), the panel
shows a thumbnail strip and stacking controls. On "Reconstruct 3D Volume"
the slices are stacked into a NIfTI and displayed in the same NiiVue viewer.
"""

from __future__ import annotations

import base64
import io
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from loaders.nifti_loader import load_nifti, normalize_slice, extract_slice, NiftiVolume
from loaders.image_loader import load_2d_image, svg_to_array, stack_slices_to_nifti


# ── NiiVue colormap options ───────────────────────────────────────────────────
COLORMAPS = [
    "gray", "hot", "cool", "inferno", "viridis",
    "plasma", "green", "red", "blue", "yellow",
]

VIEW_MODES = {
    "Multiplanar (axial + coronal + sagittal)": "multiplanar",
    "Axial only":                               "axial",
    "Coronal only":                             "coronal",
    "Sagittal only":                            "sagittal",
    "3-D render":                               "render",
}


def _build_niivue_html(
    b64_data: str,
    file_name: str,
    colormap: str,
    view_mode: str,
    show_crosshair: bool,
    height: int,
) -> str:
    """Return the full HTML string for the NiiVue iframe.

    Parameters
    ----------
    b64_data:
        Base64-encoded NIfTI file bytes.
    file_name:
        Original file name (used to set the NiiVue volume name).
    colormap:
        NiiVue colormap string.
    view_mode:
        One of 'multiplanar', 'axial', 'coronal', 'sagittal', 'render'.
    show_crosshair:
        Whether to display the crosshair overlay.
    height:
        Iframe height in pixels.
    """
    is_compressed_js = "true" if file_name.lower().endswith(".gz") else "false"

    slice_type_map = {
        "multiplanar": "nv.sliceTypeMultiplanar",
        "axial":       "nv.sliceTypeAxial",
        "coronal":     "nv.sliceTypeCoronal",
        "sagittal":    "nv.sliceTypeSagittal",
        "render":      "nv.sliceTypeRender",
    }
    slice_type_js = slice_type_map.get(view_mode, "nv.sliceTypeMultiplanar")
    crosshair_color = "[1,0,0,1]" if show_crosshair else "[0,0,0,0]"

    return f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    * {{ margin:0; padding:0; box-sizing:border-box; }}
    html, body {{ width:100%; height:{height}px; overflow:hidden;
                  background:#080c14; }}
    #gl-canvas {{ width:100%; height:100%; display:block; }}
    #status {{
        position:absolute; top:8px; left:8px;
        font-family:'JetBrains Mono',monospace; font-size:11px;
        color:#4d9de0; background:rgba(8,12,20,0.8);
        padding:4px 8px; border-radius:3px;
        border:1px solid #1e2d45;
    }}
    #coords {{
        position:absolute; bottom:8px; left:8px;
        font-family:'JetBrains Mono',monospace; font-size:10px;
        color:#8fa3bc; background:rgba(8,12,20,0.75);
        padding:3px 7px; border-radius:3px;
    }}
  </style>
</head>
<body>
  <canvas id="gl-canvas"></canvas>
  <div id="status">Loading volume…</div>
  <div id="coords">x — y — z</div>

  <script type="module">
    import {{ Niivue }} from
      "https://unpkg.com/@niivue/niivue@0.57.0/dist/index.js";

    const statusEl = document.getElementById("status");
    const coordsEl = document.getElementById("coords");

    async function main() {{
      try {{
        const nv = new Niivue({{
          isResizeCanvas:    true,
          backColor:         [0.031, 0.047, 0.078, 1],
          crosshairColor:    {crosshair_color},
          crosshairWidth:    1,
          isColorbar:        true,
          colorbarHeight:    0.04,
          colorbarMargin:    0.05,
          textHeight:        0.03,
          isOrientCube:      true,
        }});

        await nv.attachTo("gl-canvas");

        // Decode base64 → Uint8Array
        const raw    = atob("{b64_data}");
        const bytes  = new Uint8Array(raw.length);
        for (let i = 0; i < raw.length; i++) {{
          bytes[i] = raw.charCodeAt(i);
        }}

        const blob = new Blob([bytes], {{ type:"application/octet-stream" }});
        const url  = URL.createObjectURL(blob);

        await nv.loadVolumes([{{
          url:          url,
          name:         "{file_name}",
          colormap:     "{colormap}",
          isCompressed: {is_compressed_js},
          opacity:      1.0,
        }}]);

        // Set view mode
        nv.setSliceType({slice_type_js});
        nv.updateGLVolume();

        statusEl.textContent = "✓ {file_name}";

        // Show crosshair coordinates on move
        nv.onLocationChange = (data) => {{
          if (data && data.string) {{
            coordsEl.textContent = data.string;
          }}
        }};

      }} catch (err) {{
        statusEl.style.color = "#ff6b6b";
        statusEl.textContent = "Error: " + err.message;
        console.error(err);
      }}
    }}

    main();
  </script>
</body>
</html>
"""


def _render_matplotlib_viewer(vol: NiftiVolume) -> None:
    """Render a 3-panel matplotlib slice viewer as a Streamlit fallback."""
    data = vol.data if vol.data.ndim == 3 else vol.data[..., 0]

    shape = data.shape
    st.caption(
        f"Shape: `{shape}` · Zooms: `{vol.zooms}` · "
        f"Orientation: `{vol.orientation}` · dtype: `{vol.dtype}`"
    )

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)

    with col_ctrl1:
        ax_idx = st.slider("Axial (Z)", 0, shape[2] - 1, shape[2] // 2, key="sl_ax")
    with col_ctrl2:
        cor_idx = st.slider("Coronal (Y)", 0, shape[1] - 1, shape[1] // 2, key="sl_cor")
    with col_ctrl3:
        sag_idx = st.slider("Sagittal (X)", 0, shape[0] - 1, shape[0] // 2, key="sl_sag")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5),
                             facecolor="#080c14", constrained_layout=True)

    slice_configs = [
        (extract_slice(data, 2, ax_idx),  "Axial",     f"z={ax_idx}"),
        (extract_slice(data, 1, cor_idx), "Coronal",   f"y={cor_idx}"),
        (extract_slice(data, 0, sag_idx), "Sagittal",  f"x={sag_idx}"),
    ]

    for ax, (slc, title, coord) in zip(axes, slice_configs):
        norm_slc = normalize_slice(slc)
        ax.imshow(norm_slc, cmap="gray", vmin=0, vmax=1, aspect="equal",
                  interpolation="bilinear")
        ax.set_title(f"{title}  ·  {coord}",
                     color="#8fa3bc", fontsize=9,
                     fontfamily="monospace", pad=4)
        ax.axis("off")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor="#080c14")
    plt.close(fig)
    buf.seek(0)
    st.image(buf, use_container_width=True)

    # ── Metadata table ────────────────────────────────────────────────────
    with st.expander("📋 NIfTI Header", expanded=False):
        import pandas as pd
        meta_rows = [
            ("File",        vol.file_name),
            ("Shape",       str(vol.shape)),
            ("Voxel size",  f"{tuple(round(z, 3) for z in vol.zooms)} mm"),
            ("Orientation", vol.orientation),
            ("Data type",   vol.dtype),
            ("4-D volume",  str(vol.is_4d)),
        ]
        for k, v in vol.extra_meta.items():
            meta_rows.append((k, str(v)))

        df_meta = pd.DataFrame(meta_rows, columns=["Field", "Value"])
        st.dataframe(df_meta, use_container_width=True, hide_index=True)


def _render_slice_reconstruction_ui(
    slice_files: list,
    colormap: str,
    view_mode: str,
    show_crosshair: bool,
    niivue_height: int,
) -> None:
    """Render the 2D-slice stacking controls and NiiVue viewer.

    Parameters
    ----------
    slice_files:
        Streamlit UploadedFile objects (PNG, JPG, or SVG).
    colormap, view_mode, show_crosshair, niivue_height:
        Viewer settings chosen in the shared control row.
    """
    if len(slice_files) == 1:
        st.warning(
            "Only 1 slice uploaded. The reconstructed volume will be a single-slice NIfTI. "
            "Upload more files for a meaningful 3D volume."
        )

    # ── Thumbnail preview strip ───────────────────────────────────────────────
    with st.expander("Slice Preview", expanded=True):
        n_thumbs = min(len(slice_files), 8)
        thumb_cols = st.columns(n_thumbs)
        for col, sf in zip(thumb_cols, slice_files[:8]):
            with col:
                try:
                    img_bytes = sf.getvalue()
                    if sf.name.lower().endswith(".svg"):
                        arr = svg_to_array(img_bytes)
                        pil_img = Image.fromarray(
                            (np.clip(arr, 0, 1) * 255).astype(np.uint8)
                        )
                    else:
                        pil_img = Image.open(io.BytesIO(img_bytes)).convert("L")
                    pil_img.thumbnail((64, 64))
                    st.image(pil_img, caption=sf.name[:10], use_container_width=False)
                except Exception:
                    st.caption(f"[{sf.name[:8]}]")
        if len(slice_files) > 8:
            st.caption(f"… and {len(slice_files) - 8} more")

    # ── Stacking controls ─────────────────────────────────────────────────────
    with st.expander("Stacking Controls", expanded=True):
        axis_label = st.radio(
            "Stack slices along axis",
            options=["Axial (Z)", "Coronal (Y)", "Sagittal (X)"],
            horizontal=True,
            key="img_stack_axis",
        )
        axis_map = {"Axial (Z)": 2, "Coronal (Y)": 1, "Sagittal (X)": 0}
        stack_axis = axis_map[axis_label]

        vc1, vc2, vc3 = st.columns(3)
        with vc1:
            vx = st.number_input("Voxel dx (mm)", min_value=0.01, value=1.0,
                                 step=0.1, key="img_vx")
        with vc2:
            vy = st.number_input("Voxel dy (mm)", min_value=0.01, value=1.0,
                                 step=0.1, key="img_vy")
        with vc3:
            vz = st.number_input("Voxel dz (mm)", min_value=0.01, value=1.0,
                                 step=0.1, key="img_vz")

        # ── Slice ordering ────────────────────────────────────────────────
        sorted_names = sorted(f.name for f in slice_files)
        default_order = {f.name: sorted_names.index(f.name) for f in slice_files}

        st.markdown("**Slice order** (0 = first; ties broken alphabetically):")
        n_order_cols = min(len(slice_files), 4)
        order_cols = st.columns(n_order_cols)
        order_vals: dict[str, int] = {}
        for i, sf in enumerate(slice_files):
            col = order_cols[i % n_order_cols]
            with col:
                order_vals[sf.name] = st.number_input(
                    sf.name[:12],
                    min_value=0,
                    max_value=len(slice_files) - 1,
                    value=default_order[sf.name],
                    step=1,
                    key=f"img_order_{sf.name}",
                )

        ordered_files = sorted(slice_files, key=lambda f: (order_vals[f.name], f.name))

        reconstruct_btn = st.button(
            "Reconstruct 3D Volume",
            type="primary",
            key="img_reconstruct_btn",
        )

    # ── Reconstruct on button click ───────────────────────────────────────────
    if reconstruct_btn:
        with st.spinner("Stacking slices and generating NIfTI…"):
            try:
                arrays: List[np.ndarray] = []
                progress = st.progress(0)
                for i, sf in enumerate(ordered_files):
                    raw = sf.getvalue()
                    if sf.name.lower().endswith(".svg"):
                        arr = svg_to_array(raw)
                    else:
                        arr = load_2d_image(raw, sf.name)
                    arrays.append(arr)
                    progress.progress((i + 1) / len(ordered_files))
                progress.empty()

                nifti_bytes = stack_slices_to_nifti(
                    arrays,
                    axis=stack_axis,
                    voxel_size=(float(vx), float(vy), float(vz)),
                )
                st.session_state["reconstructed_nifti_bytes"] = nifti_bytes
                st.session_state["reconstructed_nifti_name"] = "reconstructed.nii"

            except Exception as exc:
                st.error(f"Reconstruction failed: {exc}")
                st.session_state.pop("reconstructed_nifti_bytes", None)
                return

    # ── Show NiiVue viewer if bytes are ready ─────────────────────────────────
    nifti_bytes = st.session_state.get("reconstructed_nifti_bytes")
    nifti_name  = st.session_state.get("reconstructed_nifti_name", "reconstructed.nii")

    if nifti_bytes:
        shape_hint = ""
        try:
            import tempfile as _tf
            import os as _os
            import nibabel as _nib
            with _tf.NamedTemporaryFile(delete=False, suffix=".nii") as _tmp:
                _tmp.write(nifti_bytes)
                _tpath = _tmp.name
            _img = _nib.load(_tpath)
            shape_hint = str(_img.shape)
            _os.remove(_tpath)
        except Exception:
            pass

        st.success(
            f"✓ Reconstructed NIfTI{' · shape ' + shape_hint if shape_hint else ''} "
            f"· axis={stack_axis}, voxels=({vx:.2f}, {vy:.2f}, {vz:.2f}) mm"
        )

        b64 = base64.b64encode(nifti_bytes).decode("utf-8")
        html_code = _build_niivue_html(
            b64_data=b64,
            file_name=nifti_name,
            colormap=colormap,
            view_mode=view_mode,
            show_crosshair=show_crosshair,
            height=niivue_height,
        )
        components.html(html_code, height=niivue_height + 8, scrolling=False)


def render_niivue_panel(
    mri_file: Optional[object],
    slice_files: Optional[list] = None,
) -> None:
    """Render the full NiiVue viewer panel.

    Parameters
    ----------
    mri_file:
        Streamlit ``UploadedFile`` or ``None`` — direct NIfTI volume.
    slice_files:
        List of Streamlit ``UploadedFile`` objects (PNG/JPG/SVG) to stack
        into a 3D volume, or ``None`` / empty list.

    Priority: if *mri_file* is provided it takes precedence and the
    slice-reconstruction workflow is not shown.
    """
    st.markdown("### ⬡ NiiVue Multiplanar Viewer")

    slice_files = slice_files or []

    if mri_file is None and not slice_files:
        st.info(
            "Upload a NIfTI file (.nii or .nii.gz) **or** "
            "multiple 2D image slices (PNG / JPG / SVG) using the sidebar to begin."
        )
        return

    # ── Shared viewer controls ────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([2, 1, 1, 1])

    with ctrl1:
        view_label = st.selectbox(
            "View mode", list(VIEW_MODES.keys()), key="nv_view_mode"
        )
        view_mode = VIEW_MODES[view_label]

    with ctrl2:
        colormap = st.selectbox("Colormap", COLORMAPS, key="nv_colormap")

    with ctrl3:
        niivue_height = st.slider("Viewer height", 400, 900, 560, step=40, key="nv_height")

    with ctrl4:
        show_crosshair = st.checkbox("Crosshair", value=True, key="nv_crosshair")

    st.divider()

    # ── NIfTI path (existing behaviour — unchanged) ───────────────────────────
    if mri_file is not None:
        with st.spinner("Loading NIfTI volume…"):
            try:
                vol = load_nifti(mri_file.getvalue(), mri_file.name)
            except ValueError as exc:
                st.error(f"Failed to load volume: {exc}")
                return

        st.success(
            f"✓ Loaded `{vol.file_name}` — shape `{vol.shape}`, "
            f"voxels `{tuple(round(z,2) for z in vol.zooms[:3])}` mm"
        )
        st.session_state.nifti_meta = vol

        tab_webgl, tab_mpl = st.tabs(["🌐 NiiVue (WebGL)", "📊 Matplotlib (fallback)"])

        with tab_webgl:
            b64 = base64.b64encode(mri_file.getvalue()).decode("utf-8")
            html_code = _build_niivue_html(
                b64_data=b64,
                file_name=mri_file.name,
                colormap=colormap,
                view_mode=view_mode,
                show_crosshair=show_crosshair,
                height=niivue_height,
            )
            components.html(html_code, height=niivue_height + 8, scrolling=False)

        with tab_mpl:
            _render_matplotlib_viewer(vol)

    # ── 2D slices path (new) ──────────────────────────────────────────────────
    elif slice_files:
        _render_slice_reconstruction_ui(
            slice_files=slice_files,
            colormap=colormap,
            view_mode=view_mode,
            show_crosshair=show_crosshair,
            niivue_height=niivue_height,
        )

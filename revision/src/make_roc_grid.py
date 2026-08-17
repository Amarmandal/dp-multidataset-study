#!/usr/bin/env python3
"""Reconcile the six per-dataset LiRA log-log ROC panels into one 2x3 figure.

    python revision/src/make_roc_grid.py

Inputs  : revision/figures/png/loglog_roc_<DS>.png   (six 900x900 rasters)
Outputs : revision/figures/pdf/loglog_roc_grid.pdf   (vector container)
          revision/figures/png/loglog_roc_grid.png   (600 dpi)

IMPORTANT -- what "vector" means here. The source ROC curves exist only as
PNG: Attack/LiRA/run_lira.py::_loglog_roc() draws them from the in-memory
`_roc_fpr` / `_roc_tpr` arrays and saves a raster; those arrays are never
written to disk (Attack/LiRA/results/<DS>/ holds the CSV + PNGs only). So this
script composes a PDF whose *frame* -- the panel labels -- is true vector text,
while each panel is the original raster embedded losslessly at native
resolution. A fully vector figure requires re-running LiRA with the curve
arrays persisted; see the note at the bottom of this file.

There is no figure-level title: the grid carries only single-line panel labels,
so the plots get the whole canvas. The wording that used to sit in the six
in-image titles ("non-private (standard) targets; points above the diagonal at
low FPR = worst-case leakage") belongs in the manuscript caption.

The in-image two-line title is removed rather than cropped. Row-scan of the
panel: title line 1 spans rows 26-49, line 2 rows 56-79 (cols 121-855), and
the topmost y-tick label "10^0" occupies rows 77-86 at cols 67-100 -- i.e. the
title's lower line shares *rows* with that label (which is why the crop in
build_figures.py bailed) but not *columns*. Whiting out only the region right
of the y-axis label column and above the top spine therefore touches no tick
label, no spine, and no plotted pixel. Every assumption is asserted per image.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "figures"
PNG_DIR = FIG / "png"
PDF_DIR = FIG / "pdf"

# Ordered by N -- dataset size is the dominant explanatory variable in this
# study, so the grid reads small -> large left-to-right, top-to-bottom.
PANELS = [
    ("GALLSTONE",    "Gallstone Status",        319),
    ("BCP",          "Breast Cancer Wisconsin", 569),
    ("CANCER_RISK",  "Cancer Risk Prediction",  1_500),
    ("KIDNEY_STONE", "Kidney Stone Risk",       4_000),
    ("LUNG_CANCER",  "Lung Cancer Risk Level",  50_000),
    ("DIABETES",     "BRFSS 2015 Diabetes",     70_692),
]
LETTERS = "abcdef"

NCOLS, NROWS = 3, 2
DPI = 600           # PNG export; the PDF embeds panels at native resolution
PAD = 6             # white px kept around each panel's ink after tight-cropping


def load_panel(path: Path) -> tuple[np.ndarray, dict]:
    """Return the panel as RGB uint8 with its in-image title erased.

    Geometry is re-derived per image and asserted, so a regenerated source PNG
    with a different layout fails loudly instead of being silently defaced.
    """
    im = Image.open(path).convert("RGB")
    arr = np.asarray(im).copy()
    h, w, _ = arr.shape
    ink = np.asarray(im.convert("L")) < 250

    # Axes frame: first row / column whose ink spans >50% of the canvas.
    rows = np.flatnonzero(ink.sum(axis=1) > 0.5 * w)
    cols = np.flatnonzero(ink.sum(axis=0) > 0.5 * h)
    assert rows.size and cols.size, f"{path.name}: no axes frame found"
    top_spine, left_spine = int(rows[0]), int(cols[0])

    # Ink above the top spine splits into two column groups: the y-tick label
    # (left of the spine) and the title (right of it). Find the gap between
    # them and cut there.
    band = ink[:top_spine]
    label_cols = np.flatnonzero(band[:, :left_spine].any(axis=0))
    assert label_cols.size, f"{path.name}: no y-tick label above the top spine"
    label_right = int(label_cols.max())

    title_cols = np.flatnonzero(band.any(axis=0))
    title_cols = title_cols[title_cols > label_right]
    assert title_cols.size, f"{path.name}: no title ink found above the spine"
    cut = (label_right + int(title_cols.min())) // 2
    assert label_right < cut < left_spine, (
        f"{path.name}: y-tick label (ends col {label_right}) and title "
        f"(starts col {title_cols.min()}) are not column-separated"
    )

    erased = arr[:top_spine, cut:].copy()
    arr[:top_spine, cut:] = 255

    # Trim the source figure's own white margin on all four sides, keeping PAD
    # px around the remaining ink. This is what buys the extra panel size: the
    # 900x900 source carries ~60-90px of matplotlib margin per side, which in a
    # 2x3 grid reads as dead space between panels.
    left = np.asarray(Image.fromarray(arr).convert("L")) < 250
    ys, xs = np.flatnonzero(left.any(axis=1)), np.flatnonzero(left.any(axis=0))
    y0, y1 = max(0, int(ys.min()) - PAD), min(h, int(ys.max()) + 1 + PAD)
    x0, x1 = max(0, int(xs.min()) - PAD), min(w, int(xs.max()) + 1 + PAD)
    out = arr[y0:y1, x0:x1]

    info = {
        "top_spine": top_spine,
        "left_spine": left_spine,
        "erase_col": cut,
        "erased_px": int((erased < 250).any(axis=2).sum()),
        "bbox": (x0, y0, x1, y1),
        "size": (out.shape[1], out.shape[0]),
    }
    return out, info


def main() -> int:
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    panels = []
    for ds, _, _ in PANELS:
        src = PNG_DIR / f"loglog_roc_{ds}.png"
        if not src.exists():
            print(f"ERROR: missing {src}", file=sys.stderr)
            return 1
        img, info = load_panel(src)
        panels.append(img)
        print(f"  {ds:<13} title erased right of col {info['erase_col']} "
              f"({info['erased_px']} px), cropped to {info['size']}")

    ph, pw = panels[0].shape[:2]
    assert all(p.shape[:2] == (ph, pw) for p in panels), \
        "panels differ in size; the grid assumes a common source geometry"

    # No suptitle and single-line panel labels: the figure-level caption in the
    # manuscript carries "non-private (standard) targets; points above the
    # diagonal at low FPR = worst-case leakage". All the space goes to the plots.
    panel_w_in = 3.55
    panel_h_in = panel_w_in * ph / pw
    fig, axes = plt.subplots(
        NROWS, NCOLS,
        figsize=(NCOLS * panel_w_in, NROWS * (panel_h_in + 0.24)),
    )

    for ax, img, (ds, nice, n), letter in zip(axes.ravel(), panels, PANELS, LETTERS):
        ax.imshow(img, interpolation="lanczos", resample=True)
        ax.set_axis_off()
        ax.set_title(f"({letter}) {nice}  (N = {n:,})", fontsize=10.5, pad=4)

    fig.tight_layout(pad=0.25, h_pad=0.9, w_pad=0.6)

    pdf = PDF_DIR / "loglog_roc_grid.pdf"
    png = PNG_DIR / "loglog_roc_grid.png"
    fig.savefig(pdf)                       # vector text + embedded rasters
    fig.savefig(png, dpi=DPI)
    plt.close(fig)
    print(f"wrote {pdf}")
    print(f"wrote {png}  ({DPI} dpi)")
    return 0


# --------------------------------------------------------------------------
# To get a genuinely all-vector version (curves as paths, not pixels), the ROC
# arrays have to survive the LiRA run. In Attack/LiRA/run_lira.py::build_report
# the `targets` list still carries "_roc_fpr" / "_roc_tpr" per target; dumping
# the standard-variant entries to results/<DS>/<DS>_roc_curves.json there would
# make a from-data grid possible without re-running the attack a second time.
# Re-running costs ~50 min/dataset and re-attacks the same exported artifacts,
# so the numbers would be unchanged apart from shadow-model seeding.
# --------------------------------------------------------------------------

if __name__ == "__main__":
    raise SystemExit(main())

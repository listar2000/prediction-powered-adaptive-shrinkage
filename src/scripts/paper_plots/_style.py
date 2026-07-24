"""Shared plot styling for paper figures.

All paper-ready scripts in this directory should import labels/colors/markers
from here so the camera-ready visuals stay coherent across figures.

The mapping below covers every CI method that may appear:

    mle_ci          Classical
    pred_mean_ci    Prediction Mean (a.k.a. Classic-Bias / "predict only")
    ppi_ci          PPI
    pt_ci           Power-Tuned PPI (PT)
    pt_oracle       Oracle PT (synthetic only)
    eb_pt_ci        Rebiased PT, parametric prior  (ours)
    eb_npmle_pt_ci  Rebiased PT, NPMLE prior       (ours)

Colleague's CSVs use slightly different naming. ``CSV_TO_INTERNAL`` is the
canonical translation to the internal keys above.
"""
from __future__ import annotations

# --- Names ------------------------------------------------------------------

CSV_TO_INTERNAL = {
    "classic":      "mle_ci",
    "classic_bias": "pred_mean_ci",
    "ppi":          "ppi_ci",
    "pt":           "pt_ci",
    "pt_oracle":    "pt_oracle",
    "pt_param":     "eb_pt_ci",
    "pt_npmle":     "eb_npmle_pt_ci",
}

METHOD_LABELS = {
    "mle_ci":         "Classical",
    "pred_mean_ci":   "Pred Mean",
    "ppi_ci":         "PPI",
    "pt_ci":          "PT",
    "pt_oracle":      "Oracle",
    # ``(ours)`` rendered via mathtext so the word lands in bold.
    "eb_pt_ci":       r"RB Normal ($\mathbf{ours}$)",
    "eb_npmle_pt_ci": r"RB NPMLE ($\mathbf{ours}$)",
}

METHOD_COLORS = {
    "mle_ci":         "#1f77b4",  # blue
    "pred_mean_ci":   "#bcbd22",  # olive
    "ppi_ci":         "#2ca02c",  # green
    "pt_ci":          "#d62728",  # red
    "pt_oracle":      "#7f7f7f",  # gray
    "eb_pt_ci":       "#9467bd",  # purple
    "eb_npmle_pt_ci": "#e377c2",  # pink
}

METHOD_MARKERS = {
    "mle_ci":         "o",
    "pred_mean_ci":   "v",
    "ppi_ci":         "s",
    "pt_ci":          "D",
    "pt_oracle":      "h",
    "eb_pt_ci":       "P",
    "eb_npmle_pt_ci": "*",
}

# --- Numerical/styling constants -------------------------------------------

# ``elinewidth`` and ``capthick`` only fire when ``yerr`` is passed to
# ``ax.errorbar``, so leaving them in ERR_KW is safe even on plots that
# do not draw error bars; they just stay dormant.
ERR_KW = dict(capsize=2.5, linewidth=1.4, markersize=6,
              elinewidth=0.9, capthick=0.9)
LABEL_FS = 13
TICK_FS = 11
LEGEND_FS = 13
COLTITLE_FS = 14
SAVE_DPI = 150

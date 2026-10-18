"""Single source of truth for matplotlib styling — IEEE TAC conventions."""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

WONG = ["#000000", "#E69F00", "#56B4E9", "#009E73",
        "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]

METHOD_COLOR = {
    "Ours": WONG[0],
    "B1":   WONG[1],
    "B2":   WONG[2],
    "B3":   WONG[3],
    "B4":   WONG[5],
    "B5":   "#7F7F7F",
}

METHOD_LINESTYLE = {
    "Ours": "-",
    "B1":   "--",
    "B2":   "-.",
    "B3":   ":",
    "B4":   "-",
    "B5":   "-",
}

METHOD_MARKER = {
    "Ours": "o",
    "B1":   "s",
    "B2":   "^",
    "B3":   "v",
    "B4":   "D",
    "B5":   "x",
}

METHOD_LABEL = {
    "Ours": "Ours",
    "B1":   "B1 (Nominal-Sim)",
    "B2":   "B2 (DR)",
    "B3":   "B3 (Direct-Real)",
    "B4":   "B4 (Black-Box)",
    "B5":   "B5 (Oracle-State)",
}

# Figure sizes (inches)
SINGLE = (3.4, 2.4)
DOUBLE = (7.0, 2.6)
LEARNING = (3.4, 2.0)


def apply():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def panel_label(ax, label, x=0.03, y=0.95):
    """Place a panel label `(a)`, `(b)`, ... inside the upper-left corner."""
    ax.text(
        x, y, f"({label})", transform=ax.transAxes,
        ha="left", va="top", fontsize=9,
    )


def savefig(fig, path, **kwargs):
    """Save a figure as PDF only — never PNG."""
    if not str(path).endswith(".pdf"):
        path = str(path) + ".pdf"
    fig.savefig(path, format="pdf", **kwargs)
    plt.close(fig)

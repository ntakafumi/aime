"""
aime_xai.style
==============

Signature visual-design system for AIME (Approximate Inverse Model Explanations).

The goal of this module is *visual identity*: every figure produced by
``aime_xai`` should be immediately recognisable as an AIME explanation and look
like a hand-crafted, publication-grade graphic rather than a default
matplotlib plot.

Design language
---------------
AIME is an **inverse-operator** method: it reads explanations *backwards*, from
the model output ``y`` to the input ``x`` through the approximate inverse
operator ``A_dagger`` (A^\\dagger).  The visual language encodes that idea:

* a warm "paper" canvas with no chart-junk spines,
* a zero-centred **diverging** palette (indigo for negative / inverse pull,
  coral for positive contribution) that is colour-vision friendly and prints
  well in greyscale,
* gradient-filled, soft-shadowed marks instead of flat bars,
* curved **ribbons** that flow from outputs back to inputs (the operator made
  visible), and
* a discreet ``A†`` brand mark on every figure.

This module deliberately depends only on numpy + matplotlib so it stays light
and so the look is fully under our control (not seaborn defaults).

matplotlib is imported **lazily**: the palette constants below are plain strings,
the colormap objects are built on first access via the module-level
``__getattr__`` (PEP 562), and every drawing helper imports matplotlib locally.
This lets ``import aime_xai`` and the pure-computation paths run with only
numpy + pandas installed.
"""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
# Palette                                                                      #
# --------------------------------------------------------------------------- #
# Publication-friendly identity colours.  Indigo <-> warm-paper <-> coral.
INK        = "#1B2A4A"   # deep navy — text, axis, default ink
INK_SOFT   = "#5C6680"   # muted ink for secondary text / ticks
PAPER      = "#FFFFFF"   # canvas — pure white for publication
PANEL      = "#FFFFFF"   # plotting panel
GRIDLINE   = "#E6E6E6"   # soft neutral grid

# Publication mode: when True, figures carry NO title/subtitle headers and no
# brand mark — only in-figure captions (axis labels, legends, value labels,
# colorbar labels).  Background is white.  Call set_publication_mode(False) to
# restore the branded/titled "signature" look.
PUBLICATION = True


def set_publication_mode(flag: bool = True) -> None:
    """Toggle publication-clean styling (no titles/brand, white background)."""
    global PUBLICATION
    PUBLICATION = bool(flag)


def _top_rect() -> float:
    """Top of the tight-layout rect — reclaim title space in publication mode."""
    return 0.985 if PUBLICATION else 0.90

# brand accents
INDIGO     = "#21307A"   # negative / inverse direction
INDIGO_LT  = "#4F66C2"
CORAL      = "#D2552E"   # positive / forward contribution
CORAL_DK   = "#A4271C"
AMBER      = "#E0A23B"   # highlight / focus instance
TEAL       = "#2E7E8C"   # neutral categorical accent

# A categorical sequence for per-class marks (works on the paper canvas)
CLASS_CYCLE = [
    "#21307A", "#D2552E", "#2E7E8C", "#E0A23B", "#7A4FA0",
    "#4F66C2", "#C2362B", "#3F8E6E", "#B0762A", "#9A4C6B",
]

# --------------------------------------------------------------------------- #
# Colormaps                                                                    #
# --------------------------------------------------------------------------- #
_CMAPS: dict = {}


def _build_cmaps() -> dict:
    """Build and register the three AIME colormaps once (lazy; needs matplotlib).

    Returns the ``{name: Colormap}`` cache.  Idempotent."""
    if _CMAPS:
        return _CMAPS
    import matplotlib as mpl
    from matplotlib.colors import LinearSegmentedColormap
    _CMAPS["AIME_DIVERGING"] = LinearSegmentedColormap.from_list(
        "aime_diverging",
        ["#16235C", "#2E4A8B", "#5C7AC4", "#A9BBE2", PAPER,
         "#F2C8AC", "#E8895C", "#D2552E", "#8F2017"], N=256)
    _CMAPS["AIME_SEQ"] = LinearSegmentedColormap.from_list(  # indigo single-hue ramp
        "aime_seq",
        [PAPER, "#CDD6EC", "#8FA2D6", "#5468B5", "#2E3E8E", "#16235C"], N=256)
    _CMAPS["AIME_SEQ_WARM"] = LinearSegmentedColormap.from_list(  # warm ramp
        "aime_seq_warm",
        [PAPER, "#F3D9B6", "#E7A85F", "#D2552E", "#A4271C", "#5E120C"], N=256)
    for _cm in _CMAPS.values():
        try:
            mpl.colormaps.register(_cm, force=True)
        except Exception:
            pass
    return _CMAPS


def __getattr__(name):
    """PEP 562 lazy module attribute access: ``style.AIME_DIVERGING`` etc. build
    the colormap on first use without importing matplotlib at module load."""
    if name in ("AIME_DIVERGING", "AIME_SEQ", "AIME_SEQ_WARM"):
        return _build_cmaps()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def signed_norm(values, vcenter: float = 0.0):
    """A zero-centred :class:`TwoSlopeNorm` covering ``values`` symmetrically."""
    from matplotlib.colors import TwoSlopeNorm
    v = np.asarray(values, dtype=float)
    m = float(np.nanmax(np.abs(v))) if v.size else 1.0
    if m <= 0 or not np.isfinite(m):
        m = 1.0
    return TwoSlopeNorm(vmin=-m, vcenter=vcenter, vmax=m)


def signed_color(value: float, vmax: float) -> tuple:
    """Map a single signed value to an AIME diverging colour."""
    if vmax <= 0 or not np.isfinite(vmax):
        vmax = 1.0
    t = 0.5 + 0.5 * np.clip(value / vmax, -1.0, 1.0)
    return _build_cmaps()["AIME_DIVERGING"](t)


# --------------------------------------------------------------------------- #
# Global style                                                                 #
# --------------------------------------------------------------------------- #
def apply_aime_style() -> None:
    """Install the AIME look as matplotlib rcParams (idempotent)."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.facecolor":  PAPER,
        "savefig.facecolor":  PAPER,
        "axes.facecolor":     PANEL,
        "figure.dpi":         120,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",

        "font.family":        ["DejaVu Sans"],
        "font.size":          11,
        "text.color":         INK,
        "axes.labelcolor":    INK,
        "axes.edgecolor":     GRIDLINE,
        "axes.linewidth":     1.0,
        "axes.titlecolor":    INK,
        "axes.titlesize":     14,
        "axes.titleweight":   "bold",
        "axes.labelsize":     11.5,
        "axes.labelweight":   "medium",

        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.spines.left":   False,
        "axes.spines.bottom": False,

        "xtick.color":        INK_SOFT,
        "ytick.color":        INK_SOFT,
        "xtick.labelsize":    10,
        "ytick.labelsize":    10,
        "xtick.major.size":   0,
        "ytick.major.size":   0,

        "axes.grid":          True,
        "axes.grid.axis":     "x",
        "grid.color":         GRIDLINE,
        "grid.linewidth":     0.9,
        "grid.alpha":         0.9,

        "legend.frameon":     False,
        "legend.fontsize":    10,
    })


# --------------------------------------------------------------------------- #
# Figure / axes construction                                                   #
# --------------------------------------------------------------------------- #
def new_figure(figsize=(10, 6), nrows=1, ncols=1, **kw):
    """Create a paper-styled figure + axes with the AIME look applied."""
    import matplotlib.pyplot as plt
    apply_aime_style()
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize,
                           facecolor=PAPER, **kw)
    for a in np.atleast_1d(np.array(ax)).ravel():
        a.set_facecolor(PANEL)
    return fig, ax


def style_title(ax, title: str, subtitle: str | None = None,
                accent: str = CORAL) -> None:
    """Editorial-style title: bold title, an accent rule, optional subtitle.

    Drawn in axes-fraction coords above the plotting area so it reads like a
    figure header rather than a matplotlib title.

    In publication mode this is a no-op (no title/subtitle text on the figure)
    and it does NOT touch any existing axes title, so per-panel in-graph
    captions set via ``ax.set_title`` are preserved.
    """
    if PUBLICATION:
        return
    ax.set_title("")  # clear any default before drawing the custom header
    ax.text(0.0, 1.16, title, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=15.5, fontweight="bold", color=INK, zorder=5)
    if subtitle:
        ax.text(0.0, 1.085, subtitle, transform=ax.transAxes, ha="left",
                va="bottom", fontsize=10.5, color=INK_SOFT, zorder=5)
    # accent rule
    ax.plot([0.0, 0.055], [1.135, 1.135], transform=ax.transAxes,
            color=accent, lw=3.2, solid_capstyle="round",
            clip_on=False, zorder=6)


def add_brand(fig, tag: str = "AIME · A†",
              note: str = "Approximate Inverse Model Explanations") -> None:
    """Stamp the AIME brand mark in the lower-right corner of the figure.

    No-op in publication mode (kept off publication figures by request)."""
    if PUBLICATION:
        return
    fig.text(0.995, 0.015, tag, ha="right", va="bottom",
             fontsize=9.5, fontweight="bold", color=INDIGO, alpha=0.9)
    fig.text(0.995, 0.048, note, ha="right", va="bottom",
             fontsize=7.0, color=INK_SOFT, alpha=0.7)


# --------------------------------------------------------------------------- #
# Custom marks                                                                 #
# --------------------------------------------------------------------------- #
def gradient_hbar(ax, y, value, *, height=0.62, vmax=1.0, zero=0.0,
                  radius=0.012, mode="diverging", label=None, zorder=3):
    """Draw a single horizontal bar from ``zero`` to ``value`` filled with an
    AIME gradient and a soft drop shadow.

    This replaces flat ``bar``/``barplot`` marks with a custom, branded look.

    Parameters
    ----------
    mode : {"diverging", "sequential"}
        ``"diverging"`` fills with the indigo/coral signed colormap (use for
        signed contributions).  ``"sequential"`` fills with the warm importance
        ramp from ``zero`` to ``value`` (use for non-negative magnitudes).
    label : str, optional
        Override for the value chip text (defaults to a signed/plain number).
    """
    from matplotlib.colors import TwoSlopeNorm, Normalize
    from matplotlib.patches import FancyBboxPatch
    _cm = _build_cmaps()
    if vmax <= 0 or not np.isfinite(vmax):
        vmax = 1.0
    # a genuinely-zero value draws NO bar (only the "0.00" label), so a flat /
    # near-machine-zero quantity never looks like a full bar.
    if abs(value - zero) < 1e-9:
        txt = label if label is not None else (
            "0.00" if mode == "sequential" else "+0.00")
        ax.text(zero + 0.018 * vmax, y, txt, ha="left", va="center",
                fontsize=9.5, fontweight="bold", color=INK_SOFT, zorder=zorder + 2)
        return
    # enforce a minimum visible span so near-zero bars still render cleanly and
    # never produce a rounded clip-path smaller than its corner radius (which
    # makes the path bbox explode under bbox_inches='tight').
    min_w = 0.012 * vmax
    if abs(value - zero) < min_w:
        draw_val = zero + (1.0 if value >= zero else -1.0) * min_w
    else:
        draw_val = value
    x0, x1 = (zero, draw_val) if draw_val >= zero else (draw_val, zero)
    width = x1 - x0
    radius = float(min(radius, 0.45 * abs(width), 0.45 * height))

    if mode == "sequential":
        # warm intensity from base (light) to tip (deep); encodes magnitude
        grad = np.linspace(0.0, abs(draw_val - zero), 256).reshape(1, -1)
        if draw_val < zero:
            grad = grad[:, ::-1]
        cmap = _cm["AIME_SEQ_WARM"]
        norm = Normalize(vmin=0.0, vmax=vmax)
    else:
        # indigo for negative span, coral for positive span
        grad = np.linspace(draw_val, zero, 256).reshape(1, -1)
        if draw_val < zero:
            grad = grad[:, ::-1]
        cmap = _cm["AIME_DIVERGING"]
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    # soft shadow
    shadow = FancyBboxPatch((x0, y - height / 2 - 0.015), width, height,
                            boxstyle=f"round,pad=0,rounding_size={radius}",
                            linewidth=0, facecolor=INK, alpha=0.10,
                            mutation_aspect=1, zorder=zorder - 1)
    shadow.set_clip_on(False)
    ax.add_patch(shadow)

    im = ax.imshow(grad, extent=(x0, x1, y - height / 2, y + height / 2),
                   aspect="auto", cmap=cmap, norm=norm, zorder=zorder)
    clip = FancyBboxPatch((x0, y - height / 2), width, height,
                          boxstyle=f"round,pad=0,rounding_size={radius}",
                          linewidth=0, transform=ax.transData)
    im.set_clip_path(clip)
    # thin outline
    outline = FancyBboxPatch((x0, y - height / 2), width, height,
                             boxstyle=f"round,pad=0,rounding_size={radius}",
                             linewidth=1.0, edgecolor="#FFFFFF",
                             facecolor="none", alpha=0.6, zorder=zorder + 1)
    ax.add_patch(outline)

    # value chip
    txt = label if label is not None else (
        f"{value:.2f}" if mode == "sequential" else f"{value:+.2f}")
    tx = value + (0.018 * vmax if value >= zero else -0.018 * vmax)
    ha = "left" if value >= zero else "right"
    ax.text(tx, y, txt, ha=ha, va="center", fontsize=9.5,
            fontweight="bold", color=INK, zorder=zorder + 2)


def class_hbar(ax, y, value, color, *, height=0.5, vmax=1.0, radius=0.01,
               zorder=3, edge="white"):
    """Draw a rounded horizontal bar 0→value in a single solid *class* colour.

    Used for per-class grouped feature-importance bars, where colour encodes the
    output class (not the sign — sign is read from the direction).  A subtle drop
    shadow keeps it from looking like a flat matplotlib bar.
    """
    from matplotlib.patches import FancyBboxPatch
    if vmax <= 0 or not np.isfinite(vmax):
        vmax = 1.0
    if abs(value) < 1e-9:        # genuinely zero → draw nothing
        return
    min_w = 0.012 * vmax
    draw = value if abs(value) >= min_w else (min_w if value >= 0 else -min_w)
    x0, x1 = (0.0, draw) if draw >= 0 else (draw, 0.0)
    w = x1 - x0
    r = float(min(radius, 0.45 * abs(w), 0.45 * height))
    shadow = FancyBboxPatch((x0, y - height / 2 - 0.012), w, height,
                            boxstyle=f"round,pad=0,rounding_size={r}",
                            linewidth=0, facecolor=INK, alpha=0.08,
                            zorder=zorder - 1)
    shadow.set_clip_on(False)
    ax.add_patch(shadow)
    bar = FancyBboxPatch((x0, y - height / 2), w, height,
                         boxstyle=f"round,pad=0,rounding_size={r}",
                         linewidth=1.0, edgecolor=edge, facecolor=color,
                         alpha=0.93, zorder=zorder)
    ax.add_patch(bar)


def ribbon(ax, x0, y0, x1, y1, width, color, alpha=0.85, zorder=2):
    """Draw a smooth flowing ribbon (cubic Bezier) of half-thickness ``width``.

    Used to render the inverse operator ``A_dagger`` as flows from output
    (class) anchors back to input (feature) anchors.
    """
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path
    mx = (x0 + x1) / 2.0
    # top edge then bottom edge, closed
    verts = [
        (x0, y0 + width),
        (mx, y0 + width), (mx, y1 + width), (x1, y1 + width),
        (x1, y1 - width),
        (mx, y1 - width), (mx, y0 - width), (x0, y0 - width),
        (x0, y0 + width),
    ]
    codes = [Path.MOVETO,
             Path.CURVE4, Path.CURVE4, Path.CURVE4,
             Path.LINETO,
             Path.CURVE4, Path.CURVE4, Path.CURVE4,
             Path.CLOSEPOLY]
    patch = PathPatch(Path(verts, codes), facecolor=color, edgecolor="none",
                      alpha=alpha, zorder=zorder)
    ax.add_patch(patch)
    return patch


def node_label(ax, x, y, text, *, color=INK, align="left", weight="medium",
               size=10.5, pad=0.0):
    ha = {"left": "left", "right": "right", "center": "center"}[align]
    ax.text(x + pad, y, text, ha=ha, va="center", color=color,
            fontsize=size, fontweight=weight, zorder=5)


def soft_legend(ax, handles_labels, loc="upper right"):
    leg = ax.legend(*handles_labels, loc=loc, frameon=True, framealpha=0.92,
                    edgecolor=GRIDLINE, fontsize=9.5)
    leg.get_frame().set_facecolor(PANEL)
    return leg

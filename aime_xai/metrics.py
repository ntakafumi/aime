"""
aime_xai.metrics
================

**Evaluation metrics for inverse-operator explanations.**

Conventional XAI metrics (fidelity, faithfulness, infidelity, sensitivity,
deletion/insertion, …) almost all measure *how well an explanation matches the
forward model*. AIME takes the opposite stance — it asks *what an approximate
inverse operator can recover* — so it admits a family of evaluation axes that
forward-problem methods cannot even define. This module implements them.

This module **does not touch how the AIME family derives `A_dagger`**. It only
reads a fitted explainer (`A_dagger`, `scaler`) plus an evaluation set `(X, Y)`
and computes diagnostics from them.

Notation
--------
* ``X'`` — standardised inputs (``scaler.transform(X)``), shape (N, n).
* ``A`` — the inverse operator ``A_dagger``, shape (n, m).
* ``Y`` — model outputs (probabilities), shape (N, m).
* Reconstruction (in standardised space):  ``X̂' = Y Aᵀ``   (since x̂'_i = A·y_i).
* Local importance (raw, un-normalised):   ``L = X̂' ⊙ X'``  (Hadamard).

Metrics implemented
-------------------
1. **IRE**  Inverse Reconstruction Error           ``‖X' − X̂'‖_F / ‖X'‖_F``
   (point-wise reconstruction error, standardised space)
2. **IRR**  Information / structure Recovery Rate   ``tr(Cov(X̂)) / tr(Cov(X))``
   (variance/structure retained, original space — an axis *independent* of IRE)
3. **RIC**  Representative Instance Consistency     mean RBF-similarity of each
   instance to the representative instance ``A eₜ`` of its **predicted/output
   class** (``t = argmax_c Y_ic``, i.e. the model's output, NOT the ground-truth
   label)
4. **EC**   Explanation Coverage                    spread of the per-class
   operator weights (L1 / (n·L∞))
5. **LGC**  Local–Global Consistency                correlation between the global
   per-class weights and the class-conditional mean local importance
6. **CSI**  Class Separability of Explanations      Fisher ratio (between/within)
   of the per-instance local explanations grouped by class
7. **rank / effective rank** of the operator ``A``
8. **IES**  Inverse Explainability Score            composite ``α(1−IRE)+β·LGC₀₁+γ·RIC``

Every assumption that goes beyond the user-supplied formulae is stated in the
relevant docstring (e.g. the choice of standardised space, the "theoretical max"
for coverage, the Gaussian/variance proxy for mutual information).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# matplotlib is intentionally NOT imported here: every plotting method routes
# through ``aime_xai.style`` (figure creation) and ``AIME._finish`` (show/save),
# both of which import matplotlib lazily.  This keeps the metrics computations
# usable with only numpy + pandas installed.
from . import style as S


# ======================================================================= #
# Metric taxonomy                                                          #
#                                                                          #
# AIME is an *explanation operator*, not an input reconstructor            #
# (reconstruction ≠ explanation).  The evaluation is therefore split into  #
# **Core explanation metrics** (the quality of the explanation itself) and #
# **Secondary explanation-fidelity** metrics (reconstruction — useful      #
# context but not AIME's purpose).  Order below is the display order.       #
# ======================================================================= #
TIER_CORE = "Core explanation metrics"
TIER_SECONDARY = "Secondary (explanation fidelity)"

METRIC_TIER = {
    # metric key       (tier,           category)
    "LGC":            (TIER_CORE,      "Consistency"),
    "RIC":            (TIER_CORE,      "Stability"),
    "RIC_purity":     (TIER_CORE,      "Stability"),
    "EC":             (TIER_CORE,      "Coverage"),
    "CSI":            (TIER_CORE,      "Separability"),
    "rank":           (TIER_CORE,      "Simplicity"),
    "effective_rank": (TIER_CORE,      "Simplicity"),
    "IES":            (TIER_CORE,      "Composite"),   # Core-only composite
    "IRE":            (TIER_SECONDARY, "Reconstruction"),
    "IRR":            (TIER_SECONDARY, "Recovery"),
}


# ======================================================================= #
# Pure metric functions (array-in / number-out) — directly unit-testable   #
# ======================================================================= #
def reconstruct(A, Y):
    """Standardised-space reconstruction  X̂' = Y·Aᵀ  (rows x̂'_i = A·y_i)."""
    return np.asarray(Y, float) @ np.asarray(A, float).T


def ire(Xp, A, Y):
    """Inverse Reconstruction Error  ‖X' − X̂'‖_F / ‖X'‖_F  (>= 0).

    Computed in the standardised space, because that is exactly the space in
    which AIME fits ``x' ≈ A·y`` (least squares).  Returns the global Frobenius
    ratio.  IRE = 0 iff every standardised feature lies in the row space of Y.
    """
    Xp = np.asarray(Xp, float)
    R = Xp - reconstruct(A, Y)
    denom = np.linalg.norm(Xp)
    return float(np.linalg.norm(R) / denom) if denom > 0 else 0.0


def ire_per_instance(Xp, A, Y):
    """Per-instance residual norm ‖x'_i − A·y_i‖ (not divided — robust)."""
    Xp = np.asarray(Xp, float)
    return np.linalg.norm(Xp - reconstruct(A, Y), axis=1)


def variance_recovery_rate(Xhat, X):
    """IRR — Information / structure **Recovery** rate

        IRR = tr(Cov(X̂)) / tr(Cov(X))  =  Σ_j Var(X̂_j) / Σ_j Var(X_j)

    the share of the input's **total variance (structure)** that the inverse
    reconstruction reproduces.  Computed in the **original feature space** (so it
    is weighted by each feature's true variance).

    This is deliberately a *different axis* from IRE, not ``1 − IRE²``: IRE is a
    point-wise reconstruction error (scale-free, unweighted), whereas IRR is a
    variance-weighted structure-retention.  They decouple — e.g. a PCA-like
    operator can retain most of the variance (high IRR) while individual points
    are poorly reconstructed (high IRE), and vice versa.  (They coincide only in
    the degenerate case of equal feature variances.)

    Range caveat.  When the reconstruction is the orthogonal projection of each
    feature column onto the span of the outputs **evaluated on the same data the
    operator was fitted from**, we have ``0 ≤ Var(X̂_j) ≤ Var(X_j)`` and hence
    IRR ∈ [0, 1], with IRR = 1 iff every feature's variance is fully recovered.
    This bound is **not guaranteed in general**: with regularised operators
    (Ridge / Huber / Bayesian), or when IRR is computed on a *held-out*
    evaluation set, or under different normalisation/projection conditions, the
    ratio can exceed 1.  IRR should therefore be read as a structure-recovery
    ratio rather than a strictly bounded score.  Forward-attribution methods
    (LIME, SHAP) have no reconstruction and cannot define this quantity.
    """
    Xhat = np.asarray(Xhat, float); X = np.asarray(X, float)
    num = float(np.var(Xhat, axis=0).sum())
    den = float(np.var(X, axis=0).sum())
    return num / den if den > 0 else 0.0


def recovery_per_feature(Xp, A, Y):
    """Per-feature variance retention ρ_j = Var(X̂_j)/Var(X_j) = 1 − SSE_j/SST_j
    (scale-invariant, identical in standardised and original space)."""
    Xp = np.asarray(Xp, float)
    R = Xp - reconstruct(A, Y)
    sse = (R ** 2).sum(0)
    sst = (Xp ** 2).sum(0)          # X' columns are zero-mean ⇒ SST = N·var
    out = np.ones_like(sse)
    nz = sst > 0
    out[nz] = 1.0 - sse[nz] / sst[nz]
    return out


def coverage(A):
    """Per-class Explanation Coverage  EC_c = (Σ_i |a_{ic}|) / (n · max_i|a_{ic}|).

    Assumption (the "theoretical maximum"): each class column is normalised by
    its own peak |A| (exactly as in ``global_feature_importance``), so the
    maximum possible L1 mass over n features is n.  EC_c ∈ [1/n, 1]:
    1 ⇒ every feature contributes equally (maximal coverage / least sparse),
    1/n ⇒ a single feature dominates (sparse).  Returns an array of length m.
    """
    A = np.asarray(A, float)
    n, m = A.shape
    out = np.zeros(m)
    for c in range(m):
        mx = np.max(np.abs(A[:, c]))
        out[c] = (np.sum(np.abs(A[:, c])) / mx) / n if mx > 0 else 0.0
    return out


def effective_rank(A, tol=None):
    """Numerical rank and **effective rank** of A.

    effective rank ``r_eff = exp(H(p))`` with ``p = σ / Σσ`` and Shannon
    entropy H (Roy & Vetterli, 2007).  r_eff ∈ [1, rank]; low ⇒ the operator is
    explained by a few latent factors, high ⇒ many.  (This decomposes the
    operator *after* it has been derived; it does not alter AIME's estimation.)
    """
    A = np.asarray(A, float)
    s = np.linalg.svd(A, compute_uv=False)
    s = s[s > 0]
    if s.size == 0:
        return dict(rank=0, effective_rank=0.0, singular_values=s)
    if tol is None:
        tol = max(A.shape) * np.finfo(float).eps * s[0]
    rank = int((s > tol).sum())
    p = s / s.sum()
    H = -np.sum(p * np.log(p))
    return dict(rank=rank, effective_rank=float(np.exp(H)), singular_values=s)


def class_separability(L, labels):
    """Class Separability of Explanations — Fisher ratio (trace form)

    ``CSI = tr(S_between) / tr(S_within)`` of the per-instance local
    explanations ``L`` grouped by ``labels``.  S_between = Σ_c N_c‖μ_c−μ‖²,
    S_within = Σ_c Σ_{i∈c}‖L_i−μ_c‖².  Higher ⇒ explanations differ more by
    class.  Returns ``np.inf`` if within-class scatter is exactly zero.
    """
    L = np.asarray(L, float); labels = np.asarray(labels)
    mu = L.mean(0)
    Sb = 0.0; Sw = 0.0
    for c in np.unique(labels):
        Lc = L[labels == c]
        muc = Lc.mean(0)
        Sb += len(Lc) * np.sum((muc - mu) ** 2)
        Sw += np.sum((Lc - muc) ** 2)
    St = Sb + Sw                                   # total scatter Σ‖L_i − μ‖²
    scale = max(float((L ** 2).sum()), 1.0)
    if St <= 1e-12 * scale:                        # all explanations identical → undefined
        return 0.0
    if Sw <= 1e-12 * St:                           # perfectly separated
        return np.inf
    return float(Sb / Sw)


def _safe_corr(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def local_global_consistency(A, L, labels):
    """Local–Global Consistency.

    For each class c, correlate (over features) the **global** per-class weight
    vector ``A[:, c]`` with the **class-conditional mean local importance**
    ``mean_{i: label_i=c} L_i``.  Because AIME's global operator and its local
    rule ``(A·y)⊙x'`` come from the *same* operator, this correlation is high by
    construction — a consistency forward-problem methods cannot guarantee
    (they average local explanations post hoc with no such link).

    Returns dict(per_class=array(m), overall=mean, vectors=list of (g_c, l_c)).
    """
    A = np.asarray(A, float); L = np.asarray(L, float); labels = np.asarray(labels)
    m = A.shape[1]
    per = np.full(m, np.nan); vecs = []
    for c in range(m):
        idx = labels == c
        g = A[:, c]
        l = L[idx].mean(0) if idx.any() else np.zeros(A.shape[0])
        per[c] = _safe_corr(g, l)
        vecs.append((g, l))
    overall = float(np.nanmean(per)) if np.isfinite(per).any() else np.nan
    return dict(per_class=per, overall=overall, vectors=vecs)


# ======================================================================= #
# Evaluator — wraps a fitted explainer + an evaluation set                 #
# ======================================================================= #
class AIMEEvaluator:
    """Compute inverse-explanation metrics for a fitted AIME explainer.

    Parameters
    ----------
    explainer : AIME
        A fitted explainer (``create_explainer`` already called).
    X, Y : arrays
        Evaluation inputs (N, n) and model outputs (N, m).
    feature_names, class_names : optional sequences
    gamma : float | 'scale' | 'auto'
        RBF bandwidth for the representative-instance similarity (RIC),
        evaluated in the standardised space.  Defaults to 'scale'.
    ignore_zero_features : bool
        If True, absent features are zeroed in the local importance used by
        LGC / CSI (Hadamard rule).  Default False (use the raw product so the
        averages are unbiased).
    """

    def __init__(self, explainer, X, Y, feature_names=None, class_names=None,
                 gamma="scale", ignore_zero_features=False):
        if explainer.A_dagger is None:
            raise ValueError("Explainer must be fitted (call create_explainer first).")
        self.expl = explainer
        self.A = np.asarray(explainer.A_dagger, float)         # (n, m)
        self.X = np.asarray(X, float)
        self.Y = np.asarray(Y, float)
        self.n, self.m = self.A.shape
        sc = explainer.scaler
        self.Xp = sc.transform(self.X) if sc is not None else self.X   # standardised
        # NOTE: ``labels`` are the model's PREDICTED / OUTPUT classes
        # (argmax over the output Y), NOT ground-truth labels.  All metrics that
        # group by class (RIC, LGC, CSI) therefore measure consistency relative
        # to the model's own decisions, which is the correct reference for an
        # explanation method.  Supply hard one-hot Y to recover true-label
        # grouping if that is desired.
        self.labels = np.argmax(self.Y, axis=1)
        self.feature_names = list(feature_names) if feature_names is not None \
            else [f"feature_{i}" for i in range(self.n)]
        self.class_names = list(class_names) if class_names is not None \
            else [f"class_{j}" for j in range(self.m)]
        self.ignore_zero_features = ignore_zero_features

        # reconstruction + raw local importances (shared building blocks)
        self.Xhat = reconstruct(self.A, self.Y)                # (N, n)
        L = self.Xhat * self.Xp                                # Hadamard
        if ignore_zero_features:
            L = L * (self.X != 0)
        self.L = L
        self.gamma = explainer._resolve_gamma(gamma, self.Xp)  # numeric bandwidth
        # representative instances in standardised space: rep_c = A·e_c
        self.reps = self.A.T                                   # (m, n)
        # original-space reconstruction (for the variance-recovery metric IRR)
        self.Xhat_orig = sc.inverse_transform(self.Xhat) if sc is not None else self.Xhat

    # ---- individual metrics ------------------------------------------- #
    def inverse_reconstruction_error(self):
        return ire(self.Xp, self.A, self.Y)

    def information_recovery_rate(self):
        """IRR = tr(Cov(X̂)) / tr(Cov(X)) in the original feature space —
        the fraction of input *structure* (variance) recovered, an axis
        independent of the point-wise reconstruction error IRE."""
        return variance_recovery_rate(self.Xhat_orig, self.X)

    def representative_instance_consistency(self):
        """RIC: mean RBF similarity of each instance to the representative
        instance of its **predicted/output class** (``argmax_c Y``, i.e. the
        model's output — not the ground-truth label), plus the per-class
        breakdown and a 'purity' (fraction of instances whose nearest
        representative is that same predicted class).
        """
        sim = self.expl.rbf_kernel(self.Xp, self.reps, self.gamma)   # (N, m)
        own = sim[np.arange(len(sim)), self.labels]
        per = np.array([own[self.labels == c].mean() if (self.labels == c).any() else np.nan
                        for c in range(self.m)])
        purity = float(np.mean(np.argmax(sim, axis=1) == self.labels))
        return dict(overall=float(np.nanmean(own)), per_class=per, purity=purity, sim=sim)

    def explanation_coverage(self):
        ec = coverage(self.A)
        return dict(per_class=ec, overall=float(np.mean(ec)))

    def local_global_consistency(self):
        return local_global_consistency(self.A, self.L, self.labels)

    def class_separability(self):
        return class_separability(self.L, self.labels)

    def operator_rank(self):
        return effective_rank(self.A)

    def inverse_explainability_score(self, weights=None):
        """**Core-only** composite IES = weighted mean of the normalised Core
        explanation metrics — Consistency, Stability (×2), Coverage, Separability,
        Simplicity.  **Reconstruction (IRE/IRR) is deliberately excluded**: AIME
        is an explanation operator, not an input reconstructor, so its overall
        explanation quality must not be diluted by reconstruction fidelity.

        Each component is mapped to [0, 1]:
        LGC→(LGC+1)/2, RIC, purity, EC already in [0,1], CSI→CSI/(1+CSI),
        simplicity→1−(r_eff−1)/(R−1) with R=max(min(n,m),2).  Default weights are
        equal.  ``weights`` may be a **partial** dict: any components it omits
        keep their default weight (no KeyError), unknown keys are ignored, and
        the final weights are renormalised to sum to 1 so the score stays in
        [0, 1] regardless of the magnitudes passed.
        """
        lgc = self.local_global_consistency()["overall"]
        ric = self.representative_instance_consistency()
        ec = self.explanation_coverage()["overall"]
        csi = self.class_separability()
        rk = self.operator_rank()
        R = max(min(self.n, self.m), 2)
        comps = {
            "Consistency (LGC)":  (np.clip(lgc, -1, 1) + 1) / 2 if np.isfinite(lgc) else 0.0,
            "Stability (RIC)":     float(ric["overall"]),
            "Stability (purity)":  float(ric["purity"]),
            "Coverage (EC)":       float(ec),
            "Separability (CSI)":  (csi / (1 + csi)) if np.isfinite(csi) else 1.0,
            "Simplicity (rank)":   float(np.clip(1 - (rk["effective_rank"] - 1) / (R - 1), 0, 1)),
        }
        # Start from equal default weights, overlay any user-supplied (possibly
        # partial) weights, drop unknown keys, then renormalise to sum to 1.
        eff = {k: 1.0 / len(comps) for k in comps}
        if weights is not None:
            for k, v in weights.items():
                if k in eff:
                    eff[k] = float(v)
        total = sum(eff.values())
        if total <= 0:
            eff = {k: 1.0 / len(comps) for k in comps}
            total = 1.0
        eff = {k: v / total for k, v in eff.items()}
        score = float(sum(eff[k] * comps[k] for k in comps))
        return dict(score=score, components=comps, weights=eff)

    def summary(self):
        """Return a flat dict of all headline metrics."""
        rank = self.operator_rank()
        ric = self.representative_instance_consistency()
        return {
            "LGC": self.local_global_consistency()["overall"],
            "RIC": ric["overall"],
            "RIC_purity": ric["purity"],
            "EC": self.explanation_coverage()["overall"],
            "CSI": self.class_separability(),
            "rank": rank["rank"],
            "effective_rank": rank["effective_rank"],
            "IRE": self.inverse_reconstruction_error(),
            "IRR": self.information_recovery_rate(),
            "IES": self.inverse_explainability_score()["score"],
        }

    def summary_frame(self):
        """Metrics table organised by tier.

        **Core explanation metrics** evaluate the operator *as an explanation*
        (does it explain consistently, stably, simply, broadly?). **Secondary
        (explanation fidelity)** measures input reconstruction — useful context,
        but *not* AIME's purpose: AIME is an explanation operator, not an input
        reconstructor (``reconstruction ≠ explanation``)."""
        s = self.summary()
        rows = []
        for key, (tier, category) in METRIC_TIER.items():
            rows.append({"metric": key, "value": s[key],
                         "tier": tier, "category": category})
        return pd.DataFrame(rows).set_index("metric")

    def core_metrics(self):
        """The Core explanation metrics (the operator-quality tier)."""
        s = self.summary()
        return {k: s[k] for k, (t, _) in METRIC_TIER.items() if t == TIER_CORE}

    def secondary_metrics(self):
        """The Secondary (explanation-fidelity / reconstruction) tier."""
        s = self.summary()
        return {k: s[k] for k, (t, _) in METRIC_TIER.items() if t == TIER_SECONDARY}

    # =================================================================== #
    # Visualisations (signature style)                                    #
    # =================================================================== #
    def plot_metrics_summary(self, save_path=None, show=True):
        """Two-tier dashboard: **Core explanation metrics** (emphasised) above
        **Secondary explanation-fidelity** (muted). All bars are mapped to a
        comparable 0–1 scale; the raw value is annotated where a surrogate is
        used (rank/r_eff/CSI). No formula is changed — only the presentation
        reflects that AIME is an explanation operator, not a reconstructor."""
        s = self.summary()
        R = max(min(self.n, self.m), 2)
        lgc01 = (np.clip(s["LGC"], -1, 1) + 1) / 2 if np.isfinite(s["LGC"]) else 0.0
        simplicity = float(np.clip(1 - (s["effective_rank"] - 1) / (R - 1), 0, 1))
        csi = s["CSI"]; sep = (csi / (1 + csi)) if np.isfinite(csi) else 1.0
        csi_raw = "∞" if not np.isfinite(csi) else f"{csi:.2f}"
        # (label, 0-1 score, raw-value annotation or None).  Bars are 0–1
        # SCORES; the right column always shows the RAW value when a surrogate
        # mapping was applied, so the two are never confused.
        core = [
            ("Composite · IES (core)",   s["IES"],         "core mean"),
            ("Consistency · LGC₀₁",      lgc01,            f"raw LGC = {s['LGC']:+.2f}"),
            ("Stability · RIC",          s["RIC"],         None),
            ("Stability · purity",       s["RIC_purity"],  None),
            ("Coverage · EC",            s["EC"],          None),
            ("Separability · CSI score", sep,              f"raw CSI = {csi_raw}"),
            ("Simplicity · rank score",  simplicity,       f"rank = {s['rank']} · r_eff = {s['effective_rank']:.2f}"),
        ]
        secondary = [
            ("Reconstruction · 1−IRE",   max(0.0, 1 - s["IRE"]), f"raw IRE = {s['IRE']:.2f}"),
            ("Recovery · IRR",           s["IRR"],         None),
        ]
        # stack top→bottom: core first, a gap, then secondary
        entries = core + [None] + secondary
        n = len(entries)
        fig, ax = S.new_figure(figsize=(9.6, 0.56 * n + 2.2))
        ylabels = []
        for k, e in enumerate(entries):
            y = n - 1 - k                                  # top row highest
            if e is None:
                ylabels.append("")
                ax.axhline(y, color=S.GRIDLINE, lw=1.2)
                continue
            label, val, ann = e
            is_core = k < len(core)
            if is_core:
                S.gradient_hbar(ax, y, float(val), vmax=1.0, height=0.6, mode="sequential")
            else:
                S.class_hbar(ax, y, float(val), "#B9BEC9", vmax=1.0, height=0.56)  # muted grey
                ax.text(float(val) + 0.02, y, f"{val:.2f}", va="center", ha="left",
                        fontsize=9, color=S.INK_SOFT)
            if ann:
                ax.text(1.10, y, ann, va="center", ha="left", fontsize=8.5,
                        color=S.INK_SOFT, transform=ax.get_yaxis_transform())
            ylabels.append(label)
        ax.set_yticks(range(n)); ax.set_yticklabels(ylabels[::-1], fontsize=9.5)
        ax.set_xlim(0, 1.0); ax.set_ylim(-0.7, n - 0.3)
        ax.axvline(1.0, color=S.GRIDLINE, lw=1.0)
        # tier band + section captions
        ax.axhspan(n - len(core) - 0.5, n - 0.5, color=S.CORAL, alpha=0.05, zorder=0)
        ax.text(0.0, n - 0.35, "CORE  ·  explanation-operator quality", fontsize=9.5,
                fontweight="bold", color=S.CORAL_DK, ha="left", va="bottom")
        ax.text(0.0, len(secondary) - 0.62, "SECONDARY  ·  explanation fidelity (reconstruction)",
                fontsize=9.5, fontweight="bold", color=S.INK_SOFT, ha="left", va="bottom")
        ax.set_xlabel("bars = 0–1 score   ·   right column = raw value")
        S.style_title(ax, f"Explanation-Operator Metrics · {self.expl.variant_name}",
                      subtitle="bars are 0–1 scores; raw values annotated at right. IES is core-only (excludes IRE/IRR)")
        self.expl._finish(fig, save_path, show)
        return s

    def plot_local_global_consistency(self, save_path=None, show=True):
        """Per-class scatter of global weight vs class-conditional mean local
        importance — the headline AIME consistency view."""
        lg = self.local_global_consistency()
        fig, axes = S.new_figure(figsize=(4.6 * self.m, 4.6), ncols=self.m, sharey=False)
        axes = np.atleast_1d(axes)
        for c in range(self.m):
            ax = axes[c]; g, l = lg["vectors"][c]
            col = S.CLASS_CYCLE[c % len(S.CLASS_CYCLE)]
            ax.axhline(0, color=S.GRIDLINE, lw=1); ax.axvline(0, color=S.GRIDLINE, lw=1)
            ax.scatter(g, l, s=42, color=col, edgecolors="white", linewidths=0.6, zorder=3)
            # least-squares guide line
            if np.std(g) > 1e-12:
                b1, b0 = np.polyfit(g, l, 1)
                xs = np.linspace(g.min(), g.max(), 50)
                ax.plot(xs, b0 + b1 * xs, color=S.INK, lw=1.4, alpha=0.5, zorder=2)
            r = lg["per_class"][c]
            ax.set_title(f"{self.class_names[c]}   r = {r:.3f}", fontsize=11,
                         color=S.INK, pad=6)
            ax.set_xlabel("global weight  A[:, c]", fontsize=9.5)
            if c == 0:
                ax.set_ylabel("mean local importance (class-conditional)", fontsize=9.5)
            ax.grid(True, color=S.GRIDLINE, lw=0.6, alpha=0.6)
        S.style_title(axes[0], f"Local–Global Consistency · overall r = {lg['overall']:.3f}",
                      subtitle="same inverse operator drives both — not a post-hoc average")
        self.expl._finish(fig, save_path, show, rect=(0, 0.04, 1, 0.88 if not S.PUBLICATION else 0.99))
        return lg

    def plot_inverse_reconstruction(self, max_points=4000, save_path=None, show=True):
        """x' vs x̂' scatter (recovery) + per-instance residual distribution."""
        rng = np.random.default_rng(0)
        N = len(self.Xp)
        idx = rng.choice(N, min(N, max_points), replace=False)
        xp = self.Xp[idx].ravel(); xh = self.Xhat[idx].ravel()
        res = ire_per_instance(self.Xp, self.A, self.Y)
        irr = self.information_recovery_rate(); irev = self.inverse_reconstruction_error()
        fig, axes = S.new_figure(figsize=(11, 4.8), ncols=2)
        ax = axes[0]
        lim = np.percentile(np.abs(np.r_[xp, xh]), 99.5)
        ax.plot([-lim, lim], [-lim, lim], color=S.INK, lw=1.2, alpha=0.5, zorder=1)
        ax.scatter(xp, xh, s=6, color=S.INDIGO, alpha=0.25, edgecolors="none", zorder=2)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel("standardised input  x'"); ax.set_ylabel("reconstruction  x̂' = A·y")
        ax.set_title(f"IRR = {irr:.3f}   (IRE = {irev:.3f})", fontsize=11, color=S.INK, pad=6)
        ax.grid(True, color=S.GRIDLINE, lw=0.6, alpha=0.6)
        ax2 = axes[1]
        ax2.hist(res, bins=40, color=S.CORAL, alpha=0.85, edgecolor="white", linewidth=0.5)
        ax2.set_xlabel("per-instance residual  ‖x'_i − A·y_i‖")
        ax2.set_ylabel("count")
        ax2.set_title("reconstruction residual distribution", fontsize=11, color=S.INK, pad=6)
        ax2.grid(True, axis="y", color=S.GRIDLINE, lw=0.6, alpha=0.6)
        S.style_title(axes[0], "Inverse Reconstruction",
                      subtitle="how much of the input the output recovers through A†")
        self.expl._finish(fig, save_path, show)
        return dict(IRE=irev, IRR=irr, residuals=res)

    def plot_representative_consistency(self, save_path=None, show=True):
        """Per-class RIC bars + purity."""
        ric = self.representative_instance_consistency()
        fig, ax = S.new_figure(figsize=(8.8, 0.5 * self.m + 2.4))
        order = np.argsort(ric["per_class"])
        for i, c in enumerate(order):
            col = S.CLASS_CYCLE[c % len(S.CLASS_CYCLE)]
            S.class_hbar(ax, i, float(ric["per_class"][c]), col, vmax=1.0, height=0.6)
            ax.text(float(ric["per_class"][c]) + 0.02, i, f"{ric['per_class'][c]:.2f}",
                    va="center", ha="left", fontsize=9.5, fontweight="bold", color=S.INK)
        ax.set_yticks(range(self.m)); ax.set_yticklabels([self.class_names[c] for c in order], fontsize=10)
        ax.set_xlim(0, 1.12); ax.set_ylim(-0.7, self.m - 0.3)
        ax.set_xlabel(f"RIC — similarity to own representative   ·   purity = {ric['purity']:.2f}")
        S.style_title(ax, "Representative Instance Consistency",
                      subtitle="are a class's instances actually explained by its representative?")
        self.expl._finish(fig, save_path, show)
        return ric

    def plot_operator_spectrum(self, save_path=None, show=True):
        """Singular-value scree of A with rank / effective rank annotated."""
        r = self.operator_rank(); s = r["singular_values"]
        fig, ax = S.new_figure(figsize=(8.6, 5))
        xs = np.arange(1, len(s) + 1)
        S.apply_aime_style()
        ax.bar(xs, s, width=0.62, color=S.INDIGO, edgecolor="white", zorder=3)
        ax.plot(xs, s, "-o", color=S.CORAL, lw=2, ms=6, zorder=4)
        ax.axvline(r["effective_rank"], color=S.AMBER, ls="--", lw=1.6,
                   label=f"effective rank = {r['effective_rank']:.2f}")
        ax.set_xticks(xs)
        ax.set_xlabel("singular value index of A†"); ax.set_ylabel("σ")
        ax.legend(loc="upper right", frameon=False, fontsize=10)
        ax.grid(True, axis="y", color=S.GRIDLINE, lw=0.6, alpha=0.7)
        ax.set_title(f"Explanation operator spectrum · rank = {r['rank']} · "
                     f"r_eff = {r['effective_rank']:.2f}", fontsize=12.5, color=S.INK, pad=8)
        self.expl._finish(fig, save_path, show)
        return r

    def plot_class_separability(self, save_path=None, show=True):
        """Project per-instance local explanations to 2-D (PCA) coloured by class
        to visualise how separable the explanations are; title shows CSI."""
        from numpy.linalg import svd
        L = self.L - self.L.mean(0)
        U, Sv, Vt = svd(L, full_matrices=False)
        Z = L @ Vt[:2].T if Vt.shape[0] >= 2 else np.c_[L @ Vt[:1].T, np.zeros(len(L))]
        csi = self.class_separability()
        fig, ax = S.new_figure(figsize=(7.8, 6.4))
        for c in range(self.m):
            idx = self.labels == c
            col = S.CLASS_CYCLE[c % len(S.CLASS_CYCLE)]
            ax.scatter(Z[idx, 0], Z[idx, 1], s=18, color=col, alpha=0.5,
                       edgecolors="none", label=self.class_names[c])
        ax.set_xlabel("local-explanation PC1"); ax.set_ylabel("local-explanation PC2")
        ax.grid(True, color=S.GRIDLINE, lw=0.6, alpha=0.6)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=9.5,
                  title="class", title_fontsize=9.5)
        ax.set_title(f"Class separability of explanations · CSI = {csi:.2f}",
                     fontsize=12.5, color=S.INK, pad=8)
        self.expl._finish(fig, save_path, show, rect=(0, 0.04, 0.86, 0.99 if S.PUBLICATION else 0.9))
        return csi

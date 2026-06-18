"""
aime_xai.operator_viz
=====================

**Inverse Explanatory Operator Visualization System.**

AIME is not "another feature-importance method" — it constructs an *explicit
inverse explanatory operator* ``A†`` (output → input).  This module visualises
that operator and the quantities it makes directly observable, none of which a
forward-problem method (LIME/SHAP) can define.  It is **purely additive**: it
reads a *fitted* explainer (``A_dagger``, ``scaler``, and — for Bayesian — the
posterior ``A_dagger_cov``) and never changes how AIME derives the operator.

Ten views, all centred on ``A†`` (``A`` below, shape ``d × m``):

1.  ``plot_operator_map``           — the learned operator ``A`` itself
2.  ``plot_class_contrast``         — ``A(e_i − e_j)`` (inverse prototype contrast)
3.  ``plot_prototype_reconstruction`` — ``scaler⁻¹(A e_k)`` (output-conditioned representative input)
4.  ``plot_prototype_morphing``     — ``scaler⁻¹(A·((1−t)e_a + t e_b))`` path
5.  ``plot_counterfactual_direction`` — ``A(e_t − y_i)`` (gradient-free proposal direction)
6.  ``plot_inverse_deviation``      — ``r_i = x'_i − A y_i`` (deviation from the inverse representative)
7.  ``plot_reliability_map``        — HuberAIME weights ``w_i = min(1, δ/‖r_i‖)``
8.  ``plot_explanation_redundancy`` — ``cos(a_p, a_q)`` between operator rows
9.  ``plot_inverse_modes``          — SVD of ``A`` (inverse explanation modes)
10. ``plot_uncertainty_map``        — BayesianAIME posterior over ``A``

Every method follows the signature publication style (white background, no
chrome) and accepts ``save_path`` / ``show``.  Quantities are computed exactly as
specified; interpretive cautions are noted in each docstring (e.g. the
counterfactual direction is a *proposal*, the deviation is *not* model error).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# matplotlib is imported lazily inside the two helpers that build figures
# directly (``_img_grid`` and ``plot_inverse_modes``); every other view routes
# through ``aime_xai.style`` + ``AIME._finish`` which also import it lazily.
from . import style as S


def _signed_matrix_show(ax, M, row_labels, col_labels, annotate=True, cbar_fig=None,
                        cbar_label="value"):
    """Render a signed matrix with the AIME diverging colormap (zero-centred)."""
    norm = S.signed_norm(M)
    im = ax.imshow(M, aspect="auto", cmap=S.AIME_DIVERGING, norm=norm)
    ax.set_xticks(range(M.shape[1])); ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(M.shape[0])); ax.set_yticklabels(row_labels, fontsize=8.5)
    ax.set_xticks(np.arange(-.5, M.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, M.shape[0], 1), minor=True)
    ax.grid(which="minor", color=S.PAPER, lw=1.2); ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    if annotate and M.size <= 240:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7.2,
                        color=(S.PAPER if abs(v) > 0.6 * norm.vmax else S.INK))
    if cbar_fig is not None:
        cb = cbar_fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.outline.set_visible(False); cb.ax.tick_params(labelsize=8, color=S.INK_SOFT)
        cb.set_label(cbar_label, fontsize=9, color=S.INK_SOFT)
    return im


def _img_grid(values, image_shape, titles, cmap, suptitle, subtitle,
              save_path, show, expl, signed=False):
    """Render a row/grid of (reshaped) images."""
    import matplotlib.pyplot as plt
    n = len(values); ncols = min(n, 5); nrows = int(np.ceil(n / ncols))
    is_rgb = (len(image_shape) == 3)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.35 * ncols, 2.6 * nrows + 1.0),
                             facecolor=S.PAPER, squeeze=False)
    S.apply_aime_style(); fig.subplots_adjust(hspace=0.3, wspace=0.12)
    vmax = float(np.max(np.abs(np.asarray(values)))) or 1.0
    for k in range(nrows * ncols):
        ax = axes[k // ncols][k % ncols]; ax.set_xticks([]); ax.set_yticks([])
        if k >= n:
            ax.set_visible(False); continue
        img = np.asarray(values[k]).reshape(image_shape)
        if is_rgb:
            mn, mx = img.min(), img.max(); ax.imshow((img - mn) / (mx - mn + 1e-12))
        elif signed:
            ax.imshow(img, cmap=S.AIME_DIVERGING, norm=S.signed_norm([-vmax, vmax]))
        else:
            ax.imshow(img, cmap=cmap or S.AIME_SEQ)
        ax.set_title(str(titles[k]), fontsize=10.5, fontweight="bold", color=S.INK, pad=5)
        for sp in ax.spines.values():
            sp.set_color(S.GRIDLINE); sp.set_linewidth(1.1)
    if not S.PUBLICATION:
        fig.text(0.012, 0.98, suptitle, ha="left", va="top", fontsize=15, fontweight="bold", color=S.INK)
        fig.text(0.012, 0.93, subtitle, ha="left", va="top", fontsize=10, color=S.INK_SOFT)
    S.add_brand(fig)
    fig.tight_layout(rect=(0, 0.02, 1, 0.99 if S.PUBLICATION else 0.9))
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=S.PAPER)
    if show:
        plt.show()
    else:
        plt.close(fig)


class OperatorVisualizer:
    """Visualise the inverse explanatory operator ``A†`` of a fitted explainer.

    Parameters
    ----------
    explainer : AIME
        A fitted explainer (``create_explainer`` already called).
    X, Y : arrays, optional
        Data used by the sample-level views (deviation, reliability,
        counterfactual). Not needed for operator-only views.
    feature_names, class_names : sequences, optional
    image_shape : tuple, optional
        e.g. ``(28, 28)`` or ``(32, 32, 3)`` to render image-domain versions.
    """

    def __init__(self, explainer, X=None, Y=None, feature_names=None,
                 class_names=None, image_shape=None):
        if explainer.A_dagger is None:
            raise ValueError("Explainer must be fitted (call create_explainer first).")
        self.expl = explainer
        self.A = np.asarray(explainer.A_dagger, float)         # (d, m)
        self.d, self.m = self.A.shape
        self.scaler = explainer.scaler
        self.image_shape = image_shape
        self.feature_names = list(feature_names) if feature_names is not None \
            else [f"feature_{i}" for i in range(self.d)]
        self.class_names = list(class_names) if class_names is not None \
            else [f"class_{j}" for j in range(self.m)]
        self.X = None if X is None else np.asarray(X, float)
        self.Y = None if Y is None else np.asarray(Y, float)
        if self.X is not None:
            self.Xp = self.scaler.transform(self.X) if self.scaler is not None else self.X
            self.Yp = self.Y
        else:
            self.Xp = self.Yp = None

    # ------------------------------------------------------------------ #
    def _need_data(self, what):
        if self.X is None or self.Y is None:
            raise ValueError(f"{what} requires X and Y to be passed to OperatorVisualizer.")

    def _inv(self, v):
        """scaler⁻¹ for a standardised-space vector/matrix (→ original units)."""
        if self.scaler is not None:
            return self.scaler.inverse_transform(np.atleast_2d(v))
        return np.atleast_2d(v)

    # ============================================================== #
    # 1. Operator Map                                                #
    # ============================================================== #
    def plot_operator_map(self, image_shape=None, save_path=None, show=True):
        """The learned inverse operator ``A`` itself (raw, not normalised).
        Tabular: signed feature×class heatmap. Image: each column ``A eₖ`` as a
        signed image. These are *inverse explanatory coefficients* (output→input),
        not importances."""
        image_shape = image_shape or self.image_shape
        if image_shape is not None:
            return _img_grid([self.A[:, k] for k in range(self.m)], image_shape,
                             self.class_names, None,
                             "Operator Map · A† eₖ", "the learned inverse explanatory operator (per class)",
                             save_path, show, self.expl, signed=True)
        fig, ax = S.new_figure(figsize=(max(6, 1.1 * self.m + 3), 0.34 * self.d + 2.2))
        _signed_matrix_show(ax, self.A, self.feature_names, self.class_names,
                            cbar_fig=fig, cbar_label="A†  (output→input coefficient)")
        S.style_title(ax, f"Operator Map · {self.expl.variant_name}",
                      subtitle="the learned inverse explanatory operator A†  (feature × class)")
        self.expl._finish(fig, save_path, show, rect=False)
        return pd.DataFrame(self.A, index=self.feature_names, columns=self.class_names)

    # ============================================================== #
    # 2. Class-Contrast Map                                          #
    # ============================================================== #
    def plot_class_contrast(self, i, j, image_shape=None, top_k=None,
                            save_path=None, show=True):
        """``c = A(e_i − e_j) = A[:,i] − A[:,j]`` — the input-space direction that
        characterises class *i* relative to class *j* (an *inverse prototype
        contrast*, NOT a decision-boundary normal)."""
        c = self.A[:, i] - self.A[:, j]
        ci, cj = self.class_names[i], self.class_names[j]
        image_shape = image_shape or self.image_shape
        if image_shape is not None:
            return _img_grid([c], image_shape, [f"{ci}  −  {cj}"], None,
                             "Class-Contrast Map · A(eᵢ−eⱼ)",
                             f"coral = toward {ci} · indigo = toward {cj}",
                             save_path, show, self.expl, signed=True)
        order = np.argsort(c)
        if top_k:
            keep = np.argsort(np.abs(c))[::-1][:top_k]
            order = keep[np.argsort(c[keep])]
        feats = [self.feature_names[k] for k in order]; vals = c[order]
        vmax = float(np.max(np.abs(vals))) or 1.0
        fig, ax = S.new_figure(figsize=(9, max(3.4, 0.42 * len(vals) + 2)))
        for r, v in enumerate(vals):
            S.gradient_hbar(ax, r, float(v), vmax=vmax, height=0.55)
        ax.axvline(0, color=S.INK, lw=1.1, alpha=0.55)
        ax.set_yticks(range(len(vals))); ax.set_yticklabels(feats, fontsize=9.5)
        ax.set_xlim(-vmax * 1.3, vmax * 1.3); ax.set_ylim(-0.7, len(vals) - 0.3)
        ax.set_xlabel(f"← toward {cj}      A(eᵢ−eⱼ)      toward {ci} →")
        S.style_title(ax, f"Class-Contrast · {ci}  vs  {cj}",
                      subtitle="input-space difference between two output states (inverse prototype contrast)")
        self.expl._finish(fig, save_path, show)
        return pd.Series(c, index=self.feature_names, name=f"{ci}-{cj}")

    # ============================================================== #
    # 3. Prototype Reconstruction Map                               #
    # ============================================================== #
    def prototypes(self):
        """``x̂_k = scaler⁻¹(A e_k)`` for every class — the output-conditioned
        representative input (NOT a data mean, NOT a generative sample)."""
        reps = self._inv(self.A.T)                              # (m, d) original units
        return pd.DataFrame(reps, index=self.class_names, columns=self.feature_names)

    def plot_prototype_reconstruction(self, image_shape=None, cmap=None,
                                      save_path=None, show=True):
        proto = self.prototypes()
        image_shape = image_shape or self.image_shape
        if image_shape is not None:
            return _img_grid([proto.iloc[k].values for k in range(self.m)], image_shape,
                             self.class_names, cmap,
                             "Prototype Reconstruction · scaler⁻¹(A† eₖ)",
                             "output-conditioned representative input per class",
                             save_path, show, self.expl)
        # tabular: per-class representative feature profile (z-scored for comparability)
        Z = proto.values.astype(float)
        Zs = (Z - Z.mean(0, keepdims=True)) / (Z.std(0, keepdims=True) + 1e-12)
        ncols = self.m; fig, axes = S.new_figure(figsize=(3.2 * ncols, 0.34 * self.d + 2.2),
                                                  ncols=ncols, sharey=True)
        axes = np.atleast_1d(axes)
        for k in range(self.m):
            ax = axes[k]; col = S.CLASS_CYCLE[k % len(S.CLASS_CYCLE)]
            vmax = float(np.max(np.abs(Zs[k]))) or 1.0
            for r in range(self.d):
                S.class_hbar(ax, self.d - 1 - r, float(Zs[k, r]), col, vmax=vmax, height=0.6)
            ax.axvline(0, color=S.INK, lw=1.0, alpha=0.5)
            ax.set_xlim(-vmax * 1.2, vmax * 1.2); ax.set_ylim(-0.7, self.d - 0.3)
            ax.set_title(self.class_names[k], fontsize=11, color=S.INK, pad=5)
            if k == 0:
                ax.set_yticks(range(self.d)); ax.set_yticklabels(self.feature_names[::-1], fontsize=8.5)
        S.style_title(axes[0], "Prototype Reconstruction · scaler⁻¹(A† eₖ)",
                      subtitle="output-conditioned representative input per class (z-scored profile)")
        self.expl._finish(fig, save_path, show, rect=(0, 0.04, 1, 0.99 if S.PUBLICATION else 0.9))
        return proto

    # ============================================================== #
    # 4. Prototype Morphing Path                                    #
    # ============================================================== #
    def plot_prototype_morphing(self, a, b, steps=9, image_shape=None,
                                top_k=8, save_path=None, show=True):
        """``x̂(t) = scaler⁻¹(A·((1−t)e_a + t e_b))`` — how a *linear* move in
        output space appears in input space (an inverse-operator path, not a data
        manifold trajectory)."""
        ts = np.linspace(0, 1, steps)
        ys = np.array([(1 - t) * np.eye(self.m)[a] + t * np.eye(self.m)[b] for t in ts])
        xs = self._inv(ys @ self.A.T)                           # (steps, d) original units
        ca, cb = self.class_names[a], self.class_names[b]
        image_shape = image_shape or self.image_shape
        if image_shape is not None:
            titles = [f"t={t:.2f}" for t in ts]
            return _img_grid([xs[s] for s in range(steps)], image_shape, titles, None,
                             f"Prototype Morphing · {ca} → {cb}",
                             "linear output interpolation mapped to input space by A†",
                             save_path, show, self.expl)
        # tabular: top-k features' value trajectories
        spread = np.abs(xs - xs.mean(0)).sum(0)
        sel = np.argsort(spread)[::-1][:min(top_k, self.d)]
        fig, ax = S.new_figure(figsize=(9, 5.4))
        for k in sel:
            ax.plot(ts, xs[:, k], "-o", lw=2, ms=4, color=S.CLASS_CYCLE[k % len(S.CLASS_CYCLE)],
                    label=self.feature_names[k])
        ax.set_xlabel(f"t   ({ca} → {cb})"); ax.set_ylabel("reconstructed feature value")
        ax.grid(True, color=S.GRIDLINE, lw=0.7, alpha=0.7)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=9)
        S.style_title(ax, f"Prototype Morphing · {ca} → {cb}",
                      subtitle="how output-space interpolation appears in input space via A†")
        self.expl._finish(fig, save_path, show, rect=(0, 0.04, 0.82, 0.99 if S.PUBLICATION else 0.9))
        return pd.DataFrame(xs, index=[f"t={t:.2f}" for t in ts], columns=self.feature_names)

    # ============================================================== #
    # 5. Counterfactual Direction Map                              #
    # ============================================================== #
    def plot_counterfactual_direction(self, x, y_current, target, image_shape=None,
                                      top_k=12, save_path=None, show=True):
        """``d = A(e_t − y_i)`` — the input-space direction AIME proposes to move
        a sample toward target class ``t`` **without gradients or perturbation
        search**. This is a *proposal direction*; whether it flips the forward
        model must be checked separately."""
        y_current = np.asarray(y_current, float)
        d = self.A @ (np.eye(self.m)[target] - y_current)       # standardised-space direction
        ct = self.class_names[target]
        image_shape = image_shape or self.image_shape
        if image_shape is not None:
            return _img_grid([d], image_shape, [f"→ {ct}"], None,
                             "Counterfactual Direction · A(eₜ−yᵢ)",
                             "coral = increase · indigo = decrease (proposal, not verified)",
                             save_path, show, self.expl, signed=True)
        order = np.argsort(np.abs(d))[::-1][:min(top_k, self.d)]
        order = order[np.argsort(d[order])]
        feats = [self.feature_names[k] for k in order]; vals = d[order]
        vmax = float(np.max(np.abs(vals))) or 1.0
        fig, ax = S.new_figure(figsize=(9, max(3.4, 0.42 * len(vals) + 2)))
        for r, v in enumerate(vals):
            S.gradient_hbar(ax, r, float(v), vmax=vmax, height=0.55)
        ax.axvline(0, color=S.INK, lw=1.1, alpha=0.55)
        ax.set_yticks(range(len(vals))); ax.set_yticklabels(feats, fontsize=9.5)
        ax.set_xlim(-vmax * 1.3, vmax * 1.3); ax.set_ylim(-0.7, len(vals) - 0.3)
        ax.set_xlabel(f"← decrease      proposed change toward {ct}      increase →")
        S.style_title(ax, f"Counterfactual Direction · → {ct}",
                      subtitle="gradient-free counterfactual proposal direction A(eₜ−yᵢ)")
        self.expl._finish(fig, save_path, show)
        return pd.Series(d, index=self.feature_names, name=f"->{ct}")

    # ============================================================== #
    # 6. Inverse Deviation Map                                      #
    # ============================================================== #
    def inverse_deviation(self):
        """Per-sample residual ``r_i = x'_i − A y_i`` (standardised space) and its
        norm ‖r_i‖.  This is deviation from the output-conditioned inverse
        representative — **not** model error and **not** explanation failure."""
        self._need_data("inverse_deviation")
        Xhat = self.Yp @ self.A.T
        R = self.Xp - Xhat
        return R, np.linalg.norm(R, axis=1)

    def plot_inverse_deviation(self, sample_index=None, image_shape=None,
                               save_path=None, show=True):
        R, rn = self.inverse_deviation()
        image_shape = image_shape or self.image_shape
        if image_shape is not None and sample_index is not None:
            i = sample_index
            orig = self.Xp[i]; rep = (self.Yp[i] @ self.A.T)
            trip = [orig, rep, R[i]]
            return _img_grid(trip, image_shape, ["input x'", "inverse rep A†y", "deviation r"],
                             None, f"Inverse Deviation · sample {i}",
                             "input vs inverse representative vs their difference",
                             save_path, show, self.expl, signed=True)
        fig, axes = S.new_figure(figsize=(11, 4.8), ncols=2)
        # left: distribution of ‖r_i‖ coloured by class
        ax = axes[0]; labels = np.argmax(self.Yp, axis=1)
        ax.hist(rn, bins=40, color=S.INDIGO, alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.set_xlabel("‖r_i‖ = ‖x'_i − A† y_i‖   (inverse deviation)")
        ax.set_ylabel("count"); ax.grid(True, axis="y", color=S.GRIDLINE, lw=0.6, alpha=0.6)
        ax.set_title("deviation magnitude distribution", fontsize=11, color=S.INK, pad=6)
        # right: per-feature deviation for one sample (most-deviating by default)
        ax2 = axes[1]; i = sample_index if sample_index is not None else int(np.argmax(rn))
        ri = R[i]; order = np.argsort(np.abs(ri))[::-1][:min(12, self.d)]
        order = order[np.argsort(ri[order])]
        vmax = float(np.max(np.abs(ri[order]))) or 1.0
        for r, k in enumerate(order):
            S.gradient_hbar(ax2, r, float(ri[k]), vmax=vmax, height=0.55)
        ax2.axvline(0, color=S.INK, lw=1.0, alpha=0.5)
        ax2.set_yticks(range(len(order))); ax2.set_yticklabels([self.feature_names[k] for k in order], fontsize=9)
        ax2.set_xlim(-vmax * 1.3, vmax * 1.3); ax2.set_ylim(-0.7, len(order) - 0.3)
        ax2.set_xlabel(f"per-feature deviation r (sample {i})")
        ax2.set_title(f"where sample {i} deviates", fontsize=11, color=S.INK, pad=6)
        S.style_title(axes[0], "Inverse Deviation Map",
                      subtitle="deviation from the inverse representative — not model error")
        self.expl._finish(fig, save_path, show)
        return rn

    # ============================================================== #
    # 7. Huber Reliability Map                                      #
    # ============================================================== #
    def reliability_weights(self, delta=None):
        """Recompute the HuberAIME IRLS reliability weights from the *final*
        operator: ``w_i = min(1, δ/‖x'_i − A y_i‖)`` (= the fixed-point weights).
        These flag samples down-weighted when estimating the operator — an
        inverse-operator estimation reliability, NOT a classification confidence."""
        self._need_data("reliability_weights")
        delta = float(self.expl.delta if delta is None else delta)
        _, rn = self.inverse_deviation()
        w = np.ones_like(rn)
        big = rn > delta
        w[big] = delta / rn[big]
        return w, delta

    def plot_reliability_map(self, delta=None, save_path=None, show=True):
        w, delta = self.reliability_weights(delta)
        labels = np.argmax(self.Yp, axis=1)
        fig, axes = S.new_figure(figsize=(11, 4.8), ncols=2)
        ax = axes[0]
        ax.hist(w, bins=40, color=S.CORAL, alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.axvline(1.0, color=S.INK, lw=1.0, alpha=0.5)
        ax.set_xlabel(f"reliability weight  w_i = min(1, δ/‖r_i‖),  δ={delta:g}")
        ax.set_ylabel("count"); ax.grid(True, axis="y", color=S.GRIDLINE, lw=0.6, alpha=0.6)
        ax.set_title(f"{(w < 1).mean()*100:.0f}% of samples down-weighted", fontsize=11, color=S.INK, pad=6)
        # right: per-class mean reliability
        ax2 = axes[1]
        means = [w[labels == c].mean() if (labels == c).any() else np.nan for c in range(self.m)]
        for c in range(self.m):
            S.class_hbar(ax2, self.m - 1 - c, float(means[c]), S.CLASS_CYCLE[c % len(S.CLASS_CYCLE)],
                         vmax=1.0, height=0.55)
            ax2.text(means[c] + 0.02, self.m - 1 - c, f"{means[c]:.2f}", va="center",
                     ha="left", fontsize=9.5, color=S.INK)
        ax2.set_yticks(range(self.m)); ax2.set_yticklabels(self.class_names[::-1], fontsize=10)
        ax2.set_xlim(0, 1.12); ax2.set_ylim(-0.7, self.m - 0.3)
        ax2.set_xlabel("mean reliability per class")
        ax2.set_title("class-wise reliability", fontsize=11, color=S.INK, pad=6)
        note = "" if self.expl.use_huber else "  (variant is not Huber — shown for reference)"
        S.style_title(axes[0], "Huber Reliability Map" + note,
                      subtitle="samples the robust operator estimation trusted least")
        self.expl._finish(fig, save_path, show)
        return w

    # ============================================================== #
    # 8. Explanation Redundancy Map                                #
    # ============================================================== #
    def redundancy_matrix(self):
        """Cosine similarity ``S_pq = cos(a_p, a_q)`` between operator rows
        (each feature's length-m vector).  Prediction-relevant feature similarity
        in ``A†`` — NOT raw input multicollinearity (that is an ``XᵀX`` property)."""
        nrm = np.linalg.norm(self.A, axis=1, keepdims=True)
        Au = self.A / np.where(nrm == 0, 1, nrm)
        return Au @ Au.T

    def plot_explanation_redundancy(self, order=True, save_path=None, show=True):
        Sm = self.redundancy_matrix()
        idx = np.arange(self.d)
        if order and self.d > 2:
            # seriation by the leading singular direction of the operator rows
            try:
                U = np.linalg.svd(self.A - self.A.mean(0), full_matrices=False)[0]
                idx = np.argsort(U[:, 0])
            except np.linalg.LinAlgError:
                pass
        M = Sm[np.ix_(idx, idx)]
        feats = [self.feature_names[k] for k in idx]
        fig, ax = S.new_figure(figsize=(0.42 * self.d + 3, 0.42 * self.d + 2.4))
        im = ax.imshow(M, cmap=S.AIME_DIVERGING, norm=S.signed_norm([-1, 1]))
        ax.set_xticks(range(self.d)); ax.set_xticklabels(feats, rotation=90, fontsize=8)
        ax.set_yticks(range(self.d)); ax.set_yticklabels(feats, fontsize=8)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=8, color=S.INK_SOFT); cb.set_label("cos(aₚ, a_q)", fontsize=9, color=S.INK_SOFT)
        S.style_title(ax, "Explanation Redundancy Map",
                      subtitle="prediction-relevant feature similarity in A†  (not input multicollinearity)")
        self.expl._finish(fig, save_path, show, rect=False)
        return pd.DataFrame(Sm, index=self.feature_names, columns=self.feature_names)

    # ============================================================== #
    # 9. Inverse Explanation Modes (SVD of A†)                      #
    # ============================================================== #
    def inverse_modes(self):
        """SVD ``A = U Σ Vᵀ`` of the (already-derived) explicit operator.
        U = input-feature modes, V = output-class modes, Σ = mode strengths.
        Reveals the low-dimensional structure of the inverse explanation (this
        is the structure of A†, not of the black box itself)."""
        U, Sg, Vt = np.linalg.svd(self.A, full_matrices=False)
        return U, Sg, Vt

    def plot_inverse_modes(self, k=2, image_shape=None, save_path=None, show=True):
        import matplotlib.pyplot as plt
        U, Sg, Vt = self.inverse_modes()
        k = min(k, len(Sg))
        image_shape = image_shape or self.image_shape
        fig = plt.figure(figsize=(12.5, 3.0 + 2.2 * k), facecolor=S.PAPER); S.apply_aime_style()
        gs = fig.add_gridspec(k + 1, 2, width_ratios=[1.0, 1.4], height_ratios=[1.1] + [1] * k,
                              hspace=0.55, wspace=0.3)
        # scree (top-left)
        axs = fig.add_subplot(gs[0, 0]); axs.set_facecolor(S.PANEL)
        xs = np.arange(1, len(Sg) + 1)
        axs.bar(xs, Sg, width=0.6, color=S.INDIGO, edgecolor="white", zorder=3)
        axs.plot(xs, Sg, "-o", color=S.CORAL, lw=2, ms=5, zorder=4)
        axs.set_xticks(xs); axs.set_xlabel("mode"); axs.set_ylabel("σ")
        axs.set_title("singular values (mode strength)", fontsize=10.5, color=S.INK, pad=5)
        axs.grid(True, axis="y", color=S.GRIDLINE, lw=0.6, alpha=0.7)
        # energy text (top-right)
        axe = fig.add_subplot(gs[0, 1]); axe.axis("off")
        energy = (Sg ** 2) / (Sg ** 2).sum()
        txt = "mode energy (σ²-share):\n" + "  ".join(f"M{i+1}={energy[i]*100:.0f}%" for i in range(len(Sg)))
        axe.text(0.0, 0.6, txt, fontsize=11, color=S.INK, va="center")
        # per-mode loadings
        for mi in range(k):
            axi = fig.add_subplot(gs[mi + 1, 0]); axi.set_facecolor(S.PANEL)
            u = U[:, mi]; vmax = float(np.max(np.abs(u))) or 1.0
            top = np.argsort(np.abs(u))[::-1][:min(10, self.d)]; top = top[np.argsort(u[top])]
            for r, f in enumerate(top):
                S.gradient_hbar(axi, r, float(u[f]), vmax=vmax, height=0.55)
            axi.axvline(0, color=S.INK, lw=1.0, alpha=0.5)
            axi.set_yticks(range(len(top))); axi.set_yticklabels([self.feature_names[t] for t in top], fontsize=8)
            axi.set_xlim(-vmax * 1.3, vmax * 1.3); axi.set_ylim(-0.7, len(top) - 0.3)
            axi.set_title(f"mode {mi+1} · input-feature loadings (U)", fontsize=10, color=S.INK, pad=4)
            axo = fig.add_subplot(gs[mi + 1, 1]); axo.set_facecolor(S.PANEL)
            v = Vt[mi]; vmx = float(np.max(np.abs(v))) or 1.0
            for c in range(self.m):
                S.gradient_hbar(axo, self.m - 1 - c, float(v[c]), vmax=vmx, height=0.5)
            axo.axvline(0, color=S.INK, lw=1.0, alpha=0.5)
            axo.set_yticks(range(self.m)); axo.set_yticklabels(self.class_names[::-1], fontsize=9)
            axo.set_xlim(-vmx * 1.3, vmx * 1.3); axo.set_ylim(-0.7, self.m - 0.3)
            axo.set_title(f"mode {mi+1} · output-class loadings (V)", fontsize=10, color=S.INK, pad=4)
        if not S.PUBLICATION:
            fig.suptitle("Inverse Explanation Modes · SVD of A†", x=0.012, ha="left",
                         fontsize=15, fontweight="bold", color=S.INK)
        S.add_brand(fig); fig.tight_layout(rect=(0, 0.02, 1, 0.97))
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=S.PAPER)
        if show:
            plt.show()
        else:
            plt.close(fig)
        return U, Sg, Vt

    # ============================================================== #
    # 10. Bayesian Uncertainty Map                                 #
    # ============================================================== #
    def plot_uncertainty_map(self, n_samples=120, save_path=None, show=True):
        """BayesianAIME: posterior over ``A``.  Shows the per-coefficient
        signal-to-uncertainty ``|μ|/σ`` (σ_k = sqrt(cov[k,k]) is shared across
        features), and a posterior spaghetti of the global importance per class.
        Requires a BayesianAIME explainer (``A_dagger_cov``)."""
        cov = getattr(self.expl, "A_dagger_cov", None)
        if cov is None:
            raise ValueError("plot_uncertainty_map requires a BayesianAIME explainer "
                             "(use_bayesian=True), which provides A_dagger_cov.")
        cov = np.asarray(cov, float)
        sigma = np.sqrt(np.clip(np.diag(cov), 0, None))         # (m,) per-class std (shared over features)
        snr = np.abs(self.A) / np.where(sigma == 0, 1, sigma)[None, :]   # (d, m)
        fig, axes = S.new_figure(figsize=(13, 0.34 * self.d + 3), ncols=2)
        # left: signal-to-uncertainty heatmap
        ax = axes[0]
        im = ax.imshow(snr, aspect="auto", cmap=S.AIME_SEQ_WARM)
        ax.set_xticks(range(self.m)); ax.set_xticklabels(self.class_names, rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(self.d)); ax.set_yticklabels(self.feature_names, fontsize=8)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=8, color=S.INK_SOFT); cb.set_label("|μ| / σ  (signal-to-uncertainty)", fontsize=9, color=S.INK_SOFT)
        ax.set_title("signal-to-uncertainty per coefficient", fontsize=11, color=S.INK, pad=6)
        # right: posterior spaghetti of global importance for class 0
        ax2 = axes[1]; rng = np.random.default_rng(0); c = 0
        order = np.argsort(np.abs(self.A[:, c]))
        yv = np.arange(self.d)
        for _ in range(n_samples):
            sample = self.A[:, c] + rng.normal(0, sigma[c], self.d)
            ax2.plot(sample[order], yv, color=S.INDIGO, alpha=0.06, lw=1)
        ax2.plot(self.A[order, c], yv, color=S.CORAL, lw=2.2, label="posterior mean")
        ax2.set_yticks(range(self.d)); ax2.set_yticklabels([self.feature_names[k] for k in order], fontsize=8)
        ax2.axvline(0, color=S.INK, lw=1.0, alpha=0.5)
        ax2.set_xlabel(f"A†[:, {self.class_names[c]}]  (posterior samples)")
        ax2.set_title(f"posterior spaghetti · {self.class_names[c]}", fontsize=11, color=S.INK, pad=6)
        ax2.legend(loc="lower right", frameon=False, fontsize=9)
        S.style_title(axes[0], "Bayesian Uncertainty Map",
                      subtitle="which parts of the inverse explanatory operator are certain vs uncertain")
        self.expl._finish(fig, save_path, show)
        return pd.DataFrame(snr, index=self.feature_names, columns=self.class_names)

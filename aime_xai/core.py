"""
aime_xai.core
=============

AIME (Approximate Inverse Model Explanations) — reference implementation with a
*signature* publication-grade visualisation layer.

Theory / implementation correspondence
---------------------------------------
AIME treats explanation as an **approximate inverse problem**: instead of
probing how the output changes when the input is perturbed (the forward problem
solved by LIME/SHAP), it builds an approximate inverse operator

        A_dagger  (written  A^\\dagger)   such that   x ≈ A^\\dagger y ,

derived directly from a dataset ``X`` and the model outputs ``Y`` via a
Moore–Penrose pseudo-inverse (or a Huber-robust IRLS variant).  From this single
operator everything follows:

* **Global feature importance** — column ``t`` of ``A_dagger`` is the feature
  weight vector for class ``t`` (obtained as ``A_dagger @ e_t`` with ``e_t`` the
  one-hot class basis).
* **Representative estimation instance** — ``A_dagger @ e_t`` *is* the "ideal
  input" the model associates with a pure prediction of class ``t``.  This is a
  genuinely generative quantity that forward-problem methods cannot produce.
* **Local feature importance** — the Hadamard product
  ``(A_dagger @ y) ⊙ x'``.  Because it multiplies by the instance itself, any
  feature that is zero in ``x`` contributes exactly zero (intuitive, no
  attribution to absent features).

The mathematics below is unchanged from the canonical AIME / HuberAIME papers;
only the *visualisation* has been redesigned.  Every public method name and
signature is preserved for drop-in compatibility; the plotting methods gain a
few **optional** keyword arguments (e.g. ``save_path``, ``show``) with defaults
that keep existing call sites working.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# matplotlib is imported lazily (inside the plotting methods / helpers below) so
# that the *computation core* — operator construction, representative instances,
# the ``*_without_viz`` methods and the metrics module — can be used with only
# numpy + pandas installed.  ``import aime_xai`` does NOT require matplotlib.
# scikit-learn is likewise imported lazily inside the few methods that need it
# (create_explainer / dim_reduce).

from . import style as S


def _show_or_close(fig, show):
    """Display the figure if ``show`` else close it (frees memory and avoids the
    'More than 20 figures' warning when many plots are produced with show=False).
    matplotlib is imported lazily here so the core stays import-light."""
    import matplotlib.pyplot as plt
    if show:
        plt.show()
    else:
        plt.close(fig)


class AIME:
    """
    AIME family — one class, five variants selected by flags:

    * **AIME**            (defaults)                      — Moore–Penrose pseudo-inverse
    * **HuberAIME**       ``use_huber=True``              — outlier-robust IRLS
    * **RidgeAIME**       ``use_ridge=True``              — ℓ2-regularised closed form
    * **Huber-RidgeAIME** ``use_huber=True, use_ridge=True``
    * **BayesianAIME**    ``use_bayesian=True``           — posterior mean + covariance
      (yields 95% credible intervals on the importances)

    The operator mathematics is identical to the canonical AIME / HuberAIME /
    RidgeAIME / BayesianAIME implementations; only the visualisation layer is the
    signature edition.  ``use_bayesian`` cannot be combined with Huber/Ridge.
    """

    def __init__(self,
                 use_huber=False,
                 delta=1.0,
                 max_iter=50,
                 tol=1e-5,
                 *,
                 use_ridge=False,
                 use_bayesian=False,
                 ridge_alpha=1e-4,
                 bayesian_sigma=1.0,
                 bayesian_tau=1.0):
        """
        Parameters
        ----------
        use_huber : bool
            If True, use Huber loss + IRLS for an outlier-robust operator.
        use_ridge : bool
            If True, add ℓ2 (Ridge) regularisation (combinable with Huber).
        use_bayesian : bool
            If True, Bayesian estimation of the operator with a posterior
            covariance (gives 95% credible intervals).  Mutually exclusive with
            Huber/Ridge.
        delta : float
            Huber threshold (ignored unless use_huber=True).
        ridge_alpha : float
            Ridge coefficient λ (ignored unless use_ridge=True).
        bayesian_sigma, bayesian_tau : float
            Likelihood-noise σ and prior-variance τ for Bayesian AIME.
        max_iter, tol : int, float
            IRLS iteration cap and Frobenius tolerance (Huber).
        """
        self.use_huber = bool(use_huber)
        self.use_ridge = bool(use_ridge)
        self.use_bayesian = bool(use_bayesian)
        self.delta = float(delta)
        self.ridge_alpha = float(ridge_alpha)
        self.bayesian_sigma = float(bayesian_sigma)
        self.bayesian_tau = float(bayesian_tau)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

        if self.use_bayesian and (self.use_huber or self.use_ridge):
            raise ValueError("Bayesian mode cannot be combined with Huber or Ridge modes.")

        self.A_dagger = None
        self.A_dagger_cov = None          # posterior covariance (Bayesian only)
        self.scaler = None
        # kept for the convenience of the new visualisations (purely optional)
        self._X = None
        self._Y = None

    @property
    def variant_name(self):
        """Human-readable name of the active variant."""
        if self.use_bayesian:
            return "BayesianAIME"
        if self.use_huber and self.use_ridge:
            return "Huber-RidgeAIME"
        if self.use_huber:
            return "HuberAIME"
        if self.use_ridge:
            return "RidgeAIME"
        return "AIME"

    # ------------------------------------------------------------------ #
    # Operator construction (mathematics UNCHANGED from canonical AIME)   #
    # ------------------------------------------------------------------ #
    def create_explainer(self, X, Y, normalize=True):
        """
        Create an explainer by deriving the approximate inverse operator
        from input X and output Y. If use_huber=True, applies IRLS with Huber loss.

        Parameters
        ----------
        X : array-like of shape (N, n)
            Input data.
        Y : array-like of shape (N, m)
            Output data.
        normalize : bool
            If True, apply standard scaling to X before computing the operator.

        Returns
        -------
        self : AIME
            Fitted explainer with self.A_dagger as the inverse operator.
        """
        if X is None or Y is None:
            raise ValueError("Both X and Y must be provided.")
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        if X.ndim != 2:
            raise ValueError(
                f"X must be 2-D (N, n_features); got shape {X.shape}.")
        if Y.ndim != 2:
            raise ValueError(
                f"Y must be 2-D (N, n_classes); got shape {Y.shape}. "
                "For a single output column reshape with Y.reshape(-1, 1).")
        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"X and Y sample sizes differ: X has {X.shape[0]} rows, "
                f"Y has {Y.shape[0]} rows.")
        if X.shape[0] < 1:
            raise ValueError("X and Y must contain at least one sample.")
        self.A_dagger, self.A_dagger_cov, self.scaler = \
            self._generate_inverse_operator_from_y(X, Y, normalize)
        self._X = np.asarray(X, dtype=float)
        self._Y = np.asarray(Y, dtype=float)
        return self

    def _generate_inverse_operator_from_y(self, X, Y, normalize=True):
        """
        Compute the approximate inverse operator (and, for Bayesian mode, its
        posterior covariance).  Routes to the variant selected by the flags.
        Returns (A_dagger, A_dagger_cov_or_None, scaler).
        """
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)

        # Optional normalization of X
        if normalize:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler().fit(X)
            X_prime = scaler.transform(X)
        else:
            scaler = None
            X_prime = X

        A_cov = None
        if self.use_bayesian:
            # Bayesian solver consumes the (N, d) / (N, m) layout directly
            A_dagger, A_cov = self._bayesian_solver(X_prime, Y)
        else:
            X_t = X_prime.T   # (n, N)
            Y_t = Y.T         # (m, N)
            if not self.use_huber and not self.use_ridge:
                A_dagger = X_t @ np.linalg.pinv(Y_t)          # AIME
            elif self.use_huber and not self.use_ridge:
                A_dagger = self._huber_inverse_operator(X_t, Y_t, ridge=False)
            elif self.use_ridge and not self.use_huber:
                A_dagger = self._ridge_solver(X_t, Y_t)       # RidgeAIME
            else:
                A_dagger = self._huber_inverse_operator(X_t, Y_t, ridge=True)

        return A_dagger, A_cov, scaler

    # ---- Ridge closed form:  M = X Yᵀ (Y Yᵀ + λI)⁻¹ -------------------- #
    def _ridge_solver(self, X_t, Y_t):
        m = Y_t.shape[0]
        G = Y_t @ Y_t.T + self.ridge_alpha * np.eye(m)
        try:
            inv_part = np.linalg.inv(G)
        except np.linalg.LinAlgError:
            inv_part = np.linalg.pinv(G)
        return X_t @ (Y_t.T @ inv_part)

    # ---- Bayesian linear regression for the operator ------------------- #
    def _bayesian_solver(self, X_proc, Y):
        """
        Posterior mean and (feature-shared) covariance of the operator under the
        model  X = Y · A_dagger.ᵀ + E,  with Gaussian likelihood (σ) and prior (τ).

        Σ_post = (σ⁻² ΦᵀΦ + τ⁻² I)⁻¹ ,   Φ = Y
        μ_post = Σ_post (σ⁻² ΦᵀX + τ⁻² Â_priorᵀ) ,  Â_prior = pseudo-inverse AIME
        Returns (mean (d, m), cov (m, m)).

        Note on the interpretation
        --------------------------
        The prior mean ``Â_prior`` is the ordinary pseudo-inverse AIME operator
        computed **from the same data**.  This is therefore an *empirical-Bayes /
        regularised posterior-style* estimate, **not** a strict subjective-prior
        Bayesian inference: the "prior" is data-derived, so the resulting
        covariance should be read as a regularised uncertainty proxy (a τ-shrunk
        estimator spread) rather than a fully calibrated posterior. The credible
        intervals are useful for *relative* comparison of operator entries.
        """
        N, d = X_proc.shape
        m = Y.shape[1]
        s2_inv = 1.0 / (self.bayesian_sigma ** 2)
        t2_inv = 1.0 / (self.bayesian_tau ** 2)

        A_prior = X_proc.T @ np.linalg.pinv(Y.T)        # standard AIME prior mean
        precision = s2_inv * (Y.T @ Y) + t2_inv * np.eye(m)
        try:
            cov_post = np.linalg.inv(precision)
        except np.linalg.LinAlgError:
            cov_post = np.linalg.pinv(precision)
        mean_post_T = cov_post @ (s2_inv * (Y.T @ X_proc) + t2_inv * A_prior.T)
        return mean_post_T.T, cov_post

    def _huber_inverse_operator(self, X_t, Y_t, ridge=False):
        """
        Solve min sum_{i=1..N} huber( ||X_i - M Y_i|| ) via IRLS, optionally with
        ℓ2 (Ridge) regularisation λI inside each weighted least-squares step
        (Huber-RidgeAIME when ``ridge=True``).
        X_t : (n, N) ; Y_t : (m, N) ; returns M (n, m).
        """
        n, N = X_t.shape
        m = Y_t.shape[0]
        if Y_t.shape[1] != N:
            raise ValueError("Dimension mismatch in X_t and Y_t.")

        lamI = self.ridge_alpha * np.eye(m) if ridge else 0.0

        # 1) Initialize M with the (optionally regularised) closed-form solution
        try:
            inv_init = np.linalg.inv(Y_t @ Y_t.T + lamI)   # shape (m, m)
        except np.linalg.LinAlgError:
            inv_init = np.linalg.pinv(Y_t @ Y_t.T + lamI)
        M = X_t @ (Y_t.T @ inv_init)

        delta = self.delta
        for _ in range(self.max_iter):
            # 2) residuals r_i = || x_i - M y_i ||
            R = X_t - (M @ Y_t)
            residuals = np.linalg.norm(R, axis=0)

            # 3) Huber weights
            w = np.ones_like(residuals)
            mask_large = (residuals > delta)
            w[mask_large] = delta / residuals[mask_large]

            # 4) weighted (optionally ridge) least squares
            W_sqrt = np.sqrt(w)
            X_w = X_t * W_sqrt
            Y_w = Y_t * W_sqrt
            try:
                tmp_inv = np.linalg.inv(Y_w @ Y_w.T + lamI)
            except np.linalg.LinAlgError:
                tmp_inv = np.linalg.pinv(Y_w @ Y_w.T + lamI)
            M_new = X_w @ (Y_w.T @ tmp_inv)

            # Convergence-break semantics are kept BYTE-IDENTICAL to the canonical
            # HuberAIME IRLS reference: when the update falls below tolerance we
            # break and return the *current* M (the iterate before the final
            # below-tolerance step), NOT M_new.  Returning M_new would be the more
            # "natural" choice and differs only by ‖M_new − M‖_F < tol (here
            # tol=1e-5), but it would break exact reproduction of published AIME-
            # family results, so the reference behaviour is retained deliberately.
            if np.linalg.norm(M_new - M, ord='fro') < self.tol:
                break
            M = M_new

        return M

    # ------------------------------------------------------------------ #
    # Shared helpers                                                      #
    # ------------------------------------------------------------------ #
    def _global_matrix(self, feature_names=None, class_names=None,
                       normalize_rows=True):
        """Return the (class x feature) global-importance DataFrame.

        Row ``t`` is ``A_dagger @ e_t`` (== column ``t`` of ``A_dagger``),
        optionally peak-normalised to [-1, 1] exactly as in the canonical
        ``global_feature_importance``.
        """
        if self.A_dagger is None:
            raise ValueError("Please create an explainer first using create_explainer.")
        dim = self.A_dagger.shape[1]
        data = []
        for t in range(dim):
            basis = np.zeros(dim)
            basis[t] = 1
            heatmap = np.dot(self.A_dagger, basis)
            if normalize_rows:
                maxval = np.max(np.abs(heatmap))
                if maxval > 0:
                    heatmap = heatmap / maxval
            data.append(heatmap)
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(self.A_dagger.shape[0])]
        if class_names is None:
            class_names = [f"class_{i}" for i in range(dim)]
        return pd.DataFrame(np.array(data), index=class_names, columns=feature_names)

    def _check_local_inputs(self, x, y):
        """Validate a single local instance ``x`` and its target output ``y``.

        Gives a clear, actionable error instead of an opaque NumPy broadcasting
        failure when the feature/class dimensionality does not match the fitted
        operator.  Returns the coerced ``(x, y)`` float arrays.
        """
        if self.A_dagger is None:
            raise ValueError("Please create an explainer first using create_explainer.")
        if x is None or y is None:
            raise ValueError("Please provide both x and y for local explanation.")
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        d, m = self.A_dagger.shape
        if x.ndim != 1 or x.shape[0] != d:
            raise ValueError(
                f"x must be a 1-D vector of length n_features={d}; got shape {x.shape}.")
        if y.ndim != 1 or y.shape[0] != m:
            raise ValueError(
                f"y must be a 1-D vector of length n_classes={m}; got shape {y.shape}.")
        return x, y

    @staticmethod
    def _finish(fig, save_path=None, show=True, rect=None):
        S.add_brand(fig)            # no-op in publication mode
        if rect is None:
            rect = (0, 0.04, 1, S._top_rect())
        if rect is not False:       # rect=False → caller manages layout itself
            try:
                fig.tight_layout(rect=rect)
            except Exception:
                pass
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=S.PAPER)
        _show_or_close(fig, show)
        return fig

    # ================================================================== #
    # GLOBAL feature importance  (signature redesign)                     #
    # ================================================================== #
    def global_feature_importance(self, feature_names=None, class_names=None,
                                  top_k=None, top_k_criterion='average',
                                  save_path=None, show=True, *,
                                  layout="vertical"):
        """
        Visualize and return a DataFrame of global feature importance.

        Signature visual: an **operator field** heatmap (the class × feature
        sub-block of A_dagger rendered with the AIME diverging colormap) paired
        with a **per-class signed feature ranking**.  Same return value
        (class × feature DataFrame) as the canonical implementation.

        ``layout`` (keyword-only):
        * ``"vertical"`` (default) — heatmap on top, ranking below, each spanning
          the full figure width.  Recommended when feature names are long, since
          each panel gets the entire left margin and labels never overlap.
        * ``"horizontal"`` — the original compact side-by-side arrangement.
        """
        df = self._global_matrix(feature_names, class_names)
        err = self._global_ci_margin(df.index, df.columns)   # None unless Bayesian

        if top_k is not None:
            # Rank by MAGNITUDE (|weight|): AIME weights are signed, and a strong
            # negative importance is just as informative as a strong positive one.
            # Selecting by the raw (signed) average/max would silently drop large
            # negative contributions, so we rank on the absolute value.
            if top_k_criterion == 'average':
                cols = df.abs().mean(axis=0).nlargest(top_k).index.tolist()
            elif top_k_criterion == 'max':
                cols = df.abs().max(axis=0).nlargest(top_k).index.tolist()
            else:
                raise ValueError(f"Unknown top_k_criterion: {top_k_criterion}")
            df = df.loc[:, cols]
            if err is not None:
                err = err.loc[:, cols]

        self._plot_global(df, title=f"Global Feature Importance · {self.variant_name}",
                          subtitle="columns of the approximate inverse operator  A†  ·  Aᵀ y = feature weights",
                          err=err, save_path=save_path, show=show, layout=layout)

        if self.use_bayesian and err is not None:
            # return mean + 95% credible-interval bounds (canonical Bayesian form)
            return pd.concat({'mean': df, 'lower_bound': df - err,
                              'upper_bound': df + err}, axis=1) \
                     .swaplevel(0, 1, axis=1).sort_index(axis=1)
        return df

    def _global_ci_margin(self, class_names, feature_names):
        """95% credible-interval half-width of the *normalised* global importance,
        per (class, feature).  Returns None unless in Bayesian mode.

        The posterior covariance is shared across features, so the operator entry
        A[i, j] has std sqrt(cov[j, j]); after the per-class peak normalisation it
        becomes a per-class constant broadcast over the features.
        """
        if not self.use_bayesian or self.A_dagger_cov is None:
            return None
        m = self.A_dagger.shape[1]
        std_per_class = np.sqrt(np.clip(np.diag(self.A_dagger_cov), 0, None))
        norm = np.array([np.max(np.abs(self.A_dagger[:, j])) or 1.0 for j in range(m)])
        margin = 1.96 * (std_per_class / norm)             # (m,)
        d = self.A_dagger.shape[0]
        return pd.DataFrame(np.tile(margin.reshape(-1, 1), (1, d)),
                            index=list(class_names), columns=list(feature_names))

    def global_feature_importance_each(self, feature_names=None, class_names=None,
                                       top_k=None, top_k_criterion='average',
                                       class_num=0, save_path=None, show=True):
        """
        Similar to global_feature_importance but only for a single output
        dimension (``class_num``).  Rendered as a signed lollipop ranking.
        """
        df_full = self._global_matrix(feature_names, class_names)
        cname = df_full.index[class_num]
        row = df_full.iloc[class_num:class_num + 1, :]

        if top_k is not None:
            # rank by magnitude so strong negative weights are not dropped
            if top_k_criterion == 'average':
                cols = row.abs().mean(axis=0).nlargest(top_k).index.tolist()
            elif top_k_criterion == 'max':
                cols = row.abs().max(axis=0).nlargest(top_k).index.tolist()
            else:
                raise ValueError(f"Unknown top_k_criterion: {top_k_criterion}")
            row = row.loc[:, cols]

        series = row.iloc[0]
        order = series.abs().sort_values(ascending=True).index
        series = series.loc[order]

        fig, ax = S.new_figure(figsize=(9.2, max(3.6, 0.42 * len(series) + 2)))
        vmax = float(np.max(np.abs(series.values))) or 1.0
        for i, (feat, val) in enumerate(series.items()):
            S.gradient_hbar(ax, i, val, vmax=vmax, height=0.5)
        ax.axvline(0, color=S.INK, lw=1.1, alpha=0.55, zorder=1)
        ax.set_yticks(range(len(series)))
        ax.set_yticklabels(series.index, fontsize=10)
        ax.set_xlim(-vmax * 1.28, vmax * 1.28)
        ax.set_ylim(-0.7, len(series) - 0.3)
        ax.set_xlabel("signed contribution to the class (A† column)")
        S.style_title(ax, f"Global Feature Importance · {cname}",
                      subtitle="how each input feature pulls the model toward this single class")
        self._finish(fig, save_path, show)
        return row

    def _global_heatmap(self, fig, axf, df, M, norm, title, subtitle, cax=None,
                        xlabels_top=False):
        """Draw the operator-field heatmap (features × classes) on ``axf``.

        ``cax`` (optional) is a dedicated colorbar axes; if ``None`` the colorbar
        is carved out of ``axf`` with a fraction (side-by-side layout).
        ``xlabels_top`` puts the class tick labels above the heatmap (used in the
        vertical layout so they don't collide with the legend below)."""
        axf.set_facecolor(S.PANEL)
        n_feat, n_classes = M.shape
        im = axf.imshow(M, aspect="auto", cmap=S.AIME_DIVERGING, norm=norm)
        axf.set_xticks(range(n_classes))
        if xlabels_top:
            axf.xaxis.set_ticks_position("top")
            axf.set_xticklabels(df.index, rotation=30, ha="left", fontsize=9.5)
        else:
            axf.set_xticklabels(df.index, rotation=30, ha="right", fontsize=9.5)
        axf.set_yticks(range(n_feat))
        axf.set_yticklabels(df.columns, fontsize=9)
        axf.set_xticks(np.arange(-.5, n_classes, 1), minor=True)
        axf.set_yticks(np.arange(-.5, n_feat, 1), minor=True)
        axf.grid(which="minor", color=S.PAPER, lw=1.4)
        axf.grid(which="major", visible=False)
        axf.tick_params(which="minor", length=0)
        if n_feat * n_classes <= 220:
            for i in range(n_feat):
                for j in range(n_classes):
                    v = M[i, j]
                    axf.text(j, i, f"{v:+.2f}", ha="center", va="center",
                             fontsize=7.4,
                             color=(S.PAPER if abs(v) > 0.6 * norm.vmax else S.INK))
        if cax is not None:
            cb = fig.colorbar(im, cax=cax)
        else:
            cb = fig.colorbar(im, ax=axf, fraction=0.046, pad=0.03)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=8, color=S.INK_SOFT)
        cb.set_label("signed weight  (A†)", fontsize=9, color=S.INK_SOFT)
        S.style_title(axf, title, subtitle=subtitle)

    def _global_bars(self, axr, df, err, vmax):
        """Draw per-class grouped horizontal feature bars on ``axr``.

        This is the defining AIME capability: a distinct signed feature-weight
        vector for EVERY output class (LIME has none; SHAP must average many
        per-instance values).  Returns the per-class legend handles."""
        from matplotlib.patches import Patch
        axr.set_facecolor(S.PANEL)
        feats = list(df.columns)
        n_f = len(feats)
        n_classes = df.shape[0]
        group_h = 0.82
        sub_h = group_h / max(n_classes, 1)
        for fi, feat in enumerate(feats):
            y_base = (n_f - 1 - fi)  # feature 0 at top (aligns with heatmap)
            for cj in range(n_classes):
                offset = (cj - (n_classes - 1) / 2.0) * sub_h
                val = df.iloc[cj][feat]
                col = S.CLASS_CYCLE[cj % len(S.CLASS_CYCLE)]
                S.class_hbar(axr, y_base + offset, val, col,
                             height=sub_h * 0.86, vmax=vmax)
                if err is not None:
                    e = float(err.iloc[cj][feat])
                    axr.errorbar(val, y_base + offset, xerr=e, fmt='none',
                                 ecolor=S.INK, elinewidth=1.0, capsize=2.5,
                                 alpha=0.7, zorder=6)
        axr.axvline(0, color=S.INK, lw=1.1, alpha=0.55, zorder=2)
        axr.set_yticks(range(n_f))
        axr.set_yticklabels(feats[::-1], fontsize=9)
        axr.set_xlim(-vmax * 1.12, vmax * 1.12)
        axr.set_ylim(-0.7, n_f - 0.3)
        axr.set_xlabel("signed feature weight, per class  (A†)")
        axr.grid(True, axis="x", color=S.GRIDLINE, lw=0.8, alpha=0.8)
        return [Patch(facecolor=S.CLASS_CYCLE[j % len(S.CLASS_CYCLE)],
                      edgecolor="white", label=str(df.index[j]))
                for j in range(n_classes)]

    def _plot_global(self, df, title, subtitle, err=None, save_path=None,
                     show=True, layout="vertical"):
        """Signature global view: operator field heatmap + per-class bars.

        ``layout`` controls how the two panels are arranged:

        * ``"vertical"`` (default) — heatmap on top, bars below, each spanning the
          **full width**.  This is the recommended layout: long feature names get
          the entire left margin and never collide with a neighbouring panel.
        * ``"horizontal"`` — the original side-by-side arrangement (more compact,
          but long feature labels can overlap between the two panels).

        When ``err`` (a class × feature DataFrame of 95 % CI half-widths) is given
        — i.e. BayesianAIME — horizontal credible-interval whiskers are drawn on
        every per-class bar."""
        import matplotlib.pyplot as plt
        n_classes, n_feat = df.shape
        M = df.values.T  # features (rows) × classes (cols)
        norm = S.signed_norm(M)
        vmax = float(np.max(np.abs(df.values))) or 1.0
        if err is not None:
            vmax = max(vmax, float(np.max(np.abs(df.values) + err.values)))

        if layout == "vertical":
            # Two stacked full-width panels: heatmap (top) + bars (bottom).
            fig = plt.figure(figsize=(10.5, max(6.4, 0.66 * n_feat + 3.0)),
                             facecolor=S.PAPER)
            S.apply_aime_style()
            gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.028],
                                  height_ratios=[1.0, 1.05], hspace=0.20,
                                  wspace=0.03, left=0.26, right=0.97,
                                  top=0.92, bottom=0.08)
            axf = fig.add_subplot(gs[0, 0])
            cax = fig.add_subplot(gs[0, 1])
            axr = fig.add_subplot(gs[1, 0])
            self._global_heatmap(fig, axf, df, M, norm, title, subtitle, cax=cax,
                                 xlabels_top=True)
            handles = self._global_bars(axr, df, err, vmax)
            # horizontal legend above the bars → does not steal panel width
            leg = axr.legend(handles=handles, loc="lower left",
                             bbox_to_anchor=(0.0, 1.005),
                             ncol=min(n_classes, 6), frameon=False,
                             fontsize=9.0, columnspacing=1.2, handlelength=1.2,
                             title="class", title_fontsize=9.0)
            leg._legend_box.align = "left"
            self._finish(fig, save_path, show, rect=False)
        else:
            # Original side-by-side layout (kept for compatibility).
            fig = plt.figure(figsize=(14.0, max(4.6, 0.34 * n_feat + 2.6)),
                             facecolor=S.PAPER)
            S.apply_aime_style()
            gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.05], wspace=0.30,
                                  right=0.90)
            axf = fig.add_subplot(gs[0, 0])
            axr = fig.add_subplot(gs[0, 1])
            self._global_heatmap(fig, axf, df, M, norm, title, subtitle, cax=None)
            handles = self._global_bars(axr, df, err, vmax)
            leg = axr.legend(handles=handles, loc="upper left",
                             bbox_to_anchor=(1.01, 1.0), frameon=False,
                             fontsize=9.5, title="class", title_fontsize=9.5)
            leg._legend_box.align = "left"
            S.style_title(axr, "Feature ranking",
                          subtitle="per-class signed weights")
            self._finish(fig, save_path, show, rect=False)

    # ================================================================== #
    # LOCAL feature importance  (signature redesign)                      #
    # ================================================================== #
    def local_feature_importance(self, x, y, feature_names=None, scale=True,
                                 scaler=None, top_k=None,
                                 ignore_zero_features=True,
                                 save_path=None, show=True):
        """
        Local feature importance for a single instance x with target y.

        Implements the canonical AIME local rule
        ``(A_dagger @ y) ⊙ x'`` (Hadamard product), then renders it as a signed
        diverging ranking.

        ``ignore_zero_features`` (default **True**, matching the original AIME
        implementation) forces features whose *raw* value is exactly ``0`` to
        contribute exactly zero.  This is natural for sparse / one-hot inputs.
        Caution: for standardised continuous features a raw 0 does **not** map to
        a standardised 0, so masking can suppress informative features — set this
        to ``False`` in that case to use the unmasked Hadamard product.
        """
        df = self.local_feature_importance_without_viz(
            x, y, feature_names=feature_names, scale=scale, scaler=scaler,
            top_k=top_k, ignore_zero_features=ignore_zero_features)

        # Bayesian 95% credible-interval half-widths, aligned to df's columns
        err_series = self._local_ci_margin(
            x, y, scale=scale, scaler=scaler,
            ignore_zero_features=ignore_zero_features,
            feature_names=feature_names)

        series = df.iloc[0]
        order = series.abs().sort_values(ascending=True).index
        series = series.loc[order]

        fig, ax = S.new_figure(figsize=(9.2, max(3.4, 0.42 * len(series) + 2)))
        vmax = float(np.max(np.abs(series.values))) or 1.0
        if err_series is not None:
            vmax = max(vmax, float(np.max(np.abs(series.values) +
                                          err_series.reindex(series.index).values)))
        for i, (feat, val) in enumerate(series.items()):
            S.gradient_hbar(ax, i, val, vmax=vmax, height=0.5)
            if err_series is not None:
                e = float(err_series.get(feat, 0.0))
                ax.errorbar(val, i, xerr=e, fmt='none', ecolor=S.INK,
                            elinewidth=1.0, capsize=3, alpha=0.75, zorder=6)
        ax.axvline(0, color=S.INK, lw=1.1, alpha=0.55, zorder=1)
        ax.set_yticks(range(len(series)))
        ax.set_yticklabels(series.index, fontsize=10)
        ax.set_xlim(-vmax * 1.3, vmax * 1.3)
        ax.set_ylim(-0.7, len(series) - 0.3)
        ax.set_xlabel("signed local contribution   (A†y) ⊙ x")
        S.style_title(ax, f"Local Feature Importance · {self.variant_name}",
                      subtitle="why THIS instance was predicted — only present features can contribute")
        self._finish(fig, save_path, show)

        if self.use_bayesian and err_series is not None:
            e = err_series.reindex(df.columns)
            return pd.DataFrame([df.iloc[0].values, df.iloc[0].values - e.values,
                                 df.iloc[0].values + e.values],
                                index=['mean', 'lower_bound', 'upper_bound'],
                                columns=df.columns)
        return df

    def _local_ci_margin(self, x, y, scale=True, scaler=None,
                         ignore_zero_features=True, feature_names=None):
        """95% credible-interval half-width of the *normalised* local importance
        per feature (BayesianAIME only; returns None otherwise).

        Mirrors the canonical derivation: with the posterior covariance shared
        across features, the local contribution (A†y)_i x'_i has std
        |x'_i| · sqrt(yᵀ Σ y); divide by the same peak-normalisation factor used
        for the mean."""
        if not self.use_bayesian or self.A_dagger_cov is None:
            return None
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        use_scaler = scaler if scaler is not None else self.scaler
        x_prime = use_scaler.transform([x])[0] if (scale and use_scaler is not None) else x

        heat = np.dot(self.A_dagger, y) * x_prime
        std_common = np.sqrt(max(float(y.T @ self.A_dagger_cov @ y), 0.0))
        err = np.abs(x_prime) * std_common * 1.96
        if ignore_zero_features:
            mask = (x != 0)
            heat = heat * mask
            err = err * mask
        maxval = np.max(np.abs(heat))
        if maxval > 0:
            err = err / maxval
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(x))]
        return pd.Series(err, index=list(feature_names))

    def local_feature_importance_without_viz(self, x, y, feature_names=None,
                                             scale=True, scaler=None, top_k=None,
                                             ignore_zero_features=True):
        """
        Same as local_feature_importance but returns the DataFrame without plotting.
        """
        x, y = self._check_local_inputs(x, y)
        use_scaler = scaler if scaler is not None else self.scaler
        if scale and (use_scaler is not None):
            x_prime = use_scaler.transform([x])[0]
        else:
            x_prime = x

        # canonical AIME local rule: (A_dagger @ y) ⊙ x'
        heatmap = np.dot(self.A_dagger, y) * x_prime

        if ignore_zero_features:
            heatmap = heatmap * (x != 0)

        maxval = np.max(np.abs(heatmap))
        if maxval > 0:
            heatmap = heatmap / maxval

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(x))]

        df = pd.DataFrame([heatmap], columns=feature_names)
        if top_k is not None:
            sorted_cols = df.iloc[0, :].abs().sort_values(ascending=False).index
            df = df.loc[:, sorted_cols[:top_k]]
        return df

    def global_feature_importance_without_viz(self, feature_names=None,
                                              class_names=None, top_k=None,
                                              top_k_criterion='average'):
        """
        Return a DataFrame of global feature importance without any plotting.
        """
        df = self._global_matrix(feature_names, class_names)
        if top_k is not None:
            # rank by magnitude so strong negative weights are not dropped
            if top_k_criterion == 'average':
                cols = df.abs().mean(axis=0).nlargest(top_k).index.tolist()
            elif top_k_criterion == 'max':
                cols = df.abs().max(axis=0).nlargest(top_k).index.tolist()
            else:
                raise ValueError(f"Unknown top_k_criterion: {top_k_criterion}")
            df = df.loc[:, cols]
        return df

    # ================================================================== #
    # Representative estimation instance  (UNIQUE to inverse-operator XAI)#
    # ================================================================== #
    def representative_instance(self, scaler=None, feature_names=None,
                                class_names=None):
        """
        Return the **representative estimation instance** of every class:
        ``A_dagger @ e_t`` for each one-hot class basis ``e_t``.

        This is the "ideal input" the model implicitly associates with a pure
        prediction of class ``t`` — a generative quantity that forward-problem
        explainers (LIME/SHAP) cannot produce.  If a ``scaler`` is supplied (or
        was fitted in ``create_explainer``) the instances are mapped back to the
        original feature space via ``inverse_transform``.

        Returns
        -------
        pandas.DataFrame  (class x feature)
        """
        if self.A_dagger is None:
            raise ValueError("Please create an explainer first using create_explainer.")
        dim = self.A_dagger.shape[1]
        repvec = np.array([np.dot(self.A_dagger, np.eye(dim)[t]) for t in range(dim)])
        use_scaler = scaler if scaler is not None else self.scaler
        rep_orig = use_scaler.inverse_transform(repvec) if use_scaler is not None else repvec
        if class_names is None:
            class_names = [f"class_{i}" for i in range(dim)]
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(self.A_dagger.shape[0])]
        return pd.DataFrame(rep_orig, index=class_names, columns=feature_names)

    def plot_representative_instance(self, scaler=None, feature_names=None,
                                     class_names=None, image_shape=None,
                                     cmap=None, ncols=None,
                                     save_path=None, show=True):
        """
        Visualise the representative estimation instances.

        * If ``image_shape`` is given (e.g. ``(28, 28)`` or ``(32, 32, 3)``)
          each class instance is reshaped and rendered as the model's
          **reconstructed "ideal input image"** — AIME's signature generative
          view.
        * Otherwise each class is drawn as a **feature fingerprint**: a signed
          profile of the representative instance across features.
        """
        rep = self.representative_instance(scaler=scaler,
                                           feature_names=feature_names,
                                           class_names=class_names)
        if image_shape is not None:
            return self._plot_rep_images(rep, image_shape, cmap, ncols,
                                         save_path, show)
        return self._plot_rep_fingerprint(rep, save_path, show)

    def plot_local_saliency(self, x, y, image_shape, scale=True, scaler=None,
                            ignore_zero_features=True, title=None,
                            save_path=None, show=True):
        """
        Image-model local explanation: overlay AIME's local importance
        ``(A_dagger @ y) ⊙ x'`` back onto the image grid.

        The original image is shown beside a **signed saliency** map (indigo =
        evidence pulling the pixel/feature down, coral = up).  For RGB inputs the
        per-pixel signed value is the dominant-channel signed magnitude.  This is
        the inverse-operator analogue of a saliency map — computed analytically,
        with no gradients or perturbations.

        Returns the signed saliency as a 2-D numpy array.
        """
        x, y = self._check_local_inputs(x, y)
        use_scaler = scaler if scaler is not None else self.scaler
        x_prime = use_scaler.transform([x])[0] if (scale and use_scaler is not None) else x
        local = np.dot(self.A_dagger, y) * x_prime
        if ignore_zero_features:
            local = local * (x != 0)

        is_rgb = (len(image_shape) == 3)
        local_img = local.reshape(image_shape)
        orig = x.reshape(image_shape)
        if is_rgb:
            absC = np.abs(local_img)
            dom = np.take_along_axis(local_img, absC.argmax(2)[..., None], 2)[..., 0]
            sal2d = np.sign(dom) * np.linalg.norm(local_img, axis=2)
            orig_disp = (orig - orig.min()) / (orig.max() - orig.min() + 1e-12)
        else:
            sal2d = local_img
            orig_disp = orig

        m = float(np.max(np.abs(sal2d))) or 1.0
        fig, axes = S.new_figure(figsize=(8.4, 4.6), ncols=2)
        axes[0].imshow(orig_disp, cmap=None if is_rgb else "gray")
        axes[0].set_title("input  x", fontsize=11, color=S.INK, pad=6)
        im = axes[1].imshow(sal2d, cmap=S.AIME_DIVERGING,
                            norm=S.signed_norm(sal2d))
        axes[1].set_title("AIME local saliency  (A†y) ⊙ x", fontsize=11,
                          color=S.INK, pad=6)
        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
        cb = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.03)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=8, color=S.INK_SOFT)
        S.style_title(axes[0], title or "Local Saliency (inverse operator)",
                      subtitle="analytic saliency — no gradients, no perturbations")
        S.add_brand(fig)
        fig.tight_layout(rect=(0, 0.04, 1, 0.99 if S.PUBLICATION else 0.88))
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=S.PAPER)
        _show_or_close(fig, show)
        return sal2d

    def _plot_rep_images(self, rep, image_shape, cmap, ncols, save_path, show):
        import matplotlib.pyplot as plt
        vals = rep.values
        n = vals.shape[0]
        ncols = ncols or min(n, 5)
        nrows = int(np.ceil(n / ncols))
        is_rgb = (len(image_shape) == 3)
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(2.35 * ncols, 2.7 * nrows + 1.2),
                                 facecolor=S.PAPER, squeeze=False)
        S.apply_aime_style()
        fig.subplots_adjust(hspace=0.32, wspace=0.12)
        cmap = cmap or S.AIME_SEQ
        for k in range(nrows * ncols):
            ax = axes[k // ncols][k % ncols]
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if k >= n:
                ax.set_visible(False)
                continue
            img = vals[k].reshape(image_shape)
            if is_rgb:
                mn, mx = img.min(), img.max()
                disp = (img - mn) / (mx - mn + 1e-12)
                ax.imshow(disp)
            else:
                ax.imshow(img, cmap=cmap)
            ax.set_title(str(rep.index[k]), fontsize=11, fontweight="bold",
                         color=S.INK, pad=6)
            # accent frame
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_color(S.GRIDLINE); sp.set_linewidth(1.2)
        if not S.PUBLICATION:
            fig.text(0.012, 0.975, "Representative Estimation Instances  ·  A† eₜ",
                     ha="left", va="top", fontsize=15.5, fontweight="bold",
                     color=S.INK)
            fig.text(0.012, 0.928, "the ideal input each class is reconstructed from — unique to inverse-operator XAI",
                     ha="left", va="top", fontsize=10, color=S.INK_SOFT)
        S.add_brand(fig)
        top = 0.99 if S.PUBLICATION else (0.90 - 0.02 * (nrows - 1))
        fig.tight_layout(rect=(0, 0.02, 1, top))
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=S.PAPER)
        _show_or_close(fig, show)
        return rep

    def _plot_rep_fingerprint(self, rep, save_path, show):
        feats = list(rep.columns)
        n_feat = len(feats)
        n_cls = rep.shape[0]
        # standardise each feature column across classes so the fingerprint is
        # comparable; this is presentation-only (does not alter returned data).
        Z = rep.values.astype(float)
        mu = Z.mean(axis=0, keepdims=True)
        sd = Z.std(axis=0, keepdims=True) + 1e-12
        Zs = (Z - mu) / sd

        angles = np.linspace(0, 2 * np.pi, n_feat, endpoint=False)
        fig, ax = S.new_figure(figsize=(8.4, 7.6),
                               subplot_kw=dict(projection="polar"))
        ax.set_facecolor(S.PANEL)
        ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
        ax.set_xticks(angles)
        ax.set_xticklabels(feats, fontsize=9, color=S.INK)
        ax.set_yticklabels([])
        ax.grid(color=S.GRIDLINE, lw=0.9)
        ax.spines["polar"].set_color(S.GRIDLINE)
        loop = np.concatenate([angles, angles[:1]])
        for i in range(n_cls):
            r = np.concatenate([Zs[i], Zs[i][:1]])
            c = S.CLASS_CYCLE[i % len(S.CLASS_CYCLE)]
            ax.plot(loop, r, color=c, lw=2.0, label=str(rep.index[i]))
            ax.fill(loop, r, color=c, alpha=0.12)
        if not S.PUBLICATION:
            ax.set_title("Representative Instance Fingerprints  ·  A† eₜ",
                         fontsize=15, fontweight="bold", color=S.INK, pad=26)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.02),
                  frameon=False, fontsize=9.5, title="class", title_fontsize=9.5)
        S.add_brand(fig)
        fig.tight_layout(rect=(0, 0.04, 1, 0.97))
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=S.PAPER)
        _show_or_close(fig, show)
        return rep

    # ================================================================== #
    # Inverse-operator flow diagram  (UNIQUE to AIME)                     #
    # ================================================================== #
    def plot_inverse_operator_flow(self, feature_names=None, class_names=None,
                                   top_k=None, threshold=0.0,
                                   save_path=None, show=True):
        """
        Render the approximate inverse operator ``A_dagger`` itself as a flow
        diagram: class (output) anchors on the right, feature (input) anchors on
        the left, connected by signed ribbons whose thickness is the magnitude
        of the operator weight and whose colour encodes direction (indigo = the
        feature is pushed *down* / negative; coral = pushed *up* / positive).

        The flow is drawn **right → left** to emphasise AIME's defining trait:
        reading explanation backwards, from output to input.  No other
        XAI method exposes an explicit operator that can be drawn this way.
        """
        df = self._global_matrix(feature_names, class_names, normalize_rows=False)
        # df: class x feature ; transpose to feature x class
        W = df.T.copy()
        if top_k is not None:
            keep = W.abs().max(axis=1).sort_values(ascending=False).index[:top_k]
            W = W.loc[keep]
        feats = list(W.index)
        classes = list(W.columns)
        n_feat, n_cls = len(feats), len(classes)

        vmax = float(np.max(np.abs(W.values))) or 1.0
        fig, ax = S.new_figure(figsize=(11.5, max(5.5, 0.46 * n_feat + 2.2)))
        ax.set_axis_off()
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

        fy = np.linspace(0.92, 0.08, n_feat)         # feature anchors (left)
        cy = np.linspace(0.86, 0.14, n_cls)          # class anchors (right)
        xf, xc = 0.30, 0.78

        # ribbons (draw weakest first so strong ones sit on top)
        order = np.argsort(np.abs(W.values).ravel())
        idx_pairs = [(o // n_cls, o % n_cls) for o in order]
        for i, j in idx_pairs:
            w = W.values[i, j]
            if abs(w) <= threshold:
                continue
            width = 0.004 + 0.05 * (abs(w) / vmax)
            color = S.signed_color(w, vmax)
            alpha = 0.30 + 0.55 * (abs(w) / vmax)
            S.ribbon(ax, xc, cy[j], xf, fy[i], width, color, alpha=alpha)

        # feature nodes
        for i, f in enumerate(feats):
            ax.scatter([xf], [fy[i]], s=70, color=S.INK, zorder=5,
                       edgecolors=S.PAPER, linewidths=1.2)
            S.node_label(ax, xf - 0.02, fy[i], str(f), align="right", size=9.5)
        # class nodes
        for j, c in enumerate(classes):
            col = S.CLASS_CYCLE[j % len(S.CLASS_CYCLE)]
            ax.scatter([xc], [cy[j]], s=150, color=col, zorder=5,
                       edgecolors=S.PAPER, linewidths=1.6)
            S.node_label(ax, xc + 0.02, cy[j], str(c), align="left",
                         size=10.5, weight="bold", color=col)

        # column captions + direction arrow
        ax.text(xf, 0.99, "INPUT  ·  features", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=S.INK_SOFT)
        ax.text(xc, 0.99, "OUTPUT  ·  classes", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=S.INK_SOFT)
        ax.annotate("", xy=(xf + 0.04, 0.035), xytext=(xc - 0.04, 0.035),
                    arrowprops=dict(arrowstyle="-|>", color=S.INDIGO, lw=2.2,
                                    alpha=0.8))
        ax.text((xf + xc) / 2, 0.012, "inverse explanation  (A† : output → input)",
                ha="center", va="bottom", fontsize=9.5, color=S.INDIGO,
                fontweight="bold")
        S.style_title(ax, "Inverse Operator Flow",
                      subtitle="A†  drawn as signed ribbons — coral pushes a feature up, indigo pulls it down")
        S.add_brand(fig)
        fig.tight_layout(rect=(0, 0.02, 1, 0.99 if S.PUBLICATION else 0.92))
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=S.PAPER)
        _show_or_close(fig, show)
        return df

    # ================================================================== #
    # Hadamard decomposition of a local explanation  (UNIQUE to AIME)     #
    # ================================================================== #
    def plot_local_hadamard_decomposition(self, x, y, feature_names=None,
                                          scale=True, scaler=None, top_k=None,
                                          ignore_zero_features=True,
                                          save_path=None, show=True):
        """
        Show *how* a local AIME explanation is built:
        ``contribution = (A_dagger @ y)  ⊙  x'``.

        Three aligned columns visualise the global pull ``A_dagger @ y``, the
        (scaled) instance ``x'``, and their Hadamard product.  This makes
        AIME's "no attribution to absent features" property visible: wherever
        ``x = 0`` the product collapses to zero — something additive
        forward-problem methods do not guarantee.
        """
        x, y = self._check_local_inputs(x, y)
        use_scaler = scaler if scaler is not None else self.scaler
        x_prime = use_scaler.transform([x])[0] if (scale and use_scaler is not None) else x

        pull = np.dot(self.A_dagger, y)          # (A_dagger @ y)
        prod = pull * x_prime                    # Hadamard product
        # mirror the canonical local rule: absent features (raw x == 0)
        # contribute exactly zero when ignore_zero_features is set.
        if ignore_zero_features:
            prod = prod * (x != 0)

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(x))]
        dfc = pd.DataFrame({"pull": pull, "x": x_prime, "prod": prod},
                           index=feature_names)
        dfc["raw"] = x
        order = dfc["prod"].abs().sort_values(ascending=False).index
        if top_k is not None:
            order = order[:top_k]
        dfc = dfc.loc[order[::-1]]
        feats = list(dfc.index)
        nf = len(feats)

        fig, axes = S.new_figure(figsize=(12.5, max(3.6, 0.44 * nf + 2)),
                                 ncols=3, sharey=True)
        cols = [("pull", "A† y   (global pull)"),
                ("x", "x′   (this instance)"),
                ("prod", "(A† y) ⊙ x′   (local importance)")]
        raw = dfc["raw"].values
        for ax, (key, sub) in zip(axes, cols):
            vals = dfc[key].values
            vmax = float(np.max(np.abs(vals))) or 1.0
            for i, v in enumerate(vals):
                absent = (key == "prod" and ignore_zero_features and raw[i] == 0)
                if absent:
                    # absent feature → exactly zero, shown as a muted marker
                    ax.text(0, i, "0", va="center", ha="center", fontsize=9.5,
                            fontweight="bold", color=S.INK_SOFT,
                            bbox=dict(boxstyle="round,pad=0.18", fc=S.PAPER,
                                      ec=S.GRIDLINE, lw=0.8))
                else:
                    S.gradient_hbar(ax, i, v, vmax=vmax, height=0.52)
            ax.axvline(0, color=S.INK, lw=1.0, alpha=0.5, zorder=1)
            ax.set_xlim(-vmax * 1.3, vmax * 1.3)
            ax.set_ylim(-0.7, nf - 0.3)
            ax.set_xlabel(sub, fontsize=10)
        axes[0].set_yticks(range(nf))
        axes[0].set_yticklabels(feats, fontsize=9.5)
        S.style_title(axes[0], "Local Explanation = Hadamard Decomposition",
                      subtitle="AIME multiplies the global pull by the instance — absent features stay exactly zero")
        S.add_brand(fig)
        fig.tight_layout(rect=(0, 0.04, 1, 0.985 if S.PUBLICATION else 0.90))
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=S.PAPER)
        _show_or_close(fig, show)
        return dfc[["pull", "x", "prod"]]

    # ================================================================== #
    # Representative instance similarity field  (signature redesign)      #
    # ================================================================== #
    def rbf_kernel(self, v1, v2, gamma):
        sq_dist = np.sum(v1 ** 2, 1).reshape(-1, 1) + np.sum(v2 ** 2, 1) - 2 * np.dot(v1, v2.T)
        return np.exp(-gamma * sq_dist)

    @staticmethod
    def _kde_grid(x, y, xr, yr, bw=None, gridsize=120):
        """Lightweight gaussian KDE evaluated on a grid (numpy only)."""
        x = np.asarray(x); y = np.asarray(y)
        n = len(x)
        if n == 0:
            return None
        if bw is None:
            sx = x.std() or 1.0
            sy = y.std() or 1.0
            bw = (n ** (-1.0 / 6.0))  # Scott-ish factor
            hx, hy = bw * sx, bw * sy
        else:
            hx = hy = bw
        hx = hx or 1e-3; hy = hy or 1e-3
        gx = np.linspace(*xr, gridsize)
        gy = np.linspace(*yr, gridsize)
        GX, GY = np.meshgrid(gx, gy)
        Z = np.zeros_like(GX)
        for xi, yi in zip(x, y):
            Z += np.exp(-0.5 * (((GX - xi) / hx) ** 2 + ((GY - yi) / hy) ** 2))
        Z /= (n * 2 * np.pi * hx * hy)
        return gx, gy, Z

    @staticmethod
    def _resolve_gamma(gamma, X_kernel):
        """Resolve a string gamma ('scale' / 'auto') to a numeric value on the
        space the kernel is evaluated in; pass numeric gamma through unchanged.

        Mirrors scikit-learn's RBF gamma conventions so the same setting behaves
        sensibly across feature spaces of very different dimensionality.
        """
        if not isinstance(gamma, str):
            return gamma
        nfeat = X_kernel.shape[1]
        if gamma == 'scale':
            var = float(np.asarray(X_kernel).var())
            return 1.0 / (nfeat * var) if var > 0 else 1.0 / nfeat
        if gamma == 'auto':
            return 1.0 / nfeat
        raise ValueError("gamma must be a float or one of {'scale', 'auto'}.")

    def plot_rep_instance_similarity(self, X, Y, x=None, feature_names=None,
                                     class_names=None, gamma=0.1, scaler=None,
                                     class_indices=[0, 1], dim_reduce=None,
                                     n_components=2, x_range=None, y_range=None,
                                     save_path=None, show=True):
        """
        Representative Instance Similarity Distribution Plot — redesigned as a
        layered **similarity field**.

        Every data point is scored by its RBF similarity to each class's
        representative instance (``A_dagger @ e_t``).  The joint distribution of
        two such similarity scores is shown as filled density contours per true
        class; the region where the classes' fields overlap is AIME's view of
        the model's "difficult / ambiguous" zone — a diagnostic LIME/SHAP do not
        offer.  Returns ``(repdf, resdf)`` exactly like the canonical method.
        """
        import matplotlib.pyplot as plt
        if self.A_dagger is None:
            raise ValueError("Please create an explainer first using create_explainer.")
        dim = self.A_dagger.shape[1]
        repvec = np.array([np.dot(self.A_dagger, np.eye(dim)[t]) for t in range(dim)])

        if class_names is None:
            vec_name = ['Class ' + str(i) + ' repvec' for i in range(dim)]
        else:
            vec_name = class_names

        repvec_in = scaler.inverse_transform(repvec) if scaler is not None else repvec
        if feature_names is None:
            repdf = pd.DataFrame(repvec_in, index=vec_name)
        else:
            repdf = pd.DataFrame(repvec_in, index=vec_name, columns=feature_names)

        X_scaled = scaler.transform(X) if scaler is not None else np.asarray(X)
        x_scaled = (scaler.transform([x]) if scaler else np.asarray(x).reshape(1, -1)) if x is not None else None

        # NOTE on gamma: the RBF similarity exp(-gamma*||.||^2) depends strongly
        # on the dimensionality of the space it is evaluated in.  A gamma that
        # works for a low-dim tabular problem (e.g. 0.1 for ~10 features) makes
        # every score underflow to ~0 in a high-dim space (e.g. 784 pixels),
        # collapsing the plot.  ``gamma`` therefore also accepts the sklearn-style
        # strings 'scale' (1 / (n_features * Var(X))) and 'auto' (1 / n_features),
        # resolved on the space actually used for the kernel.  See _resolve_gamma.

        # optional dimensionality reduction of the *feature* space before kernel
        if dim_reduce == 'pca':
            from sklearn.decomposition import PCA
            red = PCA(n_components=n_components)
            X_scaled = red.fit_transform(X_scaled)
            repvec = red.transform(repvec)
            if x is not None:
                x_scaled = red.transform(x_scaled)
        elif dim_reduce == 'umap':
            try:
                import umap.umap_ as umap
            except Exception:
                raise ImportError("umap-learn is required for dim_reduce='umap'.")
            red = umap.UMAP(n_components=n_components)
            X_scaled = red.fit_transform(X_scaled)
            repvec = red.transform(repvec)
            if x is not None:
                x_scaled = red.transform(x_scaled)
        elif dim_reduce == 'tsne':
            from sklearn.manifold import TSNE
            stack = [X_scaled, repvec] + ([x_scaled] if x is not None else [])
            total = np.concatenate(stack, axis=0)
            total = TSNE(n_components=n_components).fit_transform(total)
            X_scaled = total[:len(X_scaled)]
            repvec = total[len(X_scaled):len(X_scaled) + dim]
            if x is not None:
                x_scaled = total[-1].reshape(1, -1)

        gamma = self._resolve_gamma(gamma, X_scaled)
        res = self.rbf_kernel(X_scaled, repvec, gamma)
        if class_names is None:
            resdf = pd.DataFrame(res, columns=['score_' + str(i) for i in range(dim)])
        else:
            resdf = pd.DataFrame(res, columns=[class_names[i] + ' score' for i in range(dim)])
        resdf['result'] = np.argmax(Y, axis=1)

        ci, cj = class_indices[0], class_indices[1]
        cx, cy = resdf.columns[ci], resdf.columns[cj]

        fig, ax = S.new_figure(figsize=(9.4, 7.4))
        ax.grid(True, axis="both", color=S.GRIDLINE, lw=0.8, alpha=0.7)

        # View ranges are derived ONLY from the classes actually drawn (plus the
        # focus instance), not the full score matrix — otherwise the plotted
        # classes can collapse into a corner when other classes span a wider
        # range (e.g. with a large gamma that saturates the kernel).
        drawn = resdf[resdf['result'].isin(class_indices)]
        # warn if the similarity scores have collapsed (gamma mismatched to the
        # feature-space dimensionality) — the plot would otherwise look empty.
        sp = float(np.nanmax(drawn[[cx, cy]].values) - np.nanmin(drawn[[cx, cy]].values)) \
            if len(drawn) else 0.0
        if sp < 1e-4:
            import warnings
            warnings.warn(
                "Representative-similarity scores have nearly collapsed "
                f"(range={sp:.2e}); the RBF gamma={gamma!r} is likely mismatched "
                "to the feature dimensionality. Try gamma='scale' (recommended for "
                "high-dimensional inputs like images) or a much smaller gamma.",
                RuntimeWarning)
        xs = list(drawn[cx].values)
        ys = list(drawn[cy].values)
        if x is not None:
            fscore = self.rbf_kernel(x_scaled, repvec, gamma)
            xs.append(fscore[0][ci]); ys.append(fscore[0][cj])
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)

        def _range(vals, given):
            if given is not None:
                return given
            lo, hi = float(np.min(vals)), float(np.max(vals))
            span = hi - lo
            if span < 1e-6:                       # degenerate: expand around point
                lo, hi, span = lo - 0.05, hi + 0.05, 0.1
            return (lo - 0.08 * span, hi + 0.08 * span)

        xr = _range(xs, x_range)
        yr = _range(ys, y_range)

        cmaps = [S.AIME_SEQ, S.AIME_SEQ_WARM,
                 plt.cm.Greens, plt.cm.Purples, plt.cm.Oranges]
        for k, idx in enumerate(class_indices):
            sub = resdf[resdf['result'] == idx]
            col = S.CLASS_CYCLE[idx % len(S.CLASS_CYCLE)]
            grid = self._kde_grid(sub[cx].values, sub[cy].values, xr, yr)
            if grid is not None:
                gx, gy, Z = grid
                zmax = float(Z.max())
                if zmax > 0:
                    # threshold the lowest band so the panel is not flooded
                    levels = np.linspace(0.12 * zmax, zmax, 7)
                    cmap = cmaps[k % len(cmaps)]
                    ax.contourf(gx, gy, Z, levels=levels, cmap=cmap,
                                alpha=0.6, extend="neither", zorder=1)
                    ax.contour(gx, gy, Z, levels=levels[::2], colors=[col],
                               linewidths=1.0, alpha=0.75, zorder=3)
            ax.scatter(sub[cx], sub[cy], s=14, color=col, alpha=0.45,
                       edgecolors="white", linewidths=0.3, zorder=4,
                       label=str(resdf.columns[idx]).replace(" score", ""))

        # focus instance (fscore already computed above when x is not None)
        if x is not None:
            ax.scatter(fscore[0][ci], fscore[0][cj], s=240, marker="*",
                       color=S.AMBER, edgecolors=S.INK, linewidths=1.6,
                       zorder=6, label="focus instance")

        ax.set_xlim(xr); ax.set_ylim(yr)
        ax.set_xlabel(f"similarity to representative  ·  {str(cx).replace(' score','')}")
        ax.set_ylabel(f"similarity to representative  ·  {str(cy).replace(' score','')}")
        S.style_title(ax, "Representative Instance Similarity Field",
                      subtitle="overlap of the class fields = where the model finds the decision hard")
        # legend outside the plotting area so it never covers the density
        leg = ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
                        frameon=False, fontsize=9.5)
        S.add_brand(fig)
        fig.tight_layout(rect=(0, 0.04, 0.86, 0.985 if S.PUBLICATION else 0.92))
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=S.PAPER)
        _show_or_close(fig, show)
        return repdf, resdf

    # ================================================================== #
    # Interactive explorers (self-contained HTML, Colab-displayable)      #
    # ================================================================== #
    @staticmethod
    def _emit_html(html, path=None):
        """Write HTML to ``path`` (if given) and return a Colab/Jupyter-renderable
        object.  In a notebook, returning this object from a cell shows the
        interactive widget inline; outside a notebook it degrades to the raw
        HTML string."""
        if path is not None:
            with open(path, "w", encoding="utf-8") as f:
                f.write(html)
        # Return an IPython HTML object so the cell renders the widget inline
        # exactly ONCE (returning it as the cell's last expression).  We do NOT
        # call display() here — doing both would render the widget twice.
        try:
            from IPython.display import HTML
            return HTML(html)
        except Exception:
            return html

    def _scaler_mu_sd(self, n):
        """Return (mean, scale) of the fitted StandardScaler, or identity."""
        sc = self.scaler
        mu = np.asarray(getattr(sc, "mean_", np.zeros(n)), float)
        sd = np.asarray(getattr(sc, "scale_", np.ones(n)), float)
        if mu.shape[0] != n:
            mu = np.zeros(n)
        if sd.shape[0] != n:
            sd = np.ones(n)
        return mu, sd

    def export_interactive(self, path=None, feature_names=None, class_names=None,
                           top_k=None, title="AIME · Inverse Operator Explorer"):
        """
        Self-contained, dependency-free **interactive inverse-operator flow**.

        Renders ``A_dagger`` as inline SVG ribbons; hovering/clicking a class
        traces its ribbons back to the input features with live weight read-outs.
        Pass ``path`` to also save an .html file. Returns a Colab/Jupyter object
        that displays inline when it is the last expression in a cell.
        """
        df = self._global_matrix(feature_names, class_names, normalize_rows=False)
        W = df.T.copy()  # feature x class
        if top_k is not None:
            keep = W.abs().max(axis=1).sort_values(ascending=False).index[:top_k]
            W = W.loc[keep]
        feats = [str(f) for f in W.index]
        classes = [str(c) for c in W.columns]
        rep = self.representative_instance(class_names=class_names,
                                           feature_names=feature_names)
        import json
        payload = json.dumps({
            "feats": feats, "classes": classes, "W": W.values.tolist(),
            "rep": rep.values.tolist(), "repcols": [str(c) for c in rep.columns],
        })
        html = _INTERACTIVE_TEMPLATE.replace("__TITLE__", title).replace("__DATA__", payload)
        return self._emit_html(html, path)

    # convenient alias
    def interactive_operator_flow(self, path=None, **kw):
        """Alias of :meth:`export_interactive` (inline-displayable)."""
        return self.export_interactive(path=path, **kw)

    def interactive_reconstruction(self, path=None, feature_names=None,
                                   class_names=None, image_shape=None,
                                   title="AIME · Inverse Reconstruction Explorer"):
        """
        **Inverse Reconstruction Explorer** — the most distinctly-AIME interactive
        view, impossible for forward-problem methods (LIME/SHAP).

        The reader sets a desired output probability vector ``y`` with sliders;
        the page reconstructs, live, the input the model would need:

            x = scaler⁻¹( A_dagger · y ).

        For image models (`image_shape` given) the reconstruction is drawn on a
        canvas — drag the class sliders and watch the ideal image morph; the
        class buttons jump to each pure representative instance ``A_dagger eₜ``.
        For tabular models it is shown as a live feature-profile bar chart.

        Pass ``path`` to also save an .html file. Returns a Colab/Jupyter object
        that displays inline.
        """
        n, m = self.A_dagger.shape
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n)]
        if class_names is None:
            class_names = [f"class_{i}" for i in range(m)]
        mu, sd = self._scaler_mu_sd(n)
        import json
        payload = json.dumps({
            "A": self.A_dagger.tolist(),          # (n features, m classes)
            "mu": mu.tolist(), "sd": sd.tolist(),
            "feats": [str(f) for f in feature_names],
            "classes": [str(c) for c in class_names],
            "image_shape": list(image_shape) if image_shape is not None else None,
        })
        html = _RECONSTRUCT_TEMPLATE.replace("__TITLE__", title).replace("__DATA__", payload)
        return self._emit_html(html, path)


# --------------------------------------------------------------------------- #
# Standalone interactive HTML template (vanilla JS + inline SVG)               #
# --------------------------------------------------------------------------- #
_INTERACTIVE_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root{
    --ink:#1B2A4A; --soft:#5C6680; --paper:#FFFFFF; --panel:#FFFFFF;
    --grid:#E6E6E6; --indigo:#21307A; --coral:#D2552E;
  }
  *{box-sizing:border-box;}
  body{margin:0;background:var(--paper);color:var(--ink);
       font-family:"DejaVu Sans","Segoe UI",Helvetica,Arial,sans-serif;}
  header{padding:22px 28px 6px;}
  h1{font-size:22px;margin:0;font-weight:800;letter-spacing:.2px;}
  .sub{color:var(--soft);font-size:13px;margin-top:4px;}
  .rule{width:54px;height:4px;border-radius:3px;background:var(--coral);margin:10px 0 0;}
  .wrap{display:flex;gap:18px;padding:14px 28px 30px;flex-wrap:wrap;}
  .card{background:var(--panel);border:1px solid var(--grid);border-radius:14px;
        box-shadow:0 8px 26px rgba(27,42,74,.07);padding:14px 16px;}
  svg{display:block;}
  .feat-label{font-size:12px;fill:var(--ink);}
  .cls-label{font-size:13px;font-weight:700;}
  .cap{font-size:12px;font-weight:700;fill:var(--soft);letter-spacing:.4px;}
  .panel-side{min-width:240px;max-width:300px;flex:1;}
  .panel-side h2{font-size:14px;margin:.2em 0 .6em;}
  table{width:100%;border-collapse:collapse;font-size:12.5px;}
  td{padding:4px 6px;border-bottom:1px solid var(--grid);}
  td.val{text-align:right;font-variant-numeric:tabular-nums;font-weight:700;}
  .chip{display:inline-block;width:11px;height:11px;border-radius:3px;
        margin-right:7px;vertical-align:middle;}
  .hint{color:var(--soft);font-size:12px;margin-top:10px;line-height:1.5;}
  .brand{text-align:right;padding:4px 28px 14px;font-size:12px;font-weight:800;
          color:var(--indigo);opacity:.85;}
  .brand small{font-weight:400;color:var(--soft);font-size:9px;margin-left:8px;}
  .arrow{fill:none;stroke:var(--indigo);stroke-width:2.2;}
</style></head>
<body>
<header>
  <h1>__TITLE__</h1>
  <div class="sub">Hover or click a class node — ribbons trace the approximate inverse operator A&dagger; back to the input features.</div>
  <div class="rule"></div>
</header>
<div class="wrap">
  <div class="card"><div id="flow"></div></div>
  <div class="panel-side card">
    <h2 id="sel-title">Operator weights</h2>
    <div id="readout"></div>
    <div class="hint">Coral ribbons push a feature up for the selected class; indigo pulls it down. Width &amp; opacity scale with |weight|. This is the operator A&dagger; itself — a view unique to inverse-model XAI.</div>
  </div>
</div>
<div class="brand">AIME · A&dagger;<small>Approximate Inverse Model Explanations</small></div>
<script>
const D = __DATA__;
const feats=D.feats, classes=D.classes, W=D.W;
const nF=feats.length, nC=classes.length;
const CLS=["#21307A","#D2552E","#2E7E8C","#E0A23B","#7A4FA0","#4F66C2","#C2362B","#3F8E6E","#B0762A","#9A4C6B"];
let vmax=0; for(const r of W) for(const v of r) vmax=Math.max(vmax,Math.abs(v));
vmax=vmax||1;
function lerp(a,b,t){return a+(b-a)*t;}
function hex(c){return c;}
function signedColor(v){ // indigo (neg) -> paper -> coral (pos)
  const t=Math.max(-1,Math.min(1,v/vmax));
  if(t<0){const k=-t;return `rgb(${Math.round(lerp(251,33,k))},${Math.round(lerp(249,48,k))},${Math.round(lerp(244,122,k))})`;}
  const k=t;return `rgb(${Math.round(lerp(251,210,k))},${Math.round(lerp(249,85,k))},${Math.round(lerp(244,46,k))})`;
}
const Wd=Math.min(760, Math.max(560, 36*nF)), Hd=Math.max(360, 30*nF+60);
const mT=46,mB=30, xf=Wd*0.34, xc=Wd*0.74;
const fy=i=>lerp(mT, Hd-mB, nF<=1?0.5:i/(nF-1));
const cy=j=>lerp(mT+14, Hd-mB-14, nC<=1?0.5:j/(nC-1));
function ribbonPath(x0,y0,x1,y1,w){
  const mx=(x0+x1)/2;
  return `M ${x0} ${y0+w} C ${mx} ${y0+w}, ${mx} ${y1+w}, ${x1} ${y1+w}`
       + ` L ${x1} ${y1-w} C ${mx} ${y1-w}, ${mx} ${y0-w}, ${x0} ${y0-w} Z`;
}
let svg=`<svg id="s" width="${Wd}" height="${Hd}" viewBox="0 0 ${Wd} ${Hd}">`;
svg+=`<text x="${xf}" y="22" text-anchor="middle" class="cap">INPUT · FEATURES</text>`;
svg+=`<text x="${xc}" y="22" text-anchor="middle" class="cap">OUTPUT · CLASSES</text>`;
// ribbons
let order=[];
for(let i=0;i<nF;i++)for(let j=0;j<nC;j++)order.push([i,j]);
order.sort((a,b)=>Math.abs(W[a[0]][a[1]])-Math.abs(W[b[0]][b[1]]));
for(const [i,j] of order){
  const w=W[i][j], aw=Math.abs(w)/vmax;
  const width=1.2+9*aw, op=0.18+0.55*aw;
  svg+=`<path class="rib rib-c${j}" data-c="${j}" data-f="${i}" d="${ribbonPath(xc,cy(j),xf,fy(i),width)}" fill="${signedColor(w)}" opacity="${op}"></path>`;
}
// feature nodes
for(let i=0;i<nF;i++){
  svg+=`<circle cx="${xf}" cy="${fy(i)}" r="4.5" fill="#1B2A4A" stroke="#FBF9F4" stroke-width="1.4"></circle>`;
  svg+=`<text class="feat-label" x="${xf-10}" y="${fy(i)+4}" text-anchor="end">${feats[i]}</text>`;
}
// class nodes
for(let j=0;j<nC;j++){
  svg+=`<circle class="cnode" data-c="${j}" cx="${xc}" cy="${cy(j)}" r="9" fill="${CLS[j%CLS.length]}" stroke="#FBF9F4" stroke-width="1.8" style="cursor:pointer"></circle>`;
  svg+=`<text class="cls-label" x="${xc+14}" y="${cy(j)+4}" fill="${CLS[j%CLS.length]}" style="cursor:pointer" data-c="${j}">${classes[j]}</text>`;
}
// direction arrow
svg+=`<path class="arrow" d="M ${xf+18} ${Hd-8} L ${xc-18} ${Hd-8}" marker-end="url(#ar)"></path>`;
svg+=`<defs><marker id="ar" markerWidth="10" markerHeight="10" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#21307A"/></marker></defs>`;
svg+=`<text x="${(xf+xc)/2}" y="${Hd-12}" text-anchor="middle" fill="#21307A" font-size="11" font-weight="700">inverse explanation · A&dagger; : output → input</text>`;
svg+=`</svg>`;
document.getElementById("flow").innerHTML=svg;

function select(j){
  document.querySelectorAll(".rib").forEach(p=>{
    const on = (+p.dataset.c===j);
    p.style.opacity = on ? (0.30+0.6*(Math.abs(W[+p.dataset.f][+p.dataset.c])/vmax)) : 0.05;
  });
  const col=CLS[j%CLS.length];
  document.getElementById("sel-title").innerHTML=`<span class="chip" style="background:${col}"></span>${classes[j]} — operator weights`;
  let rows=feats.map((f,i)=>({f:f,v:W[i][j]})).sort((a,b)=>Math.abs(b.v)-Math.abs(a.v));
  let html='<table>';
  for(const r of rows){html+=`<tr><td>${r.f}</td><td class="val" style="color:${signedColor(r.v)}">${r.v>=0?'+':''}${r.v.toFixed(3)}</td></tr>`;}
  html+='</table>';
  document.getElementById("readout").innerHTML=html;
}
document.querySelectorAll(".cnode,.cls-label").forEach(el=>{
  el.addEventListener("mouseenter",()=>select(+el.dataset.c));
  el.addEventListener("click",()=>select(+el.dataset.c));
});
select(0);
</script>
</body></html>
"""


# --------------------------------------------------------------------------- #
# Inverse Reconstruction Explorer — set output y, reconstruct input x          #
#   x = scaler^{-1}( A_dagger · y ).  Unique to inverse-operator XAI.          #
# --------------------------------------------------------------------------- #
_RECONSTRUCT_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root{--ink:#1B2A4A;--soft:#5C6680;--paper:#FFFFFF;--panel:#FFFFFF;
        --grid:#E6E6E6;--indigo:#21307A;--coral:#D2552E;--amber:#E0A23B;}
  *{box-sizing:border-box;}
  body{margin:0;background:var(--paper);color:var(--ink);
       font-family:"DejaVu Sans","Segoe UI",Helvetica,Arial,sans-serif;}
  header{padding:20px 26px 4px;}
  h1{font-size:21px;margin:0;font-weight:800;}
  .sub{color:var(--soft);font-size:13px;margin-top:4px;max-width:760px;line-height:1.5;}
  .rule{width:52px;height:4px;border-radius:3px;background:var(--coral);margin:10px 0 0;}
  .wrap{display:flex;gap:20px;padding:14px 26px 28px;flex-wrap:wrap;align-items:flex-start;}
  .card{background:var(--panel);border:1px solid var(--grid);border-radius:14px;
        box-shadow:0 8px 26px rgba(27,42,74,.07);padding:16px 18px;}
  .controls{min-width:300px;max-width:360px;flex:1;}
  .controls h2{font-size:14px;margin:.1em 0 .8em;}
  .row{display:flex;align-items:center;gap:10px;margin:7px 0;}
  .row label{width:88px;font-size:12.5px;font-weight:700;}
  .row input[type=range]{flex:1;accent-color:var(--indigo);}
  .row .v{width:42px;text-align:right;font-size:12px;font-variant-numeric:tabular-nums;color:var(--soft);}
  .chip{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:6px;vertical-align:middle;}
  .btns{margin:12px 0 4px;display:flex;flex-wrap:wrap;gap:6px;}
  .btns button{font:inherit;font-size:12px;font-weight:700;border:1px solid var(--grid);
        background:#fff;color:var(--ink);border-radius:8px;padding:5px 10px;cursor:pointer;}
  .btns button:hover{border-color:var(--indigo);color:var(--indigo);}
  .stage{flex:2;min-width:320px;text-align:center;}
  canvas{image-rendering:pixelated;border:1px solid var(--grid);border-radius:10px;background:#fff;}
  .hint{color:var(--soft);font-size:12px;margin-top:12px;line-height:1.5;}
  .brand{text-align:right;padding:4px 26px 14px;font-size:12px;font-weight:800;color:var(--indigo);opacity:.85;}
  .brand small{font-weight:400;color:var(--soft);font-size:9px;margin-left:8px;}
  svg text{font-family:inherit;}
</style></head>
<body>
<header>
  <h1>__TITLE__</h1>
  <div class="sub">Set a target output <b>y</b> with the sliders — the page reconstructs the input the model would need via the inverse operator <b>x = scaler⁻¹(A&dagger;·y)</b>. This generative, output→input direction is unique to AIME.</div>
  <div class="rule"></div>
</header>
<div class="wrap">
  <div class="card controls">
    <h2>Target output  y</h2>
    <div id="sliders"></div>
    <div class="btns" id="presets"></div>
    <div class="hint">Each preset sets y to a pure class (a one-hot eₜ), reconstructing that class's <b>representative instance</b> A&dagger;eₜ. Blend classes with the sliders to morph between them.</div>
  </div>
  <div class="card stage">
    <div id="stage"></div>
  </div>
</div>
<div class="brand">AIME · A&dagger;<small>Approximate Inverse Model Explanations</small></div>
<script>
const D=__DATA__;
const A=D.A, mu=D.mu, sd=D.sd, feats=D.feats, classes=D.classes, IMG=D.image_shape;
const n=A.length, m=A[0].length;
const CLS=["#21307A","#D2552E","#2E7E8C","#E0A23B","#7A4FA0","#4F66C2","#C2362B","#3F8E6E","#B0762A","#9A4C6B"];
let y=new Array(m).fill(0); y[0]=1;

function reconstructScaled(){
  // standardized reconstruction  z = A · y  (operator output, before un-scaling)
  const z=new Array(n).fill(0);
  for(let i=0;i<n;i++){let s=0; for(let t=0;t<m;t++) s+=A[i][t]*y[t]; z[i]=s;}
  return z;
}
function reconstruct(){
  // original-feature-space reconstruction  x = scaler^{-1}(A · y) = (A·y)*sd + mu
  const z=reconstructScaled(); const x=new Array(n).fill(0);
  for(let i=0;i<n;i++) x[i]=z[i]*sd[i]+mu[i];
  return x;
}
// STABLE feature order for the tabular view (computed once from operator
// influence) so bars never reorder while sliders move.
let TAB_ORDER=null, TAB_VMAX_DOMAIN=1;
if(!IMG){
  const infl=feats.map((_,i)=>{let s=0; for(let t=0;t<m;t++) s+=Math.abs(A[i][t]); return s;});
  TAB_ORDER=feats.map((_,i)=>i).sort((a,b)=>infl[b]-infl[a]).slice(0,Math.min(n,16));
}
// ---- sliders ----
let sh='';
for(let t=0;t<m;t++){
  sh+=`<div class="row"><label><span class="chip" style="background:${CLS[t%CLS.length]}"></span>${classes[t]}</label>`
    +`<input type="range" min="0" max="1" step="0.01" value="${y[t]}" data-t="${t}">`
    +`<span class="v" id="v${t}">${y[t].toFixed(2)}</span></div>`;
}
document.getElementById('sliders').innerHTML=sh;
document.querySelectorAll('#sliders input').forEach(inp=>{
  inp.addEventListener('input',()=>{y[+inp.dataset.t]=+inp.value;
    document.getElementById('v'+inp.dataset.t).textContent=(+inp.value).toFixed(2);draw();});
});
// ---- presets ----
let pb='';
for(let t=0;t<m;t++) pb+=`<button data-t="${t}">${classes[t]}</button>`;
document.getElementById('presets').innerHTML=pb;
document.querySelectorAll('#presets button').forEach(b=>{
  b.addEventListener('click',()=>{y=new Array(m).fill(0); y[+b.dataset.t]=1;
    document.querySelectorAll('#sliders input').forEach(inp=>{inp.value=y[+inp.dataset.t];
      document.getElementById('v'+inp.dataset.t).textContent=y[+inp.dataset.t].toFixed(2);});draw();});
});
// ---- drawing ----
function colorFor(v,vmax){const t=Math.max(-1,Math.min(1,v/vmax));
  if(t<0){const k=-t;return `rgb(${Math.round(251-218*k)},${Math.round(249-201*k)},${Math.round(244-122*k)})`;}
  const k=t;return `rgb(${Math.round(251-41*k)},${Math.round(249-164*k)},${Math.round(244-198*k)})`;}
function draw(){
  const stage=document.getElementById('stage');
  if(IMG){
    const x=reconstruct();   // image: original pixel space
    const H=IMG[0],Wd=IMG[1],ch=IMG.length>2?IMG[2]:1;
    let mn=Infinity,mx=-Infinity; for(const v of x){if(v<mn)mn=v; if(v>mx)mx=v;}
    const sc=(mx-mn)||1; const px=11;
    if(!stage._c){stage.innerHTML='<canvas id="cv"></canvas>';}
    const cv=document.getElementById('cv'); cv.width=Wd*px; cv.height=H*px; stage._c=cv;
    const ctx=cv.getContext('2d');
    for(let r=0;r<H;r++)for(let c=0;c<Wd;c++){
      let R,G,B;
      if(ch===1){const v=(x[r*Wd+c]-mn)/sc*255; R=G=B=v;}
      else{const o=(r*Wd+c)*ch;
        R=(x[o]-mn)/sc*255; G=(x[o+1]-mn)/sc*255; B=(x[o+2]-mn)/sc*255;}
      ctx.fillStyle=`rgb(${R|0},${G|0},${B|0})`; ctx.fillRect(c*px,r*px,px,px);
    }
  } else {
    // tabular: STANDARDIZED reconstruction z = A·y so EVERY feature (continuous
    // and one-hot) is on a comparable scale and visibly responds to y.  In the
    // original feature space the means/scales differ by orders of magnitude
    // (e.g. Fare vs a one-hot flag), which makes most bars look frozen.
    const x=reconstructScaled();
    const idx=TAB_ORDER; const k=idx.length;
    let vmax=0; for(const i of idx) vmax=Math.max(vmax,Math.abs(x[i])); vmax=vmax||1;
    const W=440, rowH=26, L=120, H=k*rowH+24, mid=L+(W-L-20)/2, half=(W-L-30)/2;
    let s=`<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">`;
    s+=`<line x1="${mid}" y1="6" x2="${mid}" y2="${H-18}" stroke="#1B2A4A" stroke-width="1" opacity="0.5"/>`;
    idx.forEach((i,r)=>{const yy=14+r*rowH; const w=Math.abs(x[i])/vmax*half;
      const x0=x[i]>=0?mid:mid-w;
      s+=`<rect x="${x0}" y="${yy}" width="${Math.max(w,1)}" height="${rowH-10}" rx="3" fill="${colorFor(x[i],vmax)}" stroke="#fff"/>`;
      s+=`<text x="${L-8}" y="${yy+rowH-14}" text-anchor="end" font-size="12" fill="#1B2A4A">${feats[i]}</text>`;
      s+=`<text x="${x[i]>=0?mid+w+4:mid-w-4}" y="${yy+rowH-14}" text-anchor="${x[i]>=0?'start':'end'}" font-size="11" font-weight="700" fill="#1B2A4A">${x[i].toFixed(2)}</text>`;
    });
    s+=`<text x="${mid}" y="${H-4}" text-anchor="middle" font-size="11" fill="#5C6680">standardized reconstruction  z = A&dagger;·y  (per-feature σ units)</text>`;
    s+='</svg>'; stage.innerHTML=s;
  }
}
draw();
</script>
</body></html>
"""

# core.py
# ---------------------------------------------------------------------
# Unified implementation of:
#   • AIME
#   • HuberAIME
#   • RidgeAIME
#   • Huber-RidgeAIME
#   • BayesianAIME (NEW)
# Switch behaviour with flags:  use_huber / use_ridge / use_bayesian
# ---------------------------------------------------------------------

from __future__ import annotations
from typing import Optional, Sequence, Tuple, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns


# ------------------------------- #
#            AIME class           #
# ------------------------------- #
class AIME:
    """
    Approximate Inverse Model Explanations (AIME) family.

    Parameters
    ----------
    use_huber : bool, default False
        If True use Huber loss (outlier-robust).
    use_ridge : bool, default False
        If True add ℓ2 (Ridge) regularisation.
    use_bayesian : bool, default False
        If True, use Bayesian estimation to compute credible intervals.
    delta : float, default 1.0
        Huber threshold (ignored when use_huber=False).
    ridge_alpha : float, default 1e-4
        Ridge coefficient λ (ignored when use_ridge=False).
    bayesian_sigma : float, default 1.0
        Likelihood noise parameter (σ) for Bayesian AIME.
    bayesian_tau : float, default 1.0
        Prior variance parameter (τ) for Bayesian AIME.
    max_iter : int, default 50
        Maximum IRLS iterations for Huber solver.
    tol : float, default 1e-5
        Frobenius-norm tolerance for IRLS convergence.
    """

    # ----------------------------------------------------------------- #
    def __init__(
        self,
        *,
        use_huber: bool = False,
        use_ridge: bool = False,
        use_bayesian: bool = False,
        delta: float = 1.0,
        ridge_alpha: float = 1e-4,
        bayesian_sigma: float = 1.0,
        bayesian_tau: float = 1.0,
        max_iter: int = 50,
        tol: float = 1e-5,
    ):
        # --- Mode flags
        self.use_huber = bool(use_huber)
        self.use_ridge = bool(use_ridge)
        self.use_bayesian = bool(use_bayesian)
        
        # --- Hyperparameters
        self.delta = float(delta)
        self.ridge_alpha = float(ridge_alpha)
        self.bayesian_sigma = float(bayesian_sigma)
        self.bayesian_tau = float(bayesian_tau)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

        # --- Fitter results
        self.A_dagger: Optional[np.ndarray] = None
        self.A_dagger_cov: Optional[np.ndarray] = None # For Bayesian mode
        self.scaler: Optional[StandardScaler] = None

        if self.use_bayesian and (self.use_huber or self.use_ridge):
            raise ValueError("Bayesian mode cannot be combined with Huber or Ridge modes.")


    # ----------------------------------------------------------------- #
    #                         Fitting interface                         #
    # ----------------------------------------------------------------- #
    def create_explainer(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        *,
        normalize: bool = True,
    ) -> "AIME":
        """
        Fit the approximate inverse operator from (X, Y).

        X : shape (N, d)
        Y : shape (N, m)
        """
        if X is None or Y is None:
            raise ValueError("Both X and Y must be provided.")
        if len(X) != len(Y):
            raise ValueError("X and Y sample sizes differ.")

        # --- optional scaling of X
        if normalize:
            self.scaler = StandardScaler().fit(X)
            X_proc = self.scaler.transform(X)
        else:
            self.scaler = None
            X_proc = np.asarray(X, dtype=float)

        Y_proc = np.asarray(Y, dtype=float)

        # --- choose solver -------------------------------------------------------
        if self.use_bayesian:
            self.A_dagger, self.A_dagger_cov = self._bayesian_solver(X_proc, Y_proc)
        
        else: # Original AIME family solvers
            X_t: np.ndarray = X_proc.T
            Y_t: np.ndarray = Y_proc.T
            
            if not self.use_huber and not self.use_ridge:
                self.A_dagger = self._pseudo_inverse(X_t, Y_t)
            elif self.use_huber and not self.use_ridge:
                self.A_dagger = self._huber_solver(X_t, Y_t, ridge=False)
            elif not self.use_huber and self.use_ridge:
                self.A_dagger = self._ridge_solver(X_t, Y_t)
            else:  # both True  →  Huber-RidgeAIME
                self.A_dagger = self._huber_solver(X_t, Y_t, ridge=True)

        return self

    # ----------------------------------------------------------------- #
    #                       Bayesian AIME solver                        #
    # ----------------------------------------------------------------- #
    def _bayesian_solver(
        self, 
        X_proc: np.ndarray, 
        Y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Bayesian linear regression solver for AIME.
        Estimates the posterior mean and covariance of the operator.
        Model: X = Y @ A_dagger.T + E
        """
        N, d = X_proc.shape
        m = Y.shape[1]
        
        s2_inv = 1 / (self.bayesian_sigma ** 2)
        t2_inv = 1 / (self.bayesian_tau ** 2)

        # Use standard AIME solution as the prior mean
        A_dagger_prior_mean = self._pseudo_inverse(X_proc.T, Y.T)

        # Design matrix is Y (Φ in the document)
        phi_t_phi = Y.T @ Y
        
        # Posterior covariance (shared across features)
        # Σ_post = (1/σ² * ΦᵀΦ + 1/τ² * I)⁻¹
        precision_matrix = s2_inv * phi_t_phi + t2_inv * np.eye(m)
        try:
            cov_post = np.linalg.inv(precision_matrix)
        except np.linalg.LinAlgError:
            cov_post = np.linalg.pinv(precision_matrix)

        # Posterior mean
        # μ_post = Σ_post @ (1/σ² * ΦᵀX + 1/τ² * β̂_prior.T)
        phi_t_X = Y.T @ X_proc
        mean_post_T = cov_post @ (s2_inv * phi_t_X + t2_inv * A_dagger_prior_mean.T)
        
        mean_post = mean_post_T.T

        return mean_post, cov_post

    # ----------------------------------------------------------------- #
    #                        Closed-form solvers                        #
    # ----------------------------------------------------------------- #
    @staticmethod
    def _pseudo_inverse(X_t: np.ndarray, Y_t: np.ndarray) -> np.ndarray:
        """Standard AIME:  M = X Y⁺"""
        return X_t @ np.linalg.pinv(Y_t)

    def _ridge_solver(self, X_t: np.ndarray, Y_t: np.ndarray) -> np.ndarray:
        """Ridge regularised closed-form: M = X Yᵀ (Y Yᵀ + λI)⁻¹"""
        m = Y_t.shape[0]
        try:
            inv_part = np.linalg.inv(Y_t @ Y_t.T + self.ridge_alpha * np.eye(m))
        except np.linalg.LinAlgError:
            inv_part = np.linalg.pinv(Y_t @ Y_t.T + self.ridge_alpha * np.eye(m))
        return X_t @ (Y_t.T @ inv_part)

    # ----------------------------------------------------------------- #
    #                         Huber (IRLS) solver                       #
    # ----------------------------------------------------------------- #
    def _huber_solver(
        self, X_t: np.ndarray, Y_t: np.ndarray, *, ridge: bool
    ) -> np.ndarray:
        """IRLS for Huber (optionally with Ridge λI)."""
        n, N = X_t.shape
        m = Y_t.shape[0]
        λI = self.ridge_alpha * np.eye(m) if ridge else 0.0

        try:
            inv_init = np.linalg.inv(Y_t @ Y_t.T + λI)
        except np.linalg.LinAlgError:
            inv_init = np.linalg.pinv(Y_t @ Y_t.T + λI)
        M = X_t @ (Y_t.T @ inv_init)

        δ = self.delta
        for _ in range(self.max_iter):
            R = X_t - M @ Y_t
            r_norm = np.linalg.norm(R, axis=0)
            w = np.where(r_norm > δ, δ / r_norm, 1.0)
            sqrt_w = np.sqrt(w)

            X_w = X_t * sqrt_w
            Y_w = Y_t * sqrt_w

            try:
                inv_part = np.linalg.inv(Y_w @ Y_w.T + λI)
            except np.linalg.LinAlgError:
                inv_part = np.linalg.pinv(Y_w @ Y_w.T + λI)

            M_new = X_w @ (Y_w.T @ inv_part)
            if np.linalg.norm(M_new - M, ord="fro") < self.tol:
                break
            M = M_new
        return M

    # ----------------------------------------------------------------- #
    #                    Global Feature Importance (GFI)                #
    # ----------------------------------------------------------------- #
    def _norm_gfi_matrix(self) -> pd.DataFrame:
        if self.A_dagger is None:
            raise ValueError("Run create_explainer first.")
        d, m = self.A_dagger.shape
        rows = []
        for j in range(m):
            vec = self.A_dagger[:, j]
            max_abs = np.max(np.abs(vec))
            rows.append(vec / max_abs if max_abs else vec)
        return pd.DataFrame(rows)

    def global_feature_importance(
        self,
        feature_names: Optional[Sequence[str]] = None,
        class_names: Optional[Sequence[str]] = None,
        *,
        top_k: Optional[int] = None,
        top_k_criterion: str = "average",
        with_plot: bool = True,
    ) -> pd.DataFrame:
        """
        Return (and optionally plot) the normalised GFI matrix.
        If Bayesian, returns a multi-level DataFrame with mean and credible intervals.
        """
        # 1. Calculate mean importance and set names
        df_mean = self._norm_gfi_matrix()
        d, m = self.A_dagger.shape
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(d)]
        if class_names is None:
            class_names = [f"class_{j}" for j in range(m)]
        df_mean.columns = feature_names
        df_mean.index = class_names

        # This will be the final DataFrame to return
        result_df = df_mean.copy()
        
        # For plotting, we need a separate error df
        df_err_for_plot = None

        # 2. If Bayesian, calculate intervals and create multi-level df
        if self.use_bayesian:
            if self.A_dagger_cov is None:
                raise ValueError("Bayesian explainer has not been created.")
            
            # Calculate standard error for each class
            gfi_std_per_class = np.sqrt(np.diag(self.A_dagger_cov))
            
            # Normalize the error by the same factor as the mean
            norm_factors = np.array([np.max(np.abs(self.A_dagger[:, j])) for j in range(m)])
            norm_factors[norm_factors == 0] = 1.0
            gfi_std_err_normalized = gfi_std_per_class / norm_factors
            
            # Create a DataFrame for standard errors (matching df_mean's shape)
            df_err = pd.DataFrame(
                [gfi_std_err_normalized] * d, 
                columns=class_names, 
                index=feature_names
            ).T
            
            df_err_for_plot = df_err.copy() # Use this for plotting

            # Calculate CI bounds
            margin = 1.96 * df_err
            df_lower = df_mean - margin
            df_upper = df_mean + margin

            # Combine into a single multi-level column DataFrame
            result_df = pd.concat(
                {'mean': df_mean, 'lower_bound': df_lower, 'upper_bound': df_upper},
                axis=1
            )
            # Reorder columns to group by feature: (F_0, mean), (F_0, lower), (F_0, upper), ...
            result_df = result_df.swaplevel(0, 1, axis=1).sort_index(axis=1)

        # 3. Handle top_k filtering
        if top_k:
            if top_k_criterion == 'average':
                keep = df_mean.mean(0).nlargest(top_k).index
            elif top_k_criterion == 'max':
                keep = df_mean.abs().max(0).nlargest(top_k).index
            else:
                raise ValueError("top_k_criterion must be 'average' or 'max'.")
            
            # Filter the final df and the dataframes for plotting
            if self.use_bayesian:
                result_df = result_df.loc[:, (keep, slice(None))]
            else:
                result_df = result_df.loc[:, keep]
            
            df_mean = df_mean.loc[:, keep]
            if df_err_for_plot is not None:
                df_err_for_plot = df_err_for_plot.loc[:, keep]

        # 4. Plot if requested (using the simple mean and error dfs)
        if with_plot:
            self._plot_gfi(df_mean, df_err_for_plot)
        
        # 5. Return the final result
        return result_df


    def _plot_gfi(self, df: pd.DataFrame, df_err: Optional[pd.DataFrame] = None) -> None:
        # --- Bar plot with optional error bars
        df_m = (
            df.reset_index()
            .melt(id_vars="index", var_name="feature", value_name="importance")
            .rename(columns={"index": "class"})
        )
        
        err_series = None
        if df_err is not None:
            df_err_m = (
                df_err.reset_index()
                .melt(id_vars="index", var_name="feature", value_name="std_err")
            )
            df_m["err"] = 1.96 * df_err_m["std_err"] # 95% CI
            
        plt.figure(figsize=(10, max(6, len(df.columns) * 0.5)))
        
        # Draw bars
        ax = sns.barplot(
            data=df_m, x="importance", y="feature", hue="class",
            dodge=True, palette="pastel"
        )
        
        # Add error bars if available
        if 'err' in df_m.columns:
            # We need to manually add error bars as sns.barplot xerr is tricky with hue
            y_locs = {label.get_text(): i for i, label in enumerate(ax.get_yticklabels())}
            
            hue_order = sorted(df_m['class'].unique())
            num_classes = len(hue_order)
            
            # Safely get bar properties
            if ax.patches:
                bar_height = ax.patches[0].get_height()
                dodge_val = bar_height * num_classes / 2 - bar_height / 2
            else: # Fallback if no bars are drawn
                bar_height = 0.8 / num_classes if num_classes > 0 else 0.8
                dodge_val = 0.4 - bar_height / 2

            sorted_features = [label.get_text() for label in ax.get_yticklabels()]
            
            for i, feature in enumerate(sorted_features):
                for j, class_name in enumerate(hue_order):
                    val = df_m[(df_m['feature'] == feature) & (df_m['class'] == class_name)]
                    if not val.empty:
                        imp = val['importance'].iloc[0]
                        err = val['err'].iloc[0]
                        # Position of the bar center
                        y_pos = y_locs[feature] - dodge_val + j * bar_height
                        plt.errorbar(x=[imp], y=[y_pos], xerr=[err],
                                     fmt='none', c='black', capsize=3)

        plt.title("Global Feature Importance" + (" (with 95% Credible Intervals)" if df_err is not None else ""))
        plt.axvline(0, color='grey', linestyle='--')
        plt.tight_layout()
        plt.show()

        # --- Heatmap (no change)
        plt.figure(figsize=(10, 6))
        sns.heatmap(df, cmap="Blues", annot=True, fmt=".2f")
        plt.title("Global Feature Importance (heatmap)")
        plt.tight_layout()
        plt.show()


    # ----------------------------------------------------------------- #
    #                    Local Feature Importance (LFI)                 #
    # ----------------------------------------------------------------- #
    def local_feature_importance(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        feature_names: Optional[Sequence[str]] = None,
        scale: bool = True,
        top_k: Optional[int] = None,
        ignore_zero_features: bool = True,
        with_plot: bool = True,
    ) -> pd.DataFrame:
        """
        Compute local FI for a single instance (optionally plot bar).
        If Bayesian, returns a DataFrame with mean, lower_bound, and upper_bound.
        """
        if self.A_dagger is None:
            raise ValueError("Run create_explainer first.")

        # 1. Initial setup
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        if scale and self.scaler is not None:
            x_prime = self.scaler.transform([x_arr])[0]
        else:
            x_prime = x_arr
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(x_arr))]

        # 2. Calculate LFI mean and create default DataFrame
        heat = (self.A_dagger @ y_arr) * x_prime
        result_df = pd.DataFrame([heat], columns=feature_names, index=['mean'])

        lfi_err_for_plot = None

        # 3. If Bayesian, calculate bounds and create multi-row DataFrame
        if self.use_bayesian:
            if self.A_dagger_cov is None:
                raise ValueError("Bayesian explainer has not been created.")
            
            # Calculate variance and std error
            lfi_var_common = y_arr.T @ self.A_dagger_cov @ y_arr
            if lfi_var_common < 0: lfi_var_common = 0
            lfi_std_common = np.sqrt(lfi_var_common)
            lfi_stds = np.abs(x_prime) * lfi_std_common
            lfi_err = 1.96 * lfi_stds # 95% CI margin
            lfi_err_for_plot = lfi_err.copy()

            # Calculate bounds
            lower_bound = heat - lfi_err
            upper_bound = heat + lfi_err
            
            # Create the final multi-row DataFrame
            result_df = pd.DataFrame(
                [heat, lower_bound, upper_bound],
                index=['mean', 'lower_bound', 'upper_bound'],
                columns=feature_names
            )

        # 4. Apply post-processing (zero-feature masking, normalization)
        if ignore_zero_features:
            mask = (x_arr != 0)
            result_df *= mask

        mean_row_for_norm = result_df.loc['mean'] if self.use_bayesian else result_df.iloc[0]
        max_abs = np.max(np.abs(mean_row_for_norm))
        if max_abs > 0:
            result_df /= max_abs
            if lfi_err_for_plot is not None:
                 lfi_err_for_plot /= max_abs

        # 5. Handle top_k filtering
        if top_k:
            mean_values = result_df.loc['mean'] if self.use_bayesian else result_df.iloc[0]
            keep_cols = mean_values.abs().nlargest(top_k).index
            result_df = result_df.loc[:, keep_cols]

            if lfi_err_for_plot is not None:
                col_indices = [list(feature_names).index(c) for c in keep_cols]
                lfi_err_for_plot = lfi_err_for_plot[col_indices]

        # 6. Plot if requested
        if with_plot:
            # Use the 'mean' row for the bar heights
            plot_means_df = pd.DataFrame(result_df.loc['mean']).T
            
            plt.figure(figsize=(10, max(4, len(result_df.columns) * 0.4)))
            sns.barplot(data=plot_means_df, orient="h", color="lightblue")
            
            # Add error bars if Bayesian mode was used
            if self.use_bayesian:
                plt.errorbar(x=plot_means_df.iloc[0].values, y=np.arange(len(plot_means_df.columns)), 
                             xerr=lfi_err_for_plot, fmt='none', c='black', capsize=4)
            
            plt.title("Local Feature Importance" + (" (with 95% Credible Intervals)" if self.use_bayesian else ""))
            plt.axvline(0, color='grey', linestyle='--')
            plt.tight_layout()
            plt.show()

        # 7. Return the final DataFrame
        return result_df

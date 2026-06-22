# Why AIME?

[← Back to the main README](../README.md)

<p align="center">
  <img src="../assets/AIMEOverview.png"
       alt="AIME overview: global and local importance, representative patterns, similarity fields, uncertainty, and inverse-operator visualizations"
       width="100%">
</p>

> **AIME treats model explanation as an approximate inverse problem.**  
> It learns a map from model outputs back to input-side feature patterns, then derives global, local, representative, and uncertainty-aware explanations from that single operator.

## The idea in 30 seconds

Let:

- `X ∈ R^(N×d)` be the input data,
- `Y ∈ R^(N×m)` be the corresponding model outputs, such as class probabilities,
- `X_s` be the optionally standardized input data.

AIME estimates an approximate inverse explanation operator:

```text
A† = X_sᵀ (Yᵀ)⁺    with A† ∈ R^(d×m)
```

where `⁺` denotes the Moore–Penrose pseudoinverse. For an output vector `y`, the operator maps back toward an input-side pattern:

```text
x_s ≈ A† y
```

`A†` is not assumed to be an exact inverse of a nonlinear black-box model. It is a **data-dependent linear approximation** over the input representation, output representation, and samples supplied to AIME.

```text
input samples X ──► black-box model ──► model outputs Y
       │                                      │
       └──────── learn one operator A† ◄──────┘

new output y ──► A†y ──► input-side explanation
```

## One operator, several explanation views

The central benefit of AIME is that its explanation views are connected by the same operator rather than being produced by unrelated procedures.

| Question | AIME quantity | Interpretation |
|---|---|---|
| Which features characterize output or class `t`? | `A† e_t` | A signed, per-output global feature-weight vector. |
| What input pattern is associated with output or class `t`? | `A† e_t`, transformed back to the original feature scale | A representative estimation instance: a tabular feature fingerprint or an image-like class pattern. |
| Why does this instance receive output `y`? | `(A† y) ⊙ x_s` | Local feature importance for the instance, combining the output-side pull with the observed input values. |
| How does the dataset relate to representative outputs? | Similarity between samples and representative estimation instances | A representative-instance similarity field that exposes overlap, separation, and ambiguous regions. |
| How stable are the estimated weights? | A distribution over `A†` in BayesianAIME | Interval estimates for global and local explanations. |

Here, `e_t` is the basis vector for output dimension `t`, and `⊙` is the element-wise product.

## Why use the inverse-operator view?

### A coherent global-to-local explanation

AIME derives its global and local views from the same fitted object. A global class signature, a representative pattern, and an instance-level explanation therefore share one mathematical reference point.

### A distinct explanation for every output dimension

Each column of `A†` describes how one output coordinate maps back to the input-feature space. This is useful for multiclass problems because each class receives its own signed feature pattern rather than only a single overall ranking.

### Representative input-side patterns

AIME can map a pure output basis vector back into the feature space. For tabular data, this gives a class-associated feature fingerprint. For image-shaped features, it can be reshaped into a representative class pattern.

### Model-agnostic and gradient-free

AIME requires paired inputs and model outputs, not access to model parameters, gradients, or internal layers. Once the output matrix has been collected, the standard operator is computed directly rather than fitting a new local surrogate for every explained instance.

### An inspectable mathematical object

The fitted `A†` matrix can be examined directly, visualized as an operator flow, compared across datasets or model versions, and regularized or robustified when the data require it.

## AIME, LIME, and SHAP answer different questions

AIME is best viewed as a complementary explanation framework, not as a universal replacement for every attribution method.

| Dimension | AIME | LIME | SHAP |
|---|---|---|---|
| Primary starting point | A dataset of inputs and corresponding model outputs | A prediction and a local neighborhood around that instance | A prediction together with a background or reference distribution |
| Primary explanation object | One approximate inverse operator `A†` | A locally fitted interpretable surrogate | Additive feature attributions based on Shapley values |
| Global view | Direct per-output columns of `A†` | Usually assembled from selected or aggregated local explanations | Commonly assembled by aggregating local SHAP values |
| Local view | `(A† y) ⊙ x_s` | Coefficients of the local surrogate | Per-feature additive attribution values |
| Representative input pattern | Directly available from `A† e_t` | Not the primary output | Not the primary output |
| Additional model evaluations | Uses outputs collected for the chosen dataset | Usually evaluates perturbed samples for each local explanation | Depends on the SHAP explainer; model-agnostic variants may require many evaluations |

Use LIME when a sparse local surrogate around one prediction is the desired explanation. Use SHAP when Shapley-based additive attribution and its associated axioms are central. Use AIME when a **single output-to-input operator**, consistent global/local views, and representative input-side patterns are the main goals.

## When AIME is a good fit

AIME is especially useful when:

- the model exposes vector outputs such as probabilities, scores, or multiple response dimensions;
- you want class-wise or output-wise global explanations;
- you want global, local, and representative explanations in one framework;
- you want to compare the inverse operators of different models, datasets, or training runs;
- gradients are unavailable or inappropriate;
- representative feature patterns or similarity fields are useful for understanding the dataset;
- robustness, regularization, or explanation uncertainty is important.

## Choose an AIME variant

All variants in this repository use the same `AIME` class and the same visualization API.

| Variant | Use it when… | Call |
|---|---|---|
| **AIME** | You want the canonical Moore–Penrose pseudoinverse formulation. | `AIME()` |
| **HuberAIME** | Outliers may dominate the inverse operator. | `AIME(use_huber=True)` |
| **RidgeAIME** | The output-side system is ill-conditioned or regularization is needed. | `AIME(use_ridge=True)` |
| **Huber-RidgeAIME** | You need both outlier robustness and regularization. | `AIME(use_huber=True, use_ridge=True)` |
| **BayesianAIME** | You want interval estimates for operator entries and feature importance. | `AIME(use_bayesian=True)` |

The effect of Ridge regularization depends on the scale and spectrum of `YᵀY`; inspect the raw operator and tune `ridge_alpha` rather than assuming that a small default value will visibly change normalized importance plots.

## Interpret AIME responsibly

AIME explanations are conditioned on the data and representations used to build the operator. In practice:

- changing the reference dataset can change `A†`;
- feature scaling, encoding, and output calibration affect the interpretation;
- correlated features can share or redistribute signed weights;
- representative estimation instances are operator-derived patterns, not necessarily observed people or records;
- local and global importance describe associations in the fitted inverse explanation, not causal effects;
- Bayesian intervals depend on the probabilistic assumptions and should not replace external stability checks;
- high-stakes applications should compare explanations across resamples, model versions, and domain-relevant validation sets.

A useful workflow is to report the dataset, preprocessing, model-output definition, AIME variant, and principal hyperparameters together with every explanation.

## Start with a notebook

| Notebook | Purpose | Run |
|---|---|---|
| [Titanic quick start](../examples/colab/01_titanic_quickstart.ipynb) | Build AIME and inspect global, local, representative, and inverse-operator explanations. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ntakafumi/aime/blob/main/examples/colab/01_titanic_quickstart.ipynb) |
| [AIME vs SHAP/LIME](../examples/colab/02_aime_vs_shap_lime.ipynb) | Compare the explanation objects and the questions answered by each method. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ntakafumi/aime/blob/main/examples/colab/02_aime_vs_shap_lime.ipynb) |
| [BayesianAIME uncertainty](../examples/colab/03_bayesian_aime_uncertainty.ipynb) | Explore interval estimates for global and local explanations. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ntakafumi/aime/blob/main/examples/colab/03_bayesian_aime_uncertainty.ipynb) |

## Primary references

- T. Nakanishi, “[Approximate Inverse Model Explanations (AIME): Unveiling Local and Global Insights in Machine Learning Models](https://doi.org/10.1109/ACCESS.2023.3314336),” *IEEE Access*, vol. 11, pp. 101020–101044, 2023.
- T. Nakanishi, “[HuberAIME: A Robust Approach to Explainable AI in the Presence of Outliers](https://doi.org/10.1109/ACCESS.2025.3565279),” *IEEE Access*, vol. 13, pp. 76796–76810, 2025.
- T. Nakanishi, “[Bayesian-AIME: Quantifying Uncertainty and Enhancing Stability in Approximate Inverse Model Explanations](https://doi.org/10.1109/ACCESS.2025.3617984),” *IEEE Access*, vol. 13, pp. 175547–175564, 2025.
- T. Itoh and T. Nakanishi, “[Approximate Inverse Model Explanations for Metamaterial Design with Scalar-Field-Based Metal Foam Surrogates](https://doi.org/10.1109/IIAI-AAI-Winter69777.2025.00041),” *IIAI-AAI-Winter*, pp. 179–184, 2025.
- M. T. Ribeiro, S. Singh, and C. Guestrin, “[“Why Should I Trust You?”: Explaining the Predictions of Any Classifier](https://doi.org/10.1145/2939672.2939778),” *KDD*, 2016.
- S. M. Lundberg and S.-I. Lee, “[A Unified Approach to Interpreting Model Predictions](https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions),” *NeurIPS*, 2017.

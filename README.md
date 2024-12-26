# AIME:Approximate Inverse Model Explanations
<img src="https://github.com/ntakafumi/aime/assets/147581981/831db9e0-36f7-40f7-a7d8-e7cbfd0e7b64" width="30%">

The AIME methodology is detailed in the paper available at The AIME methodology is detailed in the paper available at [https://ieeexplore.ieee.org/document/10247033](https://ieeexplore.ieee.org/document/10247033). AIME is proposed to address the challenges faced by existing methods in providing intuitive explanations for black-box models. AIME offers unified global and local feature importance by deriving approximate inverse operators for black-box models. It introduces a representative instance similarity distribution plot, aiding comprehension of the predictive behavior of the model and target dataset. This software only supports the global feature importance of AIME.
## Features
- **Unified Global and Local Feature Importance**: AIME derives approximate inverse operators for black-box models, offering insights into both global and local feature importance.
* **Representative Instance Similarity Distribution Plot**: This feature aids in understanding the predictive behavior of the model and the target dataset, illustrating the relationship between different predictions.
* **Effective Across Diverse Data Types**: AIME has been tested and proven effective across various data types, including tabular data, handwritten digit images, and text data.
* **Data-Driven Model-agnostic feature importance extraction** :Feature importance can be derived in a model-independent, data-driven manner: Feature importance methods linked to machine learning, such as the XAI methods SHAP, Random Forest, and XGboost,
$Explanation=SHAP(model, X), Explanation=Random_Forest(model, X), Explanation=XGboost(model, X),$
the explanation is always affected by the bias of the black box model. In contrast, AIME is a
$Explain= AIME(X,Y)$, AIME constructs explanations “using only the correspondence between the output and input, without using the model structure or weights”. AIME avoids “the bias of dependence on the model structure in XAI itself”. Although ordinary XAI methods (e.g. LIME, SHAP) are also called “model-agnostic”, there is a possibility of bias occurring because they only look at “the neighborhood of the output” due to the way local sampling is done, etc.
Because AIME uses the entire sample (all $Y$ and $X$), it is not biased towards the “inside of the black box model”, and can handle the “ideal $Y$” across the entire range.
This approach minimizes the risk that “XAI itself depends on the model structure”, and it is possible to create explanations in a completely data-driven manner in the form of $Explanation = AIME(X,Y)$.
## PCAIME (New)
- PCAIME, or Principal Component Analysis-Enhanced Approximate Inverse Model Explanations (https://ieeexplore.ieee.org/document/10648696), is an advanced method that extends the Approximate Inverse Model Explanations (AIME) framework.
- It incorporates dimensional decomposition and expansion functionalities, such as Principal Component Analysis (PCA), to enhance the explainability of complex AI and machine learning models.
- By addressing multicollinearity and correlations among features, PCAIME allows for a comprehensive visualization of feature relationships and contributions.
- This is achieved through 2D heatmaps that highlight global and local feature importance along with their interdependencies.
- The main features of PCAIME include:
	1.	Dimensional Decomposition: PCA is used to reduce the dimensionality of data, mitigating the curse of dimensionality while preserving the relationships among features.
	2.	Expansion Functionality: PCA loadings are employed to expand and visualize the relationships between original features and principal components, facilitating a deeper understanding of feature interactions.
	3.	Heatmap Visualizations: PCAIME generates intuitive heatmaps to display feature contributions and correlations, enhancing the interpretability of both global and local feature importance.
	4.	Application Flexibility: PCAIME is particularly suited for datasets with multicollinearity and high-dimensionality, making it an essential tool for domains like healthcare, finance, and public policy.

For details on implementing PCAIME, please refer to the file PCAIME.ipynb for step-by-step guidance and code examples.

## **License**
AIME is dual-licensed under the The 2-Clause BSD License and the Commercial License. Apply the The 2-Clause BSD License only for academic or research purposes, and apply Commercial License for commercial and other purposes. You can choose which one to use.
## **Commercial License**
For those interested in Commercial License, a licensing fee may be required. Please contact us for more details at:
**Email**: [takafumi@eigenbeats.com](mailto:takafumi@eigenbeats.com)
## Installation
```
pip install aime-xai
```

## Citation
If you use this software for research or other purposes, please cite the following paper
```
@ARTICLE{10247033,
author={Nakanishi, Takafumi},
journal={IEEE Access}, 
  title={Approximate Inverse Model Explanations (AIME): Unveiling Local and Global Insights in Machine Learning Models}, 
  year={2023},
  volume={11},
  number={},
  pages={101020-101044},
 doi={10.1109/ACCESS.2023.3314336}}

@ARTICLE{10648696,
  author={Nakanishi, Takafumi},
  journal={IEEE Access}, 
  title={PCAIME: Principal Component Analysis-Enhanced Approximate Inverse Model Explanations Through Dimensional Decomposition and Expansion}, 
  year={2024},
  volume={12},
  number={},
  pages={121093-121113},
  doi={10.1109/ACCESS.2024.3450299}}
```

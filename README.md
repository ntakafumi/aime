# AIME:Approximate Inverse Model Explanations
<img src="https://github.com/ntakafumi/aime/assets/147581981/a858e613-8c30-47dd-8e61-5bf6165c385c" width"30%">

The AIME methodology is detailed in the paper available at The AIME methodology is detailed in the paper available at [https://ieeexplore.ieee.org/document/10247033](https://ieeexplore.ieee.org/document/10247033). AIME is proposed to address the challenges faced by existing methods in providing intuitive explanations for black-box models. AIME offers unified global and local feature importance by deriving approximate inverse operators for black-box models. It introduces a representative instance similarity distribution plot, aiding comprehension of the predictive behavior of the model and target dataset. This software only supports the global feature importance of AIME.
## Features
- **Unified Global and Local Feature Importance**: AIME derives approximate inverse operators for black-box models, offering insights into both global and local feature importance.
* **Representative Instance Similarity Distribution Plot**: This feature aids in understanding the predictive behavior of the model and the target dataset, illustrating the relationship between different predictions.
* **Effective Across Diverse Data Types**: AIME has been tested and proven effective across various data types, including tabular data, handwritten digit images, and text data.
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
```

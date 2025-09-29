from setuptools import setup, find_packages

setup(
    name='aime_xai',
    version='1.1.0',
    license="Apply the The 2-Clause BSD License only for academic or research purposes, and apply Commercial License for commercial and other purposes.",
    packages=find_packages(),    
    author="Takafumi Nakanishi",
    url="https://github.com/ntakafumi/aime",
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "umap-learn",
        "scipy",
        "opencv-python"
    ],
    extras_require={
        "colab": ["google-colab"]
    },
    author_email='takafumi@eigenbeats.com',
    description='AIME implementation for XAI',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
)


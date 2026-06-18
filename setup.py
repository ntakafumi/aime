# Packaging metadata now lives in pyproject.toml (PEP 621).
# This thin shim is kept only for `pip install -e .` on very old toolchains.
from setuptools import setup

setup()

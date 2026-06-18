# aime_xai/__init__.py  (signature-visualisation edition)

__version__ = "1.2.0"

from .core import AIME
from . import style
from . import metrics
from .metrics import AIMEEvaluator
from . import operator_viz
from .operator_viz import OperatorVisualizer

__all__ = [
    "AIME",
    "style",
    "metrics",
    "AIMEEvaluator",
    "operator_viz",
    "OperatorVisualizer",
]

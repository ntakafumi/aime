# aime_xai/__init__.py

# バージョン等を定義（任意）
__version__ = "1.1.0"

# core.py 内の AIME クラスをパッケージのトップレベルに公開
from .core import AIME

__all__ = [
    "AIME",
]
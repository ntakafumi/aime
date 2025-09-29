from setuptools import setup, find_packages
from pathlib import Path

# --- long_description を安全に用意 ---
here = Path(__file__).parent
readme_path = here / "README.md"
if readme_path.exists():
    long_desc = readme_path.read_text(encoding="utf-8")
    long_desc_ct = "text/markdown"
else:
    long_desc = "Approximate Inverse Model Explanations (AIME)"
    long_desc_ct = "text/markdown"  # README.md なくても指定してOK

setup(
    name="aime-xai",
    version="1.1.0",
    description="Approximate Inverse Model Explanations (AIME): unified global/local importance for XAI",
    long_description=long_desc,
    long_description_content_type=long_desc_ct,
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    license="LicenseRef-AIME-Academic-NonCommercial",  # ← ここは classifiers 外に置く
    license_files=["LICENSE*", "LICENSE-Academic*", "LICENSE-Commercial*"],
)
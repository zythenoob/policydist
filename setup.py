from setuptools import find_packages, setup
from pathlib import Path

package_path = __file__
setup(
    name="policydist",
    version="0.0.1",
    author="Jiaye Zhu",
    author_email="jiayezhu@usc.edu",
    packages=find_packages(),
    description="Suprise Policy Distillation",
    python_requires=">3.9",
    long_description=Path(package_path).parent.joinpath("README.md").read_text(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit_learn",
        "torch",
        "tqdm",
        "tensorboard",
        "matplotlib",
        "omegaconf",
        "setproctitle",
    ],
    extras_require={
        "dev": ["mypy", "pytest", "pylint", "flake8", "black"],
    },
    dependency_links=[
        # "https://download.pytorch.org/whl/torch_stable.html",
        "https://download.pytorch.org/whl/cu113",
    ],
)

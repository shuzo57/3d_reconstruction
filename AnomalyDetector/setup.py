from setuptools import find_packages, setup

setup(
    name="AnomalyDetector",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "torch",
        "torchvision",
        "torchaudio",
    ],
    license="MIT",
    description=(
        "A specialized anomaly detection library tailored "
        "for analyzing golf swings. Leveraging advanced "
        "deep learning techniques, it pinpoints irregularities "
        "in swing patterns, providing valuable insights "
        "for performance optimization and injury prevention."
    ),
)

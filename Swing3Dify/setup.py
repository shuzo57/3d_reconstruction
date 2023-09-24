from setuptools import find_packages, setup

setup(
    name="Swing3Dify",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "opencv-python",
        "pandas",
        "numpy",
        "scipy",
        "matplotlib",
        "Pillow",
        "scikit-learn",
        "scikit-image",
        "opencv-python-headless",
        "plotly",
        "nbformat",
        "tabulate",
    ],
    license="MIT",
    description="A library for 3D reconstruction of golf swings.",
)

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
        "seaborn",
        "matplotlib",
        "japanize_matplotlib",
        "Pillow",
        "plotly",
        "nbformat",
        "tabulate",
        "ultralytics",
    ],
    license="MIT",
    description="A library for 3D reconstruction of golf swings.",
)

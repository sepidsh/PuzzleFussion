from setuptools import setup

setup(
    name="puzzle_fusion",
    py_modules=["puzzle_fusion"],
    install_requires=["blobfile>=1.0.5", "torch", "tqdm"],
)

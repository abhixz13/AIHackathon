from setuptools import setup, find_packages

setup(
    name="evalflow",
    version="1.0.0",
    description="Reference implementation of an evaluation analytics library.",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.0",
        "matplotlib>=3.5.0",
        "numpy>=1.23.0",
    ],
    python_requires=">=3.9",
)

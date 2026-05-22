from setuptools import setup, find_packages

setup(
    name="materials_stock_libs",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "pandas",
        "scikit-learn",
        "matplotlib"
    ],
)

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="regressors",
    version="0.4.0",
    author="Alexander Sergian",
    author_email="alex.sergian@gmail.com",
    description="A custom implementation of linear regression and logistic regression models (binary and multi-class) with gradient descent optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/asergian/ml-regressors.git",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.19.0",
    ],
    keywords="machine-learning, regression, linear-regression, logistic-regression, softmax, multi-class, gradient-descent, classification"
)
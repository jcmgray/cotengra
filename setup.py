from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="cotengra",
    description=(
        "Hyper optimized contraction trees "
        "for large tensor networks and einsums."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jcmgray/cotengra",
    project_urls={
        "Bug Reports": "https://github.com/jcmgray/cotengra/issues",
        "Source": "https://github.com/jcmgray/cotengra/",
    },
    author="Johnnie Gray",
    author_email="johnniemcgray@gmail.com",
    license="Apache",
    packages=find_packages(exclude=["docs", "tests"]),
    include_package_data=True,
    install_requires=[
        "autoray",
        "cytoolz",
        "networkx",
        "optuna",
        "tqdm",
    ],
    extras_require={
        "recommended": [
            "opt_einsum",
            "kahypar",
            "ray",
        ],
        "docs": [
            "furo",
            "ipython!=8.7.0",
            "myst-nb",
            "setuptools_scm",
            "sphinx-autoapi",
            "sphinx-copybutton",
            "sphinx>=2.0",
        ],
        "test": [
            "altair",
            "baytune",
            "chocolate",
            "dask",
            "distributed",
            "kahypar",
            "matplotlib",
            "networkx",
            "nevergrad",
            "numpy",
            "pytest",
            "seaborn",
            "skopt",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="tensor network contraction graph hypergraph partition einsum",
)

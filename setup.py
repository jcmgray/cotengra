from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='cotengra',
    description='opt_einsum compatible contractors for large tensor networks',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/jcmgray/cotengra',
    project_urls={
        'Bug Reports': 'https://github.com/jcmgray/cotengra/issues',
        'Source': 'https://github.com/jcmgray/cotengra/',
    },
    author='Johnnie Gray',
    author_email="johnniemcgray@gmail.com",
    license="Apache",
    packages=find_packages(exclude=['docs', 'tests']),
    include_package_data=True,
    install_requires=[
        'opt_einsum',
        'tqdm',
        'cytoolz',
        'optuna',
        'autoray',
    ],
    extras_require={
        'recommended': [
            'kahypar',
            'optuna',
            'networkx',
            'autoray',
            'ray',
        ],
        'docs': [
            'sphinx>=2.0',
            'sphinx-autoapi',
            'sphinx-copybutton',
            'myst-nb',
            'furo',
            'setuptools_scm',
            'ipython!=8.7.0',
        ],
        'test': [
            'numpy',
            'kahypar',
            'matplotlib',
            'networkx'
            'altair',
            'seaborn',
            'pytest',
            'dask',
            'distributed',
            'baytune',
            'skopt',
            'chocolate',
            'nevergrad',
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    keywords='tensor network contraction graph hypergraph partition einsum',
)

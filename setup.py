from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='cotengra',
    version='0.1.0',
    description='opt_einsum compatible contractors for large tensor networks',
    long_description=long_description,
    url='https://github.com/jcmgray/cotengra',
    author='Johnnie Gray',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    keywords='tensor network contraction graph partition',
    packages=find_packages(exclude=['docs', 'tests']),
    install_requires=[
        'numpy',
        'psutil',
        'kahypar',
        'baytune>=0.3',
        'opt_einsum',
        'tqdm',
        'seaborn',
    ],
    project_urls={  # Optional
        'Bug Reports': 'https://github.com/jcmgray/cotengra/issues',
        'Source': 'https://github.com/jcmgray/cotengra/',
    },
    include_package_data=True,
)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

# Read requirements
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(req_path):
        with open(req_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return []

setup(
    name="themol",
    version="0.1.0",
    author="TheMol Team",
    author_email="themolsubmission@gmail.com",
    description="Learning Canonical Representations for Unified 3D Molecular Modeling",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/themolsubmission/TheMol",
    packages=find_packages(exclude=["tests", "tests.*"]),
    package_data={
        "unimol": ["*.txt", "*.pkl"],
        "data": ["*.txt", "*.pkl"],
    },
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "tqdm>=4.60.0",
        "lmdb>=1.0.0",
        "rdkit>=2022.03",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "flake8>=4.0",
        ],
        "optimization": [
            "cmaes>=0.9.0",
            "pyzmq>=22.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
    keywords="molecular modeling, 3D molecules, deep learning, drug discovery",
    project_urls={
        "Bug Reports": "https://github.com/themolsubmission/TheMol/issues",
        "Source": "https://github.com/themolsubmission/TheMol",
    },
)

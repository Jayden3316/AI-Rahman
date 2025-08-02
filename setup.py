#!/usr/bin/env python3
"""
Setup script for MusicGen Interpretation Library.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="musicgen-interpretation",
    version="0.1.0",
    author="MusicGen Interpretation Team",
    author_email="team@example.com",
    description="Fine-grained control over Music Generation with Activation Steering",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/musicgen-interpretation",
    project_urls={
        "Bug Tracker": "https://github.com/your-username/musicgen-interpretation/issues",
        "Documentation": "https://musicgen-interpretation.readthedocs.io",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Multimedia :: Sound/Audio :: Synthesis",
    ],
    package_dir={"": "."},
    packages=find_packages(where="."),
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "myst-parser>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "musicgen-interpretation=musicgen_interpretation.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "musicgen_interpretation": ["*.json", "*.yaml", "*.yml"],
    },
    keywords="music, generation, ai, interpretability, activation-steering, musicgen",
    zip_safe=False,
) 
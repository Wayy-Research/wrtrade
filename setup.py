"""
WRTrade - Advanced Portfolio Trading Framework
Setup script for installation and distribution.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="wrtrade",
    version="1.0.0",
    description="Advanced Portfolio Trading Framework with N-dimensional portfolios, Kelly optimization, and multi-broker support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Wayy Research",
    author_email="contact@wayy.research",
    url="https://github.com/wayy-research/wrtrade",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "polars>=0.18.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "click>=8.0.0",
        "pyyaml>=6.0.0",
        "psutil>=5.8.0",
        "aiohttp>=3.8.0",
        "requests>=2.28.0",
        "pandas>=1.3.0",  # For compatibility with some broker APIs
        "wrchart>=0.1.1",  # Financial charting
    ],
    extras_require={
        "alpaca": ["alpaca-trade-api>=3.0.0"],
        "robinhood": ["robin-stocks>=2.0.0"],
        "all": [
            "alpaca-trade-api>=3.0.0",
            "robin-stocks>=2.0.0"
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.20.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ]
    },
    entry_points={
        "console_scripts": [
            "wrtrade=wrtrade.cli:cli",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "trading", "portfolio", "quantitative", "finance", "algorithmic", 
        "kelly", "optimization", "backtesting", "deployment", "brokers"
    ],
    project_urls={
        "Bug Reports": "https://github.com/wayy-research/wrtrade/issues",
        "Documentation": "https://wrtrade.readthedocs.io/",
        "Source": "https://github.com/wayy-research/wrtrade",
    },
)
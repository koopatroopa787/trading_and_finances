"""Setup script for trading backtesting engine."""
from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="trading-backtesting-engine",
    version="1.0.0",
    author="Quant Trading System",
    description="Professional-grade multi-strategy backtesting engine with portfolio optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/trading_and_finances",
    packages=find_packages(include=['src', 'src.*']),
    package_dir={'': '.'},
    python_requires=">=3.11",
    install_requires=[
        line.strip()
        for line in open("requirements.txt")
        if line.strip() and not line.startswith("#")
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    keywords="trading backtesting quantitative-finance portfolio-optimization",
    project_urls={
        "Documentation": "https://github.com/yourusername/trading_and_finances",
        "Source": "https://github.com/yourusername/trading_and_finances",
        "Tracker": "https://github.com/yourusername/trading_and_finances/issues",
    },
)

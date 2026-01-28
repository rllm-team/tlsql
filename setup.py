"""Setup script for TLSQL package"""

from setuptools import setup
import os

# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

setup(
    name="tlsql",
    version="0.1.0",
    description="A Python library that converts custom SQL-like statements into standard SQL queries for machine learning workflows",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="TLSQL Team",
    author_email="",
    url="https://github.com/rllm-team/tlsql",
    license="MIT",
    packages=["tlsql", "tlsql.tlsql"],
    package_dir={
        "tlsql": ".",
        "tlsql.tlsql": "tlsql"
    },
    python_requires=">=3.8",
    install_requires=[
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "flake8>=5.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="sql machine-learning table-learning workflow",
)

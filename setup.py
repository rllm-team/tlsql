"""Setup script for TLSQL package"""

from setuptools import setup

setup(
    name="tlsql",
    version="0.1.0",
    description="A SQL conversion library for custom SQL statements",
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
)

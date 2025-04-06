from setuptools import setup, find_packages

setup(
    name="noteviz",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "openai>=1.0.0",
        "pypdf2>=3.0.0",
        "numpy>=1.24.0",  # For vector operations in retrieval
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "reportlab>=4.0.0",  # For generating test PDFs
        ],
    },
    entry_points={
        "console_scripts": [
            "noteviz=noteviz.cli:main",
        ],
    },
    python_requires=">=3.8",
) 
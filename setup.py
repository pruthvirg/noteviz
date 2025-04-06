from setuptools import setup, find_packages

setup(
    name="noteviz",
    version="0.1.0",
    packages=["noteviz", "noteviz.core", "noteviz.core.llm", "noteviz.core.pdf", "noteviz.core.flowchart", "noteviz.ui"],
    package_dir={"noteviz": "src/noteviz"},
    install_requires=[
        "openai>=1.0.0",
        "pypdf>=5.0.0",
        "numpy>=1.24.0",
        "streamlit>=1.32.0",
        "python-dotenv>=1.0.1",
        "aiohttp>=3.9.3",
    ],
    python_requires=">=3.9",
) 
from setuptools import setup, find_packages

setup(
    name="hallucination_detector",
    version="0.1.0",
    description="A research-grade LLM hallucination detection system",
    author="ML Engineer",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "faiss-cpu==1.7.4",
        "fastapi==0.111.0",
        "uvicorn",
        "gradio==4.36.0",
        "sentence-transformers==3.0.0",
        "transformers==4.41.0",
        "torch==2.3.0",
        "datasets==2.19.0",
        "numpy",
        "pandas",
        "scikit-learn==1.5.0",
        "scipy",
        "tqdm",
        "pydantic==2.7.0",
        "pydantic-settings",
        "python-dotenv",
        "mlflow==2.13.0",
        "pytest==8.2.0",
        "httpx",
        "rouge-score",
        "nltk"
    ],
)

from setuptools import setup, find_packages

setup(
    name="ai-agent-comparison",
    version="1.0.0",
    description="A comparison of AI agent frameworks for financial analysis",
    author="Vignesh Maradiya",
    author_email="maradiyavignesh2004@gmail.com",
    packages=find_packages(),
    install_requires=[
        "crewai",
        "langgraph", 
        "autogen-agentchat",
        "langchain-groq",
        "langchain-community",
        "yfinance",
        "tavily-python",
        "python-dotenv",
        "pydantic"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)

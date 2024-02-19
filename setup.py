from setuptools import setup, find_packages

setup(
    name="aces",
    version="2024.2.20",
    description="Agricultural Classification and Estimation Service (ACES)",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/biplovbhandari/aces",
    author="Biplov Bhandari",
    author_email="bionicbiplov45@gmail.com",
    license="GNU GPL v3.0",
    keywords="remote sensing machine learning",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "tensorflow>=2.9.3",
        "apache-beam>=2.38.0",
        "earthengine-api",
        "python-dotenv>=1.0.0",
        "matplotlib",
    ],
    python_requires='>=3.9',
)

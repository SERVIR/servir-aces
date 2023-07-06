from setuptools import setup, find_packages

setup(
    name="aces",
    version="2023.6.28",
    description="Agricultural Classification and Estimation Service",
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
        "tensorflow",
        "apache-beam",
        "earthengine-api",
    ],
)

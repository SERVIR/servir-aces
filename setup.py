from setuptools import setup, find_packages


setup(
   name="servir-aces",
   version="v0.0.1",
   description="Agricultural Classification and Estimation Service (ACES)",
   long_description=open("README.md").read(),
   long_description_content_type="text/markdown",
   url="https://github.com/SERVIR/servir-aces",
   author="Biplov Bhandari",
   author_email="bb0134@uah.edu; bionicbiplov45@gmail.com",
   license="GNU GPL v3.0",
   keywords="remote sensing, agriculture, machine learning, deep learning",
   packages=find_packages(),
   install_requires=[
       "numpy",
       "tensorflow>=2.9.3",
       "apache-beam>=2.38.0",
       "earthengine-api",
       "python-dotenv>=1.0.0",
       "matplotlib",
   ],
   # TODO: Add test dependencies
   # extras_require={
   #     'dev': [
       # "numpy",
   #         'apache-beam>=2.38.0',
   #     ]
   # }
   # TODO: Add entry points
   # TODO: Test different python versions
   python_requires='>=3.9',
)

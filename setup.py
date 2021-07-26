import setuptools
import os
from sys import platform
if platform.startswith('win'):
    os.system('py -m pip install -r requirements.txt')
else:
    os.system('pip install -r requirements.txt')

with open('requirements.txt') as f:
    requirements = f.readlines()
Packages=np.char.replace(requirements, old="\n", new="")

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

exec(open('ClustersFeatures/version.py').read())
setuptools.setup(
    name="Clusters-Features",
    version=__version__.replace('-',''),
    author="Simon Bertrand",
    author_email="simonbertrand.contact@gmail.com",
    description="The Clusters-Features package allows data science users to compute high-level linear algebra operations on any type of data set.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Simon-Bertrand",
    install_requires=Packages,
    project_urls={
        "Cluster-Features": "https://github.com/Simon-Bertrand/ClustersCharacteristics/",
    },
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.8"

)
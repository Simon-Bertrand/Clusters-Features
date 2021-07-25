import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Clusters-Features",
    version="1.0.0",
    author="Simon Bertrand",
    author_email="simonbertrand.contact@gmail.com",
    description="The Clusters-Features package allows data science users to compute high-level linear algebra operations on any type of data set.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Simon-Bertrand",
    project_urls={
        "Cluster-Features": "https://github.com/Simon-Bertrand/ClustersCharacteristics/",
    },
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.9",
)
# -*- coding: utf-8 -*-
#
# Copyright 2021 Simon Bertrand
#
# This file is part of ClusterCharacteristics.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>.


import setuptools
import os
from sys import platform

#Call the __version__ var
exec(open('ClustersFeatures/version.py').read())
#Call the settings parameters
exec(open('ClustersFeatures/settings.py').read())

if Activated_Graph:
    List_total_Packages = List_base_Packages + List_Graph
if Activated_Utils:
    List_total_Packages = List_base_Packages + List_Utils
    
#Create a requirements.txt adapted to settings options
with open("requirements.txt", "w") as file:
    for package in List_total_Packages:
        file.write(package + "\n")

#Installing all requirements : pip autodetect if already installed
if platform.startswith('win'):
    os.system('py -m pip install -r requirements.txt')
else:
    os.system('pip install -r requirements.txt')

#Saving in a list all requirements
with open('requirements.txt') as f:
    requirements = f.readlines()
Packages=[req.replace("\n", "") for req in requirements]

#Putting README.md as long_description package for PyPi
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


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
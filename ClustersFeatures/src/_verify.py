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

"""
Section: Private
Private function to verify if everything works.
__verify_scatter_matrixes : Easy way to check if the 3 matrixes are well computed. In theory, we should always have WG + BG = Total Scattering Matrix, these are calculated here independently

"""


import numpy as np

from ClustersFeatures import settings
class Verify:

    def __verify_scatter_matrixes(self):
        if (np.round(self.scatter_matrix_WG() + self.scatter_matrix_between_group_BG(), settings.precision // 2) == np.round(
                self.scatter_matrix_T(), settings.precision // 2)).sum().sum() == len(self.scatter_matrix_T()) ** 2:
            return True
        else:
            return False

import pkg_resources
import os
def install_modules():
    installed_packages = pkg_resources.working_set
    installed_packages_list = sorted(["%s" % (i.key)
                                      for i in installed_packages])
    with open('../requirements.txt') as f:
        lines = f.readlines()
    Packages = np.char.replace(lines, old="\n", new="")
    c=0
    for package in Packages:
        for installed_package in installed_packages_list:
            if not installed_packages.__contains(package):
                c+=1
                os.system("echo  ClustersFeatures - Installing missing module : " + package)
                os.system("pip install " + package)

    #Compute a second time to be sure that everything is correctly
    #installed, pip will hand already installed packages
    os.system('pip install -r requirements.txt')


# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from ClustersFeatures import settings

class __Info:
    @property
    def clusters_info(self):
        """Generate a board that gives information about the different clusters.

        :returns: A pandas dataframe.

        >>> CC.clusters_info
        """
        pd.set_option('display.float_format', ("{:." + str(settings.precision) + "f}").format)
        return self._IndexCore_create_board(['clusters','radius'])

    def general_info(self, **args):
        """Generate a board that gives general information about the dataset.

        :param bool hide_nan: Show the NaN indices and their corresponding code. If True, showing is disabled.

        :returns: A pandas dataframe.

        >>> CC.general_info(hide_nan=False)
        """
        try:
            hide_nan=args['hide_nan']
            if not(isinstance(hide_nan,bool)):
                raise ValueError('hide_nan argument is not a boolean')
        except KeyError:
            hide_nan=False

        pd.set_option('display.float_format', ("{:." + str(settings.precision) + "f}").format)
        if not hide_nan:
            print("Current NaN Index :\n")
            for name,code in self._IndexCore_nan_general_index().items():
                print(f"{name:<25}-{'':<10}{code}")
        return self._IndexCore_create_board(['general', 'score_index_GDI']).rename(columns={0:'General Informations'})




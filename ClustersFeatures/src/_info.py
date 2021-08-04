# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from ClustersFeatures import settings

class __Info:
    @property
    def clusters_info(self):
        pd.set_option('display.float_format', ("{:." + str(settings.precision) + "f}").format)
        return self._IndexCore_create_board(['clusters','radius'])

    def general_info(self, **args):
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




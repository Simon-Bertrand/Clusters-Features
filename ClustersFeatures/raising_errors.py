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

import pandas as pd
import numpy as np

##General errors
def both_clusters_in(Cluster1,Cluster2,all_clusters):
    if not(Cluster1 in all_clusters) or not(Cluster2 in all_clusters):
        raise AttributeError('Given centroids clusters label names  aren\'t found in' + str(all_clusters))

def both_element_in(El1,El2,all_El):
    if not(El1 in all_El) or not(El2 in all_El):
        raise IndexError('One of both given ElementId isn\'t in dataframe indexes')

def cluster_in(Cluster, labels_clusters):
    if not (Cluster in labels_clusters):
        raise KeyError(
            'A such cluster name "' + Cluster + '" isn\'t found in dataframe\'s clusters. Here are the available clusters : ' + str(
                list(labels_clusters)))

def column_in(col, all_cols):
    if not (col in all_cols):
        raise KeyError(
            'A such column name "' + col + '" isn\'t found in dataframe\'s columns. Here are the available clusters : ')


##__init__ errors
def verify_pandas_df_and_not_empty(pd_df):
    if not (isinstance(pd_df, pd.DataFrame)):
        raise TypeError('Given dataframe isn\'t a Pandas dataframe.')
    elif pd_df.empty:
        raise ValueError('Given Pandas dataframe is empty')

def wrong_label_target(label_target):
    raise AttributeError('A such label target name "' + label_target + '" isn\'t found in dataframe\'s columns.')

##_confusion_hypersphere errors
def CH_radius(args):
    try:
        radius_choice=args['radius']
        if not (isinstance(radius_choice, float)) and not (isinstance(radius_choice, int)):
            raise ValueError('radius argument is not numeric : float or int')
        return args['radius']
    except KeyError:
        raise ValueError('radius argument is not specified.')

def CH_counting_type(args):
    try:
        c_type=args['counting_type']
        if not (c_type in ["including", "excluding"]):
            raise ValueError('counting_type isn\'t in the following list' + str(["including", "excluding"]))
        return c_type
    except KeyError:
        raise ValueError('counting_type isn\'t specified. Available values ' + str(["including", "excluding"]))

def CH_proportion(args):
    try:
        if isinstance(args['proportion'],bool):
            return args['proportion']
        else:
            raise ValueError('Proportion arg is not boolean')
    except KeyError:
        return False

def CH_max_radius(args, default_value):
    try:
        return args['max_radius']
    except:
        return default_value

def CH_num_pts(args, default_value):
    try:
        return args['n_pts']
    except:
        return default_value

#_utils errors
def utils_seasonal_arg(args):
    try:
        seasonal=args['seasonal']
        if not(isinstance(seasonal,float)) and  not(isinstance(seasonal,int)):
            raise ValueError('seasonal argument is not numeric.')
        else:
            return seasonal

    except KeyError:
        raise ValueError('seasonal argument is not specified.')

def utils_col_arg(args, columns):
    try:
        col=args['col']
        column_in(col, columns)
        return col
    except KeyError:
        col=None

def utils_data_arg(args):
    try:
        data=args['data']
        if not(isinstance(np.array([1,2,3]), list)) and not(isinstance(np.array([1,2,3]), np.ndarray)):
            raise ValueError('Data argument is not list or np.array')
        return data
    except KeyError:
        data=None
def utils_not_botch_col_and_data():
    raise ValueError('Passing data and col argument in the same time is impossible')

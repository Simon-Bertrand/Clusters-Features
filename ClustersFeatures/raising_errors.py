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
        raise AttributeError('Given centroids clusters label names  are not found in' + str(all_clusters))

def both_element_in(El1,El2,all_El):
    if not(El1 in all_El) or not(El2 in all_El):
        raise IndexError('One of both given ElementId is not in dataframe indexes')

def cluster_in(Cluster, labels_clusters):
    if not (Cluster in labels_clusters):
        raise KeyError(
            'A such cluster name "' + str(Cluster) + '" is not found in dataframeclusters. Here are the available clusters : ' + str(
                list(labels_clusters)))

def column_in(col, all_cols):
    if not (col in all_cols):
        raise KeyError(
            'A such column name "' + str(col) + '" is not found in dataframe columns. Here are the available clusters : ')

def list_clusters(args,labels_clusters):
    try:
        l_c=args['clusters']
        if isinstance(l_c, int) or isinstance(l_c,float) or isinstance(l_c, np.int32):
            l_c=[l_c]
        elif not (isinstance(l_c, list)) and not (isinstance(l_c, np.ndarray)):
            print(type(l_c))
            raise ValueError('clusters arg is not a list or ndarray.')
        for cl in l_c:
            if not(cl in labels_clusters):
                raise ValueError(str(cl) + ' is not found in dataframe\'s labels clusters')
        return l_c
    except KeyError:
        return False

##__init__ errors
def verify_pandas_df_and_not_empty(pd_df):
    if not (isinstance(pd_df, pd.DataFrame)):
        raise TypeError('Given dataframe isn\'t a Pandas dataframe.')
    elif pd_df.empty:
        raise ValueError('Given Pandas dataframe is empty')

def verify_no_object_columns_and_delete_it(pd_df):
    if 'object' in pd_df.dtypes:
        print('Columns of object type detected, deleting them.')
        return pd_df.drop(columns=pd_df.dtypes[pd_df.dtypes.values == "object"].index)
    else:
        return pd_df

def wrong_label_target(label_target):
    raise AttributeError('A such label target name "' + label_target + '" is not found in dataframe\'s columns.')

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
def utils_period_arg(args):
    try:
        period=args['period']
        if not(isinstance(period,float)) and  not(isinstance(period,int)):
            raise ValueError('period argument is not numeric.')
        else:
            return period

    except KeyError:
        raise ValueError('period argument is not specified.')

def utils_col_arg(args, columns):
    try:
        col=args['col']
        column_in(col, columns)
        return col
    except KeyError:
        return None

def utils_data_arg(args):
    try:
        data=args['data']
        if not(isinstance(np.array([1,2,3]), list)) and not(isinstance(np.array([1,2,3]), np.ndarray)):
            raise ValueError('Data argument is not list or np.array')
        return data
    except KeyError:
        return None
def utils_not_botch_col_and_data():
    raise ValueError('Passing data and col argument in the same time is impossible')

def utils_return_KDE_model(args):
    try:
        returnKDE = args['return_KDE']
        if not(isinstance(returnKDE, bool)):
            raise ValueError('return_KDE=' + returnKDE + ' is not a boolean.')
        return returnKDE
    except KeyError:
        return False

def density_Projection_2D(args, labels_clusters):
    try:
        cluster = args['cluster']
        if isnumeric(cluster):
            cluster = [cluster]
        for el in cluster:
            if not (el in (labels_clusters + ["all"])):
                raise ValueError(str(el) + " is not in " + str(labels_clusters))
    except KeyError:
        cluster = labels_clusters


    try:
        return_clusters_density = args['return_clusters_density']
        if not (isinstance(return_clusters_density, bool)):
            raise ValueError('return_clusters_density is not boolean')
    except KeyError:
        return_clusters_density = False

    try:
        return_data = args['return_data']
        if not (isinstance(return_clusters_density, bool)):
            raise ValueError('return_data is not boolean')
    except KeyError:
        return_data = False


    return cluster, return_clusters_density, return_data


def density_Density_Projection_3D(args, labels_clusters):
    try:
        cluster = args['cluster']
        if isnumeric(cluster):
            cluster = [cluster]
        for el in cluster:
            if not (el in (labels_clusters + ["all"])):
                raise ValueError(str(el) + " is not in " + str(labels_clusters))
            if len(cluster)>2:
                raise ValueError('Computing more than 2 clusters is disabled for density 3D')
    except KeyError:
        cluster = labels_clusters


    try:
        return_clusters_density = args['return_clusters_density']
        if not (isinstance(return_clusters_density, bool)):
            raise ValueError('return_clusters_density is not boolean')
    except KeyError:
        return_clusters_density = False

    try:
        return_grid = args['return_grid']
        if not (isinstance(return_grid, bool)):
            raise ValueError('return_grid is not boolean')
    except KeyError:
        return_grid = False


    return cluster, return_clusters_density, return_grid



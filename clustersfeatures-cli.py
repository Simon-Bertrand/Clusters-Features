#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd

from ClustersFeatures import ClustersCharacteristics

parser = argparse.ArgumentParser(description='CLI for ClustersFeatures Python package.')
parser.add_argument("csv_location", type=str, help="This argument needs to be the location of the specified csv. The label target have to be the last column of the csv.")
parser.add_argument("--index", default=False, help="Returns all indexes that are in the indices.json.")
parser.add_argument("--ch_total", default=False, type=float, help="Returns the confusion hypersphere with specified radius. The counting_type is 'including' and proportion is set to false.")
parser.add_argument("--ch_proportion", default=False, type=float, help="Returns the confusion hypersphere with specified radius. The counting_type is 'including' and proportion is set to true.")
parser.add_argument("--percentile_density", default=99, type=float, help="Set the minimum contour of the density as a percentile of the reducted data. Only used for twodim and threedim densities.")
parser.add_argument("--twodim_density_PCA", default=False, help="Returns the total 2D density and the density for each cluster. Uses the PCA Reduction method.")
parser.add_argument("--threedim_density_PCA", default=False, help="Returns the total 3D density and the density for each cluster. Uses the PCA Reduction method.")
parser.add_argument("--twodim_density_UMAP", default=False, help="Returns the total 2D density and the density for each cluster. Uses the UMAP Reduction method.")
parser.add_argument("--distances_to_centroids", default=False, help="Returns the distance between each element and each cluster centroid.")
parser.add_argument("--intercentroid_distances", default=False, help="Returns the distance between each cluster centroid.")
parser.add_argument("--barycenter", default=False, help="Returns the barycenter of the dataset")
parser.add_argument("--centroids", default=False, help="Returns the clusters centroids of the dataset")
parser.add_argument("--clusters_rank", default=False, help="Returns the ranking of the clusters by index quality criteria.")
parser.add_argument("--PCA", default=False, type=int, help="Returns the n-dimensionnal PCA data where n is the option of this argument.")
parser.add_argument("--UMAP", default=False, help="Returns the 2D UMAP reduction data.")
args = parser.parse_args()




try:
    pd_df=pd.read_csv(args.csv_location)
    if pd_df.columns.str.contains('Unnamed').any():
        pd_df = pd_df.drop(columns=pd_df.columns[pd_df.columns.str.contains('Unnamed')])
except FileNotFoundError:
    raise argparse.ArgumentTypeError('Invalid location : '+str(args.location)+". No such file in that location.") from None

CC=ClustersCharacteristics(pd_df,label_target=pd_df.columns[-1])

returned_dict_as_json={}
if args.index != False:
    returned_dict_as_json['index'] = CC.IndexCore_compute_every_index()
if args.ch_total != False:
    returned_dict_as_json['ch_total'] = CC.confusion_hypersphere_matrix(radius=args.ch_total, counting_type="including", proportion=False).to_dict()
if args.ch_proportion != False:
    returned_dict_as_json['ch_proportion'] = CC.confusion_hypersphere_matrix(radius=args.ch_proportion, counting_type="including", proportion=True).to_dict()
if args.twodim_density_PCA != False:
    returned_dict_as_json['twodim_density_PCA'] = CC.density_projection_2D("PCA", args.percentile_density, cluster=['all'], return_data=False, return_clusters_density=True)
if args.twodim_density_UMAP != False:
    returned_dict_as_json['twodim_density_UMAP'] = CC.density_projection_2D("UMAP", args.percentile_density, cluster=CC.labels_clusters, return_data=False, return_clusters_density=True)
if args.twodim_density_PCA != False:
    returned_dict_as_json['twodim_density_PCA'] = CC.density_projection_2D("PCA", args.percentile_density, cluster=CC.labels_clusters, return_data=False, return_clusters_density=True)
if args.threedim_density_PCA != False:
    returned_dict_as_json['threedim_density'] = CC.density_projection_3D(args.percentile_density, cluster=CC.labels_clusters, return_data=False, return_clusters_density=True)
if args.distances_to_centroids != False:
    returned_dict_as_json['distances_to_centroids'] = CC.data_every_element_distance_to_centroids().to_dict()
if args.intercentroid_distances != False:
    returned_dict_as_json['intercentroid_distances'] = CC.data_intercentroid_distance_matrix().to_dict()
if args.barycenter != False:
    returned_dict_as_json['barycenter'] = CC.data_barycenter().to_dict()
if args.centroids != False:
    returned_dict_as_json['centroids'] = CC.data_centroids().to_dict()
if args.clusters_rank != False:
    returned_dict_as_json['clusters_rank'] = CC.utils_ClustersRank().to_dict()
if args.PCA != False:
    returned_dict_as_json['PCA'] = CC.utils_PCA(args.PCA).to_dict()
if args.UMAP != False:
    returned_dict_as_json['UMAP'] = CC.utils_UMAP().to_dict()

import json


def key_to_json(data):
    if data is None or isinstance(data, (bool, int, str)):
        return data
    if isinstance(data, np.int64):
        return int(data)
    if isinstance(data, (tuple, frozenset)):
        return str(data)
    print(type(data))
    raise TypeError('1')

def to_json(data):
    if data is None or isinstance(data, (bool, int, tuple, range, str, list)):
        return data
    if isinstance(data, (set, frozenset)):
        return sorted(data)
    if isinstance(data, pd.DataFrame):
        return data.to_dict()
    if isinstance(data, np.ndarray):
        return {key_to_json(key): to_json(data[key]) for key in data}
    if isinstance(data, dict):
        return {key_to_json(key): to_json(data[key]) for key in data}

    print(type(data))
    raise TypeError('2')

s=to_json(returned_dict_as_json)

print(s)

"""
with open("./cli-output/cli_data.json", 'w', encoding='utf-8') as f:
    json.dump(s, f, ensure_ascii=False, indent=4)
"""
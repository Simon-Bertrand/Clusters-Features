#!/usr/bin/env python3
"""
This is the command line interface manager. To initialize correctly the dataset, you have to put a csv with the label target as the last column.
The csv number of columns should be : data dimension + 1 (the +1 is corresponding to the label target concatenated).
"""

import argparse
import numpy as np
import pandas as pd

from ClustersFeatures import ClustersCharacteristics

parser = argparse.ArgumentParser(description='CLI for ClustersFeatures Python package.')
parser.add_argument("csv_location", type=str, help="This argument needs to be the location of the specified csv. The label target have to be the last column of the csv.")
parser.add_argument("--index", default=False, action="store_true", help="Returns all indexes that are in the indices.json.")
parser.add_argument("--ch_total", default=False, type=float, metavar='RADIUS', help="Returns the confusion hypersphere with specified radius. The counting_type is 'including' and proportion is set to false.")
parser.add_argument("--ch_proportion", default=False, type=float, metavar='RADIUS', help="Returns the confusion hypersphere with specified radius. The counting_type is 'including' and proportion is set to true.")
parser.add_argument("--percentile_density", default=99, type=int, metavar='PERCENTILE', help="Set the minimum contour of the density as a percentile of the reducted data. Only used for twodim and threedim densities.")
parser.add_argument("--twodim_density_PCA", default=False, action='store_true', help="Returns the total 2D density and the density for each cluster. Uses the PCA Reduction method.")
parser.add_argument("--threedim_density_PCA", default=False, dest='threedim_density_PCA', action='store_true', help="Returns the total 3D density and the density for each cluster. Uses the PCA Reduction method.")
parser.add_argument("--twodim_density_UMAP", default=False, action='store_true', help="Returns the total 2D density and the density for each cluster. Uses the UMAP Reduction method.")
parser.add_argument("--distances_to_centroids", default=False, action='store_true', help="Returns the distance between each element and each cluster centroid.")
parser.add_argument("--intercentroid_distances", default=False, action='store_true', help="Returns the distance between each cluster centroid.")
parser.add_argument("--barycenter", default=False, action='store_true', help="Returns the barycenter of the dataset")
parser.add_argument("--centroids", default=False, action='store_true', help="Returns the clusters centroids of the dataset")
parser.add_argument("--clusters_rank", default=False,action='store_true', help="Returns the ranking of the clusters by index quality criteria.")
parser.add_argument("--PCA", default=False, type=int, metavar='N-DIM', help="Returns the n-dimensionnal PCA data where n is the option of this argument.")
parser.add_argument("--UMAP", default=False, action='store_true', help="Returns the 2D UMAP reduction data.")

parser.add_argument("--O_no_json", default=False, action='store_true', help="Disable the return of the output as a json file located at ./cli-output/cli_data.json.")
parser.add_argument("--O_text", default=False, action='store_true', help="Returns the output as a terminal cat.")

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
    returned_dict_as_json['threedim_density'] = CC.density_projection_3D(args.percentile_density, return_data=False, return_clusters_density=True)
if args.distances_to_centroids != False:
    returned_dict_as_json['distances_to_centroids'] = CC.data_every_element_distance_to_centroids.to_dict()
if args.intercentroid_distances != False:
    returned_dict_as_json['intercentroid_distances'] = CC.data_intercentroid_distance_matrix.to_dict()
if args.barycenter != False:
    returned_dict_as_json['barycenter'] = CC.data_barycenter.to_dict()
if args.centroids != False:
    returned_dict_as_json['centroids'] = CC.data_centroids.to_dict()
if args.clusters_rank != False:
    returned_dict_as_json['clusters_rank'] = CC.utils_ClustersRank().to_dict()
if args.PCA != False:
    returned_dict_as_json['PCA'] = CC.utils_PCA(args.PCA).to_dict()
if args.UMAP != False:
    returned_dict_as_json['UMAP'] = CC.utils_UMAP().to_dict()

import json


def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def recursive_dict(d):
    if d is None or isinstance(d, (bool, int, tuple, range, float)):
        return np.round(d,2)
    if isinstance(d, str):
        return d
    if isinstance(d, (np.ndarray,list)):
        return recursive_dict({i:v for i,v in enumerate(d)})
    if isinstance(d, pd.DataFrame):
        return recursive_dict(d.to_dict())
    if isinstance(d, dict):
        return {key_converter(key):recursive_dict(d[key]) for key in d}

def key_converter(x):
    if isinstance(x, (np.int64, np.int32)):
        return int(x)
    elif isinstance(x, str) and isfloat(x):
        return "{:10.2f}".format(float(x))
    elif isinstance(x, (np.float32, np.float64)):
        return np.round(float(x),2)
    else:
        return x



s=recursive_dict(returned_dict_as_json)
if args.O_no_json != False:
    with open("./cli-output/cli_data.json", 'w', encoding='utf-8') as f:
        json.dump(s, f, ensure_ascii=False, indent=4)
if args.O_text:
    print(json.dumps(s,ensure_ascii=False, indent=4))

print("~ Sucessfully computed. \n")
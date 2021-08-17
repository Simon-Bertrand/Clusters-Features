#!/usr/bin/env python3
"""
This is the command line interface manager. To initialize correctly the dataset, you have to put a csv with the label target as the last column.
The csv number of columns should be : data dimension + 1 (the +1 is corresponding to the label target concatenated).

This program is just a link between the instance of ClustersFeatures and the CLI.
We initialize a ClustersCharacteristics instance with the given csv and use the method of the main class
and then convert the python dict to a valid JSON format.


How to add an output to the CLI ?
First thing to do is to implement the python function in one of the many subclasses that are contained inside the ClustersFeatures/src folder.
Then :
    1. Add an argument to the parser following the argparse python library. For boolean arguments, just let empty the type and add the store_true string argument to the action.
    2. Create the link between the function and the cli command by adding to the returned dict the output of your function. The first each key of the returned dict should be
        the name of the command used to generate the output.
    3. Be sure that you're output is in the supported format of the functions recursive_dict and key_converter. These functions are usefull when you need to convert the value of the dict to
        a valid type. We have to convert each entry to a Python dict format to avoid JSON encoding errors.

"""
#Import the libraries needed for the CLI
import argparse
import numpy as np
import pandas as pd
import os
from ClustersFeatures import ClustersCharacteristics

#Create a parser with argparse (a native Python library used to fast create CLI)
parser = argparse.ArgumentParser(description='CLI for ClustersFeatures Python package.')

#Add all the arguments with their helper
parser.add_argument("csv_location", type=str, help="This argument needs to be the location of the specified csv. The label target have to be the last column of the csv.")
parser.add_argument("--index", default=False, action="store_true", help="Returns all indexes that are in the indices.json.")
parser.add_argument("--ch_total", default=False, type=float, metavar='RADIUS', help="Returns the confusion hypersphere with specified radius. The counting_type is 'including' and proportion is set to false.")
parser.add_argument("--ch_proportion", default=False, type=float, metavar='RADIUS', help="Returns the confusion hypersphere with specified radius. The counting_type is 'including' and proportion is set to true.")
parser.add_argument("--density_for_each_observation", default=False, action='store_true', help="Returns the estimated density for each observation of the dataset.")
parser.add_argument("--density_for_each_element_for_each_cluster", default=False, action='store_true', help="Returns the estimated density of every elements for each cluster of the dataset.")
parser.add_argument("--density_for_each_cluster", default=False, action='store_true', help="Returns the estimated density for each clusters.")
parser.add_argument("--percentile_density", default=99, type=int, metavar='PERCENTILE', help="Set the minimum contour of the density as a percentile of the reducted data. Only used for twodim and threedim densities.")
parser.add_argument("--twodim_density_PCA", default=False, action='store_true', help="Returns the total 2D PCA density and the density for each cluster.")
parser.add_argument("--threedim_density_PCA", default=False, dest='threedim_density_PCA', action='store_true', help="Returns the total 3D density and the density for each cluster. Uses the PCA Reduction method.")
parser.add_argument("--twodim_density_UMAP", default=False, action='store_true', help="Returns the total 2D density and the density for each cluster. Uses the UMAP Reduction method.")
parser.add_argument("--distances_to_centroids", default=False, action='store_true', help="Returns the distance between each element and each cluster centroid.")
parser.add_argument("--intercentroid_distances", default=False, action='store_true', help="Returns the distance between each cluster centroid.")
parser.add_argument("--barycenter", default=False, action='store_true', help="Returns the barycenter of the dataset")
parser.add_argument("--centroids", default=False, action='store_true', help="Returns the clusters centroids of the dataset")
parser.add_argument("--clusters_rank", default=False,action='store_true', help="Returns the ranking of the clusters by index quality criteria.")
parser.add_argument("--PCA", default=False, type=int, metavar='N-DIM', help="Returns the n-dimensionnal PCA data where n is the option of this argument.")
parser.add_argument("--UMAP", default=False, action='store_true', help="Returns the 2D UMAP reduction data.")

#These arguments are used to manage the output format. We can for example return the output directly to the terminal or use the JSON native parser to output a json file.
parser.add_argument("--O_no_json", default=True, action='store_true', help="Disable the return of the output as a json file located at ./cli-output/cli_data.json.")
parser.add_argument("--O_text", default=False, action='store_true', help="Returns the output as a terminal cat.")

#Parse all the above arguments following the argparse library.
args = parser.parse_args()

#Make some verifications for the Pandas DataFrame.
#By default, Pandas adds a column named "Unnamed" when importing a csv.file. We detect it and if it is detected then, we drop the concerned column.
#We try to cacth the error if the file is not located at the given position.
try:
    pd_df=pd.read_csv(args.csv_location)
    if pd_df.columns.str.contains('Unnamed').any():
        pd_df = pd_df.drop(columns=pd_df.columns[pd_df.columns.str.contains('Unnamed')])
except FileNotFoundError:
    raise argparse.ArgumentTypeError('Invalid location : '+str(args.location)+". No such file in that location.") from None

#Create the instance of the ClustersFeatures library with the target as the last column.
CC=ClustersCharacteristics(pd_df,label_target=pd_df.columns[-1])

#Create the empty to return dict
returned_dict_as_json={}
#And then, create the link between the CLI commands and the methods of ClustersCharacteristics class. We just add a key to the dict for each command used in the CLI.
#If you want to implement an already programmed function that is in the src folder, you can understand what the library does by reading its official documentation.
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
if args.density_for_each_observation != False:
    returned_dict_as_json['density_for_each_observation'] = CC.density_estimation("total").to_dict()
if args.density_for_each_element_for_each_cluster != False:
    returned_dict_as_json['density_for_each_element_for_each_cluster'] = CC.density_estimation("intra").to_dict()
if args.density_for_each_cluster != False:
    returned_dict_as_json['density_for_each_cluster'] = CC.density_estimation("inter").to_dict()
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


#Import JSON native parser
import json

#Create a function that detects if a value is a float.
def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

#Browse the entire dict with a recursive technique
#This function defines which format are supported by the CLI for the ClustersCharacteristics methods.
#Each supported format should be accompanied with a dict converter in order to apply the recursive_dict function again.
def recursive_dict(d):
    if d is None or isinstance(d, (bool, int, np.int64, np.int32, np.float64, np.float32, tuple, range, float)):
        return np.round(float(d),2)
    if isinstance(d, str):
        return d
    if isinstance(d, (np.ndarray,list)):
        return recursive_dict({i:v for i,v in enumerate(d)})
    if isinstance(d, pd.DataFrame):
        return recursive_dict(d.to_dict())
    if isinstance(d, dict):
        return {key_converter(key):recursive_dict(d[key]) for key in d}

#This function converts invalid formats for dict keys to valid ones.
def key_converter(x):
    if isinstance(x, (np.int64, np.int32)):
        return int(x)
    elif isinstance(x, str) and isfloat(x):
        return "{:10.2f}".format(float(x))
    elif isinstance(x, (np.float32, np.float64)):
        return np.round(float(x),2)
    else:
        return x

#Make the conversion to a dict format for each dict value.
s=recursive_dict(returned_dict_as_json)

#Manage the output type and write the output.
if args.O_no_json != False:
    if not os.path.exists("./cli-output/"):
        os.makedirs("./cli-output/")
    with open("./cli-output/cli_data.json", 'w', encoding='utf-8') as f:
        json.dump(s, f, ensure_ascii=False, indent=4)
if args.O_text:
    print(json.dumps(s,ensure_ascii=False, indent=4))

#Echo a message if everything has succeed.
print("~ Sucessfully computed. \n")
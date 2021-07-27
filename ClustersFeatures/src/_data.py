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

import numpy as np
import pandas as pd
import re


from ClustersFeatures import raising_errors
class Data:
    def data_intercentroid_distance(self, centroid_cluster_1, centroid_cluster_2):
        """
        Compute distances between centroid of CLuster1 and centroid of Cluster2.
        :param cluster_1: Cluster1 label
        :param cluster_2: Cluster2 label
        :return: float
        """
        raising_errors.both_clusters_in(centroid_cluster_1,centroid_cluster_2,self.labels_clusters)

        return np.linalg.norm(self.data_centroids[centroid_cluster_1] - self.data_centroids[centroid_cluster_2])

    def data_intercentroid_distance_matrix(self, **args):
        """
        Return a symetric matrix (xi,j)i,j
        where xi,j is the distance between centroids of cluster i and j
        :param args: target= (bool, optional) : Concatenate the output with the data target
        :return: pd.Dataframe().shape = (num_clusters,num_clusters)
        """
        # Symetric matrix generated : it's not usefull to compute the entire matrix while using symetric matrix properties
        Matopti = pd.DataFrame(columns=self.labels_clusters, index=self.labels_clusters).fillna(0)
        for centroid_cluster_1, centroid_cluster_2 in self.data_every_possible_cluster_pairs:
                Matopti.loc[centroid_cluster_1, centroid_cluster_2] = np.linalg.norm(
                    self.data_centroids[centroid_cluster_1] - self.data_centroids[centroid_cluster_2])
        Matopti += Matopti.T
        Matopti[np.eye(len(Matopti)) > 0] = np.nan
        try:
            if args['target']:
                Matopti['target'] = self.data_target
                return (Matopti)
        except KeyError:
            return (Matopti)

    def data_interelement_distance_between_elements_of_two_clusters(self, Cluster1, Cluster2):
        """
        Return every pairwise distance between elements belong Cluster1 or Cluster2
        If Cluster1 is equal to Cluster2, than these distances are inter-clusters and the output is symetric.
        Else, these are extra-clusters and the output is not symetric.
        :param Cluster1: (str,required) - Label cluster column name
        :param Cluster2: (str,required) - Label cluster column name
        :return: pd.DataFrame().shape=(num_observations_cluster1,num_observations_cluster2)
        """
        raising_errors.both_clusters_in(Cluster1,Cluster2,self.labels_clusters)

        return pd.DataFrame(self.data_every_element_distance_to_every_element).iloc[
            self.data_clusters[Cluster1].index, self.data_clusters[Cluster2].index]

    def data_interelement_distance_for_two_element(self, ElementId1, ElementId2):
        """
        Call the distance between Element1 and Element2
        :param ElementId1: First element pandas index
        :param ElementId2: Second element pandas index
        :return: float
        """
        raising_errors.both_element_in(ElementId1,ElementId2,self.data_features.index)

        return self.data_every_element_distance_to_every_element.loc[ElementId1, ElementId2]

    def data_interelement_distance_for_clusters(self, **args):
        """
        Return a dataframe with two columns. The first column is the distance for each element belonging
        clusters in the "clusters=" list argument. The second column is a boolean column equal to True
        when both elements are inside the same cluster. We use here the Pandas Multi-Indexes to allow users
        to link the column Distance with dataset points.
        :param args: clusters= (list or int, required) : labels of clusters to compute pairwise distances
        :return: pd.DataFrame().shape= (number_of_elements_pairs, 2)
        """
        clusters = raising_errors.list_clusters(args, self.labels_clusters)

        boolean_selector = pd.concat([1 * (self.data_target == cl) for cl in clusters], axis=1).sum(axis=1)
        distances = self.data_every_element_distance_to_every_element.loc[
            boolean_selector.astype(bool), boolean_selector.astype(bool)]

        index=pd.Index([(i1, i2) for i, i1 in enumerate(distances.index) for i2 in distances.index[i + 1:]])

        result=pd.DataFrame(distances.to_numpy()[np.tri(distances.shape[0],distances.shape[1],k=-1)>0], index=index, columns=pd.Index(['Distance']))
        result['Same Cluster ?'] = [True if self.data_target[i] == self.data_target[j] else False for (i, j), dist in
                                    result['Distance'].iteritems()]

        return result
    def data_interelement_distance_minimum_matrix(self):
        """
        Return interelement minimum
        :return:
        """
        Result = pd.DataFrame(np.zeros((self.num_clusters, self.num_clusters)), index=self.labels_clusters,
                              columns=self.labels_clusters)
        Result[np.eye(self.num_clusters) > 0] = np.nan
        for Cluster1,Cluster2 in self.data_every_possible_cluster_pairs:
            Result.loc[Cluster1, Cluster2] = self.data_interelement_distance_between_elements_of_two_clusters(Cluster1,
                                                                                                           Cluster2).min().min()
        return Result + Result.T


    def data_interelement_distance_maximum_matrix(self):
        Result = pd.DataFrame(np.zeros((self.num_clusters, self.num_clusters)), index=self.labels_clusters,
                              columns=self.labels_clusters)
        Result[np.eye(self.num_clusters) > 0] = np.nan
        for Cluster1,Cluster2 in self.data_every_possible_cluster_pairs:
            Result.loc[Cluster1, Cluster2] = self.data_interelement_distance_between_elements_of_two_clusters(Cluster1,
                                                                                                           Cluster2).max().max()
        return Result + Result.T

    def data_radius_selector_specific_cluster(self, Query, Cluster):
        raising_errors.cluster_in(Cluster, self.labels_clusters)

        regex_percentile = re.compile('([0-9]+)p')
        regex_percent = re.compile('([0-9]+)%')
        if Query in ["max", "mean", "min", "median"]:
            return self.data_radiuscentroid[Query][Cluster]
        elif isinstance(Query, float) or isinstance(Query, int):
            return Query
        elif isinstance(Query, str):
            if bool(regex_percentile.match(Query)):
                return np.percentile(self.data_every_cluster_element_distance_to_centroids[Cluster],
                                     int(regex_percentile.match(Query).group(1)))
            elif bool(regex_percent.match(Query)):
                return int(regex_percent.match(Query).group(1)) * self.data_radiuscentroid['max'][Cluster] / 100
            else:
                raise ValueError('Unknown query')
        else:
            return self.data_radius_selector_specific_cluster(self, "90p", Cluster)

    def data_same_target_for_pairs_elements_matrix(self):
        func= lambda x:(self.data_target[x.index] == self.data_target[x.name])
        return pd.DataFrame(np.zeros((self.num_observations,self.num_observations)), index=self.data_features.index, columns=self.data_features.index).apply(func)

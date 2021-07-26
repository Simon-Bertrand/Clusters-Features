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
    def data_every_distance_between_elements_of_two_clusters(self, Cluster1, Cluster2):
        raising_errors.both_clusters_in(Cluster1,Cluster2,self.labels_clusters)

        return pd.DataFrame(self.data_every_element_distance_to_every_element).iloc[
            self.data_clusters[Cluster1].index, self.data_clusters[Cluster2].index]


    def data_intercentroid_distance(self, centroid_cluster_1, centroid_cluster_2):
        raising_errors.both_clusters_in(centroid_cluster_1,centroid_cluster_2,self.labels_clusters)

        return np.linalg.norm(self.data_centroids[centroid_cluster_1] - self.data_centroids[centroid_cluster_2])



    def data_intercentroid_distance_matrix(self, **args):
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


    def data_interelement_distance(self, ElementId1, ElementId2):
        raising_errors.both_element_in(ElementId1,ElementId2,self.data_features.index)

        return self.data_every_element_distance_to_every_element.loc[ElementId1, ElementId2]


    def data_interelement_distance_minimum_for_different_clusters(self):
        Result = pd.DataFrame(np.zeros((self.num_clusters, self.num_clusters)), index=self.labels_clusters,
                              columns=self.labels_clusters)
        Result[np.eye(self.num_clusters) > 0] = np.nan
        for i, Cluster1 in enumerate(self.labels_clusters):
            for Cluster2 in self.labels_clusters[:i]:
                Result.loc[Cluster1, Cluster2] = self.data_every_distance_between_elements_of_two_clusters(Cluster1,
                                                                                                           Cluster2).min().min()
        return Result + Result.T


    def data_interelement_distance_maximum_for_different_clusters(self):
        Result = pd.DataFrame(np.zeros((self.num_clusters, self.num_clusters)), index=self.labels_clusters,
                              columns=self.labels_clusters)
        Result[np.eye(self.num_clusters) > 0] = np.nan
        for i, Cluster1 in enumerate(self.labels_clusters):
            for Cluster2 in self.labels_clusters[:i]:
                Result.loc[Cluster1, Cluster2] = self.data_every_distance_between_elements_of_two_clusters(Cluster1,
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

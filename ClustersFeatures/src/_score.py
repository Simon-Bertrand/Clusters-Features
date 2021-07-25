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
 Section: Score
   _____    _____    ____    _____    ______
 / ____|  / ____|  / __ \  |  __ \  |  ____|
| (___   | |      | |  | | | |__) | | |__
 \___ \  | |      | |  | | |  _  /  |  __|
 ____) | | |____  | |__| | | | \ \  | |____
|_____/   \_____|  \____/  |_|  \_\ |______|
 Compute every score used for the ClustersFeatures class.
 All these scores have been verified with the clusterCrit package on R developped by Bernard Desgraupes, using the load_digits datasets from scikit-learn

 scatter_matrix : return the total dispersion matrix (it is self.num_observations times the variance-covariance matrix of the dataset)
 scatter_matrix_specific_cluster_WGk : return the within cluster dispersion for a specific cluster (sum square distances between cluster's elements and the centroid of the concerned cluster)
 scatter_matrix_WG : return the sum of all WGk for k
 scatter_matrix_between_group_BG : return the matrix composed with the dispersion between centroids and the barycenter
 score_total_sum_square : Trace of scatter_matrix, we can compute it differently by using variance function
 score_within_cluster_dispersion : return the trace of the WGk matrix for a specific cluster. It's the same as score_total_sum_square but computed with WGk matrix' coefficients
 score_pooled_within_cluster_dispersion : return the sum of every score_within_cluster_dispersion (for each cluster)
 """
import numpy as np
import pandas as pd


class Score:
    def scatter_matrix_T(self):  #
        T = pd.DataFrame([self.data_features[col] - self.data_features[col].mean() for col in self.data_features.columns])
        return T.dot(T.T)


    def scatter_matrix_specific_cluster_WGk(self, Cluster):  #
        if not (Cluster in self.labels_clusters):
            raise AttributeError(
                'A such cluster name "' + Cluster + '" isn\'t found in dataframe\'s clusters. Here are the available clusters : ' + str(
                    list(self.labels_clusters)))
        else:
            X = pd.DataFrame([self.data_clusters[Cluster][col] - self.data_clusters[Cluster][col].mean() for col in
                              self.data_clusters[Cluster].columns])
            return X.dot(X.T)


    def scatter_matrix_WG(self):  #
        # WG is the total sum of the matrixes WGk for all k
        WG = pd.DataFrame(np.zeros((self.data_features.shape[1], self.data_features.shape[1])))
        for Cluster in self.labels_clusters:
            WG = np.add(WG, self.scatter_matrix_specific_cluster_WGk(Cluster))
        return WG


    def scatter_matrix_between_group_BG(self):  #
        B = pd.DataFrame()
        for Cluster in self.labels_clusters:
            B = B.append(np.sqrt(self.num_observation_for_specific_cluster[Cluster]) * (
                        self.data_centroids[Cluster] - self.data_barycenter), ignore_index=True)
        return B.T.dot(B)


    def score_totalsumsquare(self):
        # Can also be seen as the trace of the scatter matrix T, we use here the pandas var func for a faster compute
        return self.data_features.shape[0] * (self.data_features.var(ddof=0)).sum()


    def score_mean_quadratic_error(self):  #
        return self.score_pooled_within_cluster_dispersion() / self.num_observations


    def score_within_cluster_dispersion(self, Cluster):  #
        # Can also be seen as the trace of the scatter matrix WGk for a specific cluster, we use here the pandas var func for a faster compute
        if not (Cluster in self.labels_clusters):
            raise AttributeError(
                'A such cluster name "' + Cluster + '" isn\'t found in dataframe\'s columns. Here are the available columns : ' + str(
                    list(self.data_features.columns.values)))
        else:
            return self.data_clusters[Cluster].shape[0] * (self.data_clusters[Cluster].var(ddof=0, axis=0)).sum()


    def score_pooled_within_cluster_dispersion(self):  ##Returned value verified with R
        return np.sum([self.score_within_cluster_dispersion(Cluster) for Cluster in self.labels_clusters])


    def score_between_group_dispersion(self):  ##Returned value verified with R
        return np.trace(self.scatter_matrix_between_group_BG())


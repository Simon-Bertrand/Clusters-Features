# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd

class __Score:
    def scatter_matrix_T(self):
        """Returns the total dispersion matrix : it is self.num_observations times the variance-covariance matrix of the dataset.

        :returns: a Pandas dataframe. """

        T = pd.DataFrame([self.data_features[col] - self.data_features[col].mean() for col in self.data_features.columns])
        return T.dot(T.T)


    def scatter_matrix_specific_cluster_WGk(self, Cluster):
        """Returns the within cluster dispersion for a specific cluster (sum square distances between cluster's elements and the centroid of the concerned cluster).

        :param Cluster: Cluster label name.

        :returns: a Pandas dataframe. """

        if not (Cluster in self.labels_clusters):
            raise AttributeError(
                'A such cluster name "' + Cluster + '" isn\'t found in dataframe\'s clusters. Here are the available clusters : ' + str(
                    list(self.labels_clusters)))
        else:
            X = pd.DataFrame([self.data_clusters[Cluster][col] - self.data_clusters[Cluster][col].mean() for col in
                              self.data_clusters[Cluster].columns])
            return X.dot(X.T)


    def scatter_matrix_WG(self):
        """Returns the sum of scatter_matrix_specific_cluster_WGk for all k, it is also called as within group matrix.

        :returns: a Pandas dataframe. """

        WG = pd.DataFrame(np.zeros((self.data_features.shape[1], self.data_features.shape[1])))
        for Cluster in self.labels_clusters:
            WG = np.add(WG, self.scatter_matrix_specific_cluster_WGk(Cluster))
        return WG


    def scatter_matrix_between_group_BG(self):
        """Return the matrix composed with the dispersion between centroids and the barycenter.

        :returns: a Pandas dataframe. """
        B = pd.DataFrame()
        for Cluster in self.labels_clusters:
            B = B.append(np.sqrt(self.num_observation_for_specific_cluster[Cluster]) * (
                        self.data_centroids[Cluster] - self.data_barycenter), ignore_index=True)
        return B.T.dot(B)


    def score_totalsumsquare(self):
        """Trace of scatter_matrix_T, we can compute it differently by using variance function.

        :returns: float."""
        return self.data_features.shape[0] * (self.data_features.var(ddof=0)).sum()


    def score_mean_quadratic_error(self):
        """Mean quadratic error, also the same as score_pooled_within_cluster_dispersion / num_observations.

        :returns: float."""
        return self.score_pooled_within_cluster_dispersion() / self.num_observations


    def score_within_cluster_dispersion(self, Cluster):
        """Returns the trace of the WGk matrix for a specific cluster. It's the same as score_total_sum_square but computed with WGk matrix' coefficients.

        :param Cluster: Cluster label name.

        :returns: float."""
        if not (Cluster in self.labels_clusters):
            raise AttributeError(
                'A such cluster name "' + Cluster + '" isn\'t found in dataframe\'s columns. Here are the available columns : ' + str(
                    list(self.data_features.columns.values)))
        else:
            return self.data_clusters[Cluster].shape[0] * (self.data_clusters[Cluster].var(ddof=0, axis=0)).sum()


    def score_pooled_within_cluster_dispersion(self):
        """Returns the sum of score_within_cluster_dispersion for each cluster.

        :returns: float. """
        return np.sum([self.score_within_cluster_dispersion(Cluster) for Cluster in self.labels_clusters])


    def score_between_group_dispersion(self):
        """Returns the between group dispersion, can also be seen as the trace of the between group matrix.

        :returns: float. """
        return np.trace(self.scatter_matrix_between_group_BG())


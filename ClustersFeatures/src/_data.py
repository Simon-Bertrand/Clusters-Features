# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import re


from ClustersFeatures import raising_errors
class __Data:
    """The ClustersCharacteristics object creates attributes that define clusters. We can find them in the Data subclass.
    To use these methods, you need to initialize a ClusterCharacteristics instance and then write the corresponding methods:

    For example:

    >>> CC=ClustersCharacteristics(pd_df,"target")
    >>> CC.data_intercentroid_distance_matrix()
    """

    def data_intercentroid_distance(self, Cluster1, Cluster2):
        """Computes distances between centroid of Cluster1 and centroid of Cluster2.

        :param Cluster1: Cluster1 label name
        :param Cluster2: Cluster2 label name
        :return: float

        >>> CC.data_intercentroid_distance(CC.labels_clusters[0], CC.labels_clusters[1])

        """
        raising_errors.both_clusters_in(Cluster1,Cluster2,self.labels_clusters)

        return np.linalg.norm(self.data_centroids[Cluster1] - self.data_centroids[Cluster2])

    def data_intercentroid_distance_matrix(self, **args):
        """Computes the distance between one centroid and another and return the matrix of this general term

        Return a symetric matrix (xi,j)i,j where xi,j is the distance between centroids of cluster i and j

        :param bool target=: Concatenate the output with the data target

        :return: A symetric pandas dataframe with the computed distances between each centroid

        >>> CC.data_intercentroid_distance_matrix()
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
        """Returns every pairwise distance between elements belonging Cluster1 or Cluster2

        If Cluster1 is equal to Cluster2, than these distances are inter-clusters and the output is symetric.
        Else, these are extra-clusters and the output is not symetric.

        :param Cluster1: Label cluster column name
        :param Cluster2: Label cluster column name
        :return: A pandas dataframe with the given clusters pairwise elements distance

        >>> CC.data_interelement_distance_between_elements_of_two_clusters(CC.labels_clusters[0], CC.labels_clusters[1])
        """
        raising_errors.both_clusters_in(Cluster1,Cluster2,self.labels_clusters)

        return pd.DataFrame(self.data_every_element_distance_to_every_element).iloc[
            self.data_clusters[Cluster1].index, self.data_clusters[Cluster2].index]

    def data_interelement_distance_for_two_element(self, ElementId1, ElementId2):
        """Calls the distance between Element1 and Element2

        :param ElementId1: First element pandas index
        :param ElementId2: Second element pandas index
        :return: float

        >>> CC.data_interelement_distance_for_two_element(CC.data_features.index[0],CC.data_features.index[1])
        """
        raising_errors.both_element_in(ElementId1,ElementId2,self.data_features.index)

        return self.data_every_element_distance_to_every_element.loc[ElementId1, ElementId2]

    def data_interelement_distance_for_clusters(self, **args):
        """Returns a dataframe with two columns. The first column is the distance for each element belonging
        clusters in the "clusters=" list argument. The second column is a boolean column equal to True
        when both elements are inside the same cluster. We use here the Pandas Multi-Indexes to allow users
        to link the column Distance with dataset points.

        :param clusters=: labels of clusters to compute pairwise distances
        :return: A pandas dataframe with two columns : one for the distance and the other named 'Same Cluster ?' is equal to True if both elements belong the same cluster


        Computing all the distances between the 3 first clusters of the dataframe

        >>> CC.data_interelement_distance_for_clusters(clusters=CC.labels_clusters[0:3])
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


    def data_radius_selector_specific_cluster(self, Query, Cluster):
        """ Returns the radius of one given cluster with different query.

        :param str Query: in the list ['max', 'min', 'median', 'mean'] or "XXp" for the XXth radius percentile or "XX%" for a percentage of the max radius.
        :param Cluster: The cluster label
        :return: a float.
        """
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
        """Returns a boolean matrix where the general term is equal to True when the index elements belong the same cluster with the column element

        :return: A boolean pandas dataframe with shape (num_observations,num_observations)

        >>> CC.data_same_target_for_pairs_elements_matrix()
        """
        df=pd.DataFrame(np.zeros((self.num_observations, self.num_observations)), index=self.data_features.index, columns=self.data_features.index)
        for Cluster in self.labels_clusters:
            df.loc[self.data_clusters[Cluster].index, self.data_clusters[Cluster].index] = np.ones(
                (self.num_observation_for_specific_cluster[Cluster], self.num_observation_for_specific_cluster[Cluster]))
        return df


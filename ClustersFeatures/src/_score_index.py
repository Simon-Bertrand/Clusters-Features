# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from ClustersFeatures import settings
class __ScoreIndex:
    def score_index_ball_hall(self):
        """Returns the Ball Hall index defined in the first reference.

        :returns: float.
        """
        return np.mean(
            [self.score_within_cluster_dispersion(Cluster) / self.num_observation_for_specific_cluster[Cluster] for Cluster
             in self.labels_clusters])


    def score_index_banfeld_Raftery(self):
        """Defined in the first reference.

        :returns: float.
        """
        if (np.array([len(self.data_clusters[Cluster]) for Cluster in self.labels_clusters]) == np.array(
                len(self.labels_clusters) * [1])).sum() != 0:
            raise ValueError(
                "One cluster has a single point, which causes an impossibility to compute Banfeld Raftery Index")
        else:
            return np.sum([self.num_observation_for_specific_cluster[Cluster] * np.log(
                self.score_within_cluster_dispersion(Cluster) / self.num_observation_for_specific_cluster[Cluster]) for
                           Cluster in self.labels_clusters])


    def score_index_davies_bouldin_for_each_cluster(self):
        """Defined in the first reference.

        :returns: np.array of davies bouldin score for each clusters.
        """
        eedc_mat = self.data_every_element_distance_to_centroids
        delta = np.array(
            [eedc_mat.iloc[self.data_clusters[Cluster1].index, Cluster1].mean() for Cluster1 in self.labels_clusters])
        return np.array([np.max(
            [(delta[Cluster1] + delta[Cluster2]) / self.data_intercentroid_distance(Cluster1, Cluster2) for Cluster2 in
             np.delete(self.labels_clusters, Cluster1)]) for Cluster1 in self.labels_clusters])


    def score_index_davies_bouldin(self):
        """Defined in the first reference.

        It is the mean of score_index_davies_bouldin_for_each_cluster.

        :returns: float.
        """
        return self.score_index_davies_bouldin_for_each_cluster().mean()


    def score_index_calinski_harabasz(self):
        """Defined in the first reference.

        :returns: float.
        """
        K = self.num_clusters
        WGSS_red = self.score_pooled_within_cluster_dispersion() / (self.num_observations - K)
        BGSS_red = self.score_between_group_dispersion() / (K - 1)
        return BGSS_red / WGSS_red


    def score_index_det_ratio(self):
        """Defined in the first reference.

        Returns NaN value when the WG matrix or Total Scatter Matrix is not invertible.
        :returns: float.
        """
        Mat_WG = self.scatter_matrix_WG()
        det_Mat_WG = np.linalg.det(Mat_WG)
        det_T = np.linalg.det(self.scatter_matrix_T())
        if det_Mat_WG == 0 or det_T == 0:
            return np.nan
        else:
            return np.linalg.det(self.scatter_matrix_T()) / np.linalg.det(Mat_WG)


    def score_index_log_ss_ratio(self):
        """Defined in the first reference.

        :returns: float.
        """
        WGSS = self.score_pooled_within_cluster_dispersion()
        return np.log(self.score_between_group_dispersion() / WGSS)


    def score_index_dunn(self):
        """Defined in the first reference.

        :returns: float.
        """
        dmin = self.data_interelement_distance_minimum_matrix.min().min()
        Dmax = np.max(
            [self.data_interelement_distance_between_elements_of_two_clusters(Cluster, Cluster).max().max() for Cluster in
             self.labels_clusters])
        return dmin / Dmax


    def score_index_silhouette(self):  ##Returned value verified with R
        """Using the scikit-learn library to fast compute Silhouette score.

        :returns: float.
        """
        return (self.score_index_silhouette_matrix['Silhouette Score'].mean())


    def score_index_silhouette_for_every_cluster(self):
        """Using the scikit-learn library to fast compute the mean for each cluster of the silhouette score.

        :returns: A pandas Series with silhouette score for each cluster.
        """
        return self.score_index_silhouette_matrix.groupby(by='Cluster').mean()


    def score_index_c(self):
        """Defined in the first reference.

        :returns: float.
        """
        pair_of_points = lambda x: x * (x - 1) / 2

        NW = np.sum(
            [pair_of_points(self.num_observation_for_specific_cluster[Cluster]) for Cluster in self.labels_clusters])
        distances_list_ordered = np.sort(self.data_every_element_distance_to_every_element.to_numpy()[
                                             np.tri(self.num_observations, self.num_observations, k=-1) > 0])

        Smin = np.sum(distances_list_ordered[:int(NW)])
        Smax = np.sum(distances_list_ordered[-int(NW):])
        SW = np.sum([self.data_interelement_distance_between_elements_of_two_clusters(Cluster, Cluster).to_numpy()[
            np.tri(self.num_observation_for_specific_cluster[Cluster], self.num_observation_for_specific_cluster[Cluster],
                   k=-1) > 0].sum().sum() for Cluster in self.labels_clusters])
        return (SW - Smin) / (Smax - Smin)


    def score_index_c_for_each_cluster(self, Cluster):
        """A variant of C Index for each cluster. The main difference is that we do not take the sum of all pairs of point but we directly take the number of pairs for the given cluster.

        :param Cluster: Cluster label name.
        :returns: float."""
        pair_of_points = lambda x: x * (x - 1) / 2

        Nw = pair_of_points(self.num_observation_for_specific_cluster[Cluster])
        distances_list_ordered = np.sort(self.data_every_element_distance_to_every_element.to_numpy()[
                                             np.tri(self.num_observations, self.num_observations, k=-1) > 0])

        Smin = np.sum(distances_list_ordered[:int(Nw)])
        Smax = np.sum(distances_list_ordered[-int(Nw):])
        SW = self.data_interelement_distance_between_elements_of_two_clusters(Cluster, Cluster).to_numpy()[
            np.tri(self.num_observation_for_specific_cluster[Cluster], self.num_observation_for_specific_cluster[Cluster],
                   k=-1) > 0].sum().sum()
        return (SW - Smin) / (Smax - Smin)


    def score_index_ray_turi(self):
        """Defined in the first reference.

        :returns: float.
        """
        return (1 / self.num_observations) * self.score_pooled_within_cluster_dispersion() / (
                    self.data_intercentroid_distance_matrix().where(
                        np.tri(self.num_clusters, self.num_clusters, k=-1) > 0) ** 2).min().min()


    def score_index_xie_beni(self):
        """Defined in the first reference.

        :returns: float.
        """
        min_distances = [self.data_interelement_distance_between_elements_of_two_clusters(Cluster1, Cluster2).min().min() ** 2 for
                         Cluster1, Cluster2 in self.data_every_possible_cluster_pairs]
        return (1 / self.num_observations) * self.score_pooled_within_cluster_dispersion() / np.min(min_distances)


    def score_index_trace_WiB(self):
        """Defined in the first reference.

        Returns NaN if WG matrix is not invertible.

        :returns: float.
        """
        try:
            return np.trace(np.linalg.inv(self.scatter_matrix_WG()).dot(self.scatter_matrix_between_group_BG()))
        except np.linalg.LinAlgError:
            return np.nan


    def score_index_wemmert_gancarski(self):
        """A special thanks to M.Gançarski who recruited me for my first traineeship at iCube, Strasbourg, its index have been implemented here:

        Defined in the first reference.

        :returns: float.
        """
        Jk_weighted = []
        for Cluster in self.labels_clusters:
            S = self.data_every_cluster_element_distance_to_centroids[Cluster]
            Jk_weighted.append(self.num_observation_for_specific_cluster[Cluster] * np.max([0, 1 - (
                        S / self.data_every_element_distance_to_centroids.drop(columns=Cluster).min(axis=1)[
                    S.index]).mean()]))
        return np.sum(Jk_weighted) / self.num_observations


    def score_index_PBM(self):
        """Defined in the first reference.

        :returns: float.
        """
        EW = np.sum(
            [self.data_every_cluster_element_distance_to_centroids[Cluster].sum() for Cluster in self.labels_clusters])
        ET = np.sqrt(((self.data_features - self.data_barycenter) ** 2).sum(axis=1)).sum()
        return (ET * self.data_intercentroid_distance_matrix().max().max() / EW / self.num_clusters) ** 2


    def score_index_generalized_dunn(self, **args):
        """Returns one of the 18 generalized dunn indices.

         :param int wc_distance: within cluster indice according to main reference. Int included in [1,2,3].
         :param int bc_distance: between cluster indice according to main reference. Int included in [1,2,3,4,5,6].

        :returns: float."""
        try:
            wc_distance = args['within_cluster_distance']
            bc_distance = args['between_cluster_distance']
        except:
            raise ValueError('within_cluster_distance or between_cluster_distance argument is not specified.')

        if wc_distance == 1:
            list_denominator = [self.data_interelement_distance_between_elements_of_two_clusters(Cluster, Cluster).max().max() for
                                Cluster in self.labels_clusters]
        elif wc_distance == 2:
            list_denominator = [
                self.data_interelement_distance_between_elements_of_two_clusters(Cluster, Cluster).sum().sum() / 2 /
                self.num_observation_for_specific_cluster[Cluster] / (
                            self.num_observation_for_specific_cluster[Cluster] - 1) for Cluster in self.labels_clusters]
        elif wc_distance == 3:
            list_denominator = [self.data_every_cluster_element_distance_to_centroids[Cluster].sum() * 2 /
                                self.num_observation_for_specific_cluster[Cluster] for Cluster in self.labels_clusters]
        else:
            raise ValueError('within_cluster_distance option isn\'t in the following list : [1,2,3]')

        if bc_distance == 1:
            numerator = self.data_interelement_distance_minimum_matrix.min().min()
        elif bc_distance == 2:
            numerator = self.data_interelement_distance_minimum_matrix.min().min()
        elif bc_distance == 3:
            numerator = np.min([self.data_interelement_distance_between_elements_of_two_clusters(Cluster1, Cluster2).sum().sum() /
                                self.num_observation_for_specific_cluster[Cluster2] /
                                self.num_observation_for_specific_cluster[Cluster1] for Cluster1, Cluster2 in
                                self.data_every_possible_cluster_pairs])
        elif bc_distance == 4:
            numerator = self.data_intercentroid_distance_matrix().to_numpy()[
                np.tri(self.num_clusters, self.num_clusters, k=-1) > 0].min()
        elif bc_distance == 5:
            numerator = np.min([np.append(self.data_every_cluster_element_distance_to_centroids[Cluster1].values,
                                          self.data_every_cluster_element_distance_to_centroids[Cluster2].values).mean() for
                                Cluster1, Cluster2 in self.data_every_possible_cluster_pairs])
        elif bc_distance == 6:
            numerator = np.min([np.max([pd.DataFrame(
                self.data_interelement_distance_between_elements_of_two_clusters(Cluster1, Cluster2)).min(axis=1).max(),
                                        pd.DataFrame(self.data_interelement_distance_between_elements_of_two_clusters(Cluster1,
                                                                                                               Cluster2)).min(
                                            axis=0).max()]) for Cluster1, Cluster2 in
                                self.data_every_possible_cluster_pairs])
        else:
            raise ValueError('between_cluster_distance option isn\'t in the following list : [1,2,3,4,5,6]')
        return numerator / np.max(list_denominator)


    def score_index_generalized_dunn_matrix(self):
        """Returns the 18 generalized dunn indices defined in the first reference.

        :returns: A pandas dataframe with shape (6,3)."""
        bc_list = np.arange(1, 7)
        wc_list = np.arange(1, 4)

        df = pd.DataFrame(columns=wc_list, index=bc_list)
        for bc in bc_list:
            for wc in wc_list:
                df.loc[bc, wc] = self.score_index_generalized_dunn(within_cluster_distance=wc, between_cluster_distance=bc)
        df.index.name = "Generalized Dunn Indexes"
        return df


    def score_index_ratkowsky_lance(self):
        """ Defined in the first reference.

        :returns: float.
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.true_divide(np.diag(self.scatter_matrix_between_group_BG()),
                                  self.num_observations * self.data_features.var(ddof=0).to_numpy()).mean()


    def score_index_SD(self):
        """ Defined in the first reference.

        Since we haven't different numbers of cluster, we can't compute the weighting coefficient :
        We will pass the average scattering for clusters and the total separation between clusters as the returned tuple.

        :returns: A tuple of float that are (Scattering, Separation).
        """

        Vk = [self.data_clusters[Cluster].var(ddof=0).to_numpy() for Cluster in self.labels_clusters]
        return (np.mean([np.linalg.norm(v) for v in Vk]) / np.linalg.norm(
            self.data_features.var(ddof=0).to_numpy()), np.sum(
            [1 / self.data_intercentroid_distance_matrix()[Cluster].sum(skipna=True) for Cluster in
             self.labels_clusters]) * self.data_intercentroid_distance_matrix().max().max() / self.data_intercentroid_distance_matrix().min().min())


    def score_index_S_Dbw(self):
        """ Defined in the first reference.

        :returns: float.
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            Vk = [np.linalg.norm(self.data_clusters[Cluster].var(ddof=0)) for Cluster in self.labels_clusters]
            std_dev = np.mean(np.sqrt(Vk))
            Rkk = []
            for Cluster1, Cluster2 in self.data_every_possible_cluster_pairs:
                mid_point = (self.data_centroids[Cluster1] + self.data_centroids[Cluster2]) / 2
                ykk_Hkk = self.confusion_hyperphere_around_specific_point_for_two_clusters(mid_point, Cluster1, Cluster2,
                                                                                           std_dev)
                ykk_Gk = self.confusion_hyperphere_around_specific_point_for_two_clusters(self.data_centroids[Cluster1],
                                                                                          Cluster1, Cluster2, std_dev)
                y_kk_Gkp = self.confusion_hyperphere_around_specific_point_for_two_clusters(self.data_centroids[Cluster2],
                                                                                            Cluster1, Cluster2, std_dev)
                Rkk.append(ykk_Hkk / np.max([ykk_Gk, y_kk_Gkp]))
            S = np.mean(Vk) / np.linalg.norm(self.data_features.var(ddof=0).to_numpy())
            return np.mean(Rkk) + S

    def score_index_Log_Det_ratio(self):
        """ Defined in the first reference.

        Returns NaN value when the WG matrix or Total Scatter Matrix is not invertible.
        :returns: float.
        """
        return self.num_observations * np.log(self.score_index_det_ratio())

    def score_index_mclain_rao(self):
        """ Defined in the first reference.

        :returns: float
        """
        same_target_matrix=self.data_same_target_for_pairs_elements_matrix()
        pair_of_points = lambda x: x * (x - 1) / 2
        NW = np.sum([pair_of_points(self.num_observation_for_specific_cluster[Cluster]) for Cluster in self.labels_clusters])
        NB= pair_of_points(self.num_observations) - NW
        #There is a unwanted 2 factor for each following SW and SB scores. As a division is made after, we can nevermind that
        SW=((same_target_matrix) * self.data_every_element_distance_to_every_element).sum().sum()
        SB= ((1-same_target_matrix)*self.data_every_element_distance_to_every_element).sum().sum()
        return NB*SW/(NW*SB)

    def score_index_point_biserial(self):
        """ Defined in the first reference.

        :returns: float.
        """
        same_target_matrix=self.data_same_target_for_pairs_elements_matrix()
        pair_of_points = lambda x: x * (x - 1) / 2
        NW = np.sum([pair_of_points(self.num_observation_for_specific_cluster[Cluster]) for Cluster in self.labels_clusters])
        NB= pair_of_points(self.num_observations) - NW
        #There is a unwanted 2 factor for each following SW and SB scores. We need to divide by two the final result
        SW=((same_target_matrix) * self.data_every_element_distance_to_every_element).sum().sum()
        SB= ((1-same_target_matrix)*self.data_every_element_distance_to_every_element).sum().sum()

        return (SW/NW - SB/NB)*np.sqrt(NB*NW)/pair_of_points(self.num_observations)/2

    def score_index_scott_symons(self):
        """ Defined in the first reference.

        Returns NaN if one of the WGk matrix is not inversible.

        :returns: float.
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            dets=np.array([np.linalg.det(self.scatter_matrix_specific_cluster_WGk(Cluster)/self.num_observation_for_specific_cluster[Cluster]) for Cluster in self.labels_clusters])
            if (dets!=0).all():
                 return (np.array([self.num_observation_for_specific_cluster[Cluster] for Cluster in self.labels_clusters]) * np.log(dets)).sum()

            else:
                return np.nan

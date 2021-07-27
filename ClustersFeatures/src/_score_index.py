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
 Section: Score Index
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
 score_index_ball_hall : Return the Ball Hall index defined in the first reference
 score_index_banfeld_Raftery : Defined in the first reference
 score_index_davies_bouldin_for_each_cluster : Defined in the first reference
 score_index_davies_bouldin : Defined in the first reference, it is the mean of score_index_davies_bouldin_for_each_cluster
 score_index_calinski_harabasz : Defined in the first reference
 score_index_det_ratio : Defined in the first reference
 score_index_log_ss_ratio : Defined in the first reference
 score_index_dunn : Defined in the first reference
 score_index_silhouette : Uses the sklearn.metrics library to fast compute the score
 score_index_silhouette_fror_every_cluster : return a list of meaned silhouette score for each cluster
 score_index_c: Defined in the first reference
 score_index_c_for_each_cluster : The same as before but we take the sum of the self.number_observations_for_specific_cluster(Cluster) elements, not the total sum.
 score_index_ray_turi : Defined in the first reference
 score_index_xie_beni : quite the same as Ray-Turi but the denominator isn`t the same intercluster distance
 score_index_PBM : Defined in the first reference
 score_index_generalized_dunn_matrix: Return the 18 indices defined in the first reference

 A special thanks to M.Gançarski who recruited me for my first traineeship at iCube, Strasbourg, its index have been implemented here:
 score_index_wemmert_gancarski : Defined in the first reference

 Reference :
 Clustering Indice - Bernard Desgraupes (University Paris Ouest, Lab Modal’X) - 2017
 https://cran.r-project.org/web/packages/clusterCrit/vignettes/clusterCrit.pdf

 Study on Different Cluster Validity Indices - Shyam Kumar K, Dr. Raju G (NSS College Rajakumari, Idukki & Kannur University, Kannur in Kerala, India) - 2018
 https://www.ripublication.com/ijaer18/ijaerv13n11_86.pdf

 Understanding of Internal Clustering Validation Measures - Yanchi Liu, Zhongmou Li, Hui Xiong, Xuedong Gao, Junjie Wu - 2010
 http://datamining.rutgers.edu/publication/internalmeasures.pdf

 """


import numpy as np
import pandas as pd
from ClustersFeatures import settings
class ScoreIndex:
    def score_index_rules(self, Query):
        if Query == "max":
            return settings.indices_max
        elif Query == "min":
            return settings.indices_min
        elif Query == 'max diff':
            return settings.indices_max_diff
        elif Query == 'min diff':
            return settings.indices_min_diff
        else:
            raise ValueError('Unknown query')


    def score_index_info(self, Query):
        clusters_info = self.general_info.loc[pd.Index(self.score_index_rules(Query))]
        if Query == "max":
            GDI = pd.DataFrame(self.score_index_generalized_dunn_matrix().stack(), columns=clusters_info.columns).rename(
                index={(i, j): f"GDI ({i},{j})" for j in range(1, 4) for i in range(1, 7)})
            clusters_info = pd.concat([clusters_info, GDI])
        return clusters_info

    def score_index_ball_hall(self):  ##Returned value verified with R
        return np.mean(
            [self.score_within_cluster_dispersion(Cluster) / self.num_observation_for_specific_cluster[Cluster] for Cluster
             in self.labels_clusters])


    def score_index_banfeld_Raftery(self):  ##Returned value verified with R
        if (np.array([len(self.data_clusters[Cluster]) for Cluster in self.labels_clusters]) == np.array(
                len(self.labels_clusters) * [1])).sum() != 0:
            raise ValueError(
                "One cluster has a single point, which causes an impossibility to compute Banfeld Raftery Index")
        else:
            return np.sum([self.num_observation_for_specific_cluster[Cluster] * np.log(
                self.score_within_cluster_dispersion(Cluster) / self.num_observation_for_specific_cluster[Cluster]) for
                           Cluster in self.labels_clusters])


    def score_index_davies_bouldin_for_each_cluster(self):
        eedc_mat = self.data_every_element_distance_to_centroids
        delta = np.array(
            [eedc_mat.iloc[self.data_clusters[Cluster1].index, Cluster1].mean() for Cluster1 in self.labels_clusters])
        return np.array([np.max(
            [(delta[Cluster1] + delta[Cluster2]) / self.data_intercentroid_distance(Cluster1, Cluster2) for Cluster2 in
             np.delete(self.labels_clusters, Cluster1)]) for Cluster1 in self.labels_clusters])


    def score_index_davies_bouldin(self):  ##Returned value verified with R
        return self.score_index_davies_bouldin_for_each_cluster().mean()


    def score_index_calinski_harabasz(self):  ##Returned value verified with R
        K = self.num_clusters
        WGSS_red = self.score_pooled_within_cluster_dispersion() / (self.num_observations - K)
        BGSS_red = self.score_between_group_dispersion() / (K - 1)
        return BGSS_red / WGSS_red


    def score_index_det_ratio(self):  ##Returned value verified with R
        Mat_WG = self.scatter_matrix_WG()
        det_Mat_WG = np.linalg.det(Mat_WG)
        det_T = np.linalg.det(self.scatter_matrix_T())
        if det_Mat_WG == 0 or det_T == 0:
            return np.nan
        else:
            return np.linalg.det(self.scatter_matrix_T()) / np.linalg.det(Mat_WG)


    def score_index_log_ss_ratio(self):  ##Returned value verified with R
        WGSS = self.score_pooled_within_cluster_dispersion()
        return np.log(self.score_between_group_dispersion() / WGSS)


    def score_index_dunn(self):  ##Returned value verified with R
        dmin = self.data_interelement_distance_minimum_matrix().min().min()
        Dmax = np.max(
            [self.data_interelement_distance_between_elements_of_two_clusters(Cluster, Cluster).max().max() for Cluster in
             self.labels_clusters])
        return dmin / Dmax


    def score_index_silhouette(self):  ##Returned value verified with R
        """This is the result of this algorithm :

        element_cluster_label=self.data_target.iloc[Observation]
        a_tab=self.data_interelement_distance_between_elements_of_two_clusters(element_cluster_label,element_cluster_label).loc[Observation]
        a=np.sum(a_tab)/(len(a_tab)-1)
        b=np.min([self.data_interelement_distance_between_elements_of_two_clusters(element_cluster_label,Cluster2).loc[Observation].mean() for Cluster2 in np.delete(self.labels_clusters, element_cluster_label)])
        return np.round((b-a)/max(a,b),settings.precision)

        but it's pretty slow so we will prefer use the scikit-learn library to compute the silhouette score
        """
        return (self.score_index_silhouette_matrix['Silhouette Score'].mean())


    def score_index_silhouette_for_every_cluster(self):
        return self.score_index_silhouette_matrix.groupby(by='Cluster').mean()


    def score_index_c(self):  ##Returned value verified with R
        pair_of_points = lambda x: x * (x - 1) / 2

        NW = np.sum(
            [pair_of_points(self.num_observation_for_specific_cluster[Cluster]) for Cluster in self.labels_clusters])
        distances_list_ordered = np.sort(self.data_every_element_distance_to_every_element.to_numpy()[
                                             np.tri(self.num_observations, self.num_observations, k=-1) > 0])

        Smin = np.sum(distances_list_ordered[:int(NW)])
        Smax = np.sum(distances_list_ordered[-int(NW):])
        SW = np.sum([self.data_interelement_distance_between_elements_of_two_clusters(Cluster, Cluster).where(
            np.tri(self.num_observation_for_specific_cluster[Cluster], self.num_observation_for_specific_cluster[Cluster],
                   k=-1) > 0).sum(skipna=True).sum() for Cluster in self.labels_clusters])
        return (SW - Smin) / (Smax - Smin)


    def score_index_c_for_each_cluster(self, Cluster):
        pair_of_points = lambda x: x * (x - 1) / 2

        Nw = pair_of_points(self.num_observation_for_specific_cluster[Cluster])
        distances_list_ordered = np.sort(self.data_every_element_distance_to_every_element.to_numpy()[
                                             np.tri(self.num_observations, self.num_observations, k=-1) > 0])

        Smin = np.sum(distances_list_ordered[:int(Nw)])
        Smax = np.sum(distances_list_ordered[-int(Nw):])
        SW = self.data_interelement_distance_between_elements_of_two_clusters(Cluster, Cluster).where(
            np.tri(self.num_observation_for_specific_cluster[Cluster], self.num_observation_for_specific_cluster[Cluster],
                   k=-1) > 0).sum(skipna=True).sum()
        return (SW - Smin) / (Smax - Smin)


    def score_index_ray_turi(self):  ##Returned value verified with R
        return (1 / self.num_observations) * self.score_pooled_within_cluster_dispersion() / (
                    self.data_intercentroid_distance_matrix().where(
                        np.tri(self.num_clusters, self.num_clusters, k=-1) > 0) ** 2).min().min()


    def score_index_xie_beni(self):  ##Returned value verified with R
        min_distances = [self.data_interelement_distance_between_elements_of_two_clusters(Cluster1, Cluster2).min().min() ** 2 for
                         Cluster1, Cluster2 in self.data_every_possible_cluster_pairs]
        return (1 / self.num_observations) * self.score_pooled_within_cluster_dispersion() / np.min(min_distances)


    def score_index_trace_WiB(self):  ##Returned value verified with R
        try:
            return np.trace(np.linalg.inv(self.scatter_matrix_WG()).dot(self.scatter_matrix_between_group_BG()))
        except np.linalg.LinAlgError:
            return np.nan


    def score_index_wemmert_gancarski(self):  ##Returned value verified with R
        Jk_weighted = []
        for Cluster in self.labels_clusters:
            S = self.data_every_cluster_element_distance_to_centroids[Cluster]
            Jk_weighted.append(self.num_observation_for_specific_cluster[Cluster] * np.max([0, 1 - (
                        S / self.data_every_element_distance_to_centroids.drop(columns=Cluster).min(axis=1)[
                    S.index]).mean()]))
        return np.sum(Jk_weighted) / self.num_observations


    def score_index_PBM(self):  ##Returned value verified with R
        EW = np.sum(
            [self.data_every_cluster_element_distance_to_centroids[Cluster].sum() for Cluster in self.labels_clusters])
        ET = np.sqrt(((self.data_features - self.data_barycenter) ** 2).sum(axis=1)).sum()
        return (ET * self.data_intercentroid_distance_matrix().max().max() / EW / self.num_clusters) ** 2


    def score_index_generalized_dunn(self, **args):  ##Returned value verified with R
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
            numerator = self.data_interelement_distance_minimum_for_different_clusters().min().min()
        elif bc_distance == 2:
            numerator = self.data_interelement_distance_maximum_for_different_clusters().min().min()
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
        bc_list = np.arange(1, 7)
        wc_list = np.arange(1, 4)

        df = pd.DataFrame(columns=wc_list, index=bc_list)
        for bc in bc_list:
            for wc in wc_list:
                df.loc[bc, wc] = self.score_index_generalized_dunn(within_cluster_distance=wc, between_cluster_distance=bc)
        df.index.name = "Generalized Dunn Indexes"
        return df


    def score_index_ratkowsky_lance(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.true_divide(np.diag(self.scatter_matrix_between_group_BG()),
                                  self.num_observations * self.data_features.var(ddof=0).to_numpy()).mean()


    def score_index_SD(self):  ##Returned value verified with R
        # Since we haven't different numbers of cluster, we can't compute the weighting coefficient :
        # We will pass the average scattering for clusters and the total separation between clusters as the returned tuple
        # return (Scattering, Separation)
        Vk = [self.data_clusters[Cluster].var(ddof=0).to_numpy() for Cluster in self.labels_clusters]
        return (np.mean([np.linalg.norm(v) for v in Vk]) / np.linalg.norm(
            self.data_features.var(ddof=0).to_numpy()), np.sum(
            [1 / self.data_intercentroid_distance_matrix()[Cluster].sum(skipna=True) for Cluster in
             self.labels_clusters]) * self.data_intercentroid_distance_matrix().max().max() / self.data_intercentroid_distance_matrix().min().min())


    def score_index_S_Dbw(self):
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
        return self.num_observations * np.log(self.score_index_det_ratio())

    def score_index_mclain_rao(self):
        pair_of_points = lambda x: x * (x - 1) / 2
        NW = np.sum([pair_of_points(self.num_observation_for_specific_cluster[Cluster]) for Cluster in self.labels_clusters])
        NB= pair_of_points(self.num_observations) - NW
        #There is a unwanted 2 factor for each following SW and SB scores. As a division is made after, we can nevermind that
        SW=((self.data_same_target_for_pairs_elements_matrix()*1) * self.data_every_element_distance_to_every_element).sum().sum()
        SB= ((1-self.data_same_target_for_pairs_elements_matrix()*1)*self.data_every_element_distance_to_every_element).sum().sum()
        return NB*SW/(NW*SB)

    def score_index_point_biserial(self):
        pair_of_points = lambda x: x * (x - 1) / 2
        NW = np.sum([pair_of_points(self.num_observation_for_specific_cluster[Cluster]) for Cluster in self.labels_clusters])
        NB= pair_of_points(self.num_observations) - NW
        #There is a unwanted 2 factor for each following SW and SB scores. We need to divide by two the final result
        SW=((self.data_same_target_for_pairs_elements_matrix()*1) * self.data_every_element_distance_to_every_element).sum().sum()
        SB= ((1-self.data_same_target_for_pairs_elements_matrix()*1)*self.data_every_element_distance_to_every_element).sum().sum()

        return (SW/NW - SB/NB)*np.sqrt(NB*NW)/pair_of_points(self.num_observations)/2

    def score_index_scott_symons(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            dets=np.array([np.linalg.det(self.scatter_matrix_specific_cluster_WGk(Cluster)/self.num_observation_for_specific_cluster[Cluster]) for Cluster in self.labels_clusters])
            if (dets!=0).all():
                 return (np.array([self.num_observation_for_specific_cluster[Cluster] for Cluster in self.labels_clusters]) * np.log(dets)).sum()

            else:
                return np.nan

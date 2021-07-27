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
from ClustersFeatures import settings

class Info:
    @property
    def clusters_info(self):
        cluster_info_df = pd.DataFrame(columns=self.labels_clusters)
        cluster_info_df.loc['Number of elements'] = [self.num_observation_for_specific_cluster[Cluster] for Cluster in
                                                     self.labels_clusters]
        cluster_info_df.loc['Centroid distance to center'] = [np.linalg.norm(self.data_centroids[Cluster]) for Cluster in
                                                              self.labels_clusters]
        cluster_info_df.loc['Centroid distance to barycenter'] = [
            np.linalg.norm(self.data_centroids[Cluster] - self.data_barycenter) for Cluster in self.labels_clusters]
        cluster_info_df.loc['Largest element distance'] = [
            self.data_interelement_distance_between_elements_of_two_clusters(Cluster, Cluster).max().max() for Cluster in
            self.labels_clusters]
        cluster_info_df.loc['Inter-element mean distance'] = [
            self.data_interelement_distance_between_elements_of_two_clusters(Cluster, Cluster).to_numpy()[
                np.tri(self.num_observation_for_specific_cluster[Cluster], k=-1) > 0].mean() for Cluster in
            self.labels_clusters]

        cluster_info_df.loc['KernelDensity mean'] = [
            self.utils_KernelDensity(clusters=Cluster).mean() for Cluster in
            self.labels_clusters]
        cluster_info_df.loc['Radius min'] = self.data_radiuscentroid['min']
        cluster_info_df.loc['Radius mean'] = self.data_radiuscentroid['mean']
        cluster_info_df.loc['Radius median'] = self.data_radiuscentroid['median']
        cluster_info_df.loc['Radius 75th Percentile'] = self.data_radiuscentroid['75p']
        cluster_info_df.loc['Radius max'] = self.data_radiuscentroid['max']
        cluster_info_df.loc['Within-Cluster Dispersion'] = [self.score_within_cluster_dispersion(Cluster) for Cluster in
                                                            self.labels_clusters]
        cluster_info_df.loc['Between-group Dispersion'] = [self.num_observation_for_specific_cluster[Cluster] * np.sum(
            (self.data_centroids[Cluster] - self.data_barycenter) ** 2) for Cluster in self.labels_clusters]
        cluster_info_df.loc['Average Silhouette'] = [
            self.score_index_silhouette_matrix[self.score_index_silhouette_matrix['Cluster'] == Cluster][
                'Silhouette Score'].mean() for Cluster in self.labels_clusters]
        cluster_info_df.loc['Ball Hall Index'] = [
            self.score_within_cluster_dispersion(Cluster) / self.num_observation_for_specific_cluster[Cluster] for Cluster
            in self.labels_clusters]
        cluster_info_df.loc['Davies Bouldin Index'] = self.score_index_davies_bouldin_for_each_cluster()
        cluster_info_df.loc['C Index'] = [self.score_index_c_for_each_cluster(Cluster) for Cluster in self.labels_clusters]
        return (cluster_info_df)

    @property
    def general_info(self):
        general_info_df = pd.DataFrame(columns=['General Informations'])
        pd.set_option('display.float_format', ("{:." + str(settings.precision) + "f}").format)
        general_info_df.loc['Between-group total dispersion'] = self.score_between_group_dispersion()
        general_info_df.loc['Mean quadratic error'] = self.score_mean_quadratic_error()
        general_info_df.loc['Trace W Index'] = self.score_pooled_within_cluster_dispersion()
        general_info_df.loc['Davies Bouldin Index'] = self.score_index_davies_bouldin()
        general_info_df.loc['Silhouette Index'] = self.score_index_silhouette_matrix['Silhouette Score'].mean()
        general_info_df.loc['C Index'] = self.score_index_c()
        general_info_df.loc['Wemmert-Gançarski Index'] = self.score_index_wemmert_gancarski()
        general_info_df.loc['Xie-Beni Index'] = self.score_index_xie_beni()
        general_info_df.loc['Ball Hall Index'] = np.mean(
            [self.score_within_cluster_dispersion(Cluster) / self.num_observation_for_specific_cluster[Cluster] for
             Cluster in self.labels_clusters])
        general_info_df.loc['Dunn Index'] = self.score_index_dunn()
        general_info_df.loc['Calinski-Harabasz Index'] = self.score_index_calinski_harabasz()
        general_info_df.loc['Banfeld-Raftery Index'] = self.score_index_banfeld_Raftery()
        general_info_df.loc['Mclain-Rao Index'] = self.score_index_mclain_rao()
        general_info_df.loc['Point Biserial Index'] = self.score_index_point_biserial()
        general_info_df.loc['Scott-Symons Index'] = self.score_index_scott_symons()
        general_info_df.loc['Log BGSS/WGSS Index'] = self.score_index_log_ss_ratio()
        general_info_df.loc['SD Index'] = [np.round(self.score_index_SD(), settings.precision)]
        general_info_df.loc['Ray-Turi Index'] = self.score_index_ray_turi()
        general_info_df.loc['PBM Index'] = self.score_index_PBM()
        general_info_df.loc['Trace WiB Index'] = self.score_index_trace_WiB()
        general_info_df.loc['Det Ratio Index'] = self.score_index_det_ratio()
        general_info_df.loc['Nlog Det Ratio Index'] = self.score_index_Log_Det_ratio()
        general_info_df.loc['Ratkowsky-Lance Index'] = self.score_index_ratkowsky_lance()
        general_info_df.loc['S_Dbw Index'] = self.score_index_S_Dbw()
        return (general_info_df)

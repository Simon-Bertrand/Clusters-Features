
import numpy as np
import pandas as pd
import json
from .settings import precision
class __IndexCore(object):

    def IndexCore_generate_output_by_info_type(self,board_type, indices_type, code):

        with open('./ClustersFeatures/indices.json') as f:
            Indices = json.load(f)
        if not(board_type in [keys for keys in Indices]):
            raise ValueError("board_type is not in " + str([keys for keys in Indices]))
            if not (indice_type in [keys for keys in Indices[board_type]]):
                raise ValueError("indice_type is not in " + str([keys for keys in Indices[board_type]]))
                if not (code in [keys for keys in Indices[board_type][code].values()]):
                    raise ValueError("code is not in " + str([keys for keys in Indices[board_type][code].values()]))
        all_index_ref=self.IndexCore_get_all_index()
        name_index = {it2: it1 for it1, it2 in all_index_ref[board_type][indices_type].items()}[code]
        if code in self._listcode_index_compute:
            return self.details_index_compute[board_type][indices_type][name_index]
        else:
            self._listcode_index_compute.append(code)
        if board_type == list(Indices.keys())[0]: #General
            if indices_type == list(Indices[board_type].keys())[0]: #Max
                if code == "G-Max-01":
                    value_to_return = self.score_between_group_dispersion()
                elif code == "G-Max-02":
                    value_to_return = self.score_mean_quadratic_error()
                elif code == "G-Max-03":
                    value_to_return = self.score_index_silhouette_matrix['Silhouette Score'].mean()
                elif code == "G-Max-04":
                    value_to_return = self.score_index_dunn()
                elif code == "G-Max-GDI":
                    GDI = self.score_index_generalized_dunn_matrix().stack()
                    GDI.index = pd.Index(["GDI " + str((idx1, idex2)) for idx1, idex2 in GDI.index])
                    value_to_return = GDI.to_dict()
                elif code == "G-Max-05":
                    value_to_return = self.score_index_wemmert_gancarski()
                elif code == "G-Max-06":
                    value_to_return = self.score_index_calinski_harabasz()
                elif code == "G-Max-07":
                    value_to_return = self.score_index_ratkowsky_lance()
                elif code == "G-Max-08":
                    value_to_return = self.score_index_point_biserial()
                elif code == "G-Max-09":
                    value_to_return = self.score_index_PBM()
                else:
                    raise ValueError('(board_type,indices_type)=' + str((board_type,indices_type)) + " - Invalid Code : " + str(code))
            elif indices_type == list(Indices[board_type].keys())[1]: #Max diff
                if code == "G-MaxD-01":
                    value_to_return = self.score_index_trace_WiB()
                elif code == "G-MaxD-02":
                    value_to_return = self.score_pooled_within_cluster_dispersion()
                else:
                    raise ValueError('(board_type,indices_type)=' + str((board_type,indices_type)) + " - Invalid Code : " + str(code))
            elif indices_type == list(Indices[board_type].keys())[2]:  # Min
                if code == "G-Min-01":
                    value_to_return = self.score_index_banfeld_Raftery()
                elif code == "G-Min-02":
                    value_to_return = np.mean([self.score_within_cluster_dispersion(Cluster) / self.num_observation_for_specific_cluster[Cluster] for Cluster in self.labels_clusters])
                elif code == "G-Min-03":
                    value_to_return = self.score_index_c()
                elif code == "G-Min-04":
                    value_to_return = self.score_index_ray_turi()
                elif code == "G-Min-05":
                    value_to_return = self.score_index_xie_beni()
                elif code == "G-Min-06":
                    value_to_return = self.score_index_davies_bouldin()
                elif code == "G-Min-07":
                    value_to_return = [np.round(self.score_index_SD(), precision)]
                elif code == "G-Min-08":
                    value_to_return = self.score_index_mclain_rao()
                elif code == "G-Min-09":
                    value_to_return = self.score_index_scott_symons()
                else:
                    raise ValueError('(board_type,indices_type)=' + str((board_type,indices_type)) + " - Invalid Code : " + str(code))
            elif indices_type == list(Indices[board_type].keys())[3]:  # Min diff
                if code == "G-MinD-01":
                    value_to_return = self.score_index_det_ratio()
                elif code == "G-MinD-02":
                    value_to_return = self.score_index_log_ss_ratio()
                elif code == "G-MinD-03":
                    value_to_return = self.score_index_S_Dbw()
                elif code == "G-MinD-04":
                    value_to_return = self.score_index_Log_Det_ratio()
                else:
                    raise ValueError('(board_type,indices_type)=' + str((board_type,indices_type)) + " - Invalid Code : " + str(code))
        elif board_type == list(Indices.keys())[1]: #Clusters
            if indices_type == list(Indices[board_type].keys())[0]:  # Max
                if code == "C-Max-01":
                    value_to_return = [np.linalg.norm(self.data_centroids[Cluster] - self.data_barycenter) for Cluster in self.labels_clusters]
                elif code == "C-Max-02":
                    value_to_return = [self.num_observation_for_specific_cluster[Cluster] * np.sum((self.data_centroids[Cluster] - self.data_barycenter) ** 2) for Cluster in self.labels_clusters]
                elif code == "C-Max-03":
                    value_to_return = [self.score_index_silhouette_matrix[self.score_index_silhouette_matrix['Cluster'] == Cluster]['Silhouette Score'].mean() for Cluster in self.labels_clusters]
                elif code == "C-Max-04":
                    value_to_return = [self.utils_KernelDensity(clusters=Cluster).mean() for Cluster in self.labels_clusters]
                elif code == "C-Max-05":
                    value_to_return = [self.score_within_cluster_dispersion(Cluster) / self.num_observation_for_specific_cluster[Cluster] for Cluster in self.labels_clusters]
                else:
                    raise ValueError('(board_type,indices_type)=' + str((board_type,indices_type)) + " - Invalid Code : " + str(code))
            elif indices_type == list(Indices[board_type].keys())[1]:  # Min
                if code == "C-Min-01":
                    value_to_return = [self.score_within_cluster_dispersion(Cluster) for Cluster in self.labels_clusters]
                elif code == "C-Min-02":
                    value_to_return = [self.data_interelement_distance_between_elements_of_two_clusters(Cluster, Cluster).max().max() for Cluster in self.labels_clusters]
                elif code == "C-Min-03":
                    value_to_return = [self.data_interelement_distance_between_elements_of_two_clusters(Cluster, Cluster).to_numpy()[np.tri(self.num_observation_for_specific_cluster[Cluster], k=-1) > 0].mean() for Cluster in self.labels_clusters]
                elif code == "C-Min-04":
                    value_to_return = self.score_index_davies_bouldin_for_each_cluster()
                elif code == "C-Min-05":
                    value_to_return = [self.score_index_c_for_each_cluster(Cluster) for Cluster in self.labels_clusters]
                else:
                    raise ValueError('(board_type,indices_type)=' + str((board_type,indices_type)) + " - Invalid Code : " + str(code))
        elif board_type == list(Indices.keys())[2]: #Radius
            if code == "R-Min-01":
                value_to_return = self.data_radiuscentroid['min']
            elif code == "R-Min-02":
                value_to_return = self.data_radiuscentroid['mean']
            elif code == "R-Min-03":
                value_to_return = self.data_radiuscentroid['median']
            elif code == "R-Min-04":
                value_to_return = self.data_radiuscentroid['75p']
            elif code == "R-Min-05":
                value_to_return = self.data_radiuscentroid['max']
            else:
                raise ValueError('(board_type,indices_type)=' + str((board_type,indices_type)) + " - Invalid Code : " + str(code))
        else:
            raise ValueError('(board_type)=' + str(board_type) + " - Invalid indices_type : " + str(code))

        self.details_index_compute[board_type][indices_type][name_index] = value_to_return
        return value_to_return

    def IndexCore_get_all_index(self):
        with open('./ClustersFeatures/indices.json') as f:
            Indices = json.load(f)

        all_index_code_ = {el: {el2: {} for el2 in list(Indices[el].keys())} for el in list(Indices.keys())}
        for k in Indices:
            for j in Indices[k]:
                for z, l in enumerate(Indices[k][j].values()):
                    all_index_code_[k][j][list(Indices[k][j].keys())[z]] = l

        return all_index_code_

    def IndexCore_compute_every_index(self):
        with open('./ClustersFeatures/indices.json') as f:
            data = json.load(f)
        all_index_ref = self.IndexCore_get_all_index()
        for k in data:
            for j in data[k]:
                for z, l in enumerate(data[k][j].values()):
                    if l in self._listcode_index_compute:
                        name_index = {it2: it1 for it1, it2 in all_index_ref[k][j].items()}[l]
                        self.details_index_compute[k][j][list(data[k][j].keys())[z]] = self.details_index_compute[k][j][name_index]
                    else:
                        print(data[k][j][{it2: it1 for it1, it2 in all_index_ref[k][j].items()}[l]], "\n")
                        self.details_index_compute[k][j][list(data[k][j].keys())[z]] = self.IndexCore_generate_output_by_info_type(k, j, l)
        return self.details_index_compute

    def IndexCore_get_number_of_index(self):
        with open('./ClustersFeatures/indices.json') as f:
            data = json.load(f)
        s = 0
        for k in data:
            for j in data[k]:
                for z, l in enumerate(data[k][j].values()):
                    s += 1
        return s

    def __IndexCore_get__board(self, boardtype):
        all_index = self.IndexCore_compute_every_index()
        if boardtype == "radius":
            radius_for_clusters = pd.DataFrame(columns=self.labels_clusters)
            for name, value in all_index['radius']['min'].items():
                radius_for_clusters.loc[name] = value

            radius_for_clusters['Type'] = len(radius_for_clusters) * ["min"]
            radius_for_clusters = radius_for_clusters.reset_index()
            return radius_for_clusters.set_index(['index', 'Type'])
        elif boardtype == "clusters":
            r = pd.DataFrame(all_index['clusters']).stack()
            clusters_indices = pd.DataFrame(columns=self.labels_clusters)
            list_type=[]
            for (name, type_), code in r.iteritems():
                clusters_indices.loc[name] = code
                list_type.append(type_)

            clusters_indices['Type'] = list_type
            clusters_indices = clusters_indices.reset_index()
            return clusters_indices.set_index(['index', 'Type'])
        elif boardtype == "score_index_GDI":
            dict_GDI = self.IndexCore_generate_output_by_info_type("general", "max", "G-Max-GDI")
            GDI = pd.DataFrame(dict_GDI.values(), pd.Index(dict_GDI))
            GDI['Type'] = len(GDI) * ["max"]
            GDI = GDI.reset_index()
            return GDI.set_index(['index', 'Type'])
        elif boardtype == "general":
            return pd.DataFrame(pd.DataFrame(self.IndexCore_compute_every_index()['general']).stack().drop('Generalized Dunn Indexes'))

    def _IndexCore_create_board(self, list_boardtype):
        if not(isinstance(list_boardtype,list)):
            raise ValueError("list_boardtype is not a list argument")
        for boardtype in list_boardtype:
            if not(boardtype in ['general', 'clusters', "radius", "score_index_GDI"]):
                raise ValueError('Invalid boardtype : ' + str(boardtype) + 'not in ' + str(['general', 'clusters', "radius", "score_index_GDI"]))
        return pd.concat([self.__IndexCore_get__board(board) for board in list_boardtype])

    def _IndexCore_nan_general_index(self):
        L = {}
        d = self.IndexCore_compute_every_index()['general']
        all_index=self.IndexCore_get_all_index()['general']
        for k_v1 in d.items():
            for k_v2 in d[k_v1[0]].items():
                if isinstance(k_v2[1], float) and np.isnan(k_v2[1]):
                    L[(k_v2[0])] = all_index[k_v1[0]][k_v2[0]]
        return L













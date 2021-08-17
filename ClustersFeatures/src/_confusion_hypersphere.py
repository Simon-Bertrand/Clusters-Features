import numpy as np
import pandas as pd
from ClustersFeatures import raising_errors

class __ConfusionHypersphere:
    def confusion_hypersphere_matrix(self, **args):
        """Returns the confusion hypersphere matrix.

        This function counts for each clusters the numbers of elements contained in each hypersphere centred on centroid.

        :param float radius_choice: The radius of the hypersphere.
        :param str counting_type: a str in ['including', 'excluding'].
            If including, then the elements belonging cluster i and contained inside the hypersphere of centroid i are counted (for i=j). If excluding, then they're not counted.
        :param bool proportion: If True, returns the proportion.
        :returns: A pandas dataframe.


        If (xi,j)i,j is the returned Matrix, then the matrix can be described as follows :

        - for proportion = False : xi,j is the number of element belonging to the cluster j contained inside (euclidian norm) the hyperpshere with specified radius of cluster i

        - for proportion = True : xi,j is the number of element belonging to the cluster j contained inside (euclidian norm) the hypersphere with specified radius of cluster i divided by the number of elements inside the cluster j


        >>> CC.confusion_hypersphere_matrix(radius=35, counting_type="including", proportion=True)

        """
        radius_choice = raising_errors.CH_radius(args)
        c_type=raising_errors.CH_counting_type(args)
        proportion=raising_errors.CH_proportion(args)

        ConfusionBooleanResult = (self.data_every_element_distance_to_centroids < radius_choice)
        ResultMat = pd.DataFrame(columns=self.labels_clusters)

        for Cluster in self.labels_clusters:
            ResultMat[Cluster] = ConfusionBooleanResult.iloc[self.data_clusters[Cluster].index].sum(axis=0)

        if c_type=="excluding":
            ResultMat.values[np.eye(len(ResultMat)) > 0] = 0

        ResultMat = ResultMat.rename(columns={col: "C:" + str(col) for col in ResultMat.columns})
        ResultMat=ResultMat.rename(index={idx: "H:" + str(idx) for idx in ResultMat.index})
        if proportion:
                return ResultMat / [self.num_observation_for_specific_cluster[Cluster] for Cluster in self.labels_clusters]
        else:
            return (ResultMat)


    def confusion_hyperphere_around_specific_point_for_two_clusters(self, point, Cluster1, Cluster2, radius):
        """ This function returns the number of elements belonging to Cluster1 or Cluster2 that are contained in the hypersphere of specific radius and centred on the given point.

        :param list,np.ndarray point: The point on which the hypersphere will be centred.
        :param Cluster1: Cluster1 label name.
        :param Cluster2: Cluster2 label name.
        :param float radius: The radius of the hyperpshere.

        :returns: int"""
        radius=raising_errors.CH_radius({'radius':radius})

        every_element_distance_to_point = np.sqrt(
            ((pd.concat([self.data_clusters[Cluster1], self.data_clusters[Cluster2]]) - pd.Series(point)) ** 2).sum(axis=1))
        return pd.DataFrame(every_element_distance_to_point < radius, index=every_element_distance_to_point.index).sum()


    def confusion_hypersphere_for_linspace_radius_each_element(self, **args):
        """ This method returns the results of the above method for a linear radius space. "="

        :param int n_pts: Allows users to set the radius range.

        :returns: A pandas dataframe"""
        max_radius = raising_errors.CH_max_radius(args,1.25 * np.max([self.data_radiuscentroid['max'][Cluster] for Cluster in self.labels_clusters]))
        num_pts=raising_errors.CH_num_pts(args, 50)
        c_type = raising_errors.CH_counting_type(args)
        proportion = raising_errors.CH_proportion(args)


        df_result = pd.DataFrame(columns=self.labels_clusters)
        radius_linspace = np.round(np.linspace(0, max_radius, num_pts), 4)
        for r in radius_linspace:
            df_result.loc[r] = self.confusion_hypersphere_for_each_element_matrix(radius=r,counting_type=c_type).sum().values
        df_result.index.name = "Radius"
        if proportion:
            return df_result / self.num_observations
        else:
            return df_result




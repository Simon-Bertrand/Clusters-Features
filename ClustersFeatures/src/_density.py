from ClustersFeatures import raising_errors
from ClustersFeatures import settings

import numpy as np
import pandas as pd
from ClustersFeatures import settings
class __Density:
    def density_projection_2D(self, reduction_method, percentile, **args):
        cluster, return_clusters_density, return_data = raising_errors.density_Projection_2D(args,
                                                                                                self.labels_clusters)

        if percentile >= 100 or percentile <= 0:
            raise ValueError('percentile is out of range [0-100]')

        if reduction_method == "UMAP":
            data = self.utils_UMAP()
        elif reduction_method == "PCA":
            data = self.utils_PCA(2)

        xmin, xmax = data[data.columns[0]].min(), data[data.columns[0]].max()
        ymin, ymax = data[data.columns[1]].min(), data[data.columns[1]].max()
        xrange = np.linspace(xmin, xmax, 200)
        yrange = np.linspace(ymin, ymax, 200)
        X, Y = np.meshgrid(xrange, yrange)
        Z = pd.DataFrame(np.zeros((len(xrange), len(yrange))), index=xrange, columns=yrange)

        std = 1
        each_cluster_density_save = {}
        for Cluster in cluster:
            Mat = np.zeros((len(xrange), len(yrange)))
            for idx, val in data[self.data_target == Cluster].T.iteritems():
                Mat += np.exp(-1 * ((X - val[0]) ** 2 + (Y - val[1]) ** 2) / (2 * std)) / (2 * np.pi * std ** 2)
                Z = Z + Mat
            each_cluster_density_save[Cluster] = Mat

        returned_var = {"Z-Grid": Z}
        if return_clusters_density:
            returned_var["Clusters Density"] = (each_cluster_density_save)
        if return_data:
            returned_var["2D PCA Data"] = data
        return returned_var

    def density_projection_3D(self, percentile, **args):
        cluster, return_clusters_density, return_grid = raising_errors.density_Density_Projection_3D(args,
                                                                                                self.labels_clusters)

        if percentile >= 100 or percentile <= 0:
            raise ValueError('percentile is out of range [0-100]')

        data = self.utils_PCA(3)

        xmin, xmax = data[data.columns[0]].min() - np.abs(data[data.columns[0]].min()) / 5, data[
            data.columns[0]].max() + np.abs(data[data.columns[0]].max()) / 5
        ymin, ymax = data[data.columns[1]].min() - np.abs(data[data.columns[1]].min()) / 5, data[
            data.columns[1]].max() + np.abs(data[data.columns[1]].max()) / 5
        zmin, zmax = data[data.columns[2]].min() - np.abs(data[data.columns[2]].min()) / 5, data[
            data.columns[2]].max() + np.abs(data[data.columns[2]].max()) / 5
        xrange = np.linspace(xmin, xmax, 50)
        yrange = np.linspace(ymin, ymax, 50)
        zrange = np.linspace(zmin, zmax, 50)

        X, Y, Z = np.meshgrid(xrange, yrange, zrange)
        A = np.zeros((len(xrange), len(yrange), len(zrange)))

        std = 1
        each_cluster_density_save = {}
        for Cluster in self.labels_clusters:
            Mat = np.zeros((len(xrange), len(yrange), len(zrange)))
            for idx, val in data[self.data_target == Cluster].T.iteritems():
                Mat += np.exp(
                    -1 * ((X - val[0]) ** 2 + (Y - val[1]) ** 2 + (Z - val[2]) ** 2) / (2 * std) / (2 * np.pi) ** (
                                3 / 2) / std ** 3)
                A = A + Mat
            each_cluster_density_save[Cluster] = Mat

        returned_var = {"A-Grid": A}
        if return_clusters_density:
            returned_var["Clusters Density"] = (each_cluster_density_save)
        if return_grid:
            returned_var["3D Grid"] = {"X":X,"Y":Y,"Z":Z}
        return returned_var

    def density_projection_2D_generate_png(self, reduction_method, percentile, **args):
        from PIL import Image
        if percentile >= 100 or percentile <= 0:
            raise ValueError('percentile is out of range [0-100]')

        if reduction_method == "UMAP":
            data = self.utils_UMAP()
        elif reduction_method == "PCA":
            data = self.utils_PCA(2)

        try:
            show = args['show_image']
            if not isinstance(show, bool):
                raise ValueError('show_image is not boolean.')
            if not settings.Activated_Graph and show:
                print(
                    'Warning : Activated_Graph is False in settings.py. Showing the graph is not possible. Put it to True and install Plotly to avoid this warning')
                show = False
        except KeyError:
            show = True

        unpacked_dict = self.density_projection_2D(reduction_method, percentile, clusters=self.labels_clusters,
                                                         return_clusters_density=True)
        Z = unpacked_dict['Z-Grid']
        clusters_density = unpacked_dict['Clusters Density']

        threshold = np.mean([np.percentile(clusters_density[Cluster], percentile) for Cluster in self.labels_clusters])

        hex_to_rgb_convert = lambda hex_string: [int(hex_string[1:3], 16), int(hex_string[3:5], 16),
                                                 int(hex_string[5:7], 16)]

        image_clusters = {}
        Base_image = Image.new('RGBA', clusters_density[list(clusters_density.keys())[0]].shape)
        for i, Cluster in enumerate(self.labels_clusters):
            image_clusters[Cluster] = np.zeros(
                (clusters_density[Cluster].shape[0], clusters_density[Cluster].shape[1], 4))
            image_clusters[Cluster][clusters_density[Cluster] > threshold] = hex_to_rgb_convert(
                settings.discrete_colors[i]) + [180]
            Cluster_image = Image.fromarray(np.uint8(image_clusters[Cluster]))
            red, g, b, alpha = Cluster_image.split()
            alpha = alpha.point(lambda i: i > 0 and 204)
            Base_image = Image.composite(Cluster_image, Base_image, alpha)
        if show:
            import plotly.express as px
            fig = px.imshow(Base_image)
            fig.show()
        return {"All clusters image data": np.asarray(Base_image), "Each cluster image data": image_clusters}


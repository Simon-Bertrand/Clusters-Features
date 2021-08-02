from ClustersFeatures import raising_errors
from ClustersFeatures import settings

if settings.Activated_Utils:
    import pandas as pd
    import numpy as np


    class __Utils:
        def utils_ts_filtering_STL(self, **args):
            """
            Filter a time-serie with STL from statsmodels.
            col argument can be specified if it is wanted to filter
            a column of self dataframe.
            Else, you can directly specify a time-serie with the data argument

            :param args: period= (int or float, required) : Specify the period between each sample
                   args: col= (str/int, required if data is None) : Specify the column of self data set to filter
                   args: data=(list/ndarray, required if col is None) : Specify the data to filter
            :return:
            """
            from statsmodels.tsa.seasonal import STL

            period = raising_errors.utils_period_arg(args)
            col = raising_errors.utils_col_arg(args, self.data_features.columns)
            data = raising_errors.utils_data_arg(args)

            if col is not None and data is None:
                decomposition = STL(self.data_features[col], period=period).fit()
                decompo = pd.DataFrame(decomposition.observed).rename({col: "observed"}, axis=1)
                decompo['seasonal'] = decomposition.seasonal
                decompo['trend'] = decomposition.trend
                decompo['resid'] = decomposition.resid
            elif data is not None and col is None:
                decomposition = STL(data, period=period).fit()
                decompo = pd.DataFrame(decomposition.observed).rename({0: "observed"}, axis=1)
                decompo['seasonal'] = decomposition.seasonal
                decompo['trend'] = decomposition.trend
                decompo['resid'] = decomposition.resid
            else:
                raising_errors.utils_not_botch_col_and_data()

            return decompo

        def utils_UMAP(self, **args):
            """
            Uniform Manifold Approximation Projection : Use the umap-learn library.
            :return:
            A 2D projection of the whole data set concatenated with the target of the data
            """
            from umap import UMAP
            from numba import config

            try:
                show_target = args['show_target']
                if not isinstance(show_target, bool):
                    raise ValueError('show_target is not boolean')
            except KeyError:
                show_target = False

            config.THREADING_LAYER = 'threadsafe'
            reducer = UMAP()
            df_drop_sort = self.data_features.dropna()
            df_reduced = pd.DataFrame(reducer.fit_transform(df_drop_sort), index=df_drop_sort.index)
            if show_target:
                df_reduced['target'] = self.data_target
            return df_reduced

        def utils_PCA(self, n_components):
            """
            Principal Component Analysis : Use the scikit learn library
            :param n_components: number of data dimension after reduction
            :return:
            A n_components-D projection of the whole data set
            """
            from sklearn.decomposition import PCA
            pca = PCA(n_components=n_components)
            pca.fit(self.data_features)
            Result = pd.DataFrame(pca.transform(self.data_features))
            return pd.DataFrame(Result, index=self.data_features.index).rename(
                columns={col: "PCA" + str(col) for col in Result.columns})

        def utils_KernelDensity(self, **args):
            """
            Function that returns an estimation of Kernel Density with the best bandwidth
            If argument return_KDE = True, so the KDE model is returned to generate samples later. It uses the Scikit Learn Library
            If no clusters specified, the KernelDensity is done on the entire data set
            :param args:
            return_KDE= (optional, bool) - Return the sklearn KDE Model
            clusters = (optional, list) - List of clusters to evaluate KernelDensity, order is not important

            :return:
            -An estimation of KernelDensity for each sample if return_KDE is false
            -An tuple with the estimation of KD for each sample and the KDE model if return_KDE is true
            """
            from sklearn.neighbors import KernelDensity
            from sklearn.model_selection import GridSearchCV
            returnKDE = raising_errors.utils_return_KDE_model(args)
            clusters = raising_errors.list_clusters(args, self.labels_clusters)
            if not (clusters):
                X = self.data_features
            else:
                boolean_selector = pd.concat([1 * (self.data_target == cl) for cl in clusters], axis=1).sum(axis=1)
                X = self.data_features[boolean_selector.astype(bool)]
            params = {'bandwidth': np.logspace(-1, 1, 20)}
            grid = GridSearchCV(KernelDensity(), params)
            grid.fit(X)
            kde = grid.best_estimator_
            if returnKDE:
                return (kde.score_samples(X), kde)
            else:
                return kde.score_samples(X)

        def utils_Projection_2D_Density(self, reduction_method, percentile, **args):
            cluster, return_clusters_density, return_data = raising_errors.utils_Density_Projection(args, self.labels_clusters)

            if percentile >= 100 or percentile <= 0:
                raise ValueError('percentile is out of range [0-100]')

            if reduction_method == "UMAP":
                data = self.utils_UMAP()
            elif reduction_method == "PCA":
                data = self.utils_PCA(2)

            xmin, xmax = data[data.columns[0]].min(), data[data.columns[0]].max()
            ymin, ymax = data[data.columns[1]].min(), data[data.columns[1]].max()
            xrange = np.linspace(xmin,xmax,200)
            yrange = np.linspace(ymin,ymax,200)
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

            contours = 1*(Z>np.percentile(Z, percentile))
            returned_var = {"Z-Grid":Z}
            if return_clusters_density:
                returned_var["Clusters Density"] = (each_cluster_density_save)
            if return_data:
                returned_var["2D PCA Data"] = data
            return returned_var

        def utils_Density_Projection_2D_generate_png(self,reduction_method, percentile,**args):
            from PIL import Image
            if percentile >= 100 or percentile <= 0:
                raise ValueError('percentile is out of range [0-100]')

            if reduction_method == "UMAP":
                data = self.utils_UMAP()
            elif reduction_method == "PCA":
                data = self.utils_PCA(2)

            try:
                show=args['show_image']
                if not isinstance(show, bool):
                    raise ValueError('show_image is not boolean.')
                if not settings.Activated_Graph  and show:
                    print('Warning : Activated_Graph is False in settings.py. Showing the graph is not possible. Put it to True and install Plotly to avoid this warning')
                    show=False
            except KeyError:
                show=True

            unpacked_dict=self.utils_Projection_2D_Density(reduction_method, percentile, clusters=self.labels_clusters,
                                           return_clusters_density=True)
            Z=unpacked_dict['Z-Grid']
            clusters_density=unpacked_dict['Clusters Density']

            threshold = np.mean([np.percentile(clusters_density[Cluster], percentile) for Cluster in self.labels_clusters])


            hex_to_rgb_convert = lambda hex_string: [int(hex_string[1:3], 16), int(hex_string[3:5], 16),
                                                     int(hex_string[5:7], 16)]

            image_clusters = {}
            Base_image = Image.new('RGBA', clusters_density[list(clusters_density.keys())[0]].shape)
            for i, Cluster in enumerate(self.labels_clusters):
                image_clusters[Cluster] = np.zeros((clusters_density[Cluster].shape[0], clusters_density[Cluster].shape[1], 4))
                image_clusters[Cluster][clusters_density[Cluster] > threshold] = hex_to_rgb_convert(settings.discrete_colors[i]) + [180]
                Cluster_image = Image.fromarray(np.uint8(image_clusters[Cluster]))
                red, g, b, alpha = Cluster_image.split()
                alpha = alpha.point(lambda i: i > 0 and 204)
                Base_image = Image.composite(Cluster_image, Base_image, alpha)
            if show:
                import plotly.express as px
                fig = px.imshow(Base_image)
                fig.show()
            return {"All clusters image data" : np.asarray(Base_image), "Each cluster image data":image_clusters}


else:
    class __Utils():
        pass

from ClustersFeatures import raising_errors
from ClustersFeatures import settings

if settings.Activated_Utils:
    import pandas as pd
    import numpy as np
    class Utils:
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

            period=raising_errors.utils_period_arg(args)
            col=raising_errors.utils_col_arg(args,self.data_features.columns)
            data=raising_errors.utils_data_arg(args)

            if col is not None and data is None:
                decomposition = STL(self.data_features[col],period=period).fit()
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

        def utils_UMAP(self):
            """
            Uniform Manifold Approximation Projection : Use the umap-learn library.
            :return:
            A 2D projection of the whole data set concatenated with the target of the data
            """
            from umap import UMAP
            from numba import config
            config.THREADING_LAYER = 'threadsafe'
            reducer = UMAP()
            df_drop_sort = self.data_features.dropna()
            df_reduced = pd.DataFrame(reducer.fit_transform(df_drop_sort), index=df_drop_sort.index)
            df_reduced['target'] = self.data_target
            return  df_reduced

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
            Result=pd.DataFrame(pca.transform(self.data_features))
            return pd.DataFrame(Result, index=self.data_features.index).rename(columns={col: "PCA"+str(col) for col in  Result.columns})

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
            clusters = raising_errors.utils_list_clusters(args,self.labels_clusters)
            if not(clusters):
                X = self.data_features
            else:
                boolean_selector=pd.concat([1*(self.data_target==cl) for cl in clusters],axis=1).sum(axis=1)
                X=self.data_features[boolean_selector.astype(bool)]
            params = {'bandwidth': np.logspace(-1, 1, 20)}
            grid = GridSearchCV(KernelDensity(), params)
            grid.fit(X)
            kde = grid.best_estimator_
            if returnKDE:
                return (kde.score_samples(X), kde)
            else:
                return kde.score_samples(X)


else:
    class Utils():
        pass
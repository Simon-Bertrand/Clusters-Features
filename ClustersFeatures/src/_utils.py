from ClustersFeatures import raising_errors
from ClustersFeatures import settings

if settings.Activated_Utils:
    import pandas as pd
    class Utils:
        def utils_ts_filtering_STL(self, **args):
            """Filters a time-serie with STL from statsmodels.
            col argument can be specified if it is wanted to filter
            a column of self dataframe
            Else, you can specify a time-serie with the data argument
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
            from umap import UMAP
            from numba import config
            config.THREADING_LAYER = 'threadsafe'
            reducer = UMAP()
            df_drop_sort = self.data_features.dropna()
            df_reduced = pd.DataFrame(reducer.fit_transform(df_drop_sort), index=df_drop_sort.index)
            df_reduced['target'] = self.data_target
            return  df_reduced

        def utils_PCA(self, n_components):
            from sklearn.decomposition import PCA
            pca = PCA(n_components=n_components)
            pca.fit(self.data_features)
            Result=pd.DataFrame(pca.transform(self.data_features))
            return pd.DataFrame(Result, index=self.data_features.index).rename(columns={col: "PCA"+str(col) for col in  Result.columns})

else:
    class Utils():
        pass
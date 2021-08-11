from ClustersFeatures import raising_errors
from ClustersFeatures import settings

if settings.Activated_Utils:
    import pandas as pd
    import numpy as np
    class __Utils:
        def utils_ts_filtering(self, filter,  **args):
            """Filter a time-serie with different filters from statsmodels.

            | Col argument can be specified if it is wanted to filter a column of self dataframe.
            | Else, you can directly specify a time-serie with the data argument.

            :param str filter: Type of filter. Have to be in the list ['STL', 'HP', 'BK', 'CF'] respectively for :
            | STL : Season-trend decomposing using LOESS. - https://www.statsmodels.org/devel/generated/statsmodels.tsa.seasonal.STL.html
            | HP : Hodrick-Prescott filter. - https://www.statsmodels.org/devel/generated/statsmodels.tsa.filters.hp_filter.hpfilter.html?highlight=hp
            | BK : Filter a time series using the Baxter-King bandpass filter. - https://www.statsmodels.org/devel/generated/statsmodels.tsa.filters.bk_filter.bkfilter.html?highlight=bk#statsmodels.tsa.filters.bk_filter.bkfilter
            | CF : Christiano Fitzgerald asymmetric, random walk filter. - https://www.statsmodels.org/devel/generated/statsmodels.tsa.filters.cf_filter.cffilter.html?highlight=cf#statsmodels.tsa.filters.cf_filter.cffilter

            :param int/float periods=: Specify the period between each sample.
            :param str/int col=: Required if data is None: Specify the column of self data set to filter.
            :param list/np.ndarray data=: Required if col is None : Specify the data to filter.

            :return: A pandas dataframe with columns as the decomposed signals.
            """
            list_filter=['STL', 'HP', 'BK', 'CF']
            if not(filter in list_filter):
                raise ValueError('Invalid filter type. Have to be in : '+str(list_filter))

            period = raising_errors.utils_period_arg(args)
            col = raising_errors.utils_col_arg(args, self.data_features.columns)
            data = raising_errors.utils_data_arg(args)

            if filter == "STL":
                from statsmodels.tsa.seasonal import STL
                if col is not None and data is None:
                    decomposition = STL(self.data_features[col], period=period).fit()
                    decompo = pd.DataFrame(decomposition.observed).rename({col: "observed"}, axis=1)
                elif data is not None and col is None:
                    decomposition = STL(data, period=period).fit()
                    decompo = pd.DataFrame(data, axis=1)
                else:
                    raising_errors.utils_not_botch_col_and_data()
                decompo['seasonal'] = decomposition.seasonal
                decompo['trend'] = decomposition.trend
                decompo['resid'] = decomposition.resid
                return decompo


            elif filter == "HP":
                import statsmodels.api as sm
                if col is not None and data is None:
                    cycle, trend = sm.tsa.filters.hpfilter(self.data_features[col],lamb=1600)  # lamb has to be dependent of the ts' period, currently it's not
                    decompo = pd.DataFrame(self.data_features[col]).rename({col: "observed"}, axis=1)
                elif data is not None and col is None:
                    cycle, trend = sm.tsa.filters.hpfilter(data,lamb=1600)  # lamb has to be dependent of the ts' period, currently it's not
                    decompo = pd.DataFrame(data).rename({0:"observed"}, axis=1)
                else:
                    raising_errors.utils_not_botch_col_and_data()
                decompo['cycle'] = cycle
                decompo['trend'] = trend
                return decompo

            elif filter == "BK":
                import statsmodels.api as sm
                if col is not None and data is None:
                    bk_cycle=sm.tsa.filters.bkfilter(self.data_features[col],K=3)
                    decompo = pd.DataFrame(self.data_features[col]).rename({col: "observed"}, axis=1)
                elif data is not None and col is None:
                    bk_cycle=sm.tsa.filters.bkfilter(data,K=3)
                    decompo = pd.DataFrame(data).rename({col: "observed"}, axis=1)
                else:
                    raising_errors.utils_not_botch_col_and_data()

                decompo['cycle'] = bk_cycle
                return decompo

            elif filter == "CF":
                import statsmodels.api as sm
                if col is not None and data is None:
                    cf_cycle = sm.tsa.filters.cffilter(self.data_features[col])
                    decompo = pd.DataFrame(self.data_features[col]).rename({col: "observed"}, axis=1)
                elif data is not None and col is None:
                    cf_cycle = sm.tsa.filters.cffilter(data)
                    decompo = pd.DataFrame(data).rename({col: "observed"}, axis=1)
                else:
                    raising_errors.utils_not_botch_col_and_data()

                decompo['cycle'], decompo['trend'] = cf_cycle[0], cf_cycle[1]
                return decompo


        def utils_UMAP(self, **args):
                """Uniform Manifold Approximation Projection : Use the umap-learn library.

                The result is cached to avoid same and repetitive calculs.

                :param bool show_target: Concatenate target to output dataframe

                :return: A pandas dataframe with the 2D projection of the whole data set.
                """
                from umap import UMAP
                from numba import config

                try :
                    return self.__UMAP_cached
                except AttributeError:
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
                    self.__UMAP_cached = df_reduced
                    return self.__UMAP_cached

        def utils_PCA(self, n_components):
            """Principal Component Analysis : Use the scikit learn library

            :param n_components: number of data dimension after reduction
            :return: A n_components-D projection of the whole data set
            """
            from sklearn.decomposition import PCA
            pca = PCA(n_components=n_components)
            pca.fit(self.data_features)
            Result = pd.DataFrame(pca.transform(self.data_features))
            return pd.DataFrame(Result, index=self.data_features.index).rename(
                columns={col: "PCA" + str(col) for col in Result.columns})

        def utils_KernelDensity(self, **args):
            """Function that returns an estimation of Kernel Density with the best bandwidth.

            :param bool return_KDE: If argument return_KDE = True, so the KDE model is returned to generate samples later. It uses the Scikit Learn Library.
            :param list clusters: List of clusters to evaluate KernelDensity, order is not important.If no clusters specified, the KernelDensity is done on the entire data set


            :return: - An estimation of KernelDensity for each sample if return_KDE is false
                - A tuple with the estimation of KD for each sample and the KDE model if return_KDE is true
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

        def utils_ClustersRank(self,**args):
            """Defines a mean rank for each cluster based on the min/max indexes of the cluster board.

            The method uses the min-max scaler to put each row of the clusters_info board at the same dimension.
            We separate the min and the max indices to output a rank for each indices.
            If the indices min i is the lower of all clusters, then its rank is self.num_observations-th.
            To generate the final rank, we compute the mean rank for each cluster with min et max type.
            Then we sum the mean rank of the min indices to the mean rank of the max indices.
            As we want a rank where first position is the better, we invert the above sum and get the final rank.
            By adding params, you can provide the mean rank for each cluster by passing cluster_rank=True.

            :param bool cluster_rank=: Returns the mean rank for each cluster

            :returns: The final leaderboard.

            >>> CC.utils_ClustersRank(mean_cluster_rank=True)
            """

            try :
                bool_cluster_rank=args['mean_cluster_rank']
                if not isinstance(bool_cluster_rank,bool):
                    raise ValueError('mean_cluster_rank is not boolean.')
            except KeyError:
                bool_cluster_rank = False

            try :
                show_result=args['show_result']
                if not isinstance(show_result,bool):
                    raise ValueError('show_result is not boolean.')
            except KeyError:
                show_result = False

            clusters_info=self.clusters_info(scaler='min_max')
            mean_cluster_rank=(1 + self.num_clusters - (clusters_info.loc[pd.IndexSlice[:, ['max']], :].rank(axis=1,method="max").mean() + clusters_info.loc[pd.IndexSlice[:, ['min']], :].rank(axis=1, method="max",ascending=False).mean()) / 2).sort_values()
            cluster_rank=mean_cluster_rank.rank()

            int_to_position_transform = lambda x: "1st" if x == 1 else "2nd" if x == 2 else "3rd" if x == 3 else str(
                int(x)) + "th"

            if show_result:
                for key,value in cluster_rank.iteritems():
                    print(f"The cluster {key} is at the {int_to_position_transform(value)} position.")

            df=pd.DataFrame()
            df['Rank']=cluster_rank.apply(int_to_position_transform)
            df.index.name="Cluster"
            if bool_cluster_rank:
                df['Mean Rank'] = mean_cluster_rank

            return df
else:
    class __Utils():
        pass

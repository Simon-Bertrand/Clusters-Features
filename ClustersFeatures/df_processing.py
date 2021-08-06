from ClustersFeatures import imputation

def impute_nan_values(pd_df_,args):
    if pd_df_.isnull().sum().sum() != 0:
        try:
            estimator = args['imputation_estimator']
            pd_df = imputation._imputation_detect_nan(pd_df_, estimator)
        except KeyError:
            pd_df = imputation._imputation_detect_nan(pd_df_, False).copy()
    else:
        pd_df = pd_df_.copy()
    return pd_df

def generate_label_target_in_case_missing_it(pd_df,args):
    try:
        output = pd_df , args['label_target']
        return output
    except KeyError:
        try:
            nb_kmean_clusters = args['n_kmeans']
        except KeyError:
            nb_kmean_clusters=5

        print(f'No specified label target. Processing to KMean with K={nb_kmean_clusters}.\n')
        from sklearn.cluster import KMeans
        pd_df['target'] = KMeans(n_clusters=nb_kmean_clusters).fit(pd_df).labels_
        label_target = "target"

    return pd_df,label_target
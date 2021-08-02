import pandas as pd
import numpy as np
class __Imputation:
    def _imputation_detect_nan(self,pd_df,estimator):
        if pd_df.isnull().sum().sum() != 0:
            print("Nan Values detected. Doing the imputation. \n")
            from sklearn.experimental import enable_iterative_imputer
            from sklearn.impute import IterativeImputer

            if estimator == "BayesianRidge":
                from sklearn.linear_model import BayesianRidge
                pd_df_imputed = IterativeImputer(estimator=BayesianRidge()).fit_transform(pd_df.values)
            elif estimator == "DecisionTreeRegressor":
                from sklearn.tree import DecisionTreeRegressor
                pd_df_imputed = IterativeImputer(estimator=DecisionTreeRegressor()).fit_transform(pd_df.values)
            elif estimator == "ExtraTreesRegressor":
                from sklearn.ensemble import ExtraTreesRegressor
                pd_df_imputed = IterativeImputer(estimator=DecisionTreeRegressor()).fit_transform(pd_df.values)
            elif estimator == "KNeighborsRegressor":
                from sklearn.neighbors import KNeighborsRegressor
                pd_df_imputed = IterativeImputer(estimator=KNeighborsRegressor()).fit_transform(pd_df.values)
            elif estimator == "KNNImputer":
                from sklearn.impute import KNNImputer
                pd_df_imputed = KNNImputer(n_neighbors=5).fit_transform(pd_df)
            else:

                print("Unknown estimator. using default BayesianRidge estimator.")
                pd_df_imputed = IterativeImputer().fit_transform(pd_df.values)


            return pd.DataFrame(pd_df_imputed, index=pd_df.index, columns=pd_df.columns)
        else:
            return pd_df
import statsmodels.api as sm
import pandas as pd

from statsmodels.tsa.seasonal import STL
from ClustersFeatures import raising_errors
class Utils:
    def utils_ts_filtering_STL(self, **args):
        """Filters a time-serie with STL from statsmodels.
        col argument can be specified if it is wanted to filter
        a column of self dataframe
        Else, you can specify a time-serie with the data argument
        """
        seasonal=raising_errors.utils_seasonal_arg(args)
        col=raising_errors.utils_col_arg(args,self.data_features.columns)
        data=raising_errors.utils_data_arg(args)

        if col is not None and data is None:
            decomposition = STL(self.data_features[col], seasonal=seasonal).fit()
            decompo = pd.DataFrame(decomposition.observed).rename({col: "observed"}, axis=1)
            decompo['seasonal'] = decomposition.seasonal
            decompo['trend'] = decomposition.trend
            decompo['resid'] = decomposition.resid
        elif data is not None and col is None:
            decomposition = STL(data, seasonal=seasonal).fit()
            decompo = pd.DataFrame(decomposition.observed).rename({col: "observed"}, axis=1)
            decompo['seasonal'] = decomposition.seasonal
            decompo['trend'] = decomposition.trend
            decompo['resid'] = decomposition.resid
        else:
            raising_errors.utils_not_botch_col_and_data()

        return decompo
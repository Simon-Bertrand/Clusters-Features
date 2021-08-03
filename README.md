```python
from ClustersFeatures import *
from sklearn.datasets import load_digits
import pandas as pd
digits = load_digits()
pd_df=pd.DataFrame(digits.data)
pd_df['target'] = digits.target
pd_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>55</th>
      <th>56</th>
      <th>57</th>
      <th>58</th>
      <th>59</th>
      <th>60</th>
      <th>61</th>
      <th>62</th>
      <th>63</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>13.0</td>
      <td>9.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>13.0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>12.0</td>
      <td>13.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>11.0</td>
      <td>16.0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>15.0</td>
      <td>12.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>11.0</td>
      <td>16.0</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>15.0</td>
      <td>13.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>13.0</td>
      <td>13.0</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>11.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>16.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1792</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>10.0</td>
      <td>13.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>14.0</td>
      <td>15.0</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>9</td>
    </tr>
    <tr>
      <th>1793</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>16.0</td>
      <td>13.0</td>
      <td>11.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>16.0</td>
      <td>14.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1794</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>11.0</td>
      <td>15.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>9.0</td>
      <td>13.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>1795</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>10.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>12.0</td>
      <td>16.0</td>
      <td>12.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>9</td>
    </tr>
    <tr>
      <th>1796</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>14.0</td>
      <td>8.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>8.0</td>
      <td>12.0</td>
      <td>14.0</td>
      <td>12.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
<p>1797 rows × 65 columns</p>
</div>




```python
CC=ClustersCharacteristics(pd_df,"target")
```


```python
CC.graph_boxplots_distances_to_centroid(5)
```


```python

```


```python

```


```python
CC=ClustersCharacteristics(pd_df, "target")
```


```python
CC.graph_reduction_density_3D(95)
```


```python
CC.graph_reduction_density_2D("UMAP", 99, "interactive")
```


```python
CC.graph_reduction_density_2D("UMAP", 99, "contour")
```


```python
CC.graph_reduction_2D("PCA")
```


```python
CC.graph_boxplots_distances_to_centroid(2)
```


```python
CC.graph_PCA_3D()
```


```python
CC.density_projection_3D(99, return_clusters_density=True)
```


```python
___
```


```python
from sklearn.datasets import make_blobs

from sklearn.preprocessing import StandardScaler

import seaborn as sns
import pandas as pd
X, y = make_blobs(n_samples=200, centers=2, n_features=3,cluster_std=0.5)
X=StandardScaler().fit_transform(X)
T=pd.DataFrame(data=X)
T['target']=y
CC2=ClustersCharacteristics(T,"target")
```


```python

    

```


```python

```

### Speed test of different scores


```python
print('score_index_ball_hall \n')
%timeit CC.score_index_ball_hall()
print('\nscore_index_banfeld_Raftery\n')
%timeit CC.score_index_banfeld_Raftery()
print('\nscore_index_c\n')
%timeit CC.score_index_c()
print('\nscore_index_c_for_each_cluster\n')
%timeit CC.score_index_c_for_each_cluster(0)
print('\nscore_index_calinski_harabasz \n')
%timeit CC.score_index_calinski_harabasz()
print('\nscore_index_davies_bouldin \n')
%timeit CC.score_index_davies_bouldin()
print('\nscore_index_davies_bouldin_for_each_cluster \n')
%timeit CC.score_index_davies_bouldin_for_each_cluster()
print('\nscore_index_det_ratio \n')
%timeit CC.score_index_det_ratio()
print('\nscore_index_dunn\n')
%timeit CC.score_index_dunn()
print('\nscore_index_generalized_dunn_matrix\n')
%timeit CC.score_index_generalized_dunn_matrix()
print('\nscore_index_Log_Det_ratio\n')
%timeit CC.score_index_Log_Det_ratio()
print('\nscore_index_log_ss_ratio \n')
%timeit CC.score_index_log_ss_ratio()
print('\nscore_index_mclain_rao \n')
%timeit CC.score_index_mclain_rao()
print('\nscore_index_PBM \n')
%timeit CC.score_index_PBM()
print('\nscore_index_point_biserial\n')
%timeit CC.score_index_point_biserial()
print('\nscore_index_ratkowsky_lance \n')
%timeit CC.score_index_ratkowsky_lance()
print('\nscore_index_ray_turi \n')
%timeit CC.score_index_ray_turi()
print('\nscore_index_S_Dbw \n')
%timeit CC.score_index_S_Dbw()
print('\nscore_index_scott_symons\n')
%timeit CC.score_index_scott_symons()
print('\nscore_index_SD \n')
%timeit CC.score_index_SD()
print('\nscore_index_trace_WiB \n')
%timeit CC.score_index_trace_WiB()
print('\nscore_index_wemmert_gancarski \n')
%timeit CC.score_index_wemmert_gancarski()
print('\nscore_index_xie_beni\n')
%timeit CC.score_index_xie_beni()
```


```python
bc_list = np.arange(1, 7)
wc_list = np.arange(1, 4)

df = pd.DataFrame(columns=wc_list, index=bc_list)
for bc in bc_list:
    for wc in wc_list:
        print(wc,bc)
        %timeit CC.score_index_generalized_dunn(within_cluster_distance=wc, between_cluster_distance=bc)
        df.loc[bc, wc] = CC.score_index_generalized_dunn(within_cluster_distance=wc, between_cluster_distance=bc)
        
df.index.name = "Generalized Dunn Indexes"
df
```

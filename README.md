```python
from ClustersFeatures import ClustersCharacteristics
```


```python
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
CC=ClustersCharacteristics(pd_df, "target")
```


```python
CC.confusion_hypersphere_matrix(radius=35,counting_type='including',proportion=True)
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
      <th>C:0</th>
      <th>C:1</th>
      <th>C:2</th>
      <th>C:3</th>
      <th>C:4</th>
      <th>C:5</th>
      <th>C:6</th>
      <th>C:7</th>
      <th>C:8</th>
      <th>C:9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>H:0</th>
      <td>0.994382</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010929</td>
      <td>0.000000</td>
      <td>0.032967</td>
      <td>0.060773</td>
      <td>0.000000</td>
      <td>0.005747</td>
      <td>0.211111</td>
    </tr>
    <tr>
      <th>H:1</th>
      <td>0.000000</td>
      <td>0.736264</td>
      <td>0.090395</td>
      <td>0.103825</td>
      <td>0.187845</td>
      <td>0.016484</td>
      <td>0.110497</td>
      <td>0.055866</td>
      <td>0.574713</td>
      <td>0.022222</td>
    </tr>
    <tr>
      <th>H:2</th>
      <td>0.000000</td>
      <td>0.142857</td>
      <td>0.881356</td>
      <td>0.355191</td>
      <td>0.000000</td>
      <td>0.005495</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.310345</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>H:3</th>
      <td>0.000000</td>
      <td>0.005495</td>
      <td>0.225989</td>
      <td>0.950820</td>
      <td>0.000000</td>
      <td>0.258242</td>
      <td>0.000000</td>
      <td>0.016760</td>
      <td>0.327586</td>
      <td>0.666667</td>
    </tr>
    <tr>
      <th>H:4</th>
      <td>0.050562</td>
      <td>0.032967</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.928177</td>
      <td>0.027473</td>
      <td>0.154696</td>
      <td>0.027933</td>
      <td>0.028736</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>H:5</th>
      <td>0.095506</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.103825</td>
      <td>0.005525</td>
      <td>0.950549</td>
      <td>0.022099</td>
      <td>0.005587</td>
      <td>0.293103</td>
      <td>0.133333</td>
    </tr>
    <tr>
      <th>H:6</th>
      <td>0.089888</td>
      <td>0.027473</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.033149</td>
      <td>0.021978</td>
      <td>0.983425</td>
      <td>0.000000</td>
      <td>0.068966</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>H:7</th>
      <td>0.000000</td>
      <td>0.071429</td>
      <td>0.028249</td>
      <td>0.071038</td>
      <td>0.060773</td>
      <td>0.005495</td>
      <td>0.000000</td>
      <td>0.882682</td>
      <td>0.201149</td>
      <td>0.055556</td>
    </tr>
    <tr>
      <th>H:8</th>
      <td>0.044944</td>
      <td>0.423077</td>
      <td>0.293785</td>
      <td>0.431694</td>
      <td>0.011050</td>
      <td>0.170330</td>
      <td>0.110497</td>
      <td>0.184358</td>
      <td>0.977011</td>
      <td>0.394444</td>
    </tr>
    <tr>
      <th>H:9</th>
      <td>0.421348</td>
      <td>0.000000</td>
      <td>0.005650</td>
      <td>0.759563</td>
      <td>0.000000</td>
      <td>0.351648</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.356322</td>
      <td>0.872222</td>
    </tr>
  </tbody>
</table>
</div>




```python
CC.score_index_info('max')
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
      <th>General Informations</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ball Hall Index</th>
      <td>695.801129</td>
    </tr>
    <tr>
      <th>Calinski-Harabasz Index</th>
      <td>144.190279</td>
    </tr>
    <tr>
      <th>Dunn Index</th>
      <td>0.258976</td>
    </tr>
    <tr>
      <th>PBM Index</th>
      <td>34.224177</td>
    </tr>
    <tr>
      <th>Ratkowsky-Lance Index</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Silhouette Index</th>
      <td>0.162943</td>
    </tr>
    <tr>
      <th>Wemmert-Gançarski Index</th>
      <td>0.250224</td>
    </tr>
    <tr>
      <th>(1, 1)</th>
      <td>0.258976</td>
    </tr>
    <tr>
      <th>(1, 2)</th>
      <td>0.907614</td>
    </tr>
    <tr>
      <th>(1, 3)</th>
      <td>0.315850</td>
    </tr>
    <tr>
      <th>(2, 1)</th>
      <td>0.874361</td>
    </tr>
    <tr>
      <th>(2, 2)</th>
      <td>3.064308</td>
    </tr>
    <tr>
      <th>(2, 3)</th>
      <td>1.066381</td>
    </tr>
    <tr>
      <th>(3, 1)</th>
      <td>0.579069</td>
    </tr>
    <tr>
      <th>(3, 2)</th>
      <td>2.029422</td>
    </tr>
    <tr>
      <th>(3, 3)</th>
      <td>0.706240</td>
    </tr>
    <tr>
      <th>(4, 1)</th>
      <td>0.287558</td>
    </tr>
    <tr>
      <th>(4, 2)</th>
      <td>1.007784</td>
    </tr>
    <tr>
      <th>(4, 3)</th>
      <td>0.350710</td>
    </tr>
    <tr>
      <th>(5, 1)</th>
      <td>0.285157</td>
    </tr>
    <tr>
      <th>(5, 2)</th>
      <td>0.999368</td>
    </tr>
    <tr>
      <th>(5, 3)</th>
      <td>0.347781</td>
    </tr>
    <tr>
      <th>(6, 1)</th>
      <td>0.603307</td>
    </tr>
    <tr>
      <th>(6, 2)</th>
      <td>2.114365</td>
    </tr>
    <tr>
      <th>(6, 3)</th>
      <td>0.735800</td>
    </tr>
  </tbody>
</table>
</div>




```python
CC.score_index_info('min')
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
      <th>General Informations</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Banfeld-Raftery Index</th>
      <td>11718.207536</td>
    </tr>
    <tr>
      <th>C Index</th>
      <td>0.147642</td>
    </tr>
    <tr>
      <th>Ray-Turi Index</th>
      <td>1.585782</td>
    </tr>
    <tr>
      <th>Xie-Beni Index</th>
      <td>1.955131</td>
    </tr>
    <tr>
      <th>Davies Bouldin Index</th>
      <td>2.151710</td>
    </tr>
  </tbody>
</table>
</div>




```python
CC.score_index_info('max diff')
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
      <th>General Informations</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Trace WiB Index</th>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
CC.score_index_info('min diff')
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
      <th>General Informations</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Det Ratio Index</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Log BGSS/WGSS Index</th>
      <td>-0.319935</td>
    </tr>
    <tr>
      <th>SD Index</th>
      <td>[0.627482, 0.070384]</td>
    </tr>
    <tr>
      <th>S_Dbw Index</th>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
CC.data_centroids
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
      <th>target</th>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.022472</td>
      <td>0.010989</td>
      <td>0.932203</td>
      <td>0.644809</td>
      <td>0.000000</td>
      <td>0.967033</td>
      <td>0.000000</td>
      <td>0.167598</td>
      <td>0.143678</td>
      <td>0.144444</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.185393</td>
      <td>2.456044</td>
      <td>9.666667</td>
      <td>8.387978</td>
      <td>0.453039</td>
      <td>9.983516</td>
      <td>1.138122</td>
      <td>5.100559</td>
      <td>5.022989</td>
      <td>5.683333</td>
    </tr>
    <tr>
      <th>3</th>
      <td>13.095506</td>
      <td>9.208791</td>
      <td>14.186441</td>
      <td>14.169399</td>
      <td>7.055249</td>
      <td>13.038462</td>
      <td>11.165746</td>
      <td>13.061453</td>
      <td>11.603448</td>
      <td>11.833333</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11.297753</td>
      <td>10.406593</td>
      <td>9.627119</td>
      <td>14.224044</td>
      <td>11.497238</td>
      <td>13.895604</td>
      <td>9.585635</td>
      <td>14.245810</td>
      <td>12.402299</td>
      <td>11.255556</td>
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
    </tr>
    <tr>
      <th>59</th>
      <td>13.561798</td>
      <td>9.137363</td>
      <td>13.966102</td>
      <td>14.650273</td>
      <td>7.812155</td>
      <td>14.736264</td>
      <td>10.685083</td>
      <td>11.659218</td>
      <td>12.695402</td>
      <td>12.044444</td>
    </tr>
    <tr>
      <th>60</th>
      <td>13.325843</td>
      <td>13.027473</td>
      <td>13.118644</td>
      <td>13.972678</td>
      <td>11.812155</td>
      <td>9.362637</td>
      <td>15.093923</td>
      <td>2.206704</td>
      <td>13.011494</td>
      <td>13.144444</td>
    </tr>
    <tr>
      <th>61</th>
      <td>5.438202</td>
      <td>8.576923</td>
      <td>11.796610</td>
      <td>8.672131</td>
      <td>1.955801</td>
      <td>2.532967</td>
      <td>13.044199</td>
      <td>0.011173</td>
      <td>6.735632</td>
      <td>8.894444</td>
    </tr>
    <tr>
      <th>62</th>
      <td>0.275281</td>
      <td>3.049451</td>
      <td>8.022599</td>
      <td>1.409836</td>
      <td>0.000000</td>
      <td>0.197802</td>
      <td>4.480663</td>
      <td>0.000000</td>
      <td>1.206897</td>
      <td>2.094444</td>
    </tr>
    <tr>
      <th>63</th>
      <td>0.000000</td>
      <td>1.494505</td>
      <td>1.932203</td>
      <td>0.065574</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.093923</td>
      <td>0.000000</td>
      <td>0.011494</td>
      <td>0.055556</td>
    </tr>
  </tbody>
</table>
<p>64 rows × 10 columns</p>
</div>




```python
CC.data_every_cluster_element_distance_to_centroids
```




    {0: 0      14.013361
     10     19.720608
     20     19.137090
     30     21.388412
     36     19.283605
               ...   
     1739   18.091624
     1745   19.299622
     1746   18.396480
     1768   24.722348
     1793   20.429465
     Name: 0, Length: 178, dtype: float64,
     1: 1      19.017525
     11     29.340485
     21     29.926215
     42     27.324792
     47     26.041098
               ...   
     1752   22.998478
     1757   26.761591
     1760   22.013639
     1766   24.926844
     1774   25.908255
     Name: 1, Length: 182, dtype: float64,
     2: 2      37.375370
     12     27.955225
     22     18.014831
     50     35.287723
     51     40.710050
               ...   
     1744   22.068658
     1751   22.441077
     1780   19.522917
     1782   21.069101
     1783   25.302190
     Name: 2, Length: 177, dtype: float64,
     3: 3      22.386098
     13     16.305509
     23     27.238187
     45     20.209307
     59     21.245783
               ...   
     1750   28.238363
     1756   22.330496
     1758   23.520848
     1765   36.034527
     1770   20.777923
     Name: 3, Length: 183, dtype: float64,
     4: 4      28.340976
     14     21.572335
     24     29.670956
     41     17.457633
     64     22.686206
               ...   
     1767   27.421024
     1777   26.523957
     1778   29.727508
     1788   24.926209
     1791   19.072727
     Name: 4, Length: 181, dtype: float64,
     5: 5      39.543609
     15     25.446222
     25     25.950805
     32     21.438240
     33     28.045378
               ...   
     1741   31.955400
     1769   26.902843
     1776   26.698650
     1784   28.695719
     1787   28.832489
     Name: 5, Length: 182, dtype: float64,
     6: 6      22.017400
     16     29.868740
     26     26.306799
     34     22.569675
     58     18.895998
               ...   
     1749   21.540740
     1755   23.057415
     1762   21.970174
     1771   27.684253
     1773   15.836811
     Name: 6, Length: 181, dtype: float64,
     7: 7      26.550397
     17     23.614138
     27     34.611033
     43     26.687444
     44     22.824852
               ...   
     1753   29.686097
     1761   17.819286
     1775   25.876622
     1779   24.550407
     1785   26.005513
     Name: 7, Length: 179, dtype: float64,
     8: 8      24.882755
     18     29.537270
     28     25.303259
     38     30.293996
     40     20.327824
               ...   
     1781   29.399777
     1789   30.636763
     1790   27.442977
     1794   25.598846
     1796   28.071988
     Name: 8, Length: 174, dtype: float64,
     9: 9      31.077677
     19     33.549960
     29     29.607241
     31     32.817506
     37     33.609520
               ...   
     1759   20.496066
     1772   24.457897
     1786   25.681377
     1792   16.959423
     1795   25.207579
     Name: 9, Length: 180, dtype: float64}



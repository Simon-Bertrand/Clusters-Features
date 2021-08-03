# Clusters-Features : a Python module to evaluate the quality of clustering #
Usefull in unsupervised learning. All criterias are used with internal validation and make no use of ground-truth labels. 




## Import the module ##
```python
from ClustersFeatures import *
```

## Load a random data set ##
We choose here the scikit-learn digits data set because it is in high dimension (64) and has a large number of observations.

```python
from sklearn.datasets import load_digits
import pandas as pd
digits = load_digits()
pd_df=pd.DataFrame(digits.data)
pd_df['target'] = digits.target
pd_df
```

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

The important thing is that the given "pd_df" dataframe in the following argument has to be concatenated with the target vector.
Then, just specify as second argument which column name has the target. The program is making automatically the separation :
```python
CC=ClustersCharacteristics(pd_df,"target")
```

## Data tools ##

The ClustersCharacteristics object creates attributes that define clusters. We can find for example the barycenter.


```python
CC.data_barycenter
```

    0      0.000000
    1      0.303840
    2      5.204786
    3     11.835838
    4     11.848080
              ...    
    59    12.089037
    60    11.809126
    61     6.764051
    62     2.067891
    63     0.364496
    Length: 64, dtype: float64


But also centroids, where the column j of the following matrix correspond to the coordinates of centroid of cluster j.
```python
CC.data_centroids
```
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


We can show the list of clusters labels :

```python
CC.labels_clusters
```
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

And look for the data with the same label target. For example we take here the first cluster label of the above list.

```python
Cluster=CC.labels_clusters[0]
CC.data_clusters[Cluster]
```

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
      <th>54</th>
      <th>55</th>
      <th>56</th>
      <th>57</th>
      <th>58</th>
      <th>59</th>
      <th>60</th>
      <th>61</th>
      <th>62</th>
      <th>63</th>
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
      <td>0.0</td>
      <td>6.0</td>
      <td>13.0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>9.0</td>
      <td>15.0</td>
      <td>11.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>10.0</td>
      <td>13.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>13.0</td>
      <td>11.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>12.0</td>
      <td>13.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>30</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>14.0</td>
      <td>11.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>11.0</td>
      <td>16.0</td>
      <td>12.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>36</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>14.0</td>
      <td>10.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>16.0</td>
      <td>11.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
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
      <th>1739</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>11.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>12.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1745</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>14.0</td>
      <td>8.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>13.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1746</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>15.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8.0</td>
      <td>15.0</td>
      <td>11.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1768</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>16.0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>15.0</td>
      <td>16.0</td>
      <td>8.0</td>
      <td>1.0</td>
      <td>0.0</td>
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
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>16.0</td>
      <td>14.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>178 rows × 64 columns</p>
</div>


Users are able to get a pairwise distance matrix generated by the Scipy library (fast). 
If (xi,j)i,j is the returned matrix, then xi,j is the distance between element of index i and element of index j. The matrix is symetric as we use Euclidian norm to evaluate distances.
```python
CC.data_every_element_distance_to_every_element
```
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
      <th>1787</th>
      <th>1788</th>
      <th>1789</th>
      <th>1790</th>
      <th>1791</th>
      <th>1792</th>
      <th>1793</th>
      <th>1794</th>
      <th>1795</th>
      <th>1796</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>59.556696</td>
      <td>54.129474</td>
      <td>47.571000</td>
      <td>50.338852</td>
      <td>43.908997</td>
      <td>48.559242</td>
      <td>56.000000</td>
      <td>44.395946</td>
      <td>40.804412</td>
      <td>...</td>
      <td>39.874804</td>
      <td>49.749372</td>
      <td>52.640289</td>
      <td>51.458721</td>
      <td>49.989999</td>
      <td>36.249138</td>
      <td>26.627054</td>
      <td>50.378567</td>
      <td>37.067506</td>
      <td>47.031904</td>
    </tr>
    <tr>
      <th>1</th>
      <td>59.556696</td>
      <td>0.000000</td>
      <td>41.629317</td>
      <td>45.475268</td>
      <td>47.906158</td>
      <td>47.127487</td>
      <td>40.286474</td>
      <td>50.960769</td>
      <td>48.620983</td>
      <td>52.820451</td>
      <td>...</td>
      <td>52.009614</td>
      <td>48.969378</td>
      <td>42.965102</td>
      <td>32.572995</td>
      <td>47.707442</td>
      <td>51.390661</td>
      <td>59.177699</td>
      <td>38.587563</td>
      <td>48.569538</td>
      <td>50.328918</td>
    </tr>
    <tr>
      <th>2</th>
      <td>54.129474</td>
      <td>41.629317</td>
      <td>0.000000</td>
      <td>53.953684</td>
      <td>52.096065</td>
      <td>55.443665</td>
      <td>45.650849</td>
      <td>49.335586</td>
      <td>42.602817</td>
      <td>54.836119</td>
      <td>...</td>
      <td>59.076222</td>
      <td>47.927028</td>
      <td>46.335731</td>
      <td>39.191836</td>
      <td>46.936127</td>
      <td>51.826634</td>
      <td>52.009614</td>
      <td>38.340579</td>
      <td>50.774009</td>
      <td>43.954522</td>
    </tr>
    <tr>
      <th>3</th>
      <td>47.571000</td>
      <td>45.475268</td>
      <td>53.953684</td>
      <td>0.000000</td>
      <td>51.215232</td>
      <td>33.660065</td>
      <td>47.254629</td>
      <td>56.824291</td>
      <td>42.449971</td>
      <td>45.166359</td>
      <td>...</td>
      <td>37.934153</td>
      <td>55.569776</td>
      <td>50.099900</td>
      <td>43.988635</td>
      <td>58.566202</td>
      <td>40.286474</td>
      <td>55.551778</td>
      <td>49.527770</td>
      <td>44.147480</td>
      <td>41.267421</td>
    </tr>
    <tr>
      <th>4</th>
      <td>50.338852</td>
      <td>47.906158</td>
      <td>52.096065</td>
      <td>51.215232</td>
      <td>0.000000</td>
      <td>54.147945</td>
      <td>36.959437</td>
      <td>59.481089</td>
      <td>52.507142</td>
      <td>55.054518</td>
      <td>...</td>
      <td>48.620983</td>
      <td>26.172505</td>
      <td>55.794265</td>
      <td>48.723711</td>
      <td>31.416556</td>
      <td>53.981478</td>
      <td>51.449004</td>
      <td>46.882833</td>
      <td>52.668776</td>
      <td>50.970580</td>
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
      <td>36.249138</td>
      <td>51.390661</td>
      <td>51.826634</td>
      <td>40.286474</td>
      <td>53.981478</td>
      <td>29.325757</td>
      <td>52.191953</td>
      <td>55.605755</td>
      <td>40.037482</td>
      <td>36.262929</td>
      <td>...</td>
      <td>31.749016</td>
      <td>54.543561</td>
      <td>55.758407</td>
      <td>48.083261</td>
      <td>55.488738</td>
      <td>0.000000</td>
      <td>41.940434</td>
      <td>46.151923</td>
      <td>23.537205</td>
      <td>40.963398</td>
    </tr>
    <tr>
      <th>1793</th>
      <td>26.627054</td>
      <td>59.177699</td>
      <td>52.009614</td>
      <td>55.551778</td>
      <td>51.449004</td>
      <td>49.325450</td>
      <td>45.354162</td>
      <td>60.456596</td>
      <td>48.041649</td>
      <td>47.265209</td>
      <td>...</td>
      <td>43.416587</td>
      <td>45.912961</td>
      <td>53.272882</td>
      <td>52.449976</td>
      <td>46.324939</td>
      <td>41.940434</td>
      <td>0.000000</td>
      <td>46.957428</td>
      <td>42.438190</td>
      <td>46.465041</td>
    </tr>
    <tr>
      <th>1794</th>
      <td>50.378567</td>
      <td>38.587563</td>
      <td>38.340579</td>
      <td>49.527770</td>
      <td>46.882833</td>
      <td>46.904158</td>
      <td>33.466401</td>
      <td>54.516053</td>
      <td>34.885527</td>
      <td>49.929951</td>
      <td>...</td>
      <td>45.077711</td>
      <td>46.421978</td>
      <td>33.896903</td>
      <td>29.189039</td>
      <td>42.602817</td>
      <td>46.151923</td>
      <td>46.957428</td>
      <td>0.000000</td>
      <td>44.158804</td>
      <td>28.879058</td>
    </tr>
    <tr>
      <th>1795</th>
      <td>37.067506</td>
      <td>48.569538</td>
      <td>50.774009</td>
      <td>44.147480</td>
      <td>52.668776</td>
      <td>32.557641</td>
      <td>48.207883</td>
      <td>55.928526</td>
      <td>37.000000</td>
      <td>28.827071</td>
      <td>...</td>
      <td>38.183766</td>
      <td>50.507425</td>
      <td>54.359912</td>
      <td>47.265209</td>
      <td>48.754487</td>
      <td>23.537205</td>
      <td>42.438190</td>
      <td>44.158804</td>
      <td>0.000000</td>
      <td>39.420807</td>
    </tr>
    <tr>
      <th>1796</th>
      <td>47.031904</td>
      <td>50.328918</td>
      <td>43.954522</td>
      <td>41.267421</td>
      <td>50.970580</td>
      <td>38.496753</td>
      <td>40.224371</td>
      <td>56.267220</td>
      <td>28.337255</td>
      <td>40.926764</td>
      <td>...</td>
      <td>38.288379</td>
      <td>50.941143</td>
      <td>38.820098</td>
      <td>38.600518</td>
      <td>49.223978</td>
      <td>40.963398</td>
      <td>46.465041</td>
      <td>28.879058</td>
      <td>39.420807</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>1797 rows × 1797 columns</p>
</div>

While centroids are not elements of the dataset, we can also compute the distance between each element to each centroid.
```python
CC.data_every_element_distance_to_centroids
```

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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>14.013361</td>
      <td>47.567376</td>
      <td>43.896678</td>
      <td>39.554151</td>
      <td>40.407399</td>
      <td>36.647929</td>
      <td>41.599287</td>
      <td>43.074401</td>
      <td>37.369109</td>
      <td>32.423583</td>
    </tr>
    <tr>
      <th>1</th>
      <td>54.059820</td>
      <td>19.017525</td>
      <td>38.701490</td>
      <td>42.313696</td>
      <td>38.269485</td>
      <td>42.273369</td>
      <td>44.388144</td>
      <td>40.861554</td>
      <td>33.800663</td>
      <td>44.312148</td>
    </tr>
    <tr>
      <th>2</th>
      <td>47.757029</td>
      <td>32.206345</td>
      <td>37.375370</td>
      <td>45.438311</td>
      <td>43.187064</td>
      <td>50.233787</td>
      <td>43.272912</td>
      <td>41.584089</td>
      <td>33.846710</td>
      <td>45.754658</td>
    </tr>
    <tr>
      <th>3</th>
      <td>44.250476</td>
      <td>36.468356</td>
      <td>33.283540</td>
      <td>22.386098</td>
      <td>48.069136</td>
      <td>36.198086</td>
      <td>41.894212</td>
      <td>45.404349</td>
      <td>36.063988</td>
      <td>31.605201</td>
    </tr>
    <tr>
      <th>4</th>
      <td>45.592148</td>
      <td>39.322928</td>
      <td>52.408033</td>
      <td>51.138040</td>
      <td>28.340976</td>
      <td>48.653228</td>
      <td>39.984571</td>
      <td>48.264247</td>
      <td>44.208386</td>
      <td>48.142841</td>
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
      <th>1792</th>
      <td>34.293071</td>
      <td>41.151239</td>
      <td>43.677849</td>
      <td>30.575459</td>
      <td>45.769008</td>
      <td>36.516206</td>
      <td>46.428263</td>
      <td>42.576472</td>
      <td>33.327387</td>
      <td>16.959423</td>
    </tr>
    <tr>
      <th>1793</th>
      <td>20.429465</td>
      <td>48.646926</td>
      <td>47.491512</td>
      <td>45.995613</td>
      <td>40.876460</td>
      <td>40.516369</td>
      <td>41.939685</td>
      <td>46.740119</td>
      <td>40.285358</td>
      <td>40.804954</td>
    </tr>
    <tr>
      <th>1794</th>
      <td>44.631741</td>
      <td>29.885611</td>
      <td>38.886808</td>
      <td>43.396579</td>
      <td>36.316532</td>
      <td>42.594489</td>
      <td>37.825320</td>
      <td>42.725794</td>
      <td>25.598846</td>
      <td>42.437926</td>
    </tr>
    <tr>
      <th>1795</th>
      <td>34.565247</td>
      <td>39.389382</td>
      <td>43.806621</td>
      <td>35.557630</td>
      <td>41.311856</td>
      <td>38.202818</td>
      <td>41.673728</td>
      <td>42.664164</td>
      <td>32.926630</td>
      <td>25.207579</td>
    </tr>
    <tr>
      <th>1796</th>
      <td>41.031409</td>
      <td>37.724803</td>
      <td>37.444086</td>
      <td>36.772758</td>
      <td>42.390657</td>
      <td>40.799218</td>
      <td>33.921312</td>
      <td>45.775584</td>
      <td>28.071988</td>
      <td>35.917805</td>
    </tr>
  </tbody>
</table>
<p>1797 rows × 10 columns</p>
</div>

It is possible to generate a matrix of intercentroid distance.
If (xi,j)i,j is the returned matrix, then xi,j is the distance between centroid of cluster i to centroid of cluster j.
These distances are not related to points of the dataset. We put NaN into the diagonal terms in order to facilitate the manipulation of min/max.

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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>nan</td>
      <td>42.026024</td>
      <td>39.274919</td>
      <td>37.062579</td>
      <td>35.981220</td>
      <td>34.078029</td>
      <td>34.274506</td>
      <td>41.772576</td>
      <td>32.909593</td>
      <td>29.617374</td>
    </tr>
    <tr>
      <th>1</th>
      <td>42.026024</td>
      <td>nan</td>
      <td>28.949723</td>
      <td>31.742287</td>
      <td>28.674700</td>
      <td>32.469295</td>
      <td>34.570287</td>
      <td>31.187817</td>
      <td>20.950348</td>
      <td>32.126942</td>
    </tr>
    <tr>
      <th>2</th>
      <td>39.274919</td>
      <td>28.949723</td>
      <td>nan</td>
      <td>26.489600</td>
      <td>42.689686</td>
      <td>32.375712</td>
      <td>36.657425</td>
      <td>35.570382</td>
      <td>25.605848</td>
      <td>32.960968</td>
    </tr>
    <tr>
      <th>3</th>
      <td>37.062579</td>
      <td>31.742287</td>
      <td>26.489600</td>
      <td>nan</td>
      <td>43.499594</td>
      <td>29.822474</td>
      <td>41.152654</td>
      <td>33.369483</td>
      <td>25.511462</td>
      <td>21.103269</td>
    </tr>
    <tr>
      <th>4</th>
      <td>35.981220</td>
      <td>28.674700</td>
      <td>42.689686</td>
      <td>43.499594</td>
      <td>nan</td>
      <td>35.577158</td>
      <td>30.756650</td>
      <td>33.444921</td>
      <td>31.858925</td>
      <td>38.689544</td>
    </tr>
    <tr>
      <th>5</th>
      <td>34.078029</td>
      <td>32.469295</td>
      <td>32.375712</td>
      <td>29.822474</td>
      <td>35.577158</td>
      <td>nan</td>
      <td>35.573804</td>
      <td>32.098017</td>
      <td>25.867262</td>
      <td>28.060732</td>
    </tr>
    <tr>
      <th>6</th>
      <td>34.274506</td>
      <td>34.570287</td>
      <td>36.657425</td>
      <td>41.152654</td>
      <td>30.756650</td>
      <td>35.573804</td>
      <td>nan</td>
      <td>43.514148</td>
      <td>31.227114</td>
      <td>39.306699</td>
    </tr>
    <tr>
      <th>7</th>
      <td>41.772576</td>
      <td>31.187817</td>
      <td>35.570382</td>
      <td>33.369483</td>
      <td>33.444921</td>
      <td>32.098017</td>
      <td>43.514148</td>
      <td>nan</td>
      <td>27.364089</td>
      <td>33.513179</td>
    </tr>
    <tr>
      <th>8</th>
      <td>32.909593</td>
      <td>20.950348</td>
      <td>25.605848</td>
      <td>25.511462</td>
      <td>31.858925</td>
      <td>25.867262</td>
      <td>31.227114</td>
      <td>27.364089</td>
      <td>nan</td>
      <td>24.630553</td>
    </tr>
    <tr>
      <th>9</th>
      <td>29.617374</td>
      <td>32.126942</td>
      <td>32.960968</td>
      <td>21.103269</td>
      <td>38.689544</td>
      <td>28.060732</td>
      <td>39.306699</td>
      <td>33.513179</td>
      <td>24.630553</td>
      <td>nan</td>
    </tr>
  </tbody>
</table>
</div>

## Scores ##
There are many indices that allow users to evaluate the quality of clusters, such as internal cluster validation indices. In Python development, some libraries compute such scores, but it is not completely done. In this library, these scores have been implemented :
- Total dispersion matrix
- Within cluster dispersion matrixes
- Between group dispersion matrix
- Total sum square
- Pooled within cluster dispersion

The implemented indexes are :
- Ball-Hall Index
- Dunn Index
- Generalized Dunn Indexes (18 indexes)
- C Index
- Banfeld-Raftery Index
- Davies-Bouldin Index
- Calinski-Harabasz Index
- Ray-Turi Index
- Xie-Beni Index
- Ratkowsky Lance Index
- SD Index
- Mclain Rao Index
- Scott-Symons Index
- PBM Index
- Point biserial Index
- Det Ratio Index
- Log SumSquare Ratio Index
- Silhouette Index (computed with scikit-learn)
- Wemmert-Gançarski Index (Thanks to M.Gançarski for this intership)

Main reference for all these scores :

Clustering Indices

Bernard Desgraupes, University Paris Ouest - Lab Modal’X , November 2017 

https://cran.r-project.org/web/packages/clusterCrit/vignettes/clusterCrit.pdf 


#### In this library, there are two types of methods to calculate these scores: Using IndexCore which automatically caches the already calculated indexes or calculating directly using the "score_index_" methods. The second method can make the calculation of the same index repetitive, which can be very slow because we know that some of these indexes have a very high computational complexity. ####


### First method using IndexCore (faster) ###


```python
CC.compute_every_index()
```
    {'general': {'max': {'Between-group total dispersion': 908297.1736053203,
    'Mean quadratic error': 696.0267765360618,
    'Silhouette Index': 0.16294320522575195,
    'Dunn Index': 0.25897601382124175,
    'Generalized Dunn Indexes': {'GDI (1, 1)': 0.25897601382124175,
    'GDI (1, 2)': 0.9076143747196692,
    'GDI (1, 3)': 0.3158503201955148,
    'GDI (2, 1)': 0.25897601382124175,
    'GDI (2, 2)': 0.9076143747196692,
    'GDI (2, 3)': 0.3158503201955148,
    'GDI (3, 1)': 0.5790691834873279,
    'GDI (3, 2)': 2.0294215944379173,
    'GDI (3, 3)': 0.7062398726473335,
    'GDI (4, 1)': 0.2875582147985151,
    'GDI (4, 2)': 1.0077843328765126,
    'GDI (4, 3)': 0.35070952278095474,
    'GDI (5, 1)': 0.28515682596025516,
    'GDI (5, 2)': 0.9993683603053982,
    'GDI (5, 3)': 0.34778075952490317,
    'GDI (6, 1)': 0.6033066382644287,
    'GDI (6, 2)': 2.1143648370097905,
    'GDI (6, 3)': 0.735800169522378},
    'Wemmert-Gancarski Index': 0.2502241827215019,
    'Calinski-Harabasz Index': 144.1902786959258,
    'Ratkowsky-Lance Index': nan,
    'Point Biserial Index': -4.064966952313242,
    'PBM Index': 34.22417733472788},
    'max diff': {'Trace WiB Index': nan, 'Trace W Index': 1250760.117435303},
    'min': {'Banfeld-Raftery Index': 11718.207536490032,
    'Ball Hall Index': 695.801129352618,
    'C Index': 0.1476415026698158,
    'Ray-Turi Index': 1.5857819700225737,
    'Xie-Beni Index': 1.9551313947642188,
    'Davies Bouldin Index': 2.1517097380390937,
    'SD Index': [array([0.627482, 0.070384])],
    'Mclain-Rao Index': 0.7267985756237975,
    'Scott-Symons Index': nan},
    'min diff': {'Det Ratio Index': nan,
    'Log BGSS/WGSS Index': -0.3199351306684197,
    'S_Dbw Index': nan,
    'Nlog Det Ratio Index': nan}},
    'clusters': {'max': {'Centroid distance to barycenter': [26.422334274375757,
    20.184062405495773,
    22.958470492954795,
    21.71559561353746,
    25.717240507145213,
    20.283308612864644,
    26.419951469008378,
    24.426658073844308,
    13.44306158441342,
    19.876908956223936],
    'Between-group Dispersion': [124268.87523421964,
    74146.1402843885,
    93295.17202553005,
    86296.77799167577,
    119709.13913372548,
    74877.09470781704,
    126340.5042480812,
    106802.4308135141,
    31444.567428645732,
    71116.47173772276],
    'Average Silhouette': [0.3608993843537291,
    0.05227459502398472,
    0.14407593888502124,
    0.15076708301431302,
    0.16517001390130848,
    0.1194825125348905,
    0.28763816949713245,
    0.19373598833558672,
    0.08488231267929798,
    0.07117051617968871],
    'KernelDensity mean': [-87.26207798353086,
    -102.79627948741418,
    -118.2807433740146,
    -102.80193279131969,
    -102.79094365877583,
    -102.79645332546204,
    -87.27879146450985,
    -102.77983243274437,
    -118.2636521439672,
    -118.29755528330563],
    'Ball Hall Index': [396.35042923873254,
    940.6359437266029,
    751.2059752944557,
    633.6276389262146,
    736.2863160465186,
    757.3853701243812,
    512.8915478770488,
    734.7467931712492,
    741.1588717135685,
    753.7224074074073]},
    'min': {'Within-Cluster Dispersion': [70550.3764044944,
    171195.74175824173,
    132963.45762711865,
    115953.85792349727,
    133267.82320441987,
    137844.13736263738,
    92833.37016574583,
    131519.67597765362,
    128961.64367816092,
    135670.03333333333],
    'Largest element distance': [54.543560573178574,
    72.85602240034794,
    67.0,
    62.3377895020348,
    71.69379331573968,
    66.53570470055908,
    61.155539405682624,
    67.93379129711516,
    61.171888968708494,
    63.773035054010094],
    'Inter-element mean distance': [27.495251790928528,
    41.577045912127325,
    37.66525398978789,
    34.81272464303223,
    37.28558306007683,
    38.08288651715454,
    31.222158502521683,
    37.241230341156786,
    37.938810358062234,
    37.830620986872184],
    'Davies Bouldin Index': array([1.55628353, 2.70948787, 2.09498538, 2.43120015, 1.96455875,
          2.09074874, 1.58102612, 1.94811882, 2.70948787, 2.43120015]),
    'C Index': [0.15780619270180213,
    0.4626045226116365,
    0.37889533673771314,
    0.31459485530776515,
    0.3693066184157008,
    0.38636193134197444,
    0.23717385124578905,
    0.36902306811086555,
    0.3857833597084178,
    0.3815092165505222]}},
    'radius': {'min': {'Radius min': {0: 11.963104233270684,
    1: 16.495963249417844,
    2: 17.228366828448973,
    3: 15.096075210359995,
    4: 15.943646753449636,
    5: 16.46455777853301,
    6: 12.786523861254974,
    7: 14.61523732739271,
    8: 18.374826032773953,
    9: 16.317673899226},
    'Radius mean': {0: 19.364954,
    1: 29.868519,
    2: 26.747682,
    3: 24.578193,
    4: 26.464614,
    5: 27.18575,
    6: 22.162453,
    7: 26.412302,
    8: 26.896195,
    9: 26.728077},
    'Radius median': {0: 19.090152,
    1: 27.705495,
    2: 25.299287,
    3: 23.495162,
    4: 26.434238,
    5: 27.194139,
    6: 21.579562,
    7: 25.358031,
    8: 26.982504,
    9: 25.201186},
    'Radius 75th Percentile': {0: 22.142983,
    1: 35.627396,
    2: 30.263862,
    3: 27.808539,
    4: 29.727508,
    5: 29.221274,
    6: 24.736136,
    7: 30.21675,
    8: 30.137334,
    9: 29.966933},
    'Radius max': {0: 35.381597,
    1: 48.76808,
    2: 48.6619,
    3: 40.02036,
    4: 51.535976,
    5: 40.584931,
    6: 42.250871,
    7: 44.424333,
    8: 38.175815,
    9: 45.985382}}}}

We can take the corresponding code in the indices.json file with this call
```python
CC._get_all_index
```
`
{'general': {'max': {'Between-group total dispersion': 'G-Max-01',
       'Mean quadratic error': 'G-Max-02',
       'Silhouette Index': 'G-Max-03',
       'Dunn Index': 'G-Max-04',
       'Generalized Dunn Indexes': 'G-Max-GDI',
       'Wemmert-Gancarski Index': 'G-Max-05',
       'Calinski-Harabasz Index': 'G-Max-06',
       'Ratkowsky-Lance Index': 'G-Max-07',
       'Point Biserial Index': 'G-Max-08',
       'PBM Index': 'G-Max-09'},
      'max diff': {'Trace WiB Index': 'G-MaxD-01', 'Trace W Index': 'G-MaxD-02'},
      'min': {'Banfeld-Raftery Index': 'G-Min-01',
       'Ball Hall Index': 'G-Min-02',
       'C Index': 'G-Min-03',
       'Ray-Turi Index': 'G-Min-04',
       'Xie-Beni Index': 'G-Min-05',
       'Davies Bouldin Index': 'G-Min-06',
       'SD Index': 'G-Min-07',
       'Mclain-Rao Index': 'G-Min-08',
       'Scott-Symons Index': 'G-Min-09'},
      'min diff': {'Det Ratio Index': 'G-MinD-01',
       'Log BGSS/WGSS Index': 'G-MinD-02',
       'S_Dbw Index': 'G-MinD-03',
       'Nlog Det Ratio Index': 'G-MinD-04'}},
     'clusters': {'max': {'Centroid distance to barycenter': 'C-Max-01',
       'Between-group Dispersion': 'C-Max-02',
       'Average Silhouette': 'C-Max-03',
       'KernelDensity mean': 'C-Max-04',
       'Ball Hall Index': 'C-Max-05'},
      'min': {'Within-Cluster Dispersion': 'C-Min-01',
       'Largest element distance': 'C-Min-02',
       'Inter-element mean distance': 'C-Min-03',
       'Davies Bouldin Index': 'C-Min-04',
       'C Index': 'C-Min-05'}},
     'radius': {'min': {'Radius min': 'R-Min-01',
       'Radius mean': 'R-Min-02',
       'Radius median': 'R-Min-03',
       'Radius 75th Percentile': 'R-Min-04',
       'Radius max': 'R-Min-05'}}}
`

These codes are usefull when you want to generate a single index using IndexCore : 

```python
CC.generate_output_by_info_type("general", "max", "G-Max-01")
908297.1736053203
```

### Second method using "score_index_" methods ###

```python
CC.score_between_group_dispersion()
908297.1736053203
```
Make the same result as above but it computes a second time the same score.


### Speed test of different scores

```
pd_df :  
shape - (1797, 65) 
 total elements=116805 


score_index_ball_hall 

5.06 ms ± 79.9 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

score_index_banfeld_Raftery

5.02 ms ± 39.2 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

score_index_c

104 ms ± 550 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

score_index_c_for_each_cluster

95.5 ms ± 720 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

score_index_calinski_harabasz 

16.5 ms ± 257 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

score_index_davies_bouldin 

12.3 ms ± 76.5 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

score_index_davies_bouldin_for_each_cluster 

12.3 ms ± 300 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

score_index_det_ratio 

181 ms ± 4.09 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

score_index_dunn

19.6 ms ± 1.06 ms per loop (mean ± std. dev. of 7 runs, 100 loops each)

score_index_generalized_dunn_matrix

994 ms ± 41.4 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

score_index_Log_Det_ratio

180 ms ± 2.86 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

score_index_log_ss_ratio 

16.3 ms ± 249 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

score_index_mclain_rao 

63.5 ms ± 6.55 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

score_index_PBM 

23.5 ms ± 3.84 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

score_index_point_biserial

50.3 ms ± 434 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

score_index_ratkowsky_lance 

12.3 ms ± 254 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

score_index_ray_turi 

23.2 ms ± 889 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

score_index_scott_symons

153 ms ± 6.25 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

score_index_SD 

211 ms ± 4.37 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

score_index_trace_WiB 

138 ms ± 2.69 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

score_index_wemmert_gancarski 

8.13 ms ± 93 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

score_index_xie_beni

85.8 ms ± 1.31 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
```

## Confusion Hypersphere ##
The confusion hypersphere subclass counts the number of element contained inside a n-dim sphere (hypersphere) of given radius and centered on each cluster centroid.
The given radius is the same for each hypersphere.

Args : "counting_type=" : ('including' or 'excluding') - If including, then the elements belonging cluster i and contained inside the hypersphere of centroid i are counted (for i=j). If excluding, then they're not counted.
"proportion=" : (bool) Return the proportion of element. Default option = False.



#### self.confusion_hypersphere_matrix
```python
CC.confusion_hypersphere_matrix(radius=35, counting_type="including", proportion=True)
```
<div>
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

To interpret this, if (xi,j)i,j is the returned matrix, then xi,j is the number of elements belonging cluster j that are contained inside the hypersphere with given radius centered on centroid of cluster i . If proportion is on True, then the number of elements becomes the proportion of elements belonging cluster j.





#### self.confusion_hypersphere_for_linspace_radius_each_element
This method returns the results of the above method for a linear radius space. "n_pts=" allows users to set the radius range.
```python
CC.confusion_hypersphere_for_linspace_radius_each_element(radius=35, counting_type="excluding", n_pts=10)
```

<div>
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
    </tr>
    <tr>
      <th>Radius</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.0000</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7.1578</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14.3155</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21.4733</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28.6311</th>
      <td>3</td>
      <td>10</td>
      <td>3</td>
      <td>36</td>
      <td>1</td>
      <td>24</td>
      <td>1</td>
      <td>0</td>
      <td>34</td>
      <td>30</td>
    </tr>
    <tr>
      <th>35.7889</th>
      <td>192</td>
      <td>171</td>
      <td>161</td>
      <td>398</td>
      <td>71</td>
      <td>211</td>
      <td>122</td>
      <td>68</td>
      <td>473</td>
      <td>325</td>
    </tr>
    <tr>
      <th>42.9466</th>
      <td>1004</td>
      <td>747</td>
      <td>802</td>
      <td>1023</td>
      <td>641</td>
      <td>940</td>
      <td>837</td>
      <td>765</td>
      <td>1285</td>
      <td>950</td>
    </tr>
    <tr>
      <th>50.1044</th>
      <td>1567</td>
      <td>1346</td>
      <td>1397</td>
      <td>1470</td>
      <td>1318</td>
      <td>1536</td>
      <td>1534</td>
      <td>1369</td>
      <td>1558</td>
      <td>1479</td>
    </tr>
    <tr>
      <th>57.2622</th>
      <td>1602</td>
      <td>1625</td>
      <td>1589</td>
      <td>1636</td>
      <td>1624</td>
      <td>1638</td>
      <td>1629</td>
      <td>1603</td>
      <td>1566</td>
      <td>1614</td>
    </tr>
    <tr>
      <th>64.4200</th>
      <td>1602</td>
      <td>1638</td>
      <td>1593</td>
      <td>1647</td>
      <td>1629</td>
      <td>1638</td>
      <td>1629</td>
      <td>1611</td>
      <td>1566</td>
      <td>1620</td>
    </tr>
  </tbody>
</table>
</div>


####confusion_hyperphere_around_specific_point_for_two_clusters
This method returns the number of elements belonging given Cluster1 or given Cluster2 that are contained inside the hypersphere of given radius and centered on given Point.

```python
Point= CC.data_features.iloc[0] #Choose an  observation  of the dataset
Cluster1= CC.labels_clusters[0] #Choose the cluster 1
Cluster2=CC.labels_clusters[1] #Choose the cluster 2
radius=110 #Large radius to capture the total of both clusters, the result should be the sum of data_clusters[Cluster1] and data_clusters[Cluster2] cardinals

CC.confusion_hyperphere_around_specific_point_for_two_clusters(Point,Cluster1,Cluster2, radius)
```
    0    360 
    dtype: int64





.. _my-reference-label:

Score
--------------------------
This section allows users to evaluate their clustering by checking values of the indices from below.


References :

`Clustering Indice - Bernard Desgraupes (University Paris Ouest, Lab Modal’X) - 2017 <https://cran.r-project.org/web/packages/clusterCrit/vignettes/clusterCrit.pdf>`_

`Study on Different Cluster Validity Indices - Shyam Kumar K, Dr. Raju G (NSS College Rajakumari, Idukki & Kannur University, Kannur in Kerala, India) - 2018 <https://www.ripublication.com/ijaer18/ijaerv13n11_86.pdf>`_

`Understanding of Internal Clustering Validation Measures - Yanchi Liu, Zhongmou Li, Hui Xiong, Xuedong Gao, Junjie Wu - 2010 <http://datamining.rutgers.edu/publication/internalmeasures.pdf>`_




Scatter Score
===========

.. autoclass:: ClustersFeatures.src._score.__Score
   :members:

Index
===========

.. autoclass:: ClustersFeatures.src._score_index.__ScoreIndex
   :members:


IndexCore
===========
In this library, there are two types of methods to calculate these scores: Using IndexCore which automatically caches the already calculated indexes or calculating directly using the score_index methods.
The second method can make the calculation of the same index repetitive, which can be very slow because we know that some of these indexes have a very high computational complexity.

.. warning::
    Special care to the indices.json structure.
    All the IndexCore class is based on this json structure.
    Modifying the aspect of indices.json brings to modifying the structure of many functions in this document.
    In other words, it is strongly discouraged to modify the global aspect of the json without having done a thorough analysis of the program.
    To add an index, it is important to add data to the json following its current structure.

    indices.json structure dependency : indices.json, _info.py, __init__

.. autoclass:: ClustersFeatures.index_core.__IndexCore
   :members:

.. autodata:: ClustersFeatures.index_core.__IndexCore
   :members:







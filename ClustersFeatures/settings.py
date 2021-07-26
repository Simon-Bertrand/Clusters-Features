#Activate and desactivate sub_classes different loadings for faster compute. For example, if you do not need any ploting library, you can pass the False argument to the subclass Graph
Activated_Graph=True
Activated_Utils=True

#Define the digits precision for the different boards
precision=6

#Indices List : need to be the same string as the general_info / cluster_info boards indexes
indices_max = ['Ball Hall Index', 'Calinski-Harabasz Index', 'Dunn Index', 'PBM Index', 'Ratkowsky-Lance Index',
                    'Silhouette Index', 'Wemmert-Gançarski Index', 'Point Biserial Index']
indices_min = ['Banfeld-Raftery Index', 'C Index', 'Ray-Turi Index', 'Xie-Beni Index', 'Davies Bouldin Index', 'SD Index', 'Mclain-Rao Index', 'Scott-Symons Index']
indices_max_diff = ['Trace WiB Index', 'Trace W Index']
indices_min_diff = ['Det Ratio Index', 'Log BGSS/WGSS Index', 'S_Dbw Index', "Nlog Det Ratio Index"]

#List_total_Packages for creating a requirements.txt before the build in order to pip install the rights librairies according to above Activated choices
#Modyfing this without knowledge of the library is quite dangerous
List_total_Packages= ["numpy","pandas","scipy","scikit-learn"]
List_Graph=["plotly"]
List_Utils=["numba","umap-learn>=0.5.1", "statsmodels"]
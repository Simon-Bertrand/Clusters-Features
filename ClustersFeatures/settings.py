#Activate and desactivate sub_classes different loadings for faster compute. For example, if you do not need any ploting library, you can pass the False argument to the subclass Graph
Activated_Graph=True
Activated_Utils=True

#Define the digits precision for the different boards
precision=6

#List_total_Packages for creating a requirements.txt before the build in order to pip install the rights librairies according to above Activated choices
#Modyfing this without knowledge of the library is quite dangerous
List_base_Packages= ["numpy","pandas","scipy","scikit-learn"]
List_Graph=["plotly"]
List_Utils=["numba","umap-learn>=0.5.1", "statsmodels", "Pillow"]

#Clusters colors for Graph section
discrete_colors=['#1abc9c','#2ecc71','#3498db','#9b59b6','#34495e','#f1c40f','#e67e22','#e74c3c','#ff5e57','#00d8d6','#0fbcf9','#DEA0FD','#FE00FA','#325A9B','#FEAF16','#F8A19F','#90AD1C','#F6222E','#1CFFCE','#2ED9FF','#B10DA1','#C075A6','#FC1CBF','#B00068','#FBE426','#FA0087']

"""
   Section: Graph
     _____   _____               _____    _    _
    / ____| |  __ \      /\     |  __ \  | |  | |
   | |  __  | |__) |    /  \    | |__) | | |__| |
   | | |_ | |  _  /    / /\ \   |  ___/  |  __  |
   | |__| | | | \ \   / ____ \  | |      | |  | |
    \_____| |_|  \_\ /_/    \_\ |_|      |_|  |_|
   Graph functions to visualize clusters characteristics. Use the library Plotly to plot the data
   """

from ClustersFeatures import settings
if settings.Activated_Graph:
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go


    class __Graph(object):
        def graph_boxplots_distances_to_centroid(self, Cluster):
            if not (Cluster in self.labels_clusters):
                raise KeyError(
                    'A such cluster name "' + Cluster + '" isn\'t found in dataframe\'s clusters. Here are the available clusters : ' + str(
                        list(self.labels_clusters)))
            else:
                Result = pd.DataFrame(data=self.data_target.values, columns=['Cluster'])
                Result['Distance'] = self.data_every_element_distance_to_centroids[Cluster]
                fig = go.Figure()
                for Cluster_ in self.labels_clusters:
                    fig.add_trace(go.Box(y=Result.Distance[Result['Cluster'] == Cluster_], boxpoints='all', text="Distribution",
                                         name="Cluster " + str(Cluster_)))
                fig.update_layout(title_text="Distance between all elements and the centroid of cluster " + str(Cluster))
                fig.show()

        def graph_density_projection_2D(self,graph_method, reduction_method, scale_method):
            from sklearn.neighbors import KernelDensity

            if reduction_method=="PCA":
                data = self.utils_PCA(2)
                xmin = data['PCA0'].min()
                xmax = data['PCA0'].max()
                ymin = data['PCA1'].min()
                ymax = data['PCA1'].max()
            elif reduction_method == "UMAP":
                data = self.utils_UMAP()
                xmin = data[0].min()
                xmax = data[0].max()
                ymin = data[1].min()
                ymax = data[1].max()
            else:
                raise ValueError('Unknown reduction method : ' + str(reduction_method) + '. Available methods = "PCA", "UMAP"')

            xgrid = np.arange(xmin, xmax, (xmax - xmin) / 100)
            ygrid = np.arange(ymin, ymax, (ymax - ymin) / 100)
            X, Y = np.meshgrid(xgrid, ygrid[::-1])
            xy = np.vstack([Y.ravel(), X.ravel()]).T

            kde = KernelDensity(bandwidth=0.04, kernel='gaussian')
            kde.fit(data)
            result = pd.DataFrame(kde.score_samples(xy).reshape(X.shape))

            if scale_method == "min_max":
                result = (result - result.min().min())/(result.max().max() - result.min().min())
            elif scale_method == "standard":
                result = (result - result.mean().mean()) / (result.std().std())
            elif scale_method == "robust":
                result = (result-result.median().median())/(result.quantile(0.75).quantile(0.75)-result.quantile(0.25).quantile(0.25))
            elif scale_method is None:
                pass
            else:
                raise ValueError('Unknown scale method : ' + str(scale_method) + '. Available methods = "min_max", "standard", "robust", None')

            import plotly.graph_objects as go
            if graph_method == "surface":
                fig = go.Figure(data=[go.Surface(z=result.values)])
                fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                                  highlightcolor="limegreen", project_z=True))
                fig.show()
            elif graph_method =="contour":
                fig = go.Figure(data=[go.Contour(
                    x=xgrid,
                    y=ygrid,
                    z=result,
                    colorscale='RdBu')])
                fig.show()
            else:
                raise ValueError('Unknown method : ' + str(graph_method) + '. Available methods = "surface", "contour"')


        def __graph_animated_dataframes(self, dict_df, **args):
            """
            Animate a dict of dataframe with plotly sliders and buttons. The key is the name of the dataframe and the item is a Pandas Dataframe
            """
            if isinstance(dict_df, dict):
                list_df = list(dict_df.values())
                # Check if there is no values in the list that are not pandas dataframes
                if np.count_nonzero([not (isinstance(x, pd.DataFrame)) for x in list_df]) != 0:
                    raise TypeError('The given dict isn\'t full of Pandas dataframes.')
            else:
                raise TypeError('The given dict hasn\'t a dict type.')

            try:
                fill_color = args['fill_color']
            except KeyError:
                fill_color = "#2980b9"
            try:
                width = args['width']
                height = args['height']
            except KeyError:
                width = 1000
                height = 550
            try:
                title = args['title']
            except KeyError:
                title = ""

            # Create figure
            fig = go.Figure(frames=[go.Frame(data=go.Table(
                header=dict(values=df.columns, fill_color=fill_color, align="center"),
                cells=dict(values=[df[i].values for i in df.columns], align="center")
            ), name=str(k)) for k, df in enumerate(list_df)])

            # Add the first dataframe
            fig.add_trace(go.Table(
                header=dict(
                    values=list(list_df[0].columns),
                    font=dict(size=10),
                    fill_color=fill_color,
                    align="center"
                ),
                cells=dict(
                    values=[list_df[0][i].values for i in list_df[0].columns],
                    align="center")
            ))

            # Create function to get the time between frames
            frame_args = lambda duration: {"frame": {"duration": duration}, "mode": "immediate", "fromcurrent": True,
                                           "transition": {"duration": duration, "easing": "linear"}}

            # Add the slider
            sliders = [dict(
                active=0,
                len=0.75,
                x=0.14,
                y=-0.05,
                currentvalue={"prefix": "Table "},
                pad={"t": 1},
                steps=[{"args": [[f.name], frame_args(0)], "label": str(k), "method": "animate"} for k, f in
                       enumerate(fig.frames)]
            )]
            # Add the buttons
            fig.update_layout(
                title=title,
                width=width,
                height=height,
                sliders=sliders,
                updatemenus=[
                    {
                        "buttons": [
                            {
                                "args": [None, frame_args(500)],
                                "label": "&#9654;",  # play symbol
                                "method": "animate",
                            },
                            {
                                "args": [[None], frame_args(0)],
                                "label": "&#9724;",  # pause symbol
                                "method": "animate",
                            },
                        ],
                        "direction": "right",
                        "pad": {"r": 10, "t": 30},
                        "type": "buttons",
                        "x": 0.12,
                        "y": 0.07,
                    }
                ]
            )

            fig.show()


        def __graph_multivariate_plot(self, df, columns, **args):
            """
            Plot severals columns in the dataframe with Plotly. This allows to add a "target=" argument in order to visualize the data with its associated cluster
            """
            xaxis_set = "date"
            if isinstance(df, pd.DataFrame):
                df.columns = df.columns.astype(str)
                if not (isinstance(df.index, pd.DatetimeIndex)):
                    xaxis_set = "-"
            else:
                raise TypeError('Given dataframe isn\'t a Pandas dataframe')

            if type(columns) == "str":
                columns = [columns]
            for col in columns:
                if col not in df.columns:
                    raise AttributeError('One of the given columns is not in dataframe\'s columns.')

            try:
                target = args['target']
                if len(target) != len(df):
                    raise ValueError('The target hasn\'t the same lenght as the given dataframe.')
                if isinstance(target, pd.Series):
                    target = target.to_list()
            except:
                target = None

            try:
                title = args['title']
            except:
                title = "Multivariate Plot"

            try:
                cluster_opacity = args['cluster_opacity']
            except:
                cluster_opacity = 0.3

            df = df.sort_index()
            cluster_colors = px.colors.qualitative.Safe
            curves_colors = px.colors.qualitative.Dark24
            yaxis_name_converter = lambda x: str(x + 1) if (x != 0) else ""

            fig = go.Figure()
            for i, col in enumerate(columns):
                fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, text=df[col], marker_color=curves_colors[i],
                                         yaxis="y" + yaxis_name_converter(i)))
            fig.update_traces(hoverinfo="name+x+text", line={"width": 0.5}, marker={"size": 8}, mode="lines+markers",
                              showlegend=False, )

            df_index = df.index.insert(len(df.index), '')

            if not (target is None):
                fig.update_layout(shapes=[
                    dict(fillcolor=cluster_colors[Cluster], opacity=cluster_opacity, line={"width": 0}, type="rect",
                         layer="below", x0=df_index[i], x1=df_index[i + 1], y0=0, y1=1, yref="paper") for i, Cluster in
                    enumerate(target)])

            xaxis = dict(autorange=True, type=xaxis_set, rangeslider=dict(autorange=True))
            fig.update_layout(xaxis=xaxis)

            fig.update_layout({"yaxis" + yaxis_name_converter(i): dict(
                anchor="x",
                range=[df[col].min(), df[col].max()],
                autorange=True,
                domain=[(i) / len(columns), (i + 1) / len(columns)],
                linecolor=curves_colors[i],
                mirror=True,
                showline=True,
                side="right",
                tickfont={"color": curves_colors[i]},
                tickmode="auto",
                ticks="",
                title=col[0:8],
                titlefont={"color": curves_colors[i]},
                type="linear",
                zeroline=False
            ) for i, col in enumerate(columns)})

            fig.update_layout(dragmode="zoom", hovermode="x", legend=dict(traceorder="reversed"), height=500, width=1450,
                              title=title, template="plotly_white", margin=dict(t=60, b=20))

            fig.show()


        def graph_PCA_3D(self):
          Mat=self.utils_PCA(3)
          data=pd.DataFrame(Mat)
          data['Cluster'] = self.data_target.astype(str)
          fig = px.scatter_3d(data, x="PCA0", y="PCA1", z="PCA2",color_discrete_sequence=px.colors.qualitative.G10,color="Cluster",title="3D PCA Graph",opacity=0.7, width=850, height=600)
          for i,trace in enumerate(fig.data):
            trace.name = "Cluster " + self.data_target.astype(str)[i]
          fig.show()


        def graph_PCA_2D(self):
          Mat=self.utils_PCA(2)
          data=pd.DataFrame(Mat)
          data['Cluster'] = self.data_target.astype(str)
          fig = px.scatter(data, x="PCA0", y="PCA1",color_discrete_sequence=px.colors.qualitative.G10,color="Cluster",title="2D PCA Graph",opacity=0.7, width=850, height=600)
          for i,trace in enumerate(fig.data):
            trace.name = "Cluster " + self.data_target.astype(str)[i]
          fig.show()



        def projection_2D(self, feature1, feature2,**args):

            import matplotlib.pyplot as plt
            import seaborn as sns
            import matplotlib
            if not(feature1 in self.data_features.columns) or not(feature1 in self.data_features.columns):
              return("Error #3, Bad features call")
            else:
                try:
                    radius_choice=args['radius']
                except KeyError:
                    radius_choice="90p"

                try:
                    zoom=args['zoom']
                except KeyError:
                    zoom=1
                data_radius_centroid= {Cluster:self.data_radius_selector_specific_cluster(radius_choice,Cluster) for Cluster in self.labels_clusters}
                figsize=5*zoom
                fig=plt.figure(2,figsize=(figsize,figsize))
                plt.title('Projection 2D')
                g=sns.scatterplot(x=self.data_features.loc[:,feature1],y=self.data_features.loc[:,feature2],hue=self.data_target,alpha=0.70,palette="tab10")
                for Cluster in self.labels_clusters:
                    Circle=plt.Circle(tuple(self.data_centroids[Cluster][[feature1,feature2]].values),radius=data_radius_centroid[Cluster],fill=False,color=matplotlib.cm.get_cmap('tab10')(Cluster))
                    g.add_patch(Circle)
                g=sns.scatterplot(x=self.data_centroids.loc[feature1,:],y=self.data_centroids.loc[feature2,:],s=12,color='#000000')
else:
    class __Graph:
        def __init(self):
            pass

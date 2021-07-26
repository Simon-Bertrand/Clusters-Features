# -*- coding: utf-8 -*-
#
# Copyright 2021 Simon Bertrand
#
# This file is part of ClusterCharacteristics.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>.

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


    class Graph(object):
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


        def PCA_3D_graph(self):
          Mat=self.PCA(3)
          data=pd.DataFrame(Mat)
          data['Cluster'] = self.data_target.astype(str)
          fig = px.scatter_3d(data, x=0, y=1, z=2,color_discrete_sequence=px.colors.qualitative.G10,color="Cluster",title="3D PCA Result",labels="test",opacity=0.7, width=850, height=600)
          fig.update_scenes(xaxis_title_text="PCA 1", yaxis_title_text="PCA 2", zaxis_title_text="PCA 3")
          for trace in fig.data:
            trace.name = "Cluster " + trace.name.split('=')[1]
          fig.show()
          return(Mat)

        def PCA_2D_graph(self):
          Mat=self.PCA(2)
          fig = plt.figure(1,figsize=(20, 10))
          ax = fig.add_subplot(111)
          plt.title('2D PCA Result',fontsize=15)
          scatter=ax.scatter(Mat[:,0], Mat[:,1], c=self.data_target, cmap="tab10",alpha=0.6)
          ax.yaxis.set_tick_params(labelsize=7)
          ax.xaxis.set_tick_params(labelsize=7)
          legend1 = ax.legend(scatter.legend_elements(), title="Clusteres")
          ax.add_artist(legend1)
          ax.set_xlabel('PCA1', fontsize=12)
          ax.set_ylabel('PCA2', fontsize=12)
          return(Mat)


        def projection_2D(self, feature1, feature2,**args):
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
            g=sns.scatterplot(data=pd.DataFrame(self.data_centroids).T,x=feature1,y=feature2,s=12,color='#000000')
else:
    class Graph:
        def __init(self):
            pass

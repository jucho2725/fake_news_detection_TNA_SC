# from igraph import *
#
#
# edges = [['a', 'b'],
#          ['a', 'b'],
#          ['a', 'b'],
#          ['b', 'a'],
#          ['a', 'c'],
#          ['c', 'a'],
#          ['c', 'd'],
#          ['c', 'd'],
#          ['d', 'c'],
#          ['d', 'c']]
#
# # collect the set of vertex names and then sort them into a list
# vertices = set()
# for line in edges:
#     vertices.update(line)
# vertices = sorted(vertices)
#
# # create an empty graph
# g = Graph()
#
# # add vertices to the graph
# g.add_vertices(vertices)
#
# # add edges to the graph
# g.add_edges(edges)
#
# # set the weight of every edge to 1
# g.es["weight"] = 1
#
# # collapse multiple edges and sum their weights
# g.simplify(combine_edges={"weight": "sum"})
#
# # plot(g)
#
# g2 = Graph.TupleList(directed=False, edges = edges)
#
# plot(g2)


import igraph as ig

G=ig.Graph.Read_GML('netscience.gml.txt')
labels=list(G.vs['label'])
# print(labels)
N=len(labels)
print("N is {0}".format(N))
E=[e.tuple for e in G.es]# list of edges
# print(E)
layt=G.layout('kk') #kamada-kawai layout
type(layt)
# print(layt)

import plotly.plotly as py
from plotly.graph_objs import *

Xn = [layt[k][0] for k in range(N)]
Yn = [layt[k][1] for k in range(N)]
Xe = []
Ye = []
for e in E:
    Xe += [layt[e[0]][0], layt[e[1]][0], None]
    Ye += [layt[e[0]][1], layt[e[1]][1], None]

trace1 = Scatter(x=Xe,
                 y=Ye,
                 mode='lines',
                 line=dict(color='rgb(210,210,210)', width=1),
                 hoverinfo='none'
                 )
trace2 = Scatter(x=Xn,
                 y=Yn,
                 mode='markers',
                 name='ntw',
                 marker=dict(symbol='circle-dot',
                             size=5,
                             color='#6959CD',
                             line=dict(color='rgb(50,50,50)', width=0.5)
                             ),
                 text=labels,
                 hoverinfo='text'
                 )

axis = dict(showline=False,  # hide axis line, grid, ticklabels and  title
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            title=''
            )

width = 800
height = 800
layout = Layout(title="Coauthorship network of scientists working on network theory and experiment",
                font=dict(size=12),
                showlegend=False,
                autosize=False,
                width=width,
                height=height,
                xaxis=layout.XAxis(axis),
                yaxis=layout.YAxis(axis),
                margin=layout.Margin(
                    l=40,
                    r=40,
                    b=85,
                    t=100,
                ),
                hovermode='closest',
                annotations=[
                    dict(
                        showarrow=False,
                        text='This igraph.Graph has the Kamada-Kawai layout',
                        xref='paper',
                        yref='paper',
                        x=0,
                        y=-0.1,
                        xanchor='left',
                        yanchor='bottom',
                        font=dict(
                            size=14
                        )
                    )
                ]
                )

data = [trace1, trace2]
fig = Figure(data=data, layout=layout)
py.iplot(fig, filename='Coautorship-network-igraph')
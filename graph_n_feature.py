"""
graph visualization of Co-Occurrence matrix
&
Meta information(=features for machine learning) extraction

May 18th, 2019
author: Jin Uk, Cho

source : https://kiddwannabe.blog.me/221362423659
"""

import networkx as nx
import matplotlib.pyplot as plt
import sys

# mat = [[('Trump', 'administration'), 1], [('Trump', 'delay'), 1], [('Trump', 'tariff'), 2], [('Trump', 'car'), 3], [('Trump', 'part'), 1], [('Trump', 'import'), 2], [('Trump', 'month'), 1], [('Trump', 'negotiates'), 1], [('Trump', 'trade'), 1]]

mat = [['Trump', 'administration', 1], ['Trump', 'delay', 1], ['Trump', 'tariff', 2], ['Trump', 'car', 3], ['Trump', 'part', 1], ['Trump', 'import', 2], ['Trump', 'month', 1], ['Trump', 'negotiates', 1], ['Trump', 'trade', 1], ['Trump', 'deal', 1], ['Trump', 'European', 1], ['Trump', 'Union', 1], ['Trump', 'Japan', 1], ['administration', 'delay', 1], ['administration', 'tariff', 1], ['administration', 'car', 2], ['administration', 'part', 1], ['administration', 'import', 1], ['administration', 'month', 1], ['administration', 'negotiates', 1], ['administration', 'trade', 1], ['administration', 'deal', 1], ['administration', 'European', 1], ['administration', 'Union', 1], ['administration', 'Japan', 1], ['delay', 'tariff', 1], ['delay', 'car', 2], ['delay', 'part', 1], ['delay', 'import', 1], ['delay', 'month', 1], ['delay', 'negotiates', 1], ['delay', 'trade', 1], ['delay', 'deal', 1], ['delay', 'European', 1], ['delay', 'Union', 1], ['delay', 'Japan', 1], ['tariff', 'car', 2], ['tariff', 'part', 1], ['tariff', 'import', 1], ['tariff', 'month', 1], ['tariff', 'negotiates', 1], ['tariff', 'trade', 1], ['tariff', 'deal', 1], ['tariff', 'European', 1], ['tariff', 'Union', 1], ['tariff', 'Japan', 1], ['car', 'part', 2], ['car', 'import', 3], ['car', 'month', 2], ['car', 'negotiates', 2], ['car', 'trade', 2], ['car', 'deal', 2], ['car', 'European', 2], ['car', 'Union', 2], ['car', 'Japan', 2], ['part', 'import', 1], ['part', 'month', 1], ['part', 'negotiates', 1], ['part', 'trade', 1], ['part', 'deal', 1], ['part', 'European', 1], ['part', 'Union', 1], ['part', 'Japan', 1], ['import', 'month', 1], ['import', 'negotiates', 1], ['import', 'trade', 1], ['import', 'deal', 1], ['import', 'European', 1], ['import', 'Union', 1], ['import', 'Japan', 1], ['month', 'negotiates', 1], ['month', 'trade', 1], ['month', 'deal', 1], ['month', 'European', 1], ['month', 'Union', 1], ['month', 'Japan', 1], ['negotiates', 'trade', 1], ['negotiates', 'deal', 1], ['negotiates', 'European', 1]]



# MST(Minum Spanning Tree)-based graph
# This is common methodology in Social Network Analysis
def vis_graph(matrix, NETWORK_MAX):
    """
    Create Co-Occurrence Matrix
    :param matrix: (list) co-occurence matrix data
    :param NETWORK_MAX: (int) number of network ????
    """
    # 공부 필요
    G = nx.Graph()
    i = 0

    # create edge
    for w1, w2, count in matrix:
        i += 1
        if i > NETWORK_MAX: break

        G.add_edge(w1, w2, weight=count)

    # create MST model
    T = nx.minimum_spanning_tree(G)
    nodes = nx.nodes(T)
    degrees = nx.degree(T)

    # set size of node
    node_size = []
    for node in nodes:
        ns = degrees[node]*100
        node_size.append(ns)

    # font configuration
    if sys.platform in ["win32", "win64"]:
        font_name = "malgun gothic"
    elif sys.platform == "darwin":
        font_name = "AppleGothic"

    # disply the result
    plt.figure(figsize=(16,12))
    nx.draw_networkx(T,
                     pos=nx.fruchterman_reingold_layout(G, k=0.5),
                     node_size=node_size,
                     node_color="yellow",
                     font_family=font_name,
                     label_pos=1, #0=head, 0.5=center, 1=tail
                     with_labels=True,
                     font_size=12
                     )

    plt.axis("off")
    plt.show()

vis_graph(mat, 50)

# Metric 구하는 함수
# def metric():
#     """
#     Create Co-Occurrence Matrix
#     :param matrix: (list) co-occurence matrix data
#     :param NETWORK_MAX: (int) number of network ????
#     :return: 머신러닝에 사용되기 위한 feature 값들
#     """

import
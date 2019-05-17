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

# Metric 구하는 함수
def metric():
    """
    Create Co-Occurrence Matrix
    :param matrix: (list) co-occurence matrix data
    :param NETWORK_MAX: (int) number of network ????
    :return: 머신러닝에 사용되기 위한 feature 값들
    """
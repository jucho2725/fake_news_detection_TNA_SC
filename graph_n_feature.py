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

import network
import pandas as pd

text = "The Trump administration will delay tariffs on cars and car part imports for up to six months as it negotiates trade deals with the European Union and Japan. In a proclamation Friday, Trump said he directed U.S.Trade Representative Robert Lighthizer to seek agreements to “address the threatened impairment” of national security from car imports. Trump could choose to move forward with tariffs during the talks. “United States defense and military superiority depend on the competitiveness of our automobile industry and the research and development that industry generates,” White House press secretary Sarah Huckabee Sanders said in a statement. “The negotiation process will be led by United States Trade Representative Robert Lighthizer and, if agreements are not reached within 180 days, the President will determine whether and what further action needs to be taken."


N = network.Processing()
lemed_content = N.lemma_whole(text)
stopped_content = N.stopword(lemed_content)
tagged_results = N.tag_content(stopped_content)

tag_filter = ['NNP', 'NN', 'NNS', 'VBP', 'VBD', 'VBN', 'JJ', 'RB', 'VB']

selected_results = N.select_results(tagged_results, tag_filter)
final_result = N.create_cooc_mat(selected_results)


print('The graph has {0} nodes'.format(len(final_result)))

# MST(Minum Spanning Tree)-based graph
# This is common methodology in Social Network Analysis
def vis_graph(matrix, NETWORK_MAX):
    """
    Create Co-Occurrence Matrix
    :param matrix: (list) co-occurence matrix data
    :param NETWORK_MAX: (int) number of network ????
    """
    # 공부 더 필요
    G = nx.Graph()
    i = 0

    # create edge
    for i in range(len(matrix)):
        # print('{0} is the number'.format(len(matrix)))
        # print(matrix['Linkage'][i])
        w1 = matrix['Linkage'][i][0]
        w2 = matrix['Linkage'][i][1]
        count = matrix['Weight'][i]
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

    # plt.axis("off")
    # plt.show()

    return G

graph_ex = vis_graph(final_result, len(final_result))

deg_cent = nx.algorithms.degree_centrality(graph_ex)
info_cent = nx.algorithms.information_centrality(graph_ex)
clo_cent = nx.algorithms.closeness_centrality(graph_ex)

print(deg_cent)

# Metric 구하는 함수
# def metric():
#     """
#     Create Co-Occurrence Matrix
#     :param matrix: (list) co-occurence matrix data
#     :param NETWORK_MAX: (int) number of network ????
#     :return: 머신러닝에 사용되기 위한 feature 값들
#     """

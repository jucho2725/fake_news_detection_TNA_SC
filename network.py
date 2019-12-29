"""
graph visualization of Co-Occurrence matrix


May 18th, 2019
author: Jin Uk, Cho

plt plotting source : https://kiddwannabe.blog.me/221362423659
"""

import sys

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
# import plotly.plotly as py
# from plotly.graph_objs import *

import ast

class Graph:
    """
    Visualize Co-Occurrence Matrix
    :param matrix: (list) co-occurence matrix data
    """

    def __init__(self):
        self.G = nx.Graph()

    def strToList(self, df):
        '''
        desc : Changing df['linkage'] data type from string to list.
        '''
        try:
            df['Linkage'] = [ast.literal_eval(str) for str in df.loc[:, 'Linkage']]
        except:
            pass
        return df

    def string_to_list(self, df):
        temp = df.loc[:, 'Linkage'].astype(str).str.split("'").str
        first = temp.get(1).array
        second = temp.get(3).array
        linkage = pd.Series(list(zip(first, second)))
        df['Linkage'] = linkage
        return df

    def create_graph(self, doc_path, string_to_list=False):
        """
        네트워크 이론에 사용할 그래프를 만들어줌
        :param string_to_list: 기존에 만들어진 csv 를 가져오려면 이걸 True로 해야함
        :return:
        """
        # MST(Minum Spanning Tree)-based graph
        # create edge

        matrix = pd.read_csv(doc_path)
        if string_to_list:
            matrix = self.strToList(matrix)
        else:
            pass

        matrix = matrix.loc[:, ('Linkage', 'Weight')]


        for i in range(len(matrix)):
            w1 = matrix.loc[i, 'Linkage'][0]
            w2 = matrix.loc[i, 'Linkage'][1]
            count = matrix.loc[i, 'Weight']
            self.G.add_edge(w1, w2, weight=count)
            # i += 1
            # if i > NETWORK_MAX: # 노드 갯수 제어
            #     break



        # create MST model
        self.T = nx.minimum_spanning_tree(self.G)
        nodes = nx.nodes(self.T)
        degrees = nx.degree(self.T)
        # set size of node
        self.node_size = []
        for node in nodes:
            ns = degrees[node] * 100
            self.node_size.append(ns)
        self.pos = nx.fruchterman_reingold_layout(self.G, k=0.5)
        return self.G, matrix


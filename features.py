"""
Meta information(=features for machine learning) extraction

June 2nd, 2019
author: Jin Uk, Cho
"""

import sklearn
import networkx as nx
import pandas as pd
import numpy as np

from network_visualization import Graph
from reweight import Reweight


# 척도 계산하기
class Measure():
    def __init__(self, graph):
        self.graph = graph

    def cal_Cent(self, g):
        # deg_cent = nx.algorithms.degree_centrality(g)
        # clo_cent = nx.algorithms.closeness_centrality(g)
        bet_cent = nx.algorithms.betweenness_centrality(g)
        # eig_cent = nx.algorithms.eigenvector_centrality_numpy(g)
        # info_cent = nx.algorithms.information_centrality(g)
        # list_deg = [k for k in deg_cent.values()]
        # self.list_deg = list_deg
        # list_clo = [k for k in clo_cent.values()]
        # self.list_clo = list_clo
        list_bet = [k for k in bet_cent.values()]
        self.list_bet = list_bet

        # return deg_cent

    def get_Info(self):
        summary_g = nx.info(self.graph)
        print(summary_g)
        return summary_g

    def create_Dataframe(self):
        """
        :return: (dataframe) table of info of each nodes
        """
        word = pd.Series(self.deg_cent.keys())
        deg_val = pd.Series(self.deg_cent.values())
        bet_val = pd.Series(self.bet_cent.values())
        clo_val = pd.Series(self.clo_cent.values())
        dataframe = pd.DataFrame({'Word': word, 'Betweeness': bet_val, 'Degree': deg_val,
                                  'Closeness': clo_val})
        return dataframe

    def deg_GroupVal(self):
        """
        :param (list) list of values:
        Cd = Summation(Max - i th value) / (g-2)(g-1)
        :return: (float)  grouped value
        """
        X = 0
        max_val = max(self.list_deg)
        g = len(self.list_deg)
        gg = (g - 2) * (g - 1)
        print('maximum value of degree cetrality is : {0}'.format(max_val))
        for i in self.list_deg:
            tmp = max_val - i
            X += tmp
        result = X / gg
        print(result)
        return result

    def clo_GroupVal(self):
        """
        :param (list) list of values:
        Cd = Summation(Max - i th value) / ((g-2)(g-1)/(2g-3))
        :return: (float)  grouped value
        """
        X = 0
        max_val = max(self.list_clo)
        g = len(self.list_clo)
        gg = ((g - 2) * (g - 1)) / (2 * g - 3)
        print('maximum value of closeness cetrality is : {0}'.format(max_val))
        for i in self.list_clo:
            tmp = max_val - i
            X += tmp
        result = X / gg
        print(result)
        return result

    def bet_GroupVal(self):  ###########
        """
        :param (list) list of values:
        Cd = Summation(Max - i th value) / ((g-2)^2(g-1)/2)
        :return: (float)  grouped value
        """
        X = 0
        max_val = max(self.list_bet)
        g = len(self.list_bet)
        gg = (pow((g - 2), 2) * (g - 1)) / (2)
        # print('maximum value of betweenness cetrality is : {0}'.format(max_val))
        for i in self.list_bet:
            tmp = max_val - i
            X += tmp
        result = X / gg
        print(result)
        return result

    #
    def get_Value(self):
        """
        calculate all values and return it as a list
        :return: (float) values
        """
        # info = self.get_Info()
        start = self.cal_Cent(self.graph)
        # deg_val = self.deg_GroupVal()
        # clo_val = self.clo_GroupVal()
        bet_val = self.bet_GroupVal()

        return bet_val


class Feature():
    def __init__(self, doc_path_list):
        self.doc_filenames = doc_path_list
        self.df_tfidf = pd.read_csv(doc_path_list[0][:-20] + 'tfidf.csv', index_col=0)

    def cal_tfidf(self):
        tfidf_mean = np.mean(self.df_tfidf['Tfidf'])
        tfidf_var = np.var(self.df_tfidf['Tfidf'])
        return tfidf_mean, tfidf_var

    def cal_edge_weight(self, matrix):
        wt_mean = np.mean(matrix['Weight'])
        wt_var = np.var(matrix['Weight'])
        return wt_mean, wt_var

    def cal_edge_num(self, matrix):
        return len(matrix['Linkage'])

    def cal_net_feature(self, G):
        net = Measure(G)
        bet_val = net.get_Value()
        common_neighbors = [len(list(nx.common_neighbors(net, u, v))) for u, v in G.edges]
        com_mean = np.mean(np.array(common_neighbors))
        com_var = np.var(np.array(common_neighbors))
        degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
        core_count = len([i for i in degree_sequence if i > np.quantile(degree_sequence, 0.75)])
        return com_mean, com_var, core_count, bet_val,

    def make_df(self, doc_path, label='fake'):
        net = Graph()
        G, matrix = net.create_graph(doc_path, string_to_list=True)  # 이미 tfidf_reweight.csv 로 된 애들을 만들어놔서 그걸로 시작해야함
        tfidf_mean, tfidf_var = self.cal_tfidf()
        wt_mean, wt_var = self.cal_edge_weight(matrix)
        edge_num = self.cal_edge_num(matrix)
        com_mean, com_var, core_count, bet_val = self.cal_net_feature(G)

        feature_df_one = {'tfidf_mean': tfidf_mean,
                          'tfidf_var': tfidf_var,
                          'wt_mean': wt_mean,
                          'wt_var': wt_var,
                          'edge_num': edge_num,
                          'com_mean': com_mean,
                          'com_var': com_var,
                          'core_count': core_count,
                          'betweeness': bet_val,
                          'label': label, 'index': doc_path[-20:-4]}

        return feature_df_one

    def make_df_from_dataset(self, label):
        idx_list = []
        row_list = []
        for doc_path in self.doc_filenames:
            idx_list.append(doc_path[-20:-4])
            row_list.append(self.make_df(doc_path, label))

        feature_df = pd.DataFrame(row_list, columns=row_list[0].keys(), index=idx_list)

        return feature_df


''' TO DO 

Longest path 구현? 
or Bidirectional 그래프로 바꿀까 

'''

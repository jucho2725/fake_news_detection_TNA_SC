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
        """
        TO DO : get other features if available
        :param g: (Graph)
        :return: caculated features
        """
        deg_cent = nx.algorithms.degree_centrality(g)
        clo_cent = nx.algorithms.closeness_centrality(g)
        bet_cent = nx.algorithms.betweenness_centrality(g)
        eig_cent = nx.algorithms.eigenvector_centrality_numpy(g)
        info_cent = nx.algorithms.information_centrality(g)


        list_deg = [k for k in deg_cent.values()]
        list_clo = [k for k in clo_cent.values()]
        list_bet = [k for k in bet_cent.values()]

        return list_deg, list_clo, list_bet

    def get_Info(self): # will be deprecated
        summary_g = nx.info(self.graph)
        print(summary_g)
        return summary_g


    def deg_GroupVal(self, list_deg):
        """
        :param (list) list of values:
        Cd = Summation(Max - i th value) / (g-2)(g-1)
        :return: (float)  grouped value
        """
        X = 0
        max_val = max(list_deg)
        g = len(list_deg)
        gg = (g - 2) * (g - 1)
        # print('maximum value of degree cetrality is : {0}'.format(max_val))
        for i in list_deg:
            tmp = max_val - i
            X += tmp
        result = X / gg
        return result

    def clo_GroupVal(self, list_clo):
        """
        :param (list) list of values:
        Cd = Summation(Max - i th value) / ((g-2)(g-1)/(2g-3))
        :return: (float)  grouped value
        """
        X = 0
        max_val = max(list_clo)
        g = len(list_clo)
        gg = ((g - 2) * (g - 1)) / (2 * g - 3)
        for i in list_clo:
            tmp = max_val - i
            X += tmp
        result = X / gg
        return result

    def bet_GroupVal(self, list_bet):
        """
        :param (list) list of values:
        Cd = Summation(Max - i th value) / ((g-2)^2(g-1)/2)
        :return: (float)  grouped value
        """
        X = 0
        max_val = max(list_bet)
        g = len(list_bet)
        gg = (pow((g - 2), 2) * (g - 1)) / (2)
        for i in list_bet:
            tmp = max_val - i
            X += tmp
        result = X / gg
        # print(result)
        return result

    #
    def get_Value(self):
        """
        calculate all values and return it as a list
        :return: (float) values
        """
        # info = self.get_Info()
        list_deg, list_clo, list_bet = self.cal_Cent(self.graph)
        deg_val = self.deg_GroupVal(list_deg)
        clo_val = self.clo_GroupVal(list_clo)
        bet_val = self.bet_GroupVal(list_bet)

        return deg_val, clo_val, bet_val

    def cal_net_features(self):
        common_neighbors = [len(list(nx.common_neighbors(self.graph, u, v))) for u, v in self.graph.edges]
        com_mean = np.mean(np.array(common_neighbors))
        com_var = np.var(np.array(common_neighbors))
        degree_sequence = sorted([d for n, d in self.graph.degree()], reverse=True)
        core_count = len([i for i in degree_sequence if i > np.quantile(degree_sequence, 0.75)])
        com_mean, com_var, core_count

class Feature():
    def __init__(self, doc_path_list):
        self.doc_filenames = doc_path_list
        # self.df_tfidf = pd.read_csv('data/tfidf.csv', index_col=0)

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
        deg_val, clo_val, bet_val = net.get_Value()
        com_mean, com_var, core_count = net.cal_net_features()
        return com_mean, com_var, core_count, deg_val, clo_val, bet_val

    def make_df(self, doc_path, label):
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
                          'label': label, 'index': doc_path[-20:-4],
                          }

        return feature_df_one

    def make_df_from_dataset(self):
        idx_list = []
        row_list = []
        for doc_path in self.doc_filenames[:2]:
            idx_list.append(doc_path[-20:-4]) # title of article
            row_list.append(self.make_df(doc_path, label=0))
        for doc_path in self.doc_filenames[2:]:
            idx_list.append(doc_path[-20:-4])  # title of article
            row_list.append(self.make_df(doc_path, label=1))

        feature_df = pd.DataFrame(row_list, columns=row_list[0].keys(), index=idx_list)

        return feature_df


''' TO DO 

Longest path 구현? 
or Bidirectional 그래프로 바꿀까 

'''

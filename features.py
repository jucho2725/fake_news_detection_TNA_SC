"""
Meta information(=features for machine learning) extraction

June 2nd, 2019
author: Jin Uk, Cho
"""



import sklearn
import networkx as nx
import pandas as pd
from network_visualization import Visualization

# 척도 계산하기
class Measure(Visualization):
    def __init__(self, graph):
        super.__init__(self)
        self.graph = graph

    def Cal_Cent(self, g):
        deg_cent = nx.algorithms.degree_centrality(g)
        clo_cent = nx.algorithms.closeness_centrality(g)
        bet_cent = nx.algorithms.betweenness_centrality(g)
        # eig_cent = nx.algorithms.eigenvector_centrality_numpy(g)
        # info_cent = nx.algorithms.information_centrality(g)
        list_deg = [k for k in deg_cent.values()]
        self.list_deg = list_deg
        list_clo = [k for k in clo_cent.values()]
        self.list_clo = list_clo
        list_bet = [k for k in bet_cent.values()]
        self.list_bet = list_bet

        return deg_cent

    def Get_Info(self):
        summary_g = nx.info(self.graph)
        print(summary_g)
        return summary_g

    def Create_Dataframe(self):
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

    def bet_GroupVal(self):###########
        """
        :param (list) list of values:
        Cd = Summation(Max - i th value) / ((g-2)^2(g-1)/2)
        :return: (float)  grouped value
        """
        X = 0
        max_val = max(self.list_bet)
        g = len(self.list_bet)
        gg = (pow((g - 2), 2) * (g - 1)) / (2)
        print('maximum value of betweenness cetrality is : {0}'.format(max_val))
        for i in self.list_bet:
            tmp = max_val - i
            X += tmp
        result = X / gg
        print(result)
        return result

    #
    def Get_Value(self):
        """
        calculate all values and return it as a list
        :return: (float) values
        """
        info = self.Get_Info()
        start = self.Cal_Cent(self.graph)
        deg_val = self.deg_GroupVal()
        clo_val = self.clo_GroupVal()
        bet_val = self.bet_GroupVal()

        return deg_val, clo_val, bet_val


# result = Measure(graph_ex)
# print(result.Get_Value())




''' TO DO '''
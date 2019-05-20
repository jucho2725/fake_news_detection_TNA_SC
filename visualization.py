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

# network 에서 호출하여 전처리
N = network.Processing()
lemed_content = N.lemma_whole(text)
stopped_content = N.stopword(lemed_content)
tagged_results = N.tag_content(stopped_content)

tag_filter = ['NNP', 'NN', 'NNS', 'VBP', 'VBD', 'VBN', 'JJ', 'RB', 'VB']

selected_results = N.select_results(tagged_results, tag_filter)
final_result = N.create_cooc_mat(selected_results)

print('The graph has {0} nodes'.format(len(final_result)))


# 그래프 생성
# MST(Minum Spanning Tree)-based graph

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
        ns = degrees[node] * 100
        node_size.append(ns)

    # font configuration
    if sys.platform in ["win32", "win64"]:
        font_name = "malgun gothic"
    elif sys.platform == "darwin":
        font_name = "AppleGothic"

    # disply the result
    plt.figure(figsize=(16, 12))
    nx.draw_networkx(T,
                     pos=nx.fruchterman_reingold_layout(G, k=0.5),
                     node_size=node_size,
                     node_color="yellow",
                     font_family=font_name,
                     label_pos=1,  # 0=head, 0.5=center, 1=tail
                     with_labels=True,
                     font_size=12
                     )

    # 그래프 생성
    # plt.axis("off")
    # plt.show()

    return G


graph_ex = vis_graph(final_result, len(final_result))


# 척도 계산하기
class Measure:
    def __init__(self, graph_ex):
        self.graph_ex = graph_ex

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
        summary_g = nx.info(graph_ex)
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

    def bet_GroupVal(self):
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
        start = self.Cal_Cent(self.graph_ex)
        deg_val = self.deg_GroupVal()
        clo_val = self.clo_GroupVal()
        bet_val = self.bet_GroupVal()


        return deg_val, clo_val, bet_val


result = Measure(graph_ex)
print(result.Get_Value())

""" To Do

객체화 - 진행중
Graph 객체로 바꿔서 super 로 이어주

GroupVal 함수 하나로 통합해서 어떤 척도 계산할지만 입력하도록 하기 - 대충해놓음


return, input 정확히 쓰기
영어로 해야하나
"""

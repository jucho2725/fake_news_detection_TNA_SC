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
import plotly.plotly as py
from plotly.graph_objs import *

from coocurrence import Processing

class Graph():
    """
    Visualize Co-Occurrence Matrix
    :param matrix: (list) co-occurence matrix data
    """
    def __init__(self, matrix):
        self.G = nx.Graph()
        self.matrix = matrix

    def string_to_list(self, df):
        temp = df['Linkage'].astype(str).str.split("'").str
        first = temp.get(1).get_values()
        second = temp.get(3).get_values()
        linkage = pd.Series(list(zip(first, second)))
        df['Linkage'] = linkage
        return df

    def create_graph(self, string_to_list=False):
        # MST(Minum Spanning Tree)-based graph
        # create edge
        if string_to_list:
            self.matrix = self.string_to_list(self.matrix)
        else:
            pass

        for i in range(len(self.matrix)):
            # print('{0} is the number'.format(len(matrix)))
            # print(matrix['Linkage'][i])
            w1 = self.matrix['Linkage'][i][0]
            w2 = self.matrix['Linkage'][i][1]
            count = self.matrix['Weight'][i]
            # i += 1
            # if i > NETWORK_MAX: # 노드 갯수 제어
            #     break

            self.G.add_edge(w1, w2, weight=count)

        # create MST model
        self.T = nx.minimum_spanning_tree(self.G)
        nodes = nx.nodes(self.T)
        print("**")
        print(nodes)
        degrees = nx.degree(self.T)
        print("**")
        print(degrees)
        # set size of node
        self.node_size = []
        for node in nodes:
            ns = degrees[node] * 100
            self.node_size.append(ns)
        self.pos = nx.fruchterman_reingold_layout(self.G, k=0.5)


class Visualization(Graph):
    def __init__(self, matrix):
        super().__init__(matrix)

    def vis_plt(self):
        # matplotlib 그래프 생성

        # font configuration
        if sys.platform in ["win32", "win64"]:
            font_name = "malgun gothic"
        elif sys.platform == "darwin":
            font_name = "AppleGothic"

        # disply the result
        plt.figure(figsize=(16, 12))
        nx.draw_networkx(self.T,
                         pos=self.pos,
                         node_size=self.node_size,
                         node_color="yellow",
                         font_family=font_name,
                         label_pos=1,  # 0=head, 0.5=center, 1=tail
                         with_labels=True,
                         font_size=12
                         )

        # 그래프 생성
        plt.axis("off")
        plt.show()

        return self.G

    def vis_plotly(self):
        """ Note : Only work in web"""
        # create edges
        E = [e for e in self.G.edges]
        pos = self.pos
        # frutcherman layout == matrix

        Xv = [pos[k][0] for k in range(len(self.matrix))]
        Yv = [pos[k][1] for k in range(len(self.matrix))]
        Xed = []
        Yed = []
        for edge in E:
            Xed += [pos[edge[0]][0], pos[edge[1]][0], None]
            Yed += [pos[edge[0]][1], pos[edge[1]][1], None]

        trace3 = Scatter(x=Xed,
                         y=Yed,
                         mode='lines',
                         line=dict(color='rgb(210,210,210)', width=1),
                         hoverinfo='none'
                         )
        trace4 = Scatter(x=Xv,
                         y=Yv,
                         mode='markers',
                         name='net',
                         marker=dict(symbol='circle-dot',
                                     size=5,
                                     color='#6959CD',
                                     line=dict(color='rgb(50,50,50)', width=0.5)
                                     ),
                         text=label,
                         hoverinfo='text'
                         )

        annot = "This networkx.Graph has the Fruchterman-Reingold layout<br>Code:" + \
                "<a href='http://nbviewer.ipython.org/gist/empet/07ea33b2e4e0b84193bd'> [2]</a>"

        data1 = [trace3, trace4]
        fig1 = Figure(data=data1, layout=layout)
        fig1['layout']['annotations'][0]['text'] = annot
        py.iplot(fig1, filename='Coautorship-network-nx')

        return self.G

    def save_graph(self, title):
        nx.write_gexf(self.G, title)


""" 테스트 """
text = "The Trump administration will delay tariffs on cars and car part imports for up to six months as it negotiates trade deals with the European Union and Japan. In a proclamation Friday, Trump said he directed U.S.Trade Representative Robert Lighthizer to seek agreements to “address the threatened impairment” of national security from car imports. Trump could choose to move forward with tariffs during the talks. “United States defense and military superiority depend on the competitiveness of our automobile industry and the research and development that industry generates,” White House press secretary Sarah Huckabee Sanders said in a statement. “The negotiation process will be led by United States Trade Representative Robert Lighthizer and, if agreements are not reached within 180 days, the President will determine whether and what further action needs to be taken."
# text = open("Proof.txt", encoding='utf-8').read()
tag_filter = ['NNP', 'NN', 'NNPS', 'NNS', 'VBG', 'VBP', 'VB']
# network 에서 호출하여 전처리
model = Processing(tag_filter)
sel_result, cooc_mat = model.cooc(text=text)

# 그래프를 그리는데 사용된 co occurrence matrix 결과(dataframe 형태)
print('The network has {0} edges'.format(len(cooc_mat)))
print(cooc_mat)
N = Visualization(cooc_mat)
grpah = N.create_graph(len(cooc_mat))
N.vis_plt()
N.save_graph("Proof.gexf")


#

""" To Do

객체화 - 진행중
Visual 이랑 Measure 랑 객체로 바꿔서 super 로 이어주기
GroupVal 함수 하나로 통합해서 어떤 척도 계산할지만 입력하도록 하기 - 대충해놓음

return, input 정확히 쓰기
"""
# def label_result(selected_result):  # label들 확인하려 했던 함수. G.nodes로 출력가능함
#     label = []
#     for sent in selected_result:
#         for i in sent:
#             if i not in label:
#                 label.append(i)
#     return label
#
#
# label = label_result(sel_result)
# print("label is")
# print(label)
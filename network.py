"""
Data processing and creating Co-Occurrence matrix

May 18th, 2019
author: Jin Uk, Cho

source : https://m.blog.naver.com/PostView.nhn?blogId=kiddwannabe&logNo=221156319157&referrerCode=4&proxyReferer=http://m.blog.naver.com/SympathyHistoryList.nhn?blogId%3Dkiddwannabe%26logNo%3D221156319157%26createTime%3D1512488368000

"""

import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

# data 읽고 불러오는 구간
data = ["examples"]
example1 = [['ab', 'bb', 'cb'],['ab', 'bd',  'cb']]

# 불용어 처리
def del_stopwords():
    return




# 형태소 분석기 뭐쓸지
# 예시로 Komoran 이용
def tag_content(contents):
    """
    Tag all words in content
    :param contents:(list) processed data
    :return: (list) tagged words divided by each sentence
    """
    results = []
    model = aaa()
    for content in contents:
        tagged_content = aaa.pos(content)
        results.append(tagged_content)
    return results

tagged_results = tag_content(data)

# 어떤 태그들만 남길지
tag_filter = ['verb', 'noun']
# 불용어 처리
stopword = ['I', 'you']


# 태그 결과에서 필터링하늖 ㅏㅁ수
def select_results(contents, tag_filter):
    """
    Select word by filtering certain tags
    :param contents: (list) processed data
    :param tag_filter: (list) tags which should be left
    :return: (list) words divided by each sentence
    """

    selected_results = []

    for content in contents:
        selection = []
        # tag 를 cat 이라 표현
        for cat in content:
            if cat in tag_filter:
                # tag 말고 안에 단어 lex 만 남겨야함
                selection.append(content.lex)

        if len(selection) > 0:
            selected_results.append(selection)
    return selected_results


selected_results = select_results(tagged_results, tag_filter)

from collections import Counter
from itertools import combinations

# Co-occurence matrix 만들기
def create_cooc_mat(contents):
    """
    Create Co-Occurrence Matrix
    :param contents: (list) processed data
    :return: (list) The number of times two words occur together in each sentence in a document. [(word1, word2), count]
    """
    word_cooc_mat = Counter()
    for line in contents:
        for w1, w2 in combinations(line, 2):
            if len(w1) == 1 or len(w2) == 1: # 1음절 단어 제외
                continue
            if w1 == w2: # 동일한 단어 벡터는 계산 x.
                continue
            elif word_cooc_mat[(w2, w1)] >= 1: # 동일한 조합이고 순서만 반대인 경우
                word_cooc_mat[(w2, w1)] += 1 # 처음 했던 조합에 카운트하겠다
            else:
                word_cooc_mat[(w1, w2)] += 1

    # dict 타입인지 몰라서 확인해봄
    # print(wc.values())
    # print(wc.items())
    # print(wc.elements())
    # print(wc.keys())

    list_key_value = [[k,v] for k, v in word_cooc_mat.items()]
    return list_key_value




print(create_cooc_mat(example1))
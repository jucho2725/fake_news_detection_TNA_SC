"""
Data processing and creating Co-Occurrence matrix

May 18th, 2019
author: Jin Uk, Cho

source : https://m.blog.naver.com/PostView.nhn?blogId=kiddwannabe&logNo=221156319157&referrerCode=4&proxyReferer=http://m.blog.naver.com/SympathyHistoryList.nhn?blogId%3Dkiddwannabe%26logNo%3D221156319157%26createTime%3D1512488368000

"""


import pandas as pd
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import sent_tokenize, word_tokenize, pos_tag
from collections import defaultdict
from nltk.corpus import stopwords
from collections import Counter
from itertools import combinations

import numpy as np
import sklearn.metrics as metrics
import tensorflow as tf
from tensorflow.python.data import Dataset


class Processing():
    def __init__(self):
        tag_map = defaultdict(lambda : wn.NOUN)
        # print(tag_map)
        tag_map['J'] = wn.ADJ
        tag_map['V'] = wn.VERB
        tag_map['R'] = wn.ADV

        text1 = "guru99 is a kind of learning experiences."

    # 문장 하나 lemmatization 함수
    def token_lemma1(self, text): # token에 is, 같은 애들을 be 로 변환 시키지 않음
        results = []
        tokens = word_tokenize(text)
        lemma_function = WordNetLemmatizer()
        for token in tokens:
            lemma = lemma_function.lemmatize(token)
            # print(token, "=>", lemma)
            results.append(lemma)
        return results

    # 문서 전체 lemmatization 함수
    def lemma_whole(self, text):
        lemma_data = []
        sentences = sent_tokenize(text)
        for sent in sentences:
            lemma_sent = self.token_lemma1(sent)
            lemma_data.append(lemma_sent)
        return lemma_data

    # 불용어 처리 함수
    def stopword(self, sentences):
        stopWords = set(stopwords.words('english'))
        added_stopword = ['“', '”', '.', ',']
        results = []

        for sent in sentences:
            wordsFiltered = []
            for w in sent:
                if w not in stopWords and w not in added_stopword and not w.isdigit():
                    wordsFiltered.append(w)
            results.append(wordsFiltered)

        # print(results)
        return results

    # 태깅 함수
    def tag_content(self, contents):
        """
        Tag all words in content
        :param contents:(list) processed data
        :return: (list) tagged words divided by each sentence
        """
        results = []
        for content in contents:
            tagged_content = pos_tag(content)
            results.append(tagged_content)

        return results

    # 태그 결과에서 필터링하는 함수
    def select_results(self, contents, tag_filter):
        """
        Select word by filtering certain tags
        :param contents: (list) processed data
        :param tag_filter: (list) tags which should be left
        :return: (list) words divided by each sentence
        """

        selected_results = []

        for content in contents:
            selection = []
            # 단어를 lex, tag 를 cat 이라 표현
            for lex, cat in content:
                if cat in tag_filter:
                    # tag 말고 안에 단어 lex 만 남겨야함
                    selection.append(lex)

            if len(selection) > 0:
                selected_results.append(selection)
        return selected_results

    # Co-occurence matrix 생성 함수
    def create_cooc_mat(self, contents):
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
        # print(word_cooc_mat.values())
        # print(word_cooc_mat.items())
        # print(word_cooc_mat.elements())
        # print(word_cooc_mat.keys())

        # list_key_value = [[k,v] for k, v in word_cooc_mat.items()]

        list_keys = [k for k in word_cooc_mat.keys()]
        list_values = [v for v in word_cooc_mat.values()]
        conv_list_keys = [[w1, w2] for w1, w2 in list_keys]
        linkages = pd.Series(two_words for two_words in conv_list_keys)
        weights = pd.Series(list_values)
        data = pd.DataFrame({'Linkage': linkages, 'Weight': weights})
        sorted_data = data.sort_values(by=['Weight'], ascending=False)
        # return list_key_value
        return sorted_data


# Example

text = "The Trump administration will delay tariffs on cars and car part imports for up to six months as it negotiates trade deals with the European Union and Japan. In a proclamation Friday, Trump said he directed U.S.Trade Representative Robert Lighthizer to seek agreements to “address the threatened impairment” of national security from car imports. Trump could choose to move forward with tariffs during the talks. “United States defense and military superiority depend on the competitiveness of our automobile industry and the research and development that industry generates,” White House press secretary Sarah Huckabee Sanders said in a statement. “The negotiation process will be led by United States Trade Representative Robert Lighthizer and, if agreements are not reached within 180 days, the President will determine whether and what further action needs to be taken."

N = Processing()
lemed_content = N.lemma_whole(text)
# print(lemed_content)

stopped_content = N.stopword(lemed_content)

tagged_results = N.tag_content(stopped_content)
# print(tagged_results)
# print('***************************************')

# 어떤 태그들만 남길지
tag_filter = ['NNP', 'NN', 'NNS', 'VBP', 'VBD', 'VBN', 'JJ', 'RB', 'VB']

selected_results = N.select_results(tagged_results, tag_filter)
# print(selected_results)

final_result = N.create_cooc_mat(selected_results)


# print(final_result)
# print(final_result['Linkage'][0])

"""  To Do List
객체화 - 완료
pd.DataFrame 으로 만들기 - 완료
corpus2graph 로 다른 객체 하나 만들기 - 
앞에 호출함수 파일 만들어서 그걸로 각각 본문 불러오기

같이 나와야하는 단어들 합치기
이상한 단어 없에기 
동사와 명사 관계 나타낼수 있는지


"""

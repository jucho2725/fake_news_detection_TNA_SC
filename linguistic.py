"""
Add linguistic features & reweight matrix

TF-IDF based on the corpus
(semantic meanings - relation between documents

June 30th, 2019
author: Jin Uk, Cho
"""

from network import Processing as ps
import pandas as pd

import os
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer


def get_document_filenames(document_path='D:\\PythonProjects\\text_network_analysis\\data\\articles'):
    """
    파일 이름 받기
    """
    return [os.path.join(document_path, each)
            for each in os.listdir(document_path)]

def get_corpus(tag_filter):########### 진행중
    """
    각 기사의 cooc를 만들고 dataframe 형식으로 저장, 그리고 tfidf를 진행할 텍스트 생성
    지금은 밑 get cooc value랑 corpus 랑 두 개로 나눠놓음
    """
    for doc in get_document_filenames():
        text, sorted_df = ps.cooc(doc, tag_filter)

        input2_corrected = [" ".join(x) for x in corpus] # 지금 문장별로 합쳐놓음. 최종적으론 전체 합치기(이중 리스트)
        print(input2_corrected)

def get_cooc_value(tag_filter): ######### 진행중
    for doc in get_document_filenames():
        text, sorted_df = ps.cooc(doc, tag_filter)



def create_vectorizer():
    """
    Tfidf 벡터 생성
    :return:
    """
    return TfidfVectorizer(input=corpus,
                           use_idf=True,  # enable inverse-document-frequency reweighting
                           smooth_idf=True)  # prevents zero division for unseen words)

def display_scores(vectorizer, tfidf_result):
    # http://stackoverflow.com/questions/16078015/
    """
    tf idf 계산한 값을 dataframe 형태로 저장. 디스플레이도 함.

    :param vectorizer:
    :param tfidf_result:
    :return:
    """
    scores = zip(vectorizer.get_feature_names(),
                 np.asarray(tfidf_result.sum(axis=0)).ravel())
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    for item in sorted_scores:
        print("{0:50} Score: {1}".format(item[0], item[1]))

    key = pd.Series([k for k, v in sorted_scores])
    value = pd.Series([v for k, v in sorted_scores])
    df_tfidf = pd.DataFrame({'word': key, 'tfidf': value})
    # print(df_tfidf)
    return df_tfidf

def doc_vetcorize(body):
    """
    한 문서에 대해서 vectorize
    :param body:
    :return:
    """
    vectorizer = create_vectorizer()
    vectorizer.fit(corpus)
    tfidf_result = vectorizer.transform(body)
    x = display_scores(vectorizer, tfidf_result)
    return x

def doc_reweight(cooc, tfidf):
    """
    cooc, tfidf dataframe 을 이용해

    reweight
    :param cooc:
    :param tfidf:
    :return:
    """
    df = cooc.copy()
    for n in tfidf.index:
        for i in cooc.index:
            if tfidf['word'][n] in cooc['linkage'][i]:
                df['weight'][i] = cooc['weight'][i] * tfidf['value'][n]



def main():
    tag_filter = ['NNP', 'NN', 'NNPS', 'NNS', 'VBG', 'VBP', 'VB']
    vectorizer = create_vectorizer()
    print(get_document_filenames())
    vectorizer.fit()
 ## 현재 각각 만들어진 body frame 에 대해 하나로 합친뒤에 말해줘
    x = display_scores(vectorizer, tfidf_result)
    print(x)


if __name__ == '__main__':
    main()
# class Reweight():
#     def __init__(self):


      # def make_corpus(self):
      #   for i in path:
      #       text, frame ps.cooc()


""" 테스트 """
example_text = "The Trump administration will delay tariffs on cars and car part imports for up to six months as it negotiates trade deals with the European Union and Japan. In a proclamation Friday, Trump said he directed U.S.Trade Representative Robert Lighthizer to seek agreements to “address the threatened impairment” of national security from car imports. Trump could choose to move forward with tariffs during the talks. “United States defense and military superiority depend on the competitiveness of our automobile industry and the research and development that industry generates,” White House press secretary Sarah Huckabee Sanders said in a statement. “The negotiation process will be led by United States Trade Representative Robert Lighthizer and, if agreements are not reached within 180 days, the President will determine whether and what further action needs to be taken."
text = [
    'The Trump administration will delay tariff on car and car part import for up to six month a it negotiate trade deal with the European Union and Japan .',
    'In a proclamation Friday , Trump say he direct U.S.Trade Representative Robert Lighthizer to seek agreement to “ address the threatened impairment ” of national security from car import .',
    'Trump could choose to move forward with tariff during the talk .',
    '“ United States defense and military superiority depend on the competitiveness of our automobile industry and the research and development that industry generates , ” White House press secretary Sarah Huckabee Sanders say in a statement .',
    '“ The negotiation process will be lead by United States Trade Representative Robert Lighthizer and , if agreement be not reach within 180 day , the President will determine whether and what further action need to be take .']

N = ps()
corpus = N.lemma_whole(example_text)
input2_corrected = [" ".join(x) for x in corpus]
print(input2_corrected)

tfidv = TfidfVectorizer().fit(input2_corrected)
print(tfidv.transform(input2_corrected).toarray())

""" 
tfidf_vectorizer = TfidfVectorizer(
    min_df=1,  # min count for relevant vocabulary
    max_features=4000,  # maximum number of features
    strip_accents='unicode',  # replace all accented unicode char 
    # by their corresponding  ASCII char
    analyzer='word',  # features made of words
    token_pattern=r'\w{1,}',  # tokenize only words of 4+ chars
    ngram_range=(1, 1),  # features made of a single tokens
    use_idf=True,  # enable inverse-document-frequency reweighting
    smooth_idf=True,  # prevents zero division for unseen words
    sublinear_tf=False)

tfidf_df = tfidf_vectorizer.fit_transform(df['text'])
"""

''' TO DO '''
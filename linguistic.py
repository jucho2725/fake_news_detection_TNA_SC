"""
Add linguistic features & reweight matrix

TF-IDF based on the corpus
(semantic meanings - relation between documents

June 30th, 2019
author: Jin Uk, Cho
"""

from network import Processing
import pandas as pd

import os
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer


def get_document_filenames(document_path='D:\\PythonProjects\\text_network_analysis\\tobedeleted'):
    """
    파일 이름 받기
    """
    return [os.path.join(document_path, each)
            for each in os.listdir(document_path)]

class CorTfidf(Processing):
    """
    corpus 에 대한 모든 단어의 tfidf 값을 dataframe 형태로 저장
    """
    def __init__(self, tag_filter):
        super().__init__(tag_filter)
        self.doc_filenames = get_document_filenames()


    def doc2list(self, text):
        """
        이중리스트로 토큰화 되어있는 문서를 공백간격으로 합쳐진 하나의 리스트로 만들어줌
        :return:
        """
        doc = text
        sent_joined = [" ".join(x) for x in doc]
        doc_joined = [" ".join(sent_joined)]
        return doc_joined

    def get_corpus(self):  ########### 진행중
        """
        각 기사를 한 문장의 리스트로 만듦. 그것들을 연결해 tfidf를 진행할 코퍼스 생성
        """
        corpus = []
        for doc in self.doc_filenames:
            processed_text, _ = self.cooc(doc)
            print(doc)
            x = self.doc2list(processed_text)[0]
            corpus.append(x)
        return corpus

    def display_scores(self, vectorizer, tfidf_result, display=False):
        # http://stackoverflow.com/questions/16078015/
        """
        tf idf 계산한 값을 {term: , tfidf값: } dataframe 형태로 저장. 디스플레이도 가능
        주의 : 여기서 tfidf 는 각 문서마다 존재하는 term 들의 tfidf를 모두 합한 것임.
        즉 특정 term 의 tfidf는 모든 문서에서의 그 term 의 tfidf 의 summation
        https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
        책에 나오는 것과 약간 다른데, 자세한 내용은 링크 참조
        :param vectorizer:
        :param tfidf_result:
        :param display:
        :return:
        """
        scores = zip(vectorizer.get_feature_names(),
                     np.asarray(tfidf_result.sum(axis=0)).ravel())
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        if display is True:
            for item in sorted_scores:
                print("{0:50} Score: {1}".format(item[0], item[1]))

        key = pd.Series([k for k, v in sorted_scores])
        value = pd.Series([v for k, v in sorted_scores])
        df_tfidf = pd.DataFrame({'Word': key, 'Tfidf': value})
        return df_tfidf

    def cor2tfidf(self, corpus):
        """
        corpus 대해서 tfidf vectorize
        :param corpus:
        :return:
        """
        vectorizer = TfidfVectorizer()
        vectorizer.fit(corpus)
        tfidf_result = vectorizer.transform(corpus)
        x = self.display_scores(vectorizer, tfidf_result)
        return x


class Reweight(CorTfidf):
    def __init__(self, tag_filter):
        super(Reweight, self).__init__(tag_filter)
        self.doc_filenames = get_document_filenames()
        self.df_tfidf = self.cor2tfidf(self.get_corpus())

    def doc_reweight_csv(self, cooc_mat, tfidf, doc_name):
        """
        cooc, tfidf dataframe 을 이용해 하나의 문서 reweight함
        그리고 reweighted 된 dataframe 을 csv파일로 저장
        :param cooc:
        :param tfidf:
        :return:
        """
        redf = cooc_mat.copy()
        for n in tfidf.index:
            for i in cooc_mat.index:
                if tfidf.loc[n, 'Word'] in cooc_mat.loc[i, 'Linkage']:
                    redf.loc[i, 'Weight'] = redf.loc[i, 'Weight'] * tfidf.loc[n, 'Tfidf']
        redf.to_csv(doc_name[:-4] + '.csv')
        return redf

    def get_docs_rew_csv(self):
        for doc in self.doc_filenames:
            _, df_cooc = self.cooc(doc)
            self.doc_reweight_csv(df_cooc, self.df_tfidf, doc)
        print("all documents are reweighted and saved to .csv files.")


def main():
    tag_filter = ['NNP', 'NN', 'NNPS', 'NNS', 'VBG', 'VBP', 'VB']
    model = Reweight(tag_filter)
    model.get_docs_rew_csv()


if __name__ == '__main__':
    main()


#
# """ 테스트 """
# example_text = "The Trump administration will delay tariffs on cars and car part imports for up to six months as it negotiates trade deals with the European Union and Japan. In a proclamation Friday, Trump said he directed U.S.Trade Representative Robert Lighthizer to seek agreements to “address the threatened impairment” of national security from car imports. Trump could choose to move forward with tariffs during the talks. “United States defense and military superiority depend on the competitiveness of our automobile industry and the research and development that industry generates,” White House press secretary Sarah Huckabee Sanders said in a statement. “The negotiation process will be led by United States Trade Representative Robert Lighthizer and, if agreements are not reached within 180 days, the President will determine whether and what further action needs to be taken."
# sent_text = [
#     'The Trump administration will delay tariff on car and car part import for up to six month a it negotiate trade deal with the European Union and Japan .',
#     'In a proclamation Friday , Trump say he direct U.S.Trade Representative Robert Lighthizer to seek agreement to “ address the threatened impairment ” of national security from car import .',
#     'Trump could choose to move forward with tariff during the talk .',
#     '“ United States defense and military superiority depend on the competitiveness of our automobile industry and the research and development that industry generates , ” White House press secretary Sarah Huckabee Sanders say in a statement .',
#     '“ The negotiation process will be lead by United States Trade Representative Robert Lighthizer and , if agreement be not reach within 180 day , the President will determine whether and what further action need to be take .']
#
# N = ps()
# corpus = N.lemma_whole(example_text)
# input2_corrected = [" ".join(x) for x in corpus]
# print(input2_corrected)
#
# tfidv = TfidfVectorizer().fit(input2_corrected)
# print(tfidv.transform(input2_corrected).toarray())
#
''' TO DO 
1. 반복문횟수 줄이기 - 너무 오래걸림

2. reweight이 안됨. 확인 필요 
SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
  redf['Weight'][i] = cooc_mat['Weight'][i] * tfidf['Tfidf'][n]


3. 각 메서드에 대한 간단한 설명 달기
'''


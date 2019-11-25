"""
Data processing and creating Co-Occurrence matrix

May 18th, 2019
author: Jin Uk, Cho

source : https://m.blog.naver.com/PostView.nhn?blogId=kiddwannabe&logNo=221156319157&referrerCode=4&proxyReferer=http://m.blog.naver.com/SympathyHistoryList.nhn?blogId%3Dkiddwannabe%26logNo%3D221156319157%26createTime%3D1512488368000

last update : Nov 26th, 2019

"""

from collections import Counter, defaultdict
from itertools import combinations
import nltk
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
import pandas as pd
import os
import ast


from nltk import sent_tokenize, word_tokenize, pos_tag, ne_chunk
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tree import Tree

from merge_colloc import MergeColloc



def strToList(df):
    '''
    desc : Changing df['linkage'] data type from string to list.
    '''
    try :
        df['Linkage']= [ast.literal_eval(str) for str in df.iloc[:,0]]
    except:
        pass
    return df

class Processing():
    def __init__(self, tag_filter):
        self.tag_map = defaultdict(lambda: wn.NOUN)
        self.tag_map['J'] = wn.ADJ
        self.tag_map['V'] = wn.VERB
        self.tag_map['R'] = wn.ADV
        self.tag_filter = tag_filter

    @staticmethod
    def apply_collocations(sentence):
        sentence = sentence.replace("George H.W. Bush", "George_H.W._Bush")
        # add any phrase here
        return sentence

    # 문장 하나 lemmatization 함수
    def lemma_sentence(self, text):
        '''
        lemmatize a sentence. If the pos tag of a token starts with 'V', 'R, 'J',
        It will be lemmatized to the word's origin (was -> be) (studied -> study)
        :param text: (string) a sentence
        :return: (list) list of tokens
        '''
        results = []
        tokens = word_tokenize(text)
        # NER_chunk 함수 넣어주기
        tokens = self.ner_chunk(tokens)  # -> pos tag / ne_chunk 포함
        lmtzr = WordNetLemmatizer()
        replace_data = {"n't": 'not'}  # lemmatatization에서 제거 되고 싶지 않은 단어 추가
        for token, tag in pos_tag(tokens):
            if token in replace_data.keys():
                token = token.replace(token, replace_data[token])
            lemma = lmtzr.lemmatize(token, self.tag_map[tag[0]])
            results.append(lemma)
        return results

    # 문서 전체 lemmatization 함수
    def lemma_text(self, text):
        '''
        lemmatize text
        :param text: (string) raw text
        :return: (list of list) tokenized and lemmatized sentences
        '''
        lemma_data = []
        sentences = sent_tokenize(text)
        for sent in sentences:
            lemma_sent = self.lemma_sentence(sent)
            lemma_data.append(lemma_sent)
        return lemma_data

    # 불용어 처리 함수
    # 여기서부턴 string형태가 아니라 이중리스트 형태이므로 sentences 와 sentence 로 구분함
    def stopword(self, sentences):
        '''
        remove stopwords from sentences.
        Note that 'not' is remained due to represent negation
        :param sentences: (list of list)
        :return: (list of list)
        '''
        stopWords = set(stopwords.words('english')) - set(['not'])
        added_stopword = ['“', '”', '.', ',', '-', "—", "–", "'s", "n't", "''", ';', '&', "``", '?', "‘", "’"]
        results = []

        for sent in sentences:
            wordsFiltered = []
            wordsStopped = []
            for w in sent:
                if w not in stopWords and w not in added_stopword and not w.isdigit():
                    wordsFiltered.append(w)
                else:
                    wordsStopped.append(w)
            results.append(wordsFiltered)

        return results

    # 태깅 함수

    # apply_collocation 수정
    def ner_chunk(self, tokens):  # George H.W. Bush는 따로 작업
        '''
        :param tokens: list of tokens in one sentence
        :return:
        '''
        chunked = ne_chunk(pos_tag(tokens), binary=True)
        continuous_chunk = []
        current_chunk = []

        for i in chunked:
            if type(i) == Tree:
                current_chunk.append("_".join([token for token, pos in i.leaves()]))
                named_entity = " ".join(current_chunk)
                continuous_chunk.append(named_entity)
                current_chunk = []
            else:
                continuous_chunk.append(i[0])
                continue
        if current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
        return continuous_chunk

    def tag_content(self, sentences):
        """
        Tag all words in content
        :param sentences:(list) processed data
        :return: (list) tagged words divided by each sentence
        """
        results = []
        for content in sentences:
            tagged_content = pos_tag(content)
            results.append(tagged_content)

        return results

    # 태그 결과에서 필터링하는 함수
    def select_results(self, sentences):
        """
        Select word by filtering certain tags
        :param sentences: (list) processed data
        :param tag_filter: (list) tags which should be left
        :return: (list) words divided by each sentence
        """

        selected_results = []

        for sentence in sentences:
            selection = []
            # 단어를 lex, tag 를 cat 이라 표현
            for lex, tag in sentence:
                if tag in self.tag_filter:
                    # tag 말고 안에 단어 lex 만 남겨야함
                    selection.append(lex)

            if len(selection) > 0:
                selected_results.append(selection)
        return selected_results

    # Co-occurence matrix 생성 함수
    def create_cooc_mat(self, sentences):
        """
        Create Co-Occurrence Matrix
        :param sentences: (list of list) processed sentences.
        :return: (list) The number of times two words occur together in each sentence in a document. [(word1, word2), count]
        """
        word_cooc_mat = Counter()
        for sentence in sentences:
            for w1, w2 in combinations(sentence, 2):
                if len(w1) == 1 or len(w2) == 1:  # 1음절 단어 제외
                    continue
                if w1 == w2:  # 동일한 단어 벡터는 계산 x.
                    continue
                elif word_cooc_mat[(w2, w1)] >= 1:  # 동일한 조합이고 순서만 반대인 경우
                    word_cooc_mat[(w2, w1)] += 1  # 처음 했던 조합에 카운트하겠다
                else:
                    word_cooc_mat[(w1, w2)] += 1

        list_keys = [k for k in word_cooc_mat.keys()]
        list_values = [v for v in word_cooc_mat.values()]
        conv_list_keys = [[w1, w2] for w1, w2 in list_keys]
        linkages = pd.Series(two_words for two_words in conv_list_keys)
        weights = pd.Series(list_values)
        data = pd.DataFrame({'Linkage': linkages, 'Weight': weights})
        sorted_data = data.sort_values(by=['Weight'], ascending=False)
        return sorted_data

    def cooc(self, filepath=None, text=None, savepath=None):
        '''
        Make cooc matrix.
        :param filepath: (default=None) If a input is saved as file, add path
        :param text: (default=None) If a input is just a raw text, add text
        :return:
        (List) list of words occur in the text
        (Dataframe) cooc matrix represented by dataframe format
        '''
        if filepath is not None:
            text = open(filepath, encoding='utf-8').read()
        else:
            text = text
        text = self.apply_collocations(text)
        lem_sents = self.lemma_text(text)
        stop_sents = self.stopword(lem_sents)
        tag_sents = self.tag_content(stop_sents)
        sel_sents = self.select_results(tag_sents)  # 단어 리스트
        cooc_mat = self.create_cooc_mat(sel_sents)  # 단어간 연결 데이터프레임

        first_class = MergeColloc(cooc_mat)
        df = first_class.df
        for index, colloc_word in enumerate(first_class.colloc_words):
            # 추가수정- duplicated word 제대로 형성 위해서 다시 선언
            c = MergeColloc(cooc_mat)

            # findindex 자체를 바꿀 가능성 생각해보기
            colloc_index, no_colloc_index, _ = c.findIndex(colloc_word, df=df)
            duplicated_word, _ = c.findLinkageWord(colloc_word, df=df)

            # 여러 colloc_word에 대해서 변경을 계속 축적하려면 row 단위가 아닌 index 단위 변경이 필요함
            a = strToList(df).loc[colloc_index, :]
            b = strToList(df).loc[no_colloc_index, :]

            # colloc, no_colloc index가 둘다 없는 index 들
            d = strToList(df).loc[[index for index in df.index if index not in colloc_index + no_colloc_index], :]

            colloc_df, no_colloc_df = c.SumDropAll(colloc_word, duplicated_word, a, b)

            # 연어 처리하는 일반화된 규칙을 고민해봐야함
            no_colloc_df.iloc[:, 0] = pd.Series(
                [ast.literal_eval(str(linkage).replace(colloc_word.split('_')[0], colloc_word)) for linkage in
                 no_colloc_df.iloc[:, 0]], index=no_colloc_df.index)

            merged_mat = pd.concat([colloc_df, no_colloc_df, d])

        if savepath is not None:
            merged_mat.to_csv(savepath)

        return sel_sents, cooc_mat





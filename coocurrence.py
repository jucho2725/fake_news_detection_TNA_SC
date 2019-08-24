"""
Data processing and creating Co-Occurrence matrix

May 18th, 2019
author: Jin Uk, Cho

source : https://m.blog.naver.com/PostView.nhn?blogId=kiddwannabe&logNo=221156319157&referrerCode=4&proxyReferer=http://m.blog.naver.com/SympathyHistoryList.nhn?blogId%3Dkiddwannabe%26logNo%3D221156319157%26createTime%3D1512488368000

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
from nltk import sent_tokenize, word_tokenize, pos_tag, ne_chunk
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tree import Tree



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
    def lemma_sentence(self, text):  # token에 is, 같은 애들을 be 로 변환 시키지 않음
        results = []
        tokens = word_tokenize(text)
        # NER_chunk 함수 넣어주기 (tokens 단위?)
        tokens = self.ner_chunk(tokens)  #-> pos tag / ne_chunk 포함
        lmtzr = WordNetLemmatizer()
        replace_data={"n't":'not'} #lemmatatization에서 제거 되고 싶지 않은 단어 추가
        for token, tag in pos_tag(tokens):
            # print("token :", token, "tag :", tag)
            if token in replace_data.keys():
                # print("pass replace_Data: ",token)
                token =token.replace(token,replace_data[token])
                # print("after replace: ",token)
            lemma = lmtzr.lemmatize(token, self.tag_map[tag[0]])
            # print(token, "=>", lemma)
            results.append(lemma)
        return results

    # 문서 전체 lemmatization 함수
    def lemma_text(self, text):
        # collocation 을 이 단에서 추가해야할 듯 (sent tokenize 되지 않도록)
        lemma_data = []
        sentences = sent_tokenize(text)
        for sent in sentences:
            lemma_sent = self.lemma_sentence(sent)
            lemma_data.append(lemma_sent)
        return lemma_data

    # 불용어 처리 함수
    # 여기서부턴 string형태가 아니라 이중리스트 형태이므로 sentences 와 sentence 로 구분함
    def stopword(self, sentences):
        stopWords = set(stopwords.words('english'))-set(['not'])
        added_stopword = ['“', '”', '.', ',', '-', "—", "–", "'s", "n't", "''", ';', '&', "``", '?', "‘", "’"]
        results = []

        for sentence in sentences:
            wordsFiltered = []
            wordsStopped = []
            for w in sentence:
                if w not in stopWords and w not in added_stopword and not w.isdigit():
                    wordsFiltered.append(w)
                else:
                    wordsStopped.append(w)
            results.append(wordsFiltered)

        # print(results)
        return results

    # 태깅 함수

    # apply_collocation 수정
    def ner_chunk(self,tokens): #George H.W. Bush는 따로 작업
        chunked = ne_chunk(pos_tag(tokens), binary=True)
        # prev = None
        continuous_chunk = []
        current_chunk = []
        # print("gcc_text in :", text)
        # print("gcc chunked in : ",chunked )
        for i in chunked:
            if type(i) == Tree:
                current_chunk.append("_".join([token for token, pos in i.leaves()]))
                # print("current chunk: ",current_chunk,"\n")
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
        for sentence in sentences:
            tagged_content = pos_tag(sentence)
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
            for lex, cat in sentence:
                if cat in self.tag_filter:
                    # tag 말고 안에 단어 lex 만 남겨야함
                    selection.append(lex)

            if len(selection) > 0:
                selected_results.append(selection)
        return selected_results

    # Co-occurence matrix 생성 함수
    def create_cooc_mat(self, sentences):
        """
        Create Co-Occurrence Matrix
        :param sentences: (list of list) processed data
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
                    # print(word_cooc_mat[(w2, w1)])
                else:
                    word_cooc_mat[(w1, w2)] += 1
                    # print(word_cooc_mat[(w1, w2)])

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

    def cooc(self, filepath=None, text=None):
        if filepath is not None:
            text = open(filepath, encoding='utf-8').read()
        else:
            text = text
        text = self.apply_collocations(text)
        lem_sents = self.lemma_text(text)
        stop_sents = self.stopword(lem_sents)
        tag_sents = self.tag_content(stop_sents)
        sel_sents = self.select_results(tag_sents) # 단어 리스트 - 사용할 품사 종류 합의 필요
        cooc_mat = self.create_cooc_mat(sel_sents) # 단어간 연결 데이터프레임
        return sel_sents, cooc_mat


# """ 테스트 """

# 어떤 태그들만 남길지
# tag_filter = ['NNP', 'NN', 'NNPS', 'NNS', 'VBG', 'VBP', 'VB', 'RB', 'JJ']
# example_text = "The Trump administration will n't delay tariffs on cars and car part imports for up to six months as it negotiates trade deals with the European Union and Japan. In a proclamation Friday, Trump said he directed U.S.Trade Representative Robert Lighthizer to seek agreements to “address the threatened impairment” of national security from car imports. Trump could choose to move forward with tariffs during the talks. “United States defense and military superiority depend on the competitiveness of our automobile industry and the research and development that industry generates,” White House press secretary Sarah Huckabee Sanders said in a statement. “The negotiation process will be led by United States Trade Representative Robert Lighthizer and, if agreements are not reached within 180 days, the President will determine whether and what further action needs to be taken."
# example_text2 = " NEW YORK — Staring down tightening polls both nationwide and in the battleground states, Hillary Clinton’s campaign manager Robby Mook on Monday circulated a private memo to donors, supporters and top volunteers that maps out the Democratic nominee’s various paths to the White House in November, paired with his analysis of Donald Trump’s own precarious path. “Here’s the story that no poll can tell: Hillary Clinton has many paths to 270 electoral votes, while Donald Trump has very few. Hillary is nearly certain to win 16 ‘ blue' states, including Washington D.C., which will garner her 191 electoral votes,” writes Mook in the nearly 2,000-word memo that was blasted out in the early evening, and which was obtained by POLITICO."
# example_text3 = "Former President George H.W. Bush is bucking his party's presidential nominee and plans to vote for Hillary Clinton in November, according to a member of another famous political family, the Kennedys. Bush. 92. had intended to stay silent on the White House race between Clinton and Donald Trump, a sign in and of itself of his distaste for the GOP nominee. "
# # text = open("./data/fake/1247033108723070.txt", encoding='utf-8-sig').read()
# # print(text)
# #
# # my_sent = "WASHINGTON -- In the wake of a string of abuses by New York police officers in the 1990s, Loretta E. Lynch, the top federal prosecutor in Brooklyn, spoke forcefully about the pain of a broken trust that African-Americans felt and said the responsibility for repairing generations of miscommunication and mistrust fell to law enforcement."
# # txt = "Barack Obama is a great person and so is Michelle Obama."
# #
# N = Processing(tag_filter)
# # lemed_content = N.lemma_text(text)
# # lemed_content = N.lemma_text(example_text3)
# # print(example_text3)
# # print(lemed_content)
# print(example_text3)
# N.cooc(text=example_text3)
# # print(example_text3.replace('George H.W. Bush','Geogre_H.W._Bush'))
#
# #
# # stopped_content = N.stopword(lemed_content)
# # print(stopped_content)
# # collocated_content = N.collocate_content(stopped_content)
# # print(collocated_content)
# #
# # tagged_results = N.tag_content(collocated_content)
# # tagged_results = N.tag_content(stopped_content)
# # print(tagged_results)
# # print('***************************************')
# #
#
# # #
# # selected_results = N.select_results(tagged_results)
# # print(selected_results)
# #
# # final_result = N.create_cooc_mat(selected_results)
# # print(final_result)
# # print(final_result['Linkage'][0])
#
# # N = Processing(tag_filter)
# # df, s_df = N.cooc()
# # print(df['Linkage'][:20].tolist())
# # print(list(df['Weight'][:20]))
#

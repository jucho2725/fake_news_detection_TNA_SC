import nltk
# nltk.download()

text = "The Trump administration will delay tariffs on cars and car's part imports for up to six months as it negotiates trade deals with the European Union and Japan. In a proclamation Friday, Trump said he directed U.S.Trade Representative Robert Lighthizer to seek agreements to “address the threatened impairment” of national security from car imports. Trump could choose to move forward with tariffs during the talks. “United States defense and military superiority depend on the competitiveness of our automobile industry and the research and development that industry generates,” White House press secretary Sarah Huckabee Sanders said in a statement. “The negotiation process will be led by United States Trade Representative Robert Lighthizer and, if agreements are not reached within 180 days, the President will determine whether and what further action needs to be taken."
sentence = nltk.sent_tokenize(text)
print(sentence)

# #
# from nltk.corpus import wordnet as wn
# from nltk.stem.wordnet import WordNetLemmatizer
# from nltk import word_tokenize, pos_tag
# from collections import defaultdict
# tag_map = defaultdict(lambda : wn.NOUN)
# print(tag_map)
# tag_map['J'] = wn.ADJ
# tag_map['V'] = wn.VERB
# tag_map['R'] = wn.ADV
#
# text = "guru99 is a totally new kind of learning experience."
# def token_lemma(text):
#
#     results = []
#     tokens = word_tokenize(text)
#     lemma_function = WordNetLemmatizer()
#     for token, tag in pos_tag(tokens):
#         lemma = lemma_function.lemmatize(token, tag_map[tag[0]])
#         print(token, "=>", lemma)
#         results.append()
#     return results
#
# for sent in sentence():
#
from nltk.tokenize import sent_tokenize, word_tokenize

# for sent in sentence:
# 	 print(nltk.pos_tag(nltk.word_tokenize(sent)))
#
# #################
# from nltk.tokenize import sent_tokenize, word_tokenize
# from nltk.corpus import stopwords
#
# data = "All work and no play makes jack dull boy. All work and no play makes jack a dull boy."
# stopWords = set(stopwords.words('english'))
# words = word_tokenize(data)
# wordsFiltered = []
#
# for w in words:
#     if w not in stopWords:
#         wordsFiltered.append(w)
#
# print(wordsFiltered)
# ######################

# import nltk
# nltk.download()


#
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from collections import defaultdict
tag_map = defaultdict(lambda : wn.NOUN)
print(tag_map)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

text = "guru99 is a totally new kind of learning experience."
tokens = word_tokenize(text)
lemma_function = WordNetLemmatizer()
for token, tag in pos_tag(tokens):
    lemma = lemma_function.lemmatize(token, tag_map[tag[0]])
    print(token, "=>", lemma)


# text = "The Trump administration will delay tariffs on cars and car's part imports for up to six months as it negotiates trade deals with the European Union and Japan. In a proclamation Friday, Trump said he directed U.S.Trade Representative Robert Lighthizer to seek agreements to “address the threatened impairment” of national security from car imports. Trump could choose to move forward with tariffs during the talks. “United States defense and military superiority depend on the competitiveness of our automobile industry and the research and development that industry generates,” White House press secretary Sarah Huckabee Sanders said in a statement. “The negotiation process will be led by United States Trade Representative Robert Lighthizer and, if agreements are not reached within 180 days, the President will determine whether and what further action needs to be taken.” In his proclamation, Trump argued in part that “domestic conditions of competition must be improved by reducing imports.” The White House had to decide by Saturday whether to slap duties on automobiles. Earlier this year, the Commerce Department said Trump could justify the move on national security grounds. By law, the administration can push back its decision by up to six months if it is negotiating with trading partners. In a statement Friday, EU Trade Commissioner Cecilia Malmstrom said “we completely reject the notion that our car exports are a national security threat.” She added that the trade bloc “is prepared to negotiate a limited trade agreement” including cars, but not so-called managed trade, in which the partners could set targets like quotas.  Malmstrom said EU officials will discuss the issue with Lighthizer next week in Paris.  Levying the automobile tariffs threatened to open new fronts in a global trade war that could drag down the U.S. economy. The EU has already prepared a list of American goods to target with tariffs if Trump goes ahead with the car duties. Automakers and some U.S. lawmakers opposed the potential tariffs. The American car industry said the duties would put jobs in jeopardy and raise prices for consumers. The decision comes after the U.S. and China fired new shots in their trade war. The White House is working to salvage a deal with Beijing to address what the U.S. calls trade abuses amid the widening conflict. Trump also used the national security justification last year to put tariffs on steel and aluminum imports, including metals coming from allies such as the EU, Canada and Mexico. Europe previously retaliated after those duties."
# sentence = nltk.sent_tokenize(text)
# print(sentence)
# for sent in sentence:
# 	 print(nltk.pos_tag(nltk.word_tokenize(sent)))
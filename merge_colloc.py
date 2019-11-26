import ast

def strToList(df):
    '''
    desc : Changing df['linkage'] data type from string to list.
    '''
    try :
        df['Linkage']= [ast.literal_eval(str) for str in df.iloc[:,0]]
    except:
        pass
    return df

class MergeColloc():
    def __init__(self, df):
        self.df = df
        # Linkage : string type에서 형성
        # for i in self.df.iloc[:10,0]:
        #     temp =str(i)

        temp_list = [i for i in set(sum([s for s in self.df.iloc[:, 0] if "_" in str(s)], [])) if '_' in i]
        # temp_list = [i for i in set(sum([ast.literal_eval(s) for s in self.df.iloc[:, 0] if "_" in s], [])) if '_' in i]

        # 두 개의 단어로만 형성된 연어만 취급
        self.colloc_words = [word for word in temp_list if len(word.split('_')) == 2]
        # print("colloc_words in corpus: ",self.colloc_words)

    def findIndex(self, colloc_word, df=[]):
        '''
        * params:
        colloc_word - Target collocation word. ex) 'United_States', 'Donald_Trump', 'Fox_News', 'Hillary_Clintons', 'Middle_East'
        df - A data frame which wasn't applied 'strToList' function.
        * return :
        colloc_index - indexes which have collocation words
        no_colloc_index - indexes which don't have collocation words, but splited word of them.
        '''
        if len(df) == 0:
            df = self.df
        a, b = colloc_word.split('_')

        colloc_index = []
        no_colloc_index = []
        nothing_index = []
        total_index = []

        # 추가수정 - row 값에서 index 값 반환으로 변경
        # string type 버젼
        for i in range(len(df)):
            if (colloc_word in df.iloc[i, 0]):
                colloc_index.append(df.index[i])

            if (a in df.iloc[i, 0]) or (b in df.iloc[i, 0]):
                total_index.append(df.index[i])

            if (a not in df.iloc[i:0]) and (b not in df.iloc[i, 0]):
                nothing_index.append(df.index[i])

        no_colloc_index = [i for i in total_index if i not in colloc_index]

        return colloc_index, no_colloc_index, nothing_index

        ### 0915 수정
        # 수정 필요 - query 문? / append 대체? / list comprehension

    def findIndexListType(self, colloc_word, df=[]):
        if len(df) == 0:
            df = self.df
        a, b = colloc_word.split('_')
        df = strToList(df)  # string to index 화

        colloc_index = []
        no_colloc_index = []
        nothing_index = []
        total_index = []

        # list type 버젼
        for i in range(len(df)):
            if (colloc_word in df.iloc[i, 0]):
                colloc_index.append(df.index[i])

            if (a in df.iloc[i, 0]) or (b in df.iloc[i, 0]):
                total_index.append(df.index[i])

            if (a not in df.iloc[i:0]) and (b not in df.iloc[i, 0]):
                nothing_index.append(df.index[i])

        no_colloc_index = [i for i in total_index if i not in colloc_index]

        return colloc_index, no_colloc_index, nothing_index

    def findLinkageWord(self, colloc_word, df=[]):
        '''
        params: colloc_word - A collocation word. Same word as findIndex function.
        return: duplicated_word - The linkage words, which are both in colloc_index and no_colloc_index.
        '''

        if len(df) == None:
            df = self.df

        col_index, no_col_index, _ = self.findIndex(colloc_word, df)
        colloc_linkage_word = set(sum(list(strToList(df).loc[col_index, 'Linkage']), []))
        no_colloc_linkage_word = set(sum(list(strToList(df).loc[no_col_index, 'Linkage']), []))

        # 추가 수정 - 연어와 그 분절된 단어가 같이 있을 경우 duplicated word에서 제거
        # string으로 추가하기에 이런 오류가 발생하는가?

        duplicated_word = [word for word in no_colloc_linkage_word if (word in colloc_linkage_word)]

        # 추가 작업 - duplicated 단어 중에 분리 시 colloc_word와 겹치는 부분이 있는지 확인하고 제거
        dup_list = []
        for word in duplicated_word:
            chk = 0
            for chunk in word.split('_'):
                if chunk not in colloc_word.split('_'):
                    chk += 0
                else:
                    chk += 1
                    # print("dup_list pass: ", chunk, "'in'", word)

            if chk == 0:
                dup_list.append(word)

        return dup_list, duplicated_word

    def SumDrop(self, colloc_word, linkage_word, df_colloc, df_no_colloc):
        '''
        params : linkage_word - A word which among the result of findLinkageWord function (duplicated_word).
        return : sum weight to colloc_word , and delete the row in no_colloc_word
        '''

        colloc_row = [index for index, data in enumerate(df_colloc.loc[:, 'Linkage']) if linkage_word in data][0]
        no_colloc_row = [index for index, data in enumerate(df_no_colloc.loc[:, 'Linkage']) if linkage_word in data][0]

        df_colloc.iloc[colloc_row, 1] += df_no_colloc.iloc[no_colloc_row, 1].item()
        df_no_colloc = df_no_colloc.drop(df_no_colloc.index[no_colloc_row], axis=0)

        return df_colloc, df_no_colloc

    def SumDropAll(self, colloc_word, duplicated_word, df_colloc, df_no_colloc):
        # 새로운 colloc_word마다 새롭게 형성
        a = df_colloc
        b = df_no_colloc

        for linkage_word in duplicated_word:
            a, b = self.SumDrop(colloc_word, linkage_word, a, b)

        return a, b


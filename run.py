import os
import time

from reweight import CorTfidf, Reweight
from features import Feature

# prev_doc_path = 'D:\\PythonProjects\\text_network_analysis\\data'

def get_doc_filenames(document_path):
    """
    파일 이름 받기
    """
    return [os.path.join(document_path, each)
            for each in os.listdir(document_path)]

def get_rew_filenames(document_path):
    """
    파일 이름 받기
    """
    return [os.path.join(document_path, each)
            for each in os.listdir(document_path)]

def main():
    tag_filter = ['NNP', 'NN', 'NNPS', 'NNS', 'VBG', 'VBP', 'VB']
    # path_fake = 'data/false'
    # path_true = 'data/true'
    path_fake = 'test/false'
    path_true = 'test/true'
    path_test = 'test/'

    startTime = time.time()
    # doc_path_list = get_doc_filenames(path_test)
    doc_path_list_f = path_fake
    doc_path_list_t = path_true
    doc_path_list = doc_path_list_f + doc_path_list_t

    print("Get tfidf value from corpus")
    reweighting_model = Reweight(tag_filter, doc_path_list=doc_path_list)
    tfidfTime = time.time()
    print("It took %d seconds" % (tfidfTime - startTime))

    print("Reweight all articles and save to .csv")
    reweighting_model.get_docs_rew_csv() # 델이 reweighted 된 linkage 가 있는 각 기사의 csv 파일을 만들어줌
    rewTime = time.time()
    print("It took %d seconds" % (rewTime - tfidfTime))

    print(" ")
    print("Reweighting Finished.. Start new job")
    print(" ")

    print("Make a graph and read tfidf result")
    rew_f_list = get_rew_filenames(document_path=path_fake + 'reweighted/')
    rew_t_list = get_rew_filenames(document_path=path_true + 'reweighted/')
    rew_path_list = rew_f_list + rew_t_list

    feature_model = Feature(doc_path_list=rew_path_list) # init - 그래프 만들고 tfidf.csv 불러옴
    graphTime = time.time()
    print("It took %d seconds" % (graphTime - rewTime))

    print("make all features and load all to dataframe ")
    df = feature_model.make_df_from_dataset(label='test')

    df.to_csv(path_test + 'result.csv')
    print("Done")
    endTime = time.time()
    print("It took %d seconds" % (endTime - graphTime))

    print("Total time : %d" % (endTime - startTime))


if __name__ == '__main__':
    main()

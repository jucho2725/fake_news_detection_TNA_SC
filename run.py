import os

from coocurrence import Processing
from features import Feature

import nltk
import pandas as pd

import argparse
from tqdm import tqdm

"""
BEFORE YOU START please make sure downloading data in nltk package.

# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')

"""

# prev_doc_path = 'D:\\PythonProjects\\text_network_analysis\\data'

def define_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default='csv')
    parser.add_argument('--filepath', type=str)

    args = parser.parse_args()
    return args

def get_doc_filenames(document_path):
    """
    파일 이름 받기 - txt 형식 일 경우
    """
    return [os.path.join(document_path, each)
            for each in os.listdir(document_path)]

def get_cooc_filenames(document_path):
    """
    파일 이름 받기 - 만들어진 cooc 파일들을 불러옴
    """
    return [os.path.join(document_path[:], each)
            for each in os.listdir(document_path[:])]

def main(args):
    print("Get cooc of each doc from corpus")
    cooc_model = Processing()

    savepath = "sample_data/"
    coocpath = savepath + 'cooc/'

    filepath = args.filepath

    if not os.path.isdir(coocpath):
        os.system('mkdir ' + coocpath)

    if args.data_type == 'csv' or 'tsv':
        if args.data_type == 'csv':
            df = pd.read_csv(filepath) # path 가 현재는 dir, 근데
        else:
            df = pd.read_csv(filepath, sep='\t',)

        with tqdm(total = len(df['text'][18087:18200])) as pbar: #change index here
            no_processed_idx = []
            f =open(savepath+"no_processed_index.txt",'a',encoding='utf-8')
            f.write("Not process index:\n")
            for idx, text in enumerate(df['text'][18087:18200]): #change index here
                try:
                    pbar.update(1)
                    cooc_model.cooc(text=text, savepath="{0}/{1}.csv".format(coocpath, idx+18087))
                except Exception as e:
                    f.write("{}, index:{}\n".format(e,idx+18087))

        f.close()
        print(" ")
        print("Creation Finished.. Starts new job")
        print(" ")

        print("Make a graph")

        feature_model = Feature(doc_path_list=coocpath, dataframe=df)

        print("Make all features and load all to dataframe ")
        df = feature_model.make_df_from_dataset()

        df.to_csv(savepath + 'result.csv')
        print("Completed")

    elif args.data_type == 'txt' or 'text':
        path_fake = savepath + '/data/fake'
        path_true = savepath + '/data/true'

        doc_path_list_f = get_doc_filenames(path_fake)
        doc_path_list_t = get_doc_filenames(path_true)

        doc_label = [0]*len(doc_path_list_f) + [1]*len(doc_path_list_t)

        df = pd.DataFrame(doc_label, columns=['label'])

        with tqdm(total = len(doc_path_list_f), desc="co-occurrence matrix creation - fake news") as pbar:
            for idx, doc_path in enumerate(doc_path_list_f):
                pbar.update(1)
                cooc_model.cooc(filepath=doc_path, savepath="{0}/{1}.csv".format(path_fake, idx))

        with tqdm(total=len(doc_path_list_t), desc="co-occurrence matrix creation - true news") as pbar:
            for idx, doc_path in enumerate(doc_path_list_t):
                pbar.update(1)
                cooc_model.cooc(filepath=doc_path, savepath="{0}/{1}.csv".format(path_true, idx))

        print(" ")
        print("Creation Finished.. Starts new job")
        print(" ")

        print("Make a graph")
        cooc_f_list = get_cooc_filenames(document_path=path_fake)
        cooc_t_list = get_cooc_filenames(document_path=path_true)
        cooc_path_list = cooc_f_list + cooc_t_list

        feature_model = Feature(doc_path_list=cooc_path_list, dataframe=df)

        print("Make all features and load all to dataframe ")
        df = feature_model.make_df_from_dataset()

        df.to_csv(savepath + '/data/' + 'result.csv')
        print("Completed")

if __name__ == '__main__':
    args = define_argparser()
    main(args)


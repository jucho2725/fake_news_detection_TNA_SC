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
    parser.add_argument('--filepath', type=str, default='data/data.csv')

    args = parser.parse_args()
    return args

def get_doc_filenames(document_path):
    return [os.path.join(document_path, each)
            for each in os.listdir(document_path)]

def get_cooc_filenames(document_path):
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

    if args.data_type == 'csv':
        df = pd.read_csv(filepath)
    else:
        df = pd.read_csv(filepath, sep='\t', )

    print("Creation Finished.. Starts new job")
    print(" ")

    print("Make a graph")

    cooc_path_list = get_cooc_filenames(coocpath)
    feature_model = Feature(doc_path_list=cooc_path_list, dataframe=df)

    print("Make all features and load all to dataframe ")
    df = feature_model.make_df_from_dataset()

    df.to_csv(savepath + 'result.csv')
    print("Completed")


if __name__ == '__main__':
    args = define_argparser()
    main(args)


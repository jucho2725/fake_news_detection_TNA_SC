import os
import pandas as pd

def get_document_filenames(document_path='D:\\PythonProjects\\text_network_analysis\\data\\190715\\true'):

    """
    파일 이름 받기
    """
    return [os.path.join(document_path, each)
            for each in os.listdir(document_path)]


x = get_document_filenames(document_path='./test/')

doc_f_list = get_document_filenames(x[0])
doc_t_list = get_document_filenames(x[1])
doc_path_list = doc_f_list + doc_t_list

print(x)
print(x[0])
# print(x[0][-20:-4])
# print(x[0][:-20])
print(x[0][:-4])

print(len(doc_path_list))


for i in x:
    # os.system('mkdir ' + x[0][:-20] + 'reweighted\\')
    path = i

path = 'test/fake/nnnnnnnnnnnnnnnn.txt'
print(path[:-25])

wd = 'reweight/'
print(os.path.isdir('/test/reweighted'))
if not os.path.isdir('/test/reweighted'):
    os.system('mkdir -p test/reweighted')
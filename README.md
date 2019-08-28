# Fake New Detection_Text Network Analysis and Social Context



This repo is implementation of "Fake News Detection Framework Using Word Networks and Propagation Patterns of News in Social Media. Jongmo Kim, Hanmin Kim, Jinuk Cho, Mye Sohn. PROCEEDING OF BIG DATA APPLICATIONS AND SERVICES, Vol.7, No.1, 2019".  



## Getting Started

We used a dataset based on the repository BuzzFace [link](https://github.com/gsantia/BuzzFace). Most of the news stories were no longer found, so we used our own crawler to get the body of the news in the link that still existed.

### Prerequisites

* Python 3.6.7
* nltk  3.2.5

* sklearn 1.2.0

* pandas 0.24.2

* numpy 1.16.4
* networkx 2.3



### Download nltk files

in python code:

```
import nltk
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')
```



## Running the code

If you want to test the code with small dataset (size 4), you need do minor change 





```
python run.py
```




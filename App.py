print("Activating App...")
import os
import pandas as pd
import numpy as np
import string
import time
# from nltk import pos_tag
# from nltk.tokenize import word_tokenize
# from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
import matplotlib.pyplot as plt
# from wordcloud import WordCloud, STOPWORDS
# from PIL import Image
# import random
import glob
from tika import parser
from tkinter import filedialog
from tkinter import *

plt.style.use("fivethirtyeight")

""" Function Design """
def analyzer(s):
    s = s.lower()
    tokens = re.findall(r'[a-z]+', s)
    # Remove stop words
    punctuation = list(string.punctuation)+['..', '...']
    pronouns = ['i', 'he', 'she', 'it', 'him', 'they', 'we', 'us', 'them']
    others   = ["'d", "co", "ed", "put", "say", "get", "can", "become",\
                "los", "sta", "la", "use", "iii", "else", "dealer","problem"]
    stop = stopwords.words('english') + punctuation + pronouns + others
    tokens = [token for token in tokens if (token not in stop) and (len(token) > 4)]
    return tokens

def pdf_parser(filepath):
    raw = parser.from_file(filepath)['content']
    analyzed_text = analyzer(raw)
    return analyzed_text


def main():
    root = Tk()
    root.withdraw()

    # Data Source file path
    DATA_SOURCE = filedialog.askdirectory(parent=root,initialdir="/",title='Locate the Data Source')
    #Input file path
    INPUT_PATH = filedialog.askopenfilename(parent=root,initialdir="/",title='Input File')

    print("Start Processing Input Document...")
    start = time.time()
    
    tfidf_vect = TfidfVectorizer(max_df=1,
                                 min_df=0, 
                                 max_features=None,
                                 lowercase=False)

    #Input data
    raw = pdf_parser(INPUT_PATH)
    tfidf_vect.fit(raw)
    input_freq_dict = tfidf_vect.vocabulary_
    #Total frequency of input document
    total_input_freq = sum(v for v in input_freq_dict.values())
    file_name = INPUT_PATH.split('\\')[-1]
    print("Title of Input file:\n{}".format(file_name))
    print("Statistics of Input File...")
    print("Total numbers of terms extracted using Tf-idf: {}".format(len(input_freq_dict)))

    file_name_list = []
    common_voc = []
    score_list = []

    print("Start Processing Documents from Data Source...")
    print()

    for file in glob.glob(DATA_SOURCE + "/*.pdf"):
        file_name = file.split('\\')[-1]
        file_name_list.append(file_name)
        print()
        print("Title:  {}".format(file_name))

        raw = pdf_parser(file)
        tfidf_vect.fit(raw)

        cur = tfidf_vect.vocabulary_
        common_set = set(cur) & set(input_freq_dict)
        common_voc.append(list(common_set))

        print("Common Words:\n {}".format(common_set))
        N = len(common_set)
        #factor = N / len(input_freq_dict)
        print("Total number of common words: {}".format(N))

        # score = 1e6 * factor / (abs(sum(v for k, v in cur.items() if k in common_set) - \
        #         sum(v for k, v in input_freq_dict.items() if k in common_set)) / N) if N else 0
        score =  sum(v for k, v in input_freq_dict.items() if k in common_set) / total_input_freq

        print("Relevant Score: {}".format(score))
        score_list.append(score)
        # break

    result = pd.DataFrame({"Document Name": file_name_list, "Common Vocabularies": common_voc, "Relevance Score": score_list}, columns=["Document Name", \
                        "Common Vocabularies", "Relevance Score"]).sort_values("Relevance Score", ascending=False).set_index("Document Name")
    #print(result)
    result.to_excel('result.xlsx')

    print("Finished! ... {0:.2f}s".format(time.time() - start))
    
    print("Opining Result....")
    os.startfile('result.xlsx')

if __name__ == '__main__':
    main()
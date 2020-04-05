import pandas as pd
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import dataStructureTools
import sklearn.cluster
import sklearn.feature_extraction.text as txtvectorizer


# This file includes any function that deals with reading in the raw
#   data and processing it before any analysis or visualization occurs.

def pre_process(kGram, token_type='word', file_name = 'SubdataSample.csv'):

    data = pd.read_csv(file_name, parse_dates = True)
    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(analyzer=token_type,ngram_range=kGram)
    dataMatrix = vectorizer.fit_transform(data['content'])

    return data, dataMatrix
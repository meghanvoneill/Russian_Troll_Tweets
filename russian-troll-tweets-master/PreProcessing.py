import pandas as pd
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import dataStructureTools
import sklearn.cluster
import sklearn.feature_extraction.text as txtvectorizer
import re


# This file includes any function that deals with reading in the raw
#   data and processing it before any analysis or visualization occurs.

def pre_process(kGram, token_type='word', file_name = 'SubdataSample.csv'):

    data = pd.read_csv(file_name, parse_dates = True)
    # Remove URLs from the data first
    for t in range(data.shape[0]):
        data['content'].loc[t] = re.sub(r"http\S+", '', data['content'][t])
    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(analyzer=token_type,ngram_range=kGram)
    dataMatrix = vectorizer.fit_transform(data['content'])
    features = (vectorizer.get_feature_names())
    print("\n\nFeatures : \n", features)

    return data, dataMatrix, features
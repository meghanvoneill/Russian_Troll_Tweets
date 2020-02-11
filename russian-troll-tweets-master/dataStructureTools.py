'''
This takes in the original set of csv files and merges them into 
a single csv file as well as subsets based on location and language. 
'''

import pandas as pd
import os.path

def mergeData():
    '''
    Reads the csv files into a single data frame and saves it as a 
    csv.
    '''
    dataSets = []
    dataSets.append(pd.read_csv('IRAhandle_tweets_1.csv'))
    dataSets.append(pd.read_csv('IRAhandle_tweets_2.csv'))
    dataSets.append(pd.read_csv('IRAhandle_tweets_3.csv'))
    dataSets.append(pd.read_csv('IRAhandle_tweets_4.csv'))
    dataSets.append(pd.read_csv('IRAhandle_tweets_5.csv'))
    dataSets.append(pd.read_csv('IRAhandle_tweets_6.csv'))
    dataSets.append(pd.read_csv('IRAhandle_tweets_7.csv'))
    dataSets.append(pd.read_csv('IRAhandle_tweets_8.csv'))
    dataSets.append(pd.read_csv('IRAhandle_tweets_9.csv'))
    dataSets.append(pd.read_csv('IRAhandle_tweets_10.csv'))
    dataSets.append(pd.read_csv('IRAhandle_tweets_11.csv'))
    dataSets.append(pd.read_csv('IRAhandle_tweets_12.csv'))
    mergedData = pd.concat(dataSets)
    mergedData.to_csv('IRAhandle_tweets_all.csv')
    return

def splitByLanguage():
    '''
    Splits the data into csv's based on language.
    '''
    if not os.path.exists('IRAhandle_tweets_all.csv'):
        mergeData()
    data = pd.read_csv('IRAhandle_tweets_all.csv')

    for lang in data.language.unique():
        # Gets the subset of the data that have the language 
        subSet = data.where(data.language == lang).dropna(how = 'all')
        # Saves the data
        subSet.to_csv('IRAhandle_tweets_' + lang + '.csv')
    return

def splitByType():
    '''
    Splits the data into csv's based on account type
    '''
    if not os.path.exists('IRAhandle_tweets_all.csv'):
        mergeData()
    data = pd.read_csv('IRAhandle_tweets_all.csv')

    for accType in data.account_type.unique():
        # Gets the subset of the data that have the language 
        subSet = data.where(data.account_type == accType).dropna(how = 'all')
        # Saves the data
        subSet.to_csv('IRAhandle_tweets_' + accType + '.csv')
    return

def splitByRegion():
    '''
    Splits the data into csv's based on Region
    '''
    if not os.path.exists('IRAhandle_tweets_all.csv'):
        mergeData()
    data = pd.read_csv('IRAhandle_tweets_all.csv')

    for reg in data.region.unique():
        # Gets the subset of the data that have the language 
        subSet = data.where(data.region == reg).dropna(how = 'all')
        # Saves the data
        subSet.to_csv('IRAhandle_tweets_' + reg + '.csv')
    return
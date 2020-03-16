#%%
%load_ext autoreload
%autoreload 2
import pandas as pd
import dataStructureTools
import matplotlib.pyplot as plt
import sklearn.cluster
import cursory_data_manipulation as cdm
# %%
dataStructureTools.mergeData()

# %%
allData = pd.read_csv('IRAhandle_tweets_all.csv')
#%%

allData['publish_date'] = allData['publish_date'].astype('datetime64[ns]') 
#%%
(allData[allData['publish_date'].dt.year ==2012])['publish_date'].hist(bins = 24) 
plt.title('Number of tweets for the year 2012') 
#%%
(allData[allData['publish_date'].dt.year ==2013])['publish_date'].hist(bins = 24) 
plt.title('Number of tweets for the year 2013') 
#%%
(allData[allData['publish_date'].dt.year ==2014])['publish_date'].hist(bins = 24) 
plt.title('Number of tweets for the year 2014') 
#%%
(allData[allData['publish_date'].dt.year ==2015])['publish_date'].hist(bins = 24) 
plt.title('Number of tweets for the year 2015') 
#%%
(allData[allData['publish_date'].dt.year ==2016])['publish_date'].hist(bins = 24) 
plt.title('Number of tweets for the year 2016') 
#%%
(allData[allData['publish_date'].dt.year ==2017])['publish_date'].hist(bins = 24) 
plt.title('Number of tweets for the year 2017') 
#%%
(allData[allData['publish_date'].dt.year ==2018])['publish_date'].hist(bins = 24) 
plt.title('Number of tweets for the year 2018') 
#%%
allData['publish_date'].hist(bins = 48) 
plt.title('Tweets from 2012 to 2018') 
# %%
subData = allData.sample(frac=0.01)
#%%
cdm.QuickClusterParamaterFinder(subData)
# %%

# %%

#%%
%load_ext autoreload
%autoreload 2
import pandas as pd
import dataStructureTools
import matplotlib.pyplot as plt
import sklearn.cluster
import cursory_data_manipulation as cdm
import ParamaterSelectionExperiments as PSE
# %%
dataStructureTools.mergeData()

# %%
allData = pd.read_csv('IRAhandle_tweets_all.csv')
#%%

allData['publish_date'] = allData['publish_date'].astype('datetime64[ns]') 

# %%
subData = allData.sample(frac=0.01)
#%%
PSE.KGramClusteringExperiement(subData)
# %%

#%%
#%load_ext autoreload
#%autoreload 2
import pandas as pd
import dataStructureTools
import matplotlib.pyplot as plt
import sklearn.cluster
import cursory_data_manipulation as cdm
import ParamaterSelectionExperiments as PSE

# %%
dataStructureTools.mergeData()

# %%
allData = pd.read_csv('IRAhandle_tweets_all.csv',parse_dates=True)
#%%
subData = allData.sample(frac=0.001)
#%%
PSE.KGramClusteringExperiment(subData)
# %%

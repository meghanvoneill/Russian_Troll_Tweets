#%%
import pandas as pd
import dataStructureTools
import matplotlib.pyplot as plt
import sklearn.cluster
# %%
dataStructureTools.mergeData()

# %%
allData = pd.read_csv('IRAhandle_tweets_all.csv')
allData['publish_date'] = allData['publish_date'].astype('datetime64[ns]') 
allData['publish_date'].hist()
#%%
subData = allData.sample(frac= 0.001)
subData['publish_date'] = subData['publish_date'].astype('datetime64[ns]') 
# %%
sklearn.cluster.k_means()
# %%

# %%

# %%

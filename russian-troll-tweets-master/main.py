#%%
import pandas as pd
import dataStructureTools
import matplotlib.pyplot as plt
# %%
dataStructureTools.mergeData()

# %%
allData = pd.read_csv('IRAhandle_tweets_all.csv')

# %%
langData = dataStructureTools.splitDFByLanguage()

# %%
langSum = dict()
for lang in langData.keys():
    langSum[lang] = langData[lang].describe(include = 'all')

# %%
for lang in langSum.keys():
    langSum[lang].to_excel(lang + ".xlsx")

# %%

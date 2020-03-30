
import pandas as pd
import dataStructureTools
import matplotlib.pyplot as plt
import sklearn.cluster
import cursory_data_manipulation as cdm
import ParamaterSelectionExperiments as PSE

subData = pd.read_csv('SubdataSample.csv', parse_dates = True)
(PSE.KGramClusteringExperiment(subData))
import pandas as pd
import dataStructureTools
import matplotlib.pyplot as plt
import sklearn.cluster
import cursory_data_manipulation as cdm
import ParamaterSelectionExperiments as PSE
import PreProcessing
import Mining
import PostAnalysis

def simple_clustering_pipeline():
    data, dataMatrix = PreProcessing.pre_process((2,3)) 
    labledData = Mining.SimpleKGram(data,dataMatrix,20)
    labledData.to_csv('simpleClusteringK20.csv')


def visualize_clustering():
    data, dataMatrix = PreProcessing.pre_process((2,3)) 
    labledData = Mining.SimpleKGram(data,dataMatrix,20)
     


if __name__ == '__main__':
    simple_clustering_pipeline()
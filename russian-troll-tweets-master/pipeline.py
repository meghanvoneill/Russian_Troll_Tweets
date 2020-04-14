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
    Mining.SimpleKGram(data,dataMatrix,20)
    data.to_csv('simpleClusteringK20.csv')


def visualize_clustering():
    data, dataMatrix = PreProcessing.pre_process((2,3)) 
    labledData = Mining.SimpleKGram(data,dataMatrix,20)


def clusters_in_two_dim():
    data, dataMatrix = PreProcessing.pre_process_content_only((1,1)) 
    Mining.SimpleKGram(data,dataMatrix,20)
    data = Mining.project_to_two_dimensions(data,dataMatrix)
    PostAnalysis.plot_2D(data,'2dPlotSimpleClusteringk20.png')
    data.to_csv('simpleClusteringK20WithCords.csv')


if __name__ == '__main__':
    clusters_in_two_dim()
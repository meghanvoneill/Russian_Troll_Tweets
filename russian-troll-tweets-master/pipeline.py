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
    data, dataMatrix, features = PreProcessing.pre_process((2,3))
    labledData = Mining.SimpleKGram(data,dataMatrix,20)
    labledData.to_csv('simpleClusteringK20.csv')


def visualize_clustering(clusters):
    data, dataMatrix, features = PreProcessing.pre_process((2,3))
    labledData = Mining.SimpleKGram(data,dataMatrix,clusters)

    # Getting top ranking features


    # get the first vector out (for the first document)
    first_vector_tfidfvectorizer = dataMatrix[0]

    # place tf-idf values in a pandas data frame
    df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=vectorizer.get_feature_names(),
                      columns=["tfidf"])
    df.sort_values(by=["tfidf"], ascending=False)
    print(df)


def clusters_in_two_dim():
    data, dataMatrix, features = PreProcessing.pre_process((2,3))
    labledData = Mining.SimpleKGram(data,dataMatrix,20)
    labledData = Mining.project_to_two_dimensions(labledData,dataMatrix)
    PostAnalysis.plot_2D(labledData,'2dPlotSimpleClusteringk20.png')
    labledData.to_csv('simpleClusteringK20WithCords.csv')


if __name__ == '__main__':
    clusters_in_two_dim()

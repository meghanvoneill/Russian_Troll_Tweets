import pandas as pd
import dataStructureTools
import matplotlib.pyplot as plt
import sklearn.cluster
import cursory_data_manipulation as cdm
import ParamaterSelectionExperiments as PSE
import PreProcessing
import Mining
import PostAnalysis
import dataStructureTools
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
    data, dataMatrix = PreProcessing.pre_process_content_only((2,2),file_name = 'IRAhandle_tweets_all.csv') 
    print('PreProcessing Done')
    #Mining.SimpleKGram(data,dataMatrix,4)
    data = Mining.project_to_two_dimensions(data,dataMatrix)
    print('Mining Done')
    PostAnalysis.plot_2D(data,'2dPlotSimpleClusteringCR22.png')
    data.to_csv('simpleClusteringK20WithCords.csv')

def clusters_in_two_dim_no_url():
    data, dataMatrix, features= PreProcessing.pre_process_content_only_no_url((2,2),file_name = 'IRAhandle_tweets_all.csv') 
    print('PreProcessing Done')
    #Mining.SimpleKGram(data,dataMatrix,4)
    data = Mining.project_to_two_dimensions(data,dataMatrix)
    print('Mining Done')
    PostAnalysis.plot_2D(data,'2dPlotSimpleClusteringCR22.png')
    data.to_csv('simpleClusteringK20WithCords.csv')

# In hide sight this should proably be in datastructure tools.
def save_striped_URL():
    data, dataMatrix, features = PreProcessing.pre_process_content_only_no_url((2,2),file_name='IRAhandle_tweets_all.csv')
    data.to_csv('IRAhandle_tweets_all_no_url.csv') 

if __name__ == '__main__':
    save_striped_URL()

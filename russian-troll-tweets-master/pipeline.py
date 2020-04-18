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
    kgram = (1,5)
    data, dataMatrix = PreProcessing.pre_process_content_only(kgram,token_type='char') 
    print('PreProcessing Done')
    data = Mining.SimpleKGram_minibatch(data,dataMatrix,15)
    data = Mining.project_to_two_dimensions(data,dataMatrix)
    print('Mining Done')
    data.to_csv('charGramTest.csv')
    PostAnalysis.plot_2D(data,'charGramPlot' + str(kgram[0])+ str(kgram[1])+ '.png')
    PostAnalysis.plot_hist_clusters(data,'charGramHist.png')


def clusters_in_two_dim_no_url():
    kgram = (2,2)
    data, dataMatrix, features= PreProcessing.pre_process_content_only_no_url(kgram) 
    print('PreProcessing Done')
    reducedDataMatrix = Mining.ReduceDim(dataMatrix)
    data = Mining.SimpleKGram(data,reducedDataMatrix,4)
    data = Mining.project_to_two_dimensions(data,dataMatrix)
    print('Mining Done')
    PostAnalysis.plot_2D(data,'2dPlotSimpleClusteringNoUrlK' + str(kgram[0])+ str(kgram[1])+ '.png')
    data.to_csv('simpleClusteringK20WithCords.csv')


# In hide sight this should proably be in datastructure tools.
def save_striped_URL():
    PreProcessing.strip_urls_and_save('IRAhandle_tweets_all_stripped.csv',file_name= 'IRAhandle_tweets_all.csv')

def hist_cluster_from_file():
    clustered_data = pd.read_csv('MiniBatchDimRedAllData15k23.csv')
    all_data = pd.read_csv('SubdataSample.csv',usecols=['publish_date'])
    data = pd.concat([all_data,clustered_data['Cluster']], axis=1)
    data['publish_date'] = pd.to_datetime(data['publish_date'],infer_datetime_format=True)
    data.dropna(how = 'any', subset = ['publish_date','Cluster'],inplace = True)
    PostAnalysis.plot_hist_clusters(data,'allData15k23Hist.png')


if __name__ == '__main__':
    hist_cluster_from_file()

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
from nltk.tokenize import word_tokenize


def main():
    clusters = 20

    #dataStructureTools.mergeData()
    #dataStructureTools.save_CSV_remove_URLs('IRAhandle_tweets_all.csv', 'IRAhandle_tweets_all_no_url.csv')
    #clusters_in_two_dim_no_url()
    visualize_clustering(clusters)


def simple_clustering_pipeline():

    data, dataMatrix, features = PreProcessing.pre_process((2,3))
    labledData = Mining.SimpleKGram(data,dataMatrix,20)
    labledData.to_csv('simpleClusteringK20.csv')



def visualize_clustering(clusters):
    # data, dataMatrix, features = PreProcessing.pre_process_content_only((2,2), file_name='IRAhandle_tweets_all_no_url.csv')
    data, dataMatrix, features = PreProcessing.pre_process_content_only((2,2), file_name='IRAhandle_tweets_1.csv')
    Mining.SimpleKGram(data, dataMatrix, clusters)


    # get the first vector out (for the first document)
    # first_vector_tfidfvectorizer = dataMatrix[0]

    # place tf-idf values in a pandas data frame
    # df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), features, columns=["tfidf"])
    # df.sort_values(by=["tfidf"], ascending=False)
    # print(df)

    print(dataMatrix)
    #print(dataMatrix[0].split(',')[1])
    #weights = [(dataMatrix[0].split(',')[1], dataMatrix[1]) for pair in dataMatrix]
    all_text = ''
    for t in data['content']:
        if data['0'] == 4:
            all_text += ' '.join(t)
            print(t)
    print('all_text created')
    PostAnalysis.make_wordcloud(all_text, 'IRAhandle_tweets_1.csv - Cluster 4')




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
    main()

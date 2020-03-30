import cursory_data_manipulation as cdm
import sklearn
import sklearn.cluster
import math
import csv

# Constants
TOKEN_TYPE = 'word' # or 'char'
NGRAM_RANGE = (2,10)
KGRAM_RANGE= range(2,10)
DFDX_THRESHOLD = 0.001


# This determines the best cluster over a number of 
#   different k-gram paramaters for characters and words. 
# Saves the results as a csv file.
def KGramClusteringExperiment(data):

    clustersForKgram = {}
    clustersForKgramBrut = {}

    for kGram in KGRAM_RANGE:
        # Converts all characters to lowercase before tokenizing, enables inverse-document-frequency reweighting, 
        #   smoothes the idf weights by adding one to document frequencies, and uses the L2 norm.
        vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(analyzer=TOKEN_TYPE,ngram_range=kGram)
        dataMatrix = vectorizer.fit_transform(data['content'])
        clustersForKgram[kGram] = FindNumberOfClusters(dataMatrix) 
        clustersForKgramBrut[kGram] = bruteForceNumOfClusterCheck(dataMatrix)

    # with open('KGramClustResults.csv','wb') as f: 
    #     w = csv.DictWriter(f,clustersForKgram.keys())
    #     w.writeheader()
    #     w.writerow(clustersForKgram)

    print(clustersForKgram)
    print(clustersForKgramBrut)

# With the elbow technique we make the assumptions that there
#   is a point in the intertia function when we plot it w/r to 
#   the number of clusters in which the slope levels out. 
# We can assume that the "elbow" lies where the slope of the curve
#   is approximently -1. With this assumption in mind we can perform a binary 
#   search of sorts and estimate the derivative with 
#   f'(x) = f(x+1) - f(x-1) / 2. 
def FindNumberOfClusters(dataMatrix):

    numbClusters = 4 
    while True:
        fPrimeX1 = dfdxApprox(numbClusters,dataMatrix)

        if abs(fPrimeX1 + 1) < DFDX_THRESHOLD:  
            return numbClusters
        elif fPrimeX1 > -1:
                return binSearchNumOfClusters(dataMatrix,int(numbClusters/2),numbClusters)
        else:
            numbClusters = numbClusters * 2

# The recursive part of the binary search. Used in 
#   conjunction with FindNumberOfClusters.
def binSearchNumOfClusters(dataMatrix,a,b):

    if b - a < 2:
        return b
    else:
        numbClusters = int(b-a / 2)
        fPrimeX1 = dfdxApprox(numbClusters,dataMatrix)

        if abs(fPrimeX1 + 1) < 1 :  
            return numbClusters 
        elif fPrimeX1 > -1:
            return binSearchNumOfClusters(dataMatrix,a,a + (b+a)/2)
        else:
            return binSearchNumOfClusters(dataMatrix,a + (b+a)/2, b)

# As a sanity check this runs on each individual cluster value 
#   until it finds one with a slope greater than -1.
def bruteForceNumOfClusterCheck(dataMatrix):
    
    numbClusters = 2 
    while True: 
        fPrimeX1 = dfdxApprox(numbClusters,dataMatrix)

        if fPrimeX1 > -1:
            return numbClusters
        numbClusters += 1


# Approximates the derivate for the number of clusters n with:
#   f'(n) = f(n+1)-f(n-1)/2
def dfdxApprox(numbClusters, dataMatrix):

        X0 = sklearn.cluster.KMeans(n_clusters = numbClusters - 1) 
        X2 = sklearn.cluster.KMeans(n_clusters = numbClusters + 1) 
        X0.fit_transform(dataMatrix)
        X2.fit_transform(dataMatrix)
        fPrimeX1 = (X2.inertia_ - X0.inertia_ ) / 2
        return fPrimeX1

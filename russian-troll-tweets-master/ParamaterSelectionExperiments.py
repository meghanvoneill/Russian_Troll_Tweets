import cursory_data_manipulation as cdm

import sklearn.cluster
import math

# This determins the best cluster over a number of 
# different kgram paramaters for char and words. 
# Saves the results as a csv file.
def KGramClusteringExperiement(data):
    clustersForKgram = dict() 
    clustersForKgramBrut = dict()
    for c in range(2,10):
       vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(analyzer='word',ngram_range=(2,c))
       dataMatrix = vectorizer.fit_transform(data['content'])
       clustersForKgram[c] = FindNumberOfClusters(dataMatrix) 
       clustersForKgramBrut[c] = bruteForceNumOfClusterCheck(dataMatrix)
    print(clustersForKgram)
    print(clustersForKgramBrut)

# With the elbow tecnique we make the assumptiont that there
# is a point in the interia fuction when we plot it w/r to 
# the number of clusters in which the slope levels out. 
# We can assume that the "elbow" lies where the slope of the curve
# is approxemently -1. With this assumption in mind we can perform a binary 
# search of sorts and estimate the derivative with f'(x) = f(x+1) - f(x-1) / 2. 
def FindNumberOfClusters(dataMatrix):
    numOfClusters = 4 
    while True:
        X0 = sklearn.cluster.MiniBatchKMeans(n_clusters = numOfClusters - 1) 
        X2 = sklearn.cluster.MiniBatchKMeans(n_clusters = numOfClusters + 1) 
        X0.fit_transform(dataMatrix)
        X2.fit_transform(dataMatrix)
        fPrimeX1 = (X2.inertia_ - X0.inertia_) / 2
        if abs(fPrimeX1 + 1) < 0.1 :  
            return numOfClusters
        elif fPrimeX1 > -1:
                return binSearchNumOfClusters(dataMatrix,int(numOfClusters/2),numOfClusters)
        else:
            numOfClusters = numOfClusters * 2

# The recursive part of te binary search. used in the 
# FindNumberOfClusters
def binSearchNumOfClusters(dataMatrix,a,b):
    if b - a < 2:
        return b
    else:
        numOfClusters = int(b-a / 2)
        X0 = sklearn.cluster.MiniBatchKMeans(n_clusters = numOfClusters - 1) 
        X2 = sklearn.cluster.MiniBatchKMeans(n_clusters = numOfClusters + 1) 
        X0.fit_transform(dataMatrix)
        X2.fit_transform(dataMatrix)
        fPrimeX1 = (X2.inertia_ - X0.inertia_ ) / 2
        if abs(fPrimeX1 + 1) < 1 :  
            return numOfClusters
        elif fPrimeX1 > -1:
            return binSearchNumOfClusters(dataMatrix,a,a + (b+a)/2)
        else:
            return binSearchNumOfClusters(dataMatrix,a + (b+a)/2, b)

# As a sanity check this runs on each individual cluster value 
# until it finds one with a slope greater than -1.
def bruteForceNumOfClusterCheck(dataMatrix):
    numbClusters = 2 
    while True: 
        X0 = sklearn.cluster.MiniBatchKMeans(n_clusters = numbClusters - 1) 
        X2 = sklearn.cluster.MiniBatchKMeans(n_clusters = numbClusters + 1) 
        X0.fit_transform(dataMatrix)
        X2.fit_transform(dataMatrix)
        fPrimeX1 = (X2.inertia_ - X0.inertia_ ) / 2
        if fPrimeX1 > -1:
            return numbClusters
        numbClusters += 1



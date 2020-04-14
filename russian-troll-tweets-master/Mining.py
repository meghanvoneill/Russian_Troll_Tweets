import pandas as pd
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import dataStructureTools
import sklearn.cluster
import sklearn.decomposition as DC
import sklearn.feature_extraction.text as txtvectorizer
from scipy import linalg as LA


# Constants
TOKEN_TYPE = 'word' # or 'char'
KGRAM_RANGE= range(2,10)
DFDX_THRESHOLD = 0.1

def project_to_two_dimensions(data, data_matrix):
    PCA = DC.TruncatedSVD()
    reducedData = PCA.fit_transform(data_matrix) 
    labeledData = pd.concat([data, pd.DataFrame(reducedData[:,0],columns = ['xcord'])], axis=1)
    labeledData = pd.concat([labeledData, pd.DataFrame(reducedData[:,1],columns = ['ycord'])], axis=1)
    return labeledData


# Takes in a matrix A and an integer k_max
def find_smallest_k_10(A, percent = 0.10):
    U, s, Vt = LA.svd(A, full_matrices=False)
    k_max = len(s)
    S = np.diag(s)
    A_norm = LA.norm(A, 2)
    smallest_k = math.inf

    for k in range(k_max):
        Uk = U[:, :k]
        Sk = S[:k, :k]
        Vtk = Vt[:k, :]
        Ak = Uk @ Sk @ Vtk

        A_minus_Ak_norm = LA.norm(A-Ak, 2)

        if A_minus_Ak_norm < percent * A_norm:
            if k < smallest_k:
                smallest_k = k
                return smallest_k, Uk, Sk, Vtk, Ak
            
    Uk = U[:, :smallest_k]
    Sk = S[:smallest_k, :smallest_k]
    Vtk = Vt[:smallest_k, :]
    Ak = Uk @ Sk @ Vtk

    return smallest_k, Uk, Sk, Vtk, Ak


def SimpleKGram(data, data_matrix, number_clusters):
    KMean = sklearn.cluster.KMeans(n_clusters = number_clusters) 
    labels = KMean.fit_predict(data_matrix)
    labeledData = pd.concat([data, pd.DataFrame(labels)], axis=1)
    labeledData = labeledData.rename(columns = {'0': 'Cluster'})
    return labeledData


# This determines the best cluster over a number of 
#   different k-gram parameters for characters and words.
# Saves the results as a csv file.
def KGramClusteringExperiment(data,data_matrix):

    clustersForKgram = {}
    clustersForKgramBrut = {}

    for kGram in KGRAM_RANGE:
        # Converts all characters to lowercase before tokenizing, enables inverse-document-frequency reweighting, 
        #   smoothes the idf weights by adding one to document frequencies, and uses the L2 norm.
        clustersForKgram[kGram] = FindNumberOfClusters(data_matrix) 
        clustersForKgramBrut[kGram] = bruteForceNumOfClusterCheck(data_matrix)

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
#   is approximately -1. With this assumption in mind we can perform a binary 
#   search of sorts and estimate the derivative with 
#   f'(x) = f(x+1) - f(x-1) / 2. 
def FindNumberOfClusters(dataMatrix):

    numbClusters = 4 
    while True:
        fPrimeX1 = dfdxApprox(numbClusters,dataMatrix)

        if fPrimeX1 + 1 < DFDX_THRESHOLD:  
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

        if fPrimeX1 + 1 < DFDX_THRESHOLD :  
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
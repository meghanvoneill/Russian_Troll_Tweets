import pandas as pd
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import dataStructureTools
import sklearn.cluster
import sklearn.feature_extraction.text as txtvectorizer

_memomask = {}


def main():

    k = 2
    t = 30
    m = 100000
    all_k_grams = []

    #data = pd.read_csv('IRAhandle_tweets_all.csv')
    #data = pd.read_csv('IRAhandle_tweets_2.csv')

    for x in range(0, len(data)):
        tweet = data.loc[x]['content']
        k_grams_found = k_gram_strings(tweet, k)
        all_k_grams.append(k_grams_found)

    max_dist = 0

    for i in range(0, len(all_k_grams)):
        for j in range(0, len(all_k_grams)):
            if i == j:
                continue
            dist = min_hash_dist(all_k_grams[i], all_k_grams[j], m, t)

            if dist > max_dist:
                max_dist = dist
                print("input1: " + str(all_k_grams[i]))
                print("input2: " + str(all_k_grams[j]))
                print("dist: " + str(dist))
                print()

    return


# Takes two tweets and defines min-hash distance.
def min_hash_dist(input1, input2, m, t):

    input1_mins = min_hash(input1, m, t)
    input2_mins = min_hash(input2, m, t)

    # dist = 1 - jaccard_similarity(input1_mins, input2_mins)
    dist = 1 - cumulative_jaccard_similarity(input1_mins, input2_mins, t)

    return dist


def min_hash(x, m, t):

    min_found = math.inf
    x_mins = [0] * len(x)

    for i in range(0, len(x)):

        # Use the hash family of functions to get every hash for this k-gram, x[i].
        for hash_function_index in range(0, t):

            new_hash = hash_function(i, hash_function_index, m)

            # If the new hash is smaller than the min hash found, update the min hash and store it.
            if new_hash < min_found:
                min_found = new_hash
                x_mins[i] = new_hash

    return x_mins


# Given a data set dictionary and a test set size, partitions the data into a
# test set of the given size and a training set of the data that remains.
def build_training_set(data, size_of_test_set):

    test_set = {}
    training_set = {}
    t = 0

    # Copy the data set for the training set.
    for x in data.keys():
        training_set[x] = data[x]

    # Randomly select data for test set.
    while len(test_set.keys()) <= size_of_test_set:
        # Choose from the training set's keys so that the pool of potential options shrinks.
        random_x = random.choice(list(training_set.keys()))
        test_set[random_x] = data[random_x]
        del training_set[random_x]

    return training_set, test_set


def k_gram_strings(contents, k):

    k_grams = []

    contents_string_list = contents.split(" ")
    contents_string_list = list(filter(None, contents_string_list))

    for index in range(0, len(contents_string_list) - k + 1):
        new_k_gram = " ".join(contents_string_list[index:index + k])
        k_grams.append(new_k_gram)

    return set(k_grams)


# The Jaccard similarity between A and B is:
#   |(A n B)| / |(A u B)|
def jaccard_similarity(set_A, set_B):

    AnB = set(set_A).intersection(set(set_B))
    AuB = set(set_A).union(set(set_B))

    j_similarity = (len(AnB) / len(AuB))

    return j_similarity


def cumulative_jaccard_similarity(setA, setB, t):

    summation = 0

    for i in range(min(len(setA), len(setB))):
        # If a and b exist, check whether they have the same value.
        if i < len(setA) and i < len(setB):
            if setA[i] == setB[i]:
                summation += 1

    j_similarity = float(float(1 / t) * float(summation))
    return j_similarity


# Code on hash function family adapted from:
#   https://stackoverflow.com/questions/2255604/hash-functions-family-generator-in-python
# Credit to Alex Martelli.
def hash_function(x, n, m):

    mask = _memomask.get(n)

    if mask is None:
        random.seed(n)
        mask = _memomask[n] = random.getrandbits(32)

    val = (hash(x) ^ mask) % m
    return val


if __name__ == '__main__':
    main()

# Fits performs clustering with kmeans on vectorized data
# as we continue we can add additional argument or paramaters to this 
# when we get tot he fine tuneing stage.
def kmeansWrapper(vectorizedData,n_clusters):
    km = sklearn.cluster.KMeans(n_clusters = n_clusters)
    km.fit(vectorizedData)
    return km

# Takes in a pd.series object that represents the tweet contents for the main 
# data set and a tupple that represents the allowed size of the k-grams used.
#
# Returns a matrix where each column represents a unique k-gram and each row 
# represents a document. All N/A's are filled with the empty string.
# 
# This method is essentially a wrapper for the HashingVectorizer class from
# sklearn.
def vectorizeStrings(documents, ngramRange):
    vectorizer = txtvectorizer.HashingVectorizer(analyzer= 'char', ngram_range= ngramRange)
    vectors = vectorizer.transform(documents.fillna(""))
    return vectors

# Runs the minibatchkmeans algorythm up to 100 times and returns a list 
# that contains the "inertia" for each size of the cluster, where the batch with
# n clusters is at index n-1.
def QuickClusterParamaterFinder(data):
    Cost = list()
    vectorizer = txtvectorizer.HashingVectorizer(analyzer= 'char', ngram_range= (2,2))
    vectors = vectorizer.transform(data['content'].dropna())

    for c in range(1,100):
        kmeans = sklearn.cluster.MiniBatchKMeans(n_clusters=c)
        kmeans.fit(vectors)
        Cost.append(kmeans.inertia_)
        if c % 5 == 0:
            print(str(c) + " / 100")
    plt.plot(range(1,100),Cost)
    return Cost
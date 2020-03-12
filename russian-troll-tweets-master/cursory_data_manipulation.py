import pandas as pd
import math
import random
import matplotlib.pyplot as plt
import numpy as np

_memomask = {}


def main():

    return


# Takes two tweets and defines min-hash distance.



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
        random_x = random.choice(list(data.keys()))
        test_set[random_x] = data[random_x]
        del training_set[random_x]

    return training_set, test_set


def k_gram_strings(file_name, k):

    k_grams = []
    f = open(file_name, "r")

    if f.mode == 'r':
        contents = f.read()
    else:
        return

    contents_string_list = contents.split(" ")
    contents_string_list = list(filter(None, contents_string_list))

    for index in range(0, len(contents_string_list) - k + 1):
        new_k_gram = " ".join(contents_string_list[index:index + k])
        k_grams.append(new_k_gram)

    return set(k_grams)


# The Jaccard similarity between A and B is:
#   |(A n B)| / |(A u B)|
def jaccard_similarity(set_A, set_B):

    AnB = set_A.intersection(set_B)
    AuB = set_A.union(set_B)

    j_similarity = (len(AnB) / len(AuB))

    return j_similarity


def cumulative_jaccard_similarity(setA, setB, t):

    summation = 0

    for i in range(min(len(setA), len(setB))):
        # If a and b exist, check whether they have the same value.
        if i < len(setA) and i < len(setB):
            print("a: " + str(setA[i]) + " b: " + str(setB[i]))
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
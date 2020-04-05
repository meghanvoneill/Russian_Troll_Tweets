import cursory_data_manipulation as cdm
import sklearn
import sklearn.cluster
import math
import csv
import PreProcessing
import Mining

# Constants
TOKEN_TYPE = 'word' # or 'char'
KGRAM_RANGE= range(2,10)
DFDX_THRESHOLD = 0.1

def main():

    pre_processed_data, pre_processed_data_matrix = PreProcessing.pre_process(KGRAM_RANGE, TOKEN_TYPE)
    processed_data = Mining.KGramClusteringExperiment(pre_processed_data, pre_processed_data_matrix)

    return


if __name__ == '__main__':
    main()
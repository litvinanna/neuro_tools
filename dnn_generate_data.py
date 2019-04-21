from data_loading import *
import os
import pickle

genome_file = "../data/ecoli.genbank"
seq_np =  get_seq_np(genome_file)


def save_data():
    data_objects = []
    for i in range(30):
        data = get_data_sets(seq_np, length, n_train, n_test, genome_split = 0.1)
        data_objects.append(data)
    with open("../results/dnn/ecoli_{}_{}_{}.data".format(length, n_train, n_test), "wb") as file:
        pickle.dump(data_objects, file)
    

length = 100
n_train = 10000
n_test=1000
save_data()


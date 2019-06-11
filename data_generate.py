from data_loading import *
import os
#import pickle
import numpy as np


genome_file = "../data/ecoli.genbank"
path = "../results/"
how = "uniform" #coding noncoding uniform
n_train = 1000000
n_test = 100000

print(how, genome_file, path, "{:,} {:,}".format(n_train, n_test))
ans = input("proceeed?(y/N) ")
if ans != "y":
    print("Exit")
    exit()

genome_file_type = genome_file.split('.')[-1]
record = SeqIO.read(genome_file, genome_file_type)
seq_np = hot_encode_seq(record.seq)
seq_np_complement = hot_encode_seq(record.seq.complement())

genome_l = len(seq_np)
train_ids_list, validate_ids_list, test_ids_list  = [], [], []




for i in range(30):
    print(i, end = " ")

    train_ids, test_ids, validate_ids = get_ids(n_train, n_test, how, genome_file)
    
    train_ids_list.append(   train_ids   )
    validate_ids_list.append(validate_ids)
    test_ids_list.append(    test_ids    )


train_ids_list = np.array(train_ids_list)
test_ids_list = np.array(test_ids_list)
validate_ids_list = np.array(validate_ids_list)


name = os.path.join(path, "ecoli_{}_{}_{}".format(n_train, n_test, how))
if not os.path.exists(name):
    os.mkdir(name)
    
np.save(os.path.join(name, "seq_np.npy"),        seq_np)
np.save(os.path.join(name, "seq_np_complement.npy"),  seq_np_complement)
np.save(os.path.join(name, "train_ids_list.npy"),     train_ids_list)
np.save(os.path.join(name, "test_ids_list.npy"),      test_ids_list)
np.save(os.path.join(name, "validate_ids_list.npy"),  validate_ids_list)





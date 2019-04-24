from data_loading import *
import os
#import pickle
import numpy as np


genome_file = "../data/ecoli.genbank"
path = "../results"



def main(n_train , n_test, ):
    seq_np =  get_seq_np(genome_file)
    
    
    train_ids_list, validate_ids_list, test_ids_list  = [], [], []

    
    for i in range(30):
        print(i, end = " ")
        train_ids, test_ids, validate_ids = get_data_ids(seq_np, n_train , n_test)
        train_ids_list.append(   train_ids   )
        validate_ids_list.append(validate_ids)
        test_ids_list.append(    test_ids    )
        
        
    
    
    train_ids_list = np.array(train_ids_list)
    test_ids_list = np.array(test_ids_list)
    validate_ids_list = np.array(validate_ids_list)
    
   
    name = os.path.join(path, "ecoli_{}_{}".format(n_train, n_test))
    if not os.path.exists(name):
        os.mkdir(name)
    
    np.save(os.path.join(name, "seq_np.npy"),        seq_np)
    np.save(os.path.join(name, "train_ids_list.npy"),     train_ids_list)
    np.save(os.path.join(name, "test_ids_list.npy"),      test_ids_list)
    np.save(os.path.join(name, "validate_ids_list.npy"),  validate_ids_list)

    

main(n_train = 10000, 
     n_test  = 1000
    )


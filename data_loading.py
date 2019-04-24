import numpy as np
from Bio import SeqIO
import os
import matplotlib.pyplot as plt

def get_seq_np(genome_file):
    
    genome_file_type = genome_file.split('.')[-1]
    record = SeqIO.read(genome_file, genome_file_type)
    length = len(record.seq)
    
    bases_dict     = {"A": 0, "T": 1, "C": 2, "G": 3}
    bases_list = ["A", "T", "C", "G"]
    bases_np        = {
                        "A": np.array([1, 0, 0, 0], dtype = np.float32),
                        "T": np.array([0, 1, 0, 0], dtype = np.float32),
                        "C": np.array([0, 0, 1, 0], dtype = np.float32),
                        "G": np.array([0, 0, 0, 1], dtype = np.float32)
                        }

    seq_np = np.zeros((length, 4), dtype = np.float32) 
    for index in range(length):
        base = record.seq[index]
        if base in bases_list:
            channel = bases_dict[base]
            seq_np[index, channel] = 1
        else:
            print("alternative base")
    return seq_np


    

def get_data_ids(seq_np, n_train = 100000, n_test=10000):

    n_validate = n_train // 10
    N = n_train + n_test + n_validate

    genome_l = len(seq_np)
    ids = np.sort(np.random.choice(np.arange(start = 150, stop = genome_l - 150), N, replace=False))


    test_start = np.random.choice(N - n_test - n_validate - 1)
    test_end = test_start + n_test
    validate_start = test_end
    validate_end = validate_start + n_validate

    test_ids = ids[test_start:test_end]
    validate_ids = ids[validate_start:validate_end]
    train_ids = np.concatenate((ids[0:test_start], ids[validate_end:genome_l])) 
    
    assert len(np.intersect1d(test_ids,     train_ids))     == 0
    assert len(np.intersect1d(validate_ids, train_ids))     == 0
    assert len(np.intersect1d(test_ids,     validate_ids))  == 0

    
    return train_ids, test_ids, validate_ids
    
def generate_data_set(seq_np, train_ids, test_ids, validate_ids, enviroment_size = 6, shift = 0):   
    class Data(): pass
    
#    data = type('', (), {})()
    data = Data()
    
    a = enviroment_size
    data.train1    = np.array([seq_np[x-a:x, ...] for x in train_ids    ])
    data.test1     = np.array([seq_np[x-a:x, ...] for x in test_ids     ])
    data.validate1 = np.array([seq_np[x-a:x, ...] for x in validate_ids ])
    
    data.train2    = np.array([seq_np[x+1 : x+a+1, ...] for x in train_ids    ])
    data.test2     = np.array([seq_np[x+1 : x+a+1, ...] for x in test_ids     ])
    data.validate2 = np.array([seq_np[x+1 : x+a+1, ...] for x in validate_ids ])
    
    data.train_ans    = np.array([seq_np[x + shift, ...]  for x in train_ids    ])
    data.test_ans     = np.array([seq_np[x + shift, ...]  for x in test_ids     ])
    data.validate_ans = np.array([seq_np[x + shift, ...]  for x in validate_ids ])
    
    return data





def generate_data(datapath, enviroment_size = 6, shift = 0):
    
    name = datapath
    
    seq_np = np.load(os.path.join(name, "seq_np.npy"))
    train_ids_list = np.load(os.path.join(name, "train_ids_list.npy"))
    test_ids_list = np.load(os.path.join(name, "test_ids_list.npy"))
    validate_ids_list= np.load(os.path.join(name, "validate_ids_list.npy"))
    
    data_list = []
    
    for i in range(train_ids_list.shape[0]):
        print(i, end = " ")
        train_ids     = train_ids_list[i]
        test_ids      = test_ids_list[i]
        validate_ids  = validate_ids_list[i]
        
        data = generate_data_set(seq_np, train_ids, test_ids, validate_ids, enviroment_size, shift)
        data_list.append(data)
                           
    return data_list
   
import matplotlib.pyplot as plt

def plot_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))

    a = ax[0]
    a.plot(history['acc'])
    a.plot(history['val_acc'])
    a.set_title('Model accuracy')
    a.set_ylabel('Accuracy')
    a.set_xlabel('Epoch')
    a.legend(['Train', 'Validation'], loc='upper left')

    b = ax[1]
    b.plot(history['loss'])
    b.plot(history['val_loss'])
    b.set_title('Model loss')
    b.set_ylabel('Loss')
    b.set_xlabel('Epoch')
    b.legend(['Train', 'Validation'], loc='upper left')

    fig.show()

    
import pickle
import glob
import scipy.stats
import matplotlib.pyplot as plt


def plot_hist(path):
    statistics_file = glob.glob(os.path.join(path, "*all_runs*"))[0]
    print("file {}".format(statistics_file))
    with open(statistics_file, "rb") as file:
        (test_accs, train_accs) = pickle.load(file)  
    print("Number of runs {}".format(len(test_accs)))
    print(scipy.stats.mannwhitneyu(test_accs, train_accs))

    
    plt.boxplot([test_accs, train_accs])
    plt.xticks([1,2], ('test', 'train'))
    plt.ylabel("accuracy")
    plt.show()

    return test_accs, train_accs


    
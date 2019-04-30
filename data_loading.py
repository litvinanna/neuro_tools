import numpy as np
from Bio import SeqIO
import os
import matplotlib.pyplot as plt

def get_seq_np(genome_file):
    
    genome_file_type = genome_file.split('.')[-1]
    record = SeqIO.read(genome_file, genome_file_type)
    length = len(record.seq)
    seq_np = hot_encode_seq(record.seq)
            
    return seq_np

def hot_encode_seq(seq):
    length = len(seq)
    seq_np = np.zeros((length, 4), dtype = np.float64)
    
    bases_dict     = {"A": 0, "T": 1, "C": 2, "G": 3}
    bases_list = ["A", "T", "C", "G"]
    bases_np        = {
                        "A": np.array([1, 0, 0, 0], dtype = np.float64),
                        "T": np.array([0, 1, 0, 0], dtype = np.float64),
                        "C": np.array([0, 0, 1, 0], dtype = np.float64),
                        "G": np.array([0, 0, 0, 1], dtype = np.float64)
                        }
    for index in range(length):
        base = seq[index]
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






def generate_data(datapath, enviroment_size = 6, shift = 0, t = 30):
    
    name = datapath
    
    seq_np = np.load(os.path.join(name, "seq_np.npy"))
    train_ids_list = np.load(os.path.join(name, "train_ids_list.npy"))
    test_ids_list = np.load(os.path.join(name, "test_ids_list.npy"))
    validate_ids_list= np.load(os.path.join(name, "validate_ids_list.npy"))
    
    data_list = []
    
    for i in range(min(train_ids_list.shape[0], t)):
        print(i, end = " ")
        train_ids     = train_ids_list[i]
        test_ids      = test_ids_list[i]
        validate_ids  = validate_ids_list[i]
        
        data = generate_data_set(seq_np, train_ids, test_ids, validate_ids, enviroment_size, shift)
        data_list.append(data)
                           
    return data_list
   
#---------------------------------------------------------
from Bio.Seq import reverse_complement
from Bio import SeqIO
import numpy as np
import os
from data_loading import *

def generate_data_cds_1(datapath, enviroment_size = 6, shift = 0, genome_file = "../data/ecoli.genbank", t = 1):
    
    genome_file_type = genome_file.split('.')[-1]
    record = SeqIO.read(genome_file, genome_file_type) 
    
    seq_cds  = np.zeros(len(record.seq))  
    for f in record.features:
        if f.type == 'CDS':
            if f.location.strand == 1:
                seq_cds[f.location.start + 24 : f.location.end-24] = 1
            else:
                seq_cds[f.location.start + 24 : f.location.end-24] = 2
                
                
    name = datapath    
#     seq_np            = np.load(os.path.join(name, "seq_np.npy"))
    train_ids_list    = np.load(os.path.join(name, "train_ids_list.npy"))
    test_ids_list     = np.load(os.path.join(name, "test_ids_list.npy"))
    validate_ids_list = np.load(os.path.join(name, "validate_ids_list.npy"))
    
    data_list = []
    times = min(train_ids_list.shape[0], t)
    
    for i in range(times):
        print(i, end = " ")
        train_ids     = train_ids_list[i]
        test_ids      = test_ids_list[i]
        validate_ids  = validate_ids_list[i]
        
        data_cds = generate_data_cds_2(record, seq_cds, train_ids, test_ids, validate_ids, enviroment_size, shift)
        data_list.append(data_cds)
                           
    return data_list


def generate_ids_cds(ids, seq_cds):
    
    ids_cds_plus = []
    ids_cds_minus = []
    ids_non = []
    
    for i in ids:
        if  seq_cds[i] == 1:
            ids_cds_plus.append(i)
        elif seq_cds[i] == 2:
            ids_cds_minus.append(i)
        elif seq_cds[i] == 0:
            ids_non.append(i)
     
    return ids_cds_plus, ids_cds_minus, ids_non


def generate_data_cds_3(record, seq_cds, ids, enviroment_size = 6, shift = 0):
    
    ids_cds_plus, ids_cds_minus, ids_non =  generate_ids_cds(ids, seq_cds)
    
    seq_np          = hot_encode_seq(record.seq)      
    seq_np_reversed = hot_encode_seq(reverse_complement(record.seq))
    
    a = enviroment_size    
    def left(seq_np, ids):
        return np.array([seq_np[x-a:x,       ...] for x in ids])
    
    def right(seq_np, ids):
        return np.array([seq_np[x+1 : x+a+1, ...] for x in ids])
    
    def ans(seq_np, ids):
        return np.array([seq_np[x + shift,   ...] for x in ids])
    

    print(len(ids_cds_plus) + len(ids_cds_minus))
    set1_cds    = np.concatenate(( left(seq_np, ids_cds_plus),  left(seq_np_reversed, ids_cds_minus)))
    set2_cds    = np.concatenate((right(seq_np, ids_cds_plus), right(seq_np_reversed, ids_cds_minus)))
    set_ans_cds = np.concatenate((  ans(seq_np, ids_cds_plus),   ans(seq_np_reversed, ids_cds_minus)))
    
#     print("non start")
#     set1_non    =  left(seq_np, ids_non)
#     set2_non    = right(seq_np, ids_non)
#     set_ans_non =   ans(seq_np, ids_non)
 
    
    return set1_cds, set2_cds, set_ans_cds  #set1_non, set2_non, set_ans_non 


def generate_data_cds_2(record, seq_cds, train_ids, test_ids, validate_ids, enviroment_size = 6, shift = 0):   
    class Data(): pass
    data_cds = Data()
    print("train")
    data_cds.train1, data_cds.train2, data_cds.train_ans          = generate_data_cds_3(record, seq_cds, train_ids, enviroment_size, shift)
    print("test")
    data_cds.test1, data_cds.test2, data_cds.test_ans             = generate_data_cds_3(record, seq_cds, test_ids, enviroment_size, shift)
    print("validate" )
    data_cds.validate1, data_cds.validate2, data_cds.validate_ans = generate_data_cds_3(record, seq_cds, validate_ids, enviroment_size, shift)
    
    return data_cds

#-----------------------------------------------------------------------------

 
    
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

def validation_accs(path):
    val_accs = []
    for i in range(100):
        f = os.path.join(path, "{}_history.pyob".format(i))
        if not os.path.exists(f):
            break
        with open(f, "rb") as file:
            history = pickle.load(file)
            val_accs.append(history['val_acc'][-1])
            
    return val_accs

def test_accs(path):
    if path.endswith('.pyob'):
        path, f = os.path.split(path)
    statistics_file = glob.glob(os.path.join(path, "*all_runs*"))[0]
    print("file {}".format(statistics_file))
    with open(statistics_file, "rb") as file:
        (test_accs, train_accs) = pickle.load(file) 
    return test_accs

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Computer Modern Roman'], 'weight' : 200, 'size'   : 26})
# rc('font',**{'family':'serif','serif':['Times'], 'weight' : 700, 'size'   : 26})
rc('text', usetex=True)


def plot_hist(path, funcs = []):
    if path.endswith('.pyob'):
        path, f = os.path.split(path)
    statistics_file = glob.glob(os.path.join(path, "*all_runs*"))[0]
    print("file {}".format(statistics_file))
    with open(statistics_file, "rb") as file:
        (test_accs, train_accs) = pickle.load(file) 
        
    val_accs = validation_accs(path)
    print("Number of runs {}".format(len(test_accs)))
    
    print(scipy.stats.mannwhitneyu(val_accs, train_accs)) 
    
    for i in range(len(test_accs)):
        plt.plot([1, 2, 3], [train_accs[i], val_accs[i], test_accs[i]], alpha = 0.5)

    
    plt.boxplot([train_accs, val_accs, test_accs],
                medianprops=dict(color="black", linewidth = 2),
                boxprops=dict(linewidth = 2),
                capprops=dict(linewidth = 2),
                whiskerprops=dict(linewidth = 2),)
    
    
    plt.xticks([1,2, 3], ('train', 'validate', 'test'))
#    plt.ylabel("accuracy")
    for f in funcs:
        f
    plt.savefig("../results/pics/{}.png".format(statistics_file.split("/")[-1].split(".")[0]), dpi=300, bbox_inches='tight')
    plt.show()

    return train_accs, val_accs, test_accs


    
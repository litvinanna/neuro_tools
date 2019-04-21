import numpy as np
from Bio import SeqIO

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

class Data():
    def __init__(self):
        self.train1 = None

def get_data_sets(seq_np, length = 10, n_train = 100000, n_test=10000, genome_split = 0.1):
    genome_l = len(seq_np)
    
    train_ids = np.random.choice(int(genome_l * (1-genome_split)) - 300, n_train, replace=False) + 200
    test_ids = np.random.choice(int(genome_l * genome_split) - 300, n_test, replace=False) + int(genome_l * (1-genome_split))

#    data = type('', (), {})()
    data = Data()
    
    data.train1 = np.array([seq_np[x-length:x, ...] for x in train_ids])
    data.test1 = np.array([seq_np[x-length:x, ...] for x in test_ids])

    data.train2 = np.array([seq_np[x+1:x+length+1, ...] for x in train_ids])
    data.test = np.array([seq_np[x+1:x+length+1, ...] for x in test_ids])
        
    data.train_ans = np.array([seq_np[x, ...]  for x in train_ids])
    data.test_ans = np.array([seq_np[x, ...]  for x in test_ids])
    
    return data
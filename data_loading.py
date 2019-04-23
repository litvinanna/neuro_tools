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

def get_data_sets(seq_np, n_train = 100000, n_test=10000):

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

#    data = type('', (), {})()
    data = Data()
    
    data.train1    = np.array([seq_np[x-10:x, ...] for x in train_ids    ])
    data.test1     = np.array([seq_np[x-10:x, ...] for x in test_ids     ])
    data.validate1 = np.array([seq_np[x-10:x, ...] for x in validate_ids ])
    
    
    data.train2    = np.array([seq_np[x+1:x+11, ...] for x in train_ids    ])
    data.test2     = np.array([seq_np[x+1:x+11, ...] for x in test_ids     ])
    data.validate2 = np.array([seq_np[x+1:x+11, ...] for x in validate_ids ])
    
    data.train3    = np.array([seq_np[x-100:x, ...] for x in train_ids    ])
    data.test3     = np.array([seq_np[x-100:x, ...] for x in test_ids     ])
    data.validate3 = np.array([seq_np[x-100:x, ...] for x in validate_ids ])
    
    data.train4    = np.array([seq_np[x+1:x+101, ...] for x in train_ids    ])
    data.test4     = np.array([seq_np[x+1:x+101, ...] for x in test_ids     ])
    data.validate4 = np.array([seq_np[x+1:x+101, ...] for x in validate_ids ])
    
    data.train_ans    = np.array([seq_np[x, ...]  for x in train_ids    ])
    data.test_ans     = np.array([seq_np[x, ...]  for x in test_ids     ])
    data.validate_ans = np.array([seq_np[x, ...]  for x in validate_ids ])
    
    return data
import numpy as np
# import IPython.display
import random 

from Bio import SeqIO
import math
from collections import Counter
import datetime
import time
import os

fasta_file = "data/myco_genome.fasta"
local_genome = "/Users/pochtalionizm/Projects/neuro/data/vibrio.gbff"
remote_genome = "data/myco_genome.gbff"
myco = "/Users/pochtalionizm/Projects/neuro/data/myco.gbff"
vibrio = "/Users/pochtalionizm/Projects/neuro/data/vibrio.gbff"
homo = "data/homos_2.fasta"



    
        
class Container:
    def __init__(self):
        print("...", end = "\r")
        self.record = None #Seqrecord
        self.length = None #int
        self.seq = None #np.array of chars
        self.seq_np = None #np.array of [1, 0, 0, 0]
        
        
        self.bases_dict     = {"A": 0, "T": 1, "C": 2, "G": 3}
        self.bases_list = ["A", "T", "C", "G"]
        self.bases_np        = {
                                "A": np.array([1, 0, 0, 0], dtype = np.float32),
                                "T": np.array([0, 1, 0, 0], dtype = np.float32),
                                "C": np.array([0, 0, 1, 0], dtype = np.float32),
                                "G": np.array([0, 0, 0, 1], dtype = np.float32)
                            }
        self.freqs = None # dict {'A':0.34, ...}
        self.counter = {}
        self.inpaintinglog = None
        self.title = None
        print("container created")
        
        
    def read_seq(self, genome_file = remote_genome, genome_file_type = "genbank"):
        print("...", end = "\r")
        iterator = SeqIO.parse(genome_file, genome_file_type)
        self.record = next(iterator)
        print("read seq from file {}, length = {}".format(genome_file, len(self.record.seq)))
    
    def cut_seq(self, length = None, start = None):
        print("...", end = "\r")
        if start == None:
            start = 0
        if length == None:
            length = len(self.record.seq)
        self.seq = np.asarray(self.record.seq[start:start+length]) 
        self.length = length
        self.start = start
        length_genome = len(self.record.seq)
        self.genome_part = ( start/length_genome*100, (start+length)/length_genome*100)
        
        self.short_title = "{:.1f}-{:.1f}_".format(self.genome_part[0], self.genome_part[1])
        
        self.title = "{:07d}_{:09d}_{:.1f}-{:.1f}_".format(self.length, self.start, self.genome_part[0], self.genome_part[1])
        print("cuted seq for analysis, length = {}, start = {}, part = {:.1f}-{:.1f}".format(self.length, self.start,
                                                                                              self.genome_part[0], 
                                                                                              self.genome_part[1] 
                                                                                              ))
        
    def generate_seq(self):
        print("...", end = "\r")
 
        seq_np = np.zeros((4, self.length), dtype = np.float32) 
        for index in range(self.length):
            base = self.seq[index]
            if base in self.bases_list:
                channel = self.bases_dict[base]
                seq_np[channel][index] = 1
            else:
                print("alternative base")
        self.seq_np = seq_np
        
        print("generated seq_np")
        
        
    def generate_mask(self, seed=None):
        print("...", end = "\r")
        length = self.length
        length_mask = math.ceil(self.length * 0.1)
        
        if seed != None:
            random.seed(seed)
        mask_np = np.zeros((4, length), dtype=np.float32)
        mask = np.zeros(length)
        
        mask_np.fill(1)
        for n in range(length_mask):
            spot = 1
            index = random.randint(0, length-spot)
            for i in range(index, index+spot):
                mask_np[:, i] = [0,0,0,0]
                mask[i] = 1
                
        self.mask_np = mask_np
        self.length_mask = int(sum(mask)) # true mask length!!
        self.mask = mask
        self.short_title = self.short_title + '_'.join(self.record.description.split(' ')[0:2])
        self.title = self.title + "{}_{}".format(self.length_mask, '_'.join(self.record.description.split(' ')))

        print("generated mask with {} spots of {} bp".format(self.length_mask, spot))          
    
    def _get_freqs(self):
        counter = Counter(self.seq[0:self.length])
        self.freqs = {letter : value / self.length for (letter, value) in counter.items()}
    
    def _baseline(self): #count mistakes under mask if using random predictor with frequences
        counter = 0
        for i in range(self.length):
            if self.mask[i] == 1: #if its under mask
                w = [self.freqs[x] for x in self.bases_list]
                letter = random.choices(self.bases_list, weights=w)[0]
                if letter != self.seq[i]:
                        counter +=1
        return counter
    
    
    def baseline(self):
        self._get_freqs()
        baselines = []
        for i in range(200):
            print("{:03d}/200".format(i), end = "\r")
            baselines.append(self._baseline())
        mean = np.mean(baselines)
        sd = np.std(baselines)
        self.counter["baseline_mean"] = mean
        self.counter["baseline_sd"] = sd
        c = self.counter
        c["baseline_part"] = c["baseline_mean"] / self.length_mask
        c["baseline_part_sd"] = c["baseline_sd"] / self.length_mask
        print("got baseline")
        
    def write_folder(self, folder):
        self.datetime = datetime.datetime.now().strftime("%m-%d_%H:%M")
        
        self.log_f = os.path.join(folder, self.datetime + "_" + self.short_title)
        if not os.path.isdir(self.log_f):
            os.mkdir(self.log_f)                    
        file = open(os.path.join(self.log_f, "info.txt"), "w")
        
        file.write(self.datetime + "\n") 
        file.write(self.title + "\n")
        file.close()
                                  
        np.save(os.path.join(self.log_f, "seq_np.npy"), self.seq_np)
        np.save(os.path.join(self.log_f,  "mask.npy"), self.mask)
        print("wrote folder {}".format(self.log_f))
                                  
                                  
def compare(seq_np, out_np):
    if len(seq_np) != len(out_np):
        print("error")
        return None
    
    length = len(seq_np[0])

    out_array = np.zeros((4,length)) #array analog to seq_np
    for i in range(length):
        n = np.argmax(out_np[:, i])
        out_array[n, i] = 1
            
    diff = np.zeros(length)
    for i in range(length):
        if not np.array_equal(out_array[:, i], seq_np[:, i]):
            diff[i] = 1
    return diff 
            


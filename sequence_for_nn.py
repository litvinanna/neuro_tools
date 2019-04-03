import numpy as np
# import IPython.display
import random 

from Bio import SeqIO
import math
from collections import Counter
import datetime
import time
import os
import pickle

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
        self.coding = None
        
        self.bases_dict     = {"A": 0, "T": 1, "C": 2, "G": 3}
        self.bases_list = ["A", "T", "C", "G"]
        self.bases_np        = {
                                "A": np.array([1, 0, 0, 0], dtype = np.float32),
                                "T": np.array([0, 1, 0, 0], dtype = np.float32),
                                "C": np.array([0, 0, 1, 0], dtype = np.float32),
                                "G": np.array([0, 0, 0, 1], dtype = np.float32)
                            }
        self.freqs = None # dict {'A':0.34, ...}

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
        
    def generate_coding(self):
        coding = np.zeros(self.length)
        for f in self.record.features:
            if f.type == "CDS":
                start = max(f.location.start, self.start)
                end = min(f.location.end, self.start + self.length)
                if start < end:
                    coding[start - self.start: end - self.start] = 1 
        self.coding = coding
        print("generated coding part")
            
    def write_folder(self, folder):
        self.datetime = datetime.datetime.now().strftime("%m-%d-%H-%M")
        
        self.log_f = os.path.join(folder, self.datetime + "_" + self.short_title)
        if not os.path.isdir(self.log_f):
            os.mkdir(self.log_f)                    
        with open(os.path.join(self.log_f, "info.txt"), "w") as file:
            file.write(self.datetime + "\n") 
            file.write(self.title + "\n")
            
        properties = {"start": self.start,
                     "length": self.length,
                     "record": self.record}
        with open(os.path.join(self.log_f, "props.pyob"), "wb") as file:
            pickle.dump(properties, file)

                            
        np.save(os.path.join(self.log_f, "seq_np.npy"), self.seq_np)
        np.save(os.path.join(self.log_f,  "mask_np.npy"), self.mask)
        if self.coding is not None:
            np.save(os.path.join(self.log_f,  "coding.npy"), self.coding)
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
            

def get_seq_np(path):
    seq_np = np.load(path + "/seq_np.npy")
    return seq_np

def get_mask(path):
    mask = np.load(path + "/mask_np.npy")
    for i in range(mask.shape[0]):
        if mask[i] == 0:
            mask[i] = 1
        else:
            mask[i] = 0
    return mask

def get_coding(path):
    filename = os.path.join(path, "coding.npy")
    if os.path.exists(filename):
        return np.load(filename)
    else:
        return None
    
def get_loss(path):
    return np.load(os.path.join(path, "loss.npy"))
    
def iterate_outs(path):
    counter = 0
    for i in range(4001):
            out_np_file = "{}/{:05d}_out_np.npy".format(path,i)
            if os.path.exists(out_np_file):
                counter += 1
                out_np = np.load(out_np_file)
                yield out_np, counter - 1
                
def number_of_outs(path):
    c = 0
    for i in range(4001):
        out_np_file = "{}/{:05d}_out_np.npy".format(path,i)
        if os.path.exists(out_np_file):
            c +=1
    return c

def distance_to_right(seq_np, out_np):
    if len(seq_np) != len(out_np):
        print("error in lenght")
        return None
    length = len(seq_np[1])      
    distances_to_right = np.zeros(length)
    for i in range(length):
        a = out_np[:, i]
        b = seq_np[:, i]
        r = np.linalg.norm(a-b)
        distances_to_right[i] = r
    return distances_to_right

def distance_to_all(seq_np, out_np):
    if len(seq_np) != len(out_np):
        print("error in lenght")
        return None
    length = len(seq_np[1])      
    distances = np.zeros((4, length))
    for i in range(length):
        b = seq_np[:, i]
        for j in range(4):
            a = np.zeros(4)
            a[j] = 1
            r = np.linalg.norm(a-b)
            distances[j, i] = r
    return distances

def check_file(filename, data = "np"):
    if os.path.exists(filename):
        print("found file {}".format(filename))
        if data == "np":
            return np.load(filename)
        elif data == "pyob":
            with open(filename, "rb") as file:
                return pickle.load(file)   
    else:
        return None
    
    
def get_distances_to_right(path):
    file = os.path.join(path, "distances_to_right.npy")
    array = check_file(file)
    if array is not None:
        return array
    
    else:
        print("calculating {}".format(file))
        seq_np = get_seq_np(path)
        distances_to_right = np.zeros((number_of_outs(path), seq_np.shape[1]))
        for out_np, i in iterate_outs(path):
            print(i, end = " ")
            distances_to_right[i,...] = distance_to_right(seq_np, out_np)
        np.save(file, distances_to_right)
        return distances_to_right
    
def get_distances_to_all(path):
    file = os.path.join(path, "distances_to_all.npy")
    array = check_file(file)
    if array is not None:
        return array        
    else:
        print("calculating {}".format(file))
        seq_np = get_seq_np(path)
        distances_to_all = np.zeros((number_of_outs(path), 4, seq_np.shape[1]))
        for out_np, i in iterate_outs(path):
            print(i, end = " ")
            distances_to_all[i,...] = distance_to_all(seq_np, out_np)
        np.save(file, distances_to_all)
        return distances_to_all
    
    
    
def get_close_is_right(path):
    file = os.path.join(path, "close_is_right.npy")
    array = check_file(file)
    if array is not None:
        return array 
    else:
        print("calculating {}".format(file))
        seq_np = get_seq_np(path)
        distances_to_all = get_distances_to_all(path)
        
        length = len(seq_np[1])      
        is_close = np.zeros(length)
        for i in range(length):
            min_d = np.argmin(distances_to_all[:, i])
            real = np.argmax(seq_np[:, i])
            if real == min_d:
                is_close[i] = 1
        np.save(file, distances_to_all)        
        return is_close


def generate_diff(seq_np, out_np):
    diff = np.zeros(seq_np.shape[1])
    for i in range(seq_np.shape[1]):
        ans = np.argmax(out_np[:, i])
        right = np.argmax(seq_np[:, i])
        if ans!= right:
            diff[i] = 1
    return diff
    
def get_diff(path):
    file = os.path.join(path, "diffs.npy")
    array = check_file(file)
    if array is not None:
        return array
    else:
        print("calculating {}".format(file))
        seq_np = get_seq_np(path)
        all_diff = np.zeros((number_of_outs(path), seq_np.shape[1]))
        for out_np, i in iterate_outs(path):
            print(i, end = " ")
            all_diff[i,...] = generate_diff(seq_np, out_np)
        np.save(file, all_diff)
        return all_diff



def generate_counter(diff, mask, coding = None):
    if len(diff) != len(mask):
        print("error in length")
        return None
    length = len(diff)
    length_mask = sum(mask)
    
    c = {}
    c["all_mist"] = sum(diff)
    c["mask_mist"] = sum(diff * mask)
    c["free_mist"] = sum(diff) - sum(diff * mask)

    if c["all_mist"] != c["mask_mist"] + c["free_mist"]:
        print("error in counter")
    
    c["mask_part"] = c["mask_mist"] / length_mask
    c["free_part"] = c["free_mist"] / (length - length_mask)
    
       
    if coding is not None:
        c["coding_mask"] = sum(mask * coding)
        c["noncoding_mask"] = sum(mask) - sum(mask * coding)
        
        if c["coding_mask"]+ c["noncoding_mask"] != length_mask:
            print("error in counter")
        
        c["coding_mask_mist"] = sum(diff * mask * coding)
        c["noncoding_mask_mist"] = sum(diff * mask) - sum(diff * mask * coding)
        
        if c["coding_mask_mist"]+c["noncoding_mask_mist"] != c["mask_mist"]:
            print("error in counter")

        c["coding_part"] = c["coding_mask_mist"] / c["coding_mask"]
        c["noncoding_part"] = c["noncoding_mask_mist"] / c["noncoding_mask"]
    
#    print("generated counter")
    return c   
    
    
def get_counters(path):
    filename = os.path.join(path, "counters.pyob")
    ob = check_file(filename, data = "pyob")
    if ob is not None:
        return ob
    else:
        mask = get_mask(path)
        coding = get_coding(path)
        diffs = get_diff(path)
        counters = []
        for i in range(diffs.shape[0]):
            print(i, end = "   ")
            diff = diffs[i, ...]
            counters.append(generate_counter(diff, mask, coding))
        with open(filename, "wb") as file:
            pickle.dump(counters, file)
        return counters
    
def get_properties(path):
    filename = os.path.join(path, "props.pyob")
    ob = check_file(filename, data = "pyob")
    if ob is not None:
        return ob
    else:
        print("no properties file")
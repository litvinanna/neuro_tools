from __future__ import print_function
import os
import numpy as np
from sequence_for_nn import *
import matplotlib
import matplotlib.pyplot as plt
from collections import Counter
import random


    

def baseline_codons(path):
    def find_cds_starts(coding):
        starts = []
        current = 0
        for i in range(coding.shape[0]):
            if coding[i] == 1 and current == 0:
                starts.append([i])
            if coding[i] == 0 and current == 1:
                starts[-1].append(i)

            current = coding[i]
        starts[-1].append(len(coding))
        return starts
    seq_np = get_seq_np(path)
    mask = get_mask(path)
    coding = get_coding(path)
    starts = find_cds_starts(coding)
    
    position_freqs = np.zeros((3, 4))
    for start, end in starts:
        for i in range(start, end):
            pos = i % 3
            base = np.argmax(seq_np[..., i])
            position_freqs[pos,base] += 1

    for i in range(3):
        position_freqs[i, ...] = position_freqs[i, ...]/sum(position_freqs[i , ...])
    counter = 0
    
    k = 0
    for start, end in starts:
        if start // 100000 != k:
            k =  start // 100000
            print(k*100000, end = "  ")
        for i in range(start, end):
            if mask[i] == 1:
                pos = i % 3
                base = np.argmax(seq_np[..., i])
                predicted = np.random.choice([0, 1, 2, 3], p=position_freqs[pos, ...])
                if predicted != base:
                    counter += 1

    return counter / sum(mask * coding)


def baseline(path, n = 10):
    def _baseline(seq_np, mask, bases): #count mistakes under mask if using random predictor with frequences
        length = seq_np.shape[1]
        predicted = np.zeros(seq_np.shape)
        for i in range(length):
            base = np.random.choice([0, 1, 2, 3], p=bases)
            predicted[base, i] = 1    
        counter = len(mask) - sum(np.multiply(sum(np.multiply(seq_np, predicted)), mask))
        return counter
    
    seq_np = get_seq_np(path)
    mask= get_mask(path)
    bases = []
    for i in range(4):
        bases.append(sum(seq_np[i, ...])/(seq_np.shape[1]))
        
    baselines = []
    for i in range(n):
        print(i, end = "  ")
        baselines.append(_baseline(seq_np, mask, bases))
        
    mean = np.mean(baselines)
    sd = np.std(baselines)
    c = {}
    c["baseline_mean"] = mean
    c["baseline_sd"] = sd
    c["baseline_part"] = mean / sum(mask)
    c["baseline_part_sd"] = c["baseline_sd"] / sum(mask)
    return c
    
def get_good_parts(path, tr = 0.7, l = 2000, step = 500):
    coding = get_coding(path)
    mask = get_mask(path)
    diff = get_diff(path)
    counters = get_counters(path)
    cod = []
    mist = []
    non = []
    mist_n = []
    props = get_properties(path)
    length = props['length']
    good_parts = np.zeros(length)
    for start in range(0, length - l, step):
        stop = start + l
        part_c = coding[start:stop]
        coding_part = (sum(part_c))/l
        part_diff = diff[-1][start:stop]
        part_mask = mask[start:stop]
        mist_part = sum(part_diff * part_mask)/sum(part_mask)
        if mist_part < tr:
            cod.append(coding_part)
            mist.append(mist_part)
            good_parts[start:stop] = 1
        else:
            non.append(coding_part)
            mist_n.append(mist_part)
    fig, ax = plt.subplots()
    ax.plot(cod, mist, "ro", non, mist_n, "bo")
    plt.ylabel("mistakes in short pieces")
    plt.xlabel("coding percentage in pieces")
    plt.show()
    return good_parts

def get_proteins(path, good_parts):
    proteins = []
    props = get_properties(path)
    record = props['record']
    for f in props['record'].features:
        if f.type == "CDS":
            start = max(f.location.start,props['start'])
            end = min(f.location.end, props['start'] + props['length'])
            if start < end:
                if np.all(good_parts[start:end]) == 1:
                    protein = f.qualifiers['product']
                    if protein != ['hypothetical protein']:
                        proteins += protein
    return proteins

def plot_mistakes(path):
    counters = get_counters(path)
    fig, ax = plt.subplots()
    ax.plot([counter['mask_part'] for counter in counters], "r", 
            [counter['coding_part'] for counter in counters], "b", 
            [counter['noncoding_part'] for counter in counters], "g" )
    plt.legend(["all mask", "coding mask", "noncoding mask"], loc = 0, frameon = False)
    plt.ylabel("mistakes")
    #plt.ylim(0.5, 1)

    plt.show()
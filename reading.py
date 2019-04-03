from __future__ import print_function
import os
import numpy as np
from sequence_for_nn import *
import matplotlib
import matplotlib.pyplot as plt
from collections import Counter
import random


    
def _baseline(seq_np, mask, bases): #count mistakes under mask if using random predictor with frequences

    length = seq_np.shape[1]
    predicted = np.zeros(seq_np.shape)
    for i in range(length):
        base = np.random.choice([0, 1, 2, 3], p=bases)
        predicted[base, i] = 1    
    counter = sum(np.multiply(sum(np.multiply(seq_np, predicted)), mask))
    return counter

def baseline(_baseline, seq_np, mask, bases):
    bases = []
    for i in range(4):
        bases.append(sum(seq_np[i, ...])/(seq_np.shape[1]))
        
    baselines = []
    for i in range(10):
        print(i, end = "  ")
        baselines.append(_baseline(seq_np, mask, bases))
    mean = np.mean(baselines)
    sd = np.std(baselines)
    c = {}
    c["baseline_mean"] = mean
    c["baseline_sd"] = sd
    c["baseline_part"] = c["baseline_mean"] / sum(mask)
    c["baseline_part_sd"] = c["baseline_sd"] / sum(mask)
    return c
    
def get_good_parts(path, tr = 0.7):
    coding = get_coding(path)
    mask = get_mask(path)
    diff = get_diff(path)
    counters = get_counters(path)
    l = 2000
    cod = []
    mist = []
    non = []
    mist_n = []
    props = get_properties(path)
    length = props['length']
    good_parts = np.zeros(length)
    for start in range(0, length - l, 500):
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
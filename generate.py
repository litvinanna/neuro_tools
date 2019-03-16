import argparse
from Bio import SeqIO
import math
from collections import Counter
import datetime
import time
import os
from sequence_for_nn import Container

parser = argparse.ArgumentParser(description='Generates folder with np array for seq and mask for sequence file')

parser.add_argument( "-f", "--file", required=True, type=str, 
                    help='input genome')
parser.add_argument( "-o", "--out", required=True, type=str, 
                    help='out folder')
parser.add_argument( "-l", "--length", default=0, type=int, 
                    help='sequence length')
parser.add_argument( "-s", "--start", default=0, type=int, 
                    help='sequence start coordinate')
args = parser.parse_args()

print("input file {}".format(args.file))

container = Container()
file_format = args.file.split(".")[-1]
print("file format {}".format(file_format))
container.read_seq(genome_file = args.file, genome_file_type = file_format)
if args.length == 0:
    length = len(container.seq)
else:
    length = args.length
start = args.start
container.cut_seq(length = length, start = start)
container.generate_seq()
container.generate_mask()
container.write_folder(folder = args.out)



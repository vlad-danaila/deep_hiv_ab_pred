import json
from Bio import SeqIO

def dump_json(obj, path):
    with open(path, mode='w') as file:
        json.dump(obj, file)

def read_fasta(path):
    return SeqIO.parse(path, "fasta")
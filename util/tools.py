import json
from Bio import SeqIO
import torch as t
import os
import urllib
from math import log10
import time
import yaml

def dump_json(obj, path):
    with open(path, mode='w') as file:
        json.dump(obj, file)

def read_json_file(path):
    with open(path) as file:
        return json.loads(file.read())

def read_fasta(path):
    return SeqIO.parse(path, "fasta")

device = 'cuda' if t.cuda.is_available() else 'cpu'

"""This function places the tensor on the CPU; it detaches it from gradient computation and then transforms it to NumPy array."""
def to_numpy(x):
    return x.cpu().detach().numpy()

"""Creates a folder with a given name if not existing already."""
def create_folder(path):
    if not os.path.isdir(path):
        os.mkdir(path)

"""Function to download a file from an URL."""
def download_file(url, file):
    if not os.path.isfile(file):
        urllib.request.urlretrieve(url, file)

"""Functions to measure execution time, the timer_start can be regarded as starting a chronometer, and the function timer can be regarded as stopping and resetting the chronometer."""

time_start = 0

def timer_start():
    global time_start
    time_start = time.time()

def timer(msg = 'Timing:'):
    print(msg, time.time() - time_start)
    timer_start()

def log_transform(x):
    return log10(1 + x)

def log_transform_back(x):
    return (10 ** x) - 1

def normalize(x, mean, std, epsilon = 0):
    if x is None:
        return None
    return (x - mean) / (std + epsilon)

def unnormalize(x, mean, std):
    if x is None:
        return None
    return (x * std) + mean

def to_numpy(tensor: t.Tensor):
    return tensor.detach().cpu().numpy()

def to_torch(x, type = t.float64, device = 'cpu', grad = False):
    return t.tensor(x, dtype = type, device = device, requires_grad = grad)

def read_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
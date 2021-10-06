import json
from Bio import SeqIO
import torch as t
import os
import urllib

def dump_json(obj, path):
    with open(path, mode='w') as file:
        json.dump(obj, file)

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

"""The ProgressDisplay class shows a better-looking loader for time-consuming processes, such as training or evaluating a neural network."""

class ProgressDisplay():

    def __init__(self, steps, label):
        self.steps = steps
        self.counter = 0
        self.label = label
        self.progressbar = display(self.__html(), display_id=True)

    def __html(self):
        return HTML(f'{self.label} {self.counter * 100 / self.steps:.2f} %')

    def step(self):
        self.counter += 1
        self.progressbar.update(self.__html())

    def reset(self):
        self.counter = 0
        self.progressbar.update(HTML(''))

# To use:
# progress_display = ProgressDisplay(100, 'training')
# for ii in range(100):
#     time.sleep(0.02)
#     progress_display.step()
# progress_display.reset()
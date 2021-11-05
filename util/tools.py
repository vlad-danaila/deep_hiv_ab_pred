import json
from Bio import SeqIO
import torch as t
import os
import urllib
from math import log10
import time
import yaml
import mlflow
import optuna
from os.path import join
from deep_hiv_ab_pred.hyperparameters.constants import HYPERPARAMETERS_FOLDER
import logging

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
    logging.info(msg + ' ' + (time.time() - time_start))
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

def get_experiment(name):
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        return mlflow.create_experiment(name)
    return experiment.experiment_id

'''ex: write_study_best_params_to_json('ICERI2021_v2', 'hyperparameters_iceri_2021_v2_30_10_2021.json')'''
def write_study_best_params_to_json(study_name, hyperparameters_file):
    study = optuna.create_study(study_name = study_name, direction = 'maximize', load_if_exists = True, storage = f'sqlite:///{study_name}.db')
    dump_json(study.best_params, join(HYPERPARAMETERS_FOLDER, hyperparameters_file))

# didn't test this yet
def write_specific_study_params_to_json(study_name, study_number, hyperparameters_file):
    study = optuna.create_study(study_name = study_name, direction = 'maximize', load_if_exists = True, storage = f'sqlite:///{study_name}.db')
    trial_params = [t for t in study.get_trials() if t.number == study_number][0].params
    dump_json(trial_params, join(HYPERPARAMETERS_FOLDER, hyperparameters_file))
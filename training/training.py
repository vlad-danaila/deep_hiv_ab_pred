import os

import mlflow

from deep_hiv_ab_pred.training.constants import LOSS, ACCURACY, MATTHEWS_CORRELATION_COEFFICIENT
import numpy as np
import sklearn.metrics
import sklearn as sk
from tqdm import tqdm
from deep_hiv_ab_pred.util.tools import to_numpy
import torch as t
import math
from tqdm import tqdm
import sys

# This function performs a forward pass of the network and records the metrics.
# If training is ebabled, a backword pass and network parameter updates are also performed.
def run_network(model, conf, loader, loss_fn, optimizer = None, isTrain = False):
    # metrics will hold the loss and accuracy
    metrics = np.zeros(3)
    # we calculate a weighted average by the number of samples in each batch,
    # all batches will have the same number of elements (weight one), except
    # for the last one which will have less elements (will have subunitary weight)
    total_weight = 0

    desc = 'training' if isTrain else 'evaluating'
    # Iterate through the dataset using the data loader
    for i, (ab_light, ab_heavy, virus, pngs_mask, ground_truth) in enumerate(loader):
        # Network forward pass
        pred = model.forward(ab_light, ab_heavy, virus, pngs_mask)
        # Calculate loss value
        loss = loss_fn(pred, ground_truth)

        # Backword pass
        if isTrain:
            assert optimizer != None
            # Gradient backwords propagation
            loss.backward()
            # Gradient clipping protects against explosive gradients
            t.nn.utils.clip_grad_norm_(model.parameters(), conf['GRAD_NORM_CLIP'], norm_type=1)
            # Network parameter updates
            optimizer.step()
            # Refresh optimizer state
            optimizer.zero_grad()

        pred = to_numpy(pred) > .5
        ground_truth = to_numpy(ground_truth)

        # The last batch have fewer elements then the rest.
        # For this reason we weight each metric by the population size of the batch using the variable named 'weight'
        weight = len(ground_truth) / conf['BATCH_SIZE']
        metrics[LOSS] += (loss.item() * weight)
        accuracy = sk.metrics.accuracy_score(ground_truth, pred)
        metrics[ACCURACY] += (accuracy * weight)
        correlation = sk.metrics.matthews_corrcoef(ground_truth, pred)
        metrics[MATTHEWS_CORRELATION_COEFFICIENT] += (correlation * weight)
        total_weight += weight
    metrics /= total_weight
    return metrics

# Evaluate
def eval_network(model, conf, loader, loss_fn):
    model.eval()
    # During testing we do not calculate any gradients, nor perform any network parameter updates
    with t.no_grad():
        test_metrics = run_network(model, conf, loader, loss_fn, isTrain = False)
        return test_metrics

# Train
def train_network(model, conf, loader_train, loader_val, cross_validation_round, epochs, model_title = 'model', model_path = '', save_model = True):
    loss_fn = t.nn.BCELoss()
    optimizer = t.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr = conf['LEARNING_RATE'])
    metrics_train_per_epochs, metrics_test_per_epochs = [], []
    best = [math.inf, 0, -math.inf]
    try:
        for epoch in range(epochs):
            model.train()
            train_metrics = run_network(model, conf, loader_train, loss_fn, optimizer, isTrain = True)
            metrics_train_per_epochs.append(train_metrics)
            if loader_val:
                test_metrics = eval_network(model, conf, loader_val, loss_fn)
                metrics_test_per_epochs.append(test_metrics)
                # We save a model chekpoint if we find any improvement
                if test_metrics[MATTHEWS_CORRELATION_COEFFICIENT] > best[MATTHEWS_CORRELATION_COEFFICIENT]:
                    best = test_metrics
                    if save_model:
                        t.save({'model': model.state_dict()}, os.path.join(model_path, f'{model_title} cv {cross_validation_round + 1}.tar'))
                print(f'Epoch {epoch + 1}, Correlation: {test_metrics[MATTHEWS_CORRELATION_COEFFICIENT]}, Accuracy: {test_metrics[ACCURACY]}')
            else:
                # We save a model chekpoint if we find any improvement
                if train_metrics[MATTHEWS_CORRELATION_COEFFICIENT] > best[MATTHEWS_CORRELATION_COEFFICIENT]:
                    best = train_metrics
                    if save_model:
                        t.save({'model': model.state_dict()}, os.path.join(model_path, f'{model_title}.tar'))
                print(f'Epoch {epoch + 1}, Correlation: {train_metrics[MATTHEWS_CORRELATION_COEFFICIENT]}, Accuracy: {train_metrics[ACCURACY]}')
        print('-' * 10)
        if cross_validation_round is not None:
            print(f'Cross validation round {cross_validation_round + 1}, Correlation: {best[MATTHEWS_CORRELATION_COEFFICIENT]}, Accuracy: {best[ACCURACY]}')
        else:
            print(f'Correlation: {best[MATTHEWS_CORRELATION_COEFFICIENT]}, Accuracy: {best[ACCURACY]}')
        print('-' * 10)
        return metrics_train_per_epochs, metrics_test_per_epochs, best
    except KeyboardInterrupt as e:
        print('Training interrupted at epoch', epoch)

def train_network_n_times(model, conf, loader_train, loader_val, cross_validation_round, epochs, model_title = 'model', model_path = '', save_model = True):
    loss_fn = t.nn.BCELoss()
    optimizer = t.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr = conf['LEARNING_RATE'])
    metrics_train_per_epochs, metrics_test_per_epochs = [], []
    try:
        for epoch in range(epochs):
            model.train()
            train_metrics = run_network(model, conf, loader_train, loss_fn, optimizer, isTrain = True)
            metrics_train_per_epochs.append(train_metrics)
            if loader_val:
                test_metrics = eval_network(model, conf, loader_val, loss_fn)
                metrics_test_per_epochs.append(test_metrics)
                print(f'Epoch {epoch + 1}, Correlation: {test_metrics[MATTHEWS_CORRELATION_COEFFICIENT]}, Accuracy: {test_metrics[ACCURACY]}')
            else:
                metrics_train_per_epochs.append(train_metrics)
                print(f'Epoch {epoch + 1}, Correlation: {train_metrics[MATTHEWS_CORRELATION_COEFFICIENT]}, Accuracy: {train_metrics[ACCURACY]}')
        last = test_metrics if loader_val else train_metrics
        print('-' * 10)
        if cross_validation_round is not None:
            print(f'Cross validation round {cross_validation_round + 1}, Correlation: {last[MATTHEWS_CORRELATION_COEFFICIENT]}, Accuracy: {last[ACCURACY]}')
        else:
            print(f'Correlation: {last[MATTHEWS_CORRELATION_COEFFICIENT]}, Accuracy: {last[ACCURACY]}')
        print('-' * 10)
        return metrics_train_per_epochs, metrics_test_per_epochs, last
    except KeyboardInterrupt as e:
        print('Training interrupted at epoch', epoch)
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
def run_network(model, conf, loader, loss_fn, optimizer = None, isTrain = False, pbar = None):
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

        if pbar:
            pbar.update(1)

    metrics /= total_weight
    return metrics

# Evaluate
def eval_network(model, conf, loader, loss_fn, pbar = None):
    model.eval()
    # During testing we do not calculate any gradients, nor perform any network parameter updates
    with t.no_grad():
        test_metrics = run_network(model, conf, loader, loss_fn, isTrain = False, pbar = pbar)
        return test_metrics

# Train
def train_network(model, conf, loader_train, loader_val, cross_validation_round, epochs, model_title = 'model', model_path = '', ml_flow_prefix = '', save_model = True):

    len_validation = 0 if loader_val is None else len(loader_val)
    # pbar = tqdm(total = epochs * (len(loader_train) + len_validation),
    #             desc = 'Training',
    #             file = sys.stdout)
    pbar = None

    loss_fn = t.nn.BCELoss()
    # The optimizer updates the parameters of the model during training
    optimizer = t.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr = conf['LEARNING_RATE'])

    # We'll store the metrics per each epoch for plotting
    metrics_train_per_epochs, metrics_test_per_epochs = [], []
    # If testing on a certain epoch yields better accuracy, then we checkpoint the model
    best = [math.inf, 0, -math.inf]

    try:
        for epoch in range(epochs):
            # Train
            # We enable the training mode on the model, this activates all dropout
            # layers and makes batch normalization layers record statistics.
            model.train()
            train_metrics = run_network(model, conf, loader_train, loss_fn, optimizer, isTrain = True, pbar = pbar)
            metrics_train_per_epochs.append(train_metrics)

            if loader_val:
                test_metrics = eval_network(model, conf, loader_val, loss_fn, pbar = pbar)
                metrics_test_per_epochs.append(test_metrics)

                # We save a model chekpoint if we find any improvement
                if test_metrics[MATTHEWS_CORRELATION_COEFFICIENT] > best[MATTHEWS_CORRELATION_COEFFICIENT]:
                    best = test_metrics
                    if save_model:
                        t.save({'model': model.state_dict()}, os.path.join(model_path, f'{model_title} cv {cross_validation_round + 1}.tar'))

                # Logging
                print(f'Epoch {epoch + 1}, Correlation: {test_metrics[MATTHEWS_CORRELATION_COEFFICIENT]}, Accuracy: {test_metrics[ACCURACY]}')
            else:
                # We save a model chekpoint if we find any improvement
                if train_metrics[MATTHEWS_CORRELATION_COEFFICIENT] > best[MATTHEWS_CORRELATION_COEFFICIENT]:
                    best = train_metrics
                    if save_model:
                        t.save({'model': model.state_dict()}, os.path.join(model_path, f'{model_title}.tar'))

                # Logging
                print(f'Epoch {epoch + 1}, Correlation: {train_metrics[MATTHEWS_CORRELATION_COEFFICIENT]}, Accuracy: {train_metrics[ACCURACY]}')

            if pbar:
                pbar.refresh()

        if pbar:
            pbar.close()

        print('-' * 10)
        if cross_validation_round is not None:
            print(f'Cross validation round {cross_validation_round + 1}, Correlation: {best[MATTHEWS_CORRELATION_COEFFICIENT]}, Accuracy: {best[ACCURACY]}')
        else:
            print(f'Correlation: {best[MATTHEWS_CORRELATION_COEFFICIENT]}, Accuracy: {best[ACCURACY]}')
        print('-' * 10)

        mlflow.log_metrics({
            f'{ml_flow_prefix} mcc': best[MATTHEWS_CORRELATION_COEFFICIENT],
            f'{ml_flow_prefix} acc': best[ACCURACY]
        })
        if save_model:
            checkpoint_path = f'{model_title} cv {cross_validation_round + 1}.tar' if loader_val else f'{model_title}.tar'

        return metrics_train_per_epochs, metrics_test_per_epochs, best

    except KeyboardInterrupt as e:
        print('Training interrupted at epoch', epoch)
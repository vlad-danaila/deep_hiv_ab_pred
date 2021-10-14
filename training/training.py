from training.constants import LOSS, ACCURACY, MATTHEWS_CORRELATION_COEFFICIENT
import numpy as np
import sklearn as sk
from tqdm import tqdm
from util.tools import to_numpy
import torch as t
import math

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
    for i, (ab_light, ab_heavy, virus, pngs_mask, ground_truth) in tqdm(enumerate(loader), desc = desc):
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
def train_network(model, conf, loader_train, loader_val, cross_validation_round, epochs):
    loss_fn = t.nn.BCELoss()
    # The optimizer updates the parameters of the model during training
    optimizer = t.optim.RMSprop(model.parameters(), lr = conf['LEARNING_RATE'])

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
            train_metrics = run_network(model, conf, loader_train, loss_fn, optimizer, isTrain = True)
            metrics_train_per_epochs.append(train_metrics)

            test_metrics = eval_network(model, conf, loader_val, loss_fn)
            metrics_test_per_epochs.append(test_metrics)

            # We save a model chekpoint if we find any improvement
            if test_metrics[MATTHEWS_CORRELATION_COEFFICIENT] > best[MATTHEWS_CORRELATION_COEFFICIENT]:
                best = test_metrics
                t.save({'model': model.state_dict()}, 'model cross validation {}.tar'.format(cross_validation_round))

            # Logging
            print(f'Epoch {epoch + 1}, Correlation: {test_metrics[MATTHEWS_CORRELATION_COEFFICIENT]}, Accuracy: {test_metrics[ACCURACY]}')

        print('-' * 10)
        print(f'Cross validation round {cross_validation_round + 1}, Correlation: {best[MATTHEWS_CORRELATION_COEFFICIENT]}, Accuracy: {best[ACCURACY]}')
        print('-' * 10)
        return metrics_train_per_epochs, metrics_test_per_epochs, best

    except KeyboardInterrupt as e:
        print('Training interrupted at epoch', epoch)
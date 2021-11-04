import os
from deep_hiv_ab_pred.training.constants import LOSS, ACCURACY, MATTHEWS_CORRELATION_COEFFICIENT
import numpy as np
import torch as t
import math
from deep_hiv_ab_pred.util.metrics import compute_metrics

def run_network(model, conf, loader, loss_fn, optimizer = None, isTrain = False):
    metrics = np.zeros(4)
    # we calculate a weighted average by the number of samples in each batch,
    # all batches will have the same number of elements (weight one), except
    # for the last one which will have less elements (will have subunitary weight)
    total_weight = 0
    for i, (ab_light, ab_heavy, virus, pngs_mask, ground_truth) in enumerate(loader):
        pred = model.forward(ab_light, ab_heavy, virus, pngs_mask)
        loss = loss_fn(pred, ground_truth)
        if isTrain:
            assert optimizer != None
            loss.backward()
            t.nn.utils.clip_grad_norm_(model.parameters(), conf['GRAD_NORM_CLIP'], norm_type=1)
            optimizer.step()
            optimizer.zero_grad()
        # The last batch have fewer elements then the rest.
        # For this reason we weight each metric by the population size of the batch using the variable named 'weight'
        weight = len(ground_truth) / conf['BATCH_SIZE']
        total_weight += weight
        metrics += compute_metrics(ground_truth, pred, loss) * weight
    return metrics / total_weight

# Evaluate
def eval_network(model, conf, loader, loss_fn):
    model.eval()
    # During testing we do not calculate any gradients, nor perform any network parameter updates
    with t.no_grad():
        test_metrics = run_network(model, conf, loader, loss_fn, isTrain = False)
        return test_metrics

# Train
def train_network(model, conf, loader_train, loader_val, cross_validation_round, epochs, model_title = 'model', model_path = '', save_model = True, log_every_epoch = True):
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
                if log_every_epoch:
                    print(f'Epoch {epoch + 1}, Correlation: {test_metrics[MATTHEWS_CORRELATION_COEFFICIENT]}, Accuracy: {test_metrics[ACCURACY]}')
            else:
                # We save a model chekpoint if we find any improvement
                if train_metrics[MATTHEWS_CORRELATION_COEFFICIENT] > best[MATTHEWS_CORRELATION_COEFFICIENT]:
                    best = train_metrics
                    if save_model:
                        t.save({'model': model.state_dict()}, os.path.join(model_path, f'{model_title}.tar'))
                if log_every_epoch:
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

def run_net_with_frozen_antibody_and_embedding(model, conf, loader, loss_fn, optimizer = None, isTrain = False):
    metrics = np.zeros(4)
    total_weight = 0
    for i, (ab_light, ab_heavy, virus, pngs_mask, ground_truth) in enumerate(loader):
        batch_size = len(ab_light)
        with t.no_grad():
            ab_light, ab_heavy, virus = model.module.forward_embeddings(ab_light, ab_heavy, virus, batch_size)
            ab_hidden = model.module.forward_antibodyes(ab_light, ab_heavy, batch_size)
        pred = model.module.forward_virus(virus, pngs_mask, ab_hidden)
        loss = loss_fn(pred, ground_truth)
        if isTrain:
            assert optimizer != None
            loss.backward()
            t.nn.utils.clip_grad_norm_(model.parameters(), conf['GRAD_NORM_CLIP'], norm_type=1)
            optimizer.step()
            optimizer.zero_grad()
        weight = len(ground_truth) / conf['BATCH_SIZE']
        total_weight += weight
        metrics += compute_metrics(ground_truth, pred, loss) * weight
    return metrics / total_weight

def train_with_frozen_antibody_and_embedding(model, conf, loader_train, loader_val, cross_validation_round, epochs, model_title = 'model', model_path = '', save_model = True, log_every_epoch = True):
    # Freezing the embeddings and antibody subnetworks
    for param in model.module.aminoacid_embedding.parameters():
        param.requires_grad = False
    model.module.embedding_dropout = t.nn.Dropout(p = 0)
    model.module.embedding_dropout.requires_grad = False

    for param in model.module.light_ab_gru.parameters():
        param.requires_grad = False
    model.module.light_ab_gru.dropout = 0

    for param in model.module.heavy_ab_gru.parameters():
        param.requires_grad = False
    model.module.heavy_ab_gru.dropout = 0

    loss_fn = t.nn.BCELoss()
    optimizer = t.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr = conf['LEARNING_RATE'])
    metrics_train_per_epochs, metrics_test_per_epochs = [], []
    best = [math.inf, 0, -math.inf]
    try:
        for epoch in range(epochs):
            model.module.aminoacid_embedding.eval()
            model.module.light_ab_gru.eval()
            model.module.heavy_ab_gru.eval()
            model.module.virus_gru.train()
            model.module.embedding_dropout.eval()
            model.module.fc_dropout.train()
            model.module.fully_connected.train()

            train_metrics = run_net_with_frozen_antibody_and_embedding(model, conf, loader_train, loss_fn, optimizer, isTrain = True)
            metrics_train_per_epochs.append(train_metrics)

            test_metrics = eval_network(model, conf, loader_val, loss_fn)
            metrics_test_per_epochs.append(test_metrics)
            # We save a model chekpoint if we find any improvement
            if test_metrics[MATTHEWS_CORRELATION_COEFFICIENT] > best[MATTHEWS_CORRELATION_COEFFICIENT]:
                best = test_metrics
                if save_model:
                    t.save({'model': model.state_dict()}, os.path.join(model_path, f'{model_title} cv {cross_validation_round + 1}.tar'))
            if log_every_epoch:
                print(f'Epoch {epoch + 1}, Correlation: {test_metrics[MATTHEWS_CORRELATION_COEFFICIENT]}, Accuracy: {test_metrics[ACCURACY]}')

        print('-' * 10)
        print(f'Cross validation round {cross_validation_round + 1}, Correlation: {best[MATTHEWS_CORRELATION_COEFFICIENT]}, Accuracy: {best[ACCURACY]}')
        print('-' * 10)
        return metrics_train_per_epochs, metrics_test_per_epochs, best
    except KeyboardInterrupt as e:
        print('Training interrupted at epoch', epoch)

def run_net_with_frozen_net_except_of_last_layer(model, conf, loader, loss_fn, optimizer = None, isTrain = False):
    metrics = np.zeros(4)
    total_weight = 0
    for i, (ab_light, ab_heavy, virus, pngs_mask, ground_truth) in enumerate(loader):
        batch_size = len(ab_light)
        with t.no_grad():
            ab_light, ab_heavy, virus = model.module.forward_embeddings(ab_light, ab_heavy, virus, batch_size)
            ab_hidden = model.module.forward_antibodyes(ab_light, ab_heavy, batch_size)
            virus_and_pngs = t.cat([virus, pngs_mask], axis = 2)
            model.module.virus_gru.flatten_parameters()
            virus_ab_all_output, _ = model.module.virus_gru(virus_and_pngs, ab_hidden)
            virus_output = virus_ab_all_output[:, -1]
        virus_output = model.module.fc_dropout(virus_output)
        pred = model.module.sigmoid(model.module.fully_connected(virus_output).squeeze())
        loss = loss_fn(pred, ground_truth)
        if isTrain:
            assert optimizer != None
            loss.backward()
            t.nn.utils.clip_grad_norm_(model.parameters(), conf['GRAD_NORM_CLIP'], norm_type=1)
            optimizer.step()
            optimizer.zero_grad()
        weight = len(ground_truth) / conf['BATCH_SIZE']
        total_weight += weight
        metrics += compute_metrics(ground_truth, pred, loss) * weight
    return metrics / total_weight

def train_with_fozen_net_except_of_last_layer(model, conf, loader_train, loader_val, cross_validation_round, epochs, model_title = 'model', model_path = '', save_model = True, log_every_epoch = True):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.module.fully_connected.parameters():
        param.requires_grad = True
    model.module.fc_dropout.requires_grad = True
    model.module.embedding_dropout = t.nn.Dropout(p = 0)
    model.module.light_ab_gru.dropout = 0
    model.module.heavy_ab_gru.dropout = 0
    model.module.virus_gru.dropout = 0

    loss_fn = t.nn.BCELoss()
    optimizer = t.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr = conf['LEARNING_RATE'])
    metrics_train_per_epochs, metrics_test_per_epochs = [], []
    best = [math.inf, 0, -math.inf]
    try:
        for epoch in range(epochs):
            model.eval()
            model.module.fc_dropout.train()
            model.module.fully_connected.train()

            train_metrics = run_net_with_frozen_net_except_of_last_layer(model, conf, loader_train, loss_fn, optimizer, isTrain = True)
            metrics_train_per_epochs.append(train_metrics)

            test_metrics = eval_network(model, conf, loader_val, loss_fn)
            metrics_test_per_epochs.append(test_metrics)
            # We save a model chekpoint if we find any improvement
            if test_metrics[MATTHEWS_CORRELATION_COEFFICIENT] > best[MATTHEWS_CORRELATION_COEFFICIENT]:
                best = test_metrics
                if save_model:
                    t.save({'model': model.state_dict()}, os.path.join(model_path, f'{model_title} cv {cross_validation_round + 1}.tar'))
            if log_every_epoch:
                print(f'Epoch {epoch + 1}, Correlation: {test_metrics[MATTHEWS_CORRELATION_COEFFICIENT]}, Accuracy: {test_metrics[ACCURACY]}')

        print('-' * 10)
        print(f'Cross validation round {cross_validation_round + 1}, Correlation: {best[MATTHEWS_CORRELATION_COEFFICIENT]}, Accuracy: {best[ACCURACY]}')
        print('-' * 10)
        return metrics_train_per_epochs, metrics_test_per_epochs, best
    except KeyboardInterrupt as e:
        print('Training interrupted at epoch', epoch)

def train_network_n_times(model, conf, loader_train, loader_val, cross_validation_round, epochs, model_title = 'model', model_path = ''):
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
        t.save({'model': model.state_dict()}, os.path.join(model_path, f'{model_title}.tar'))
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
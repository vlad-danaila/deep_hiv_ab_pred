import optuna
import numpy as np

class CrossValidationPruner:

    def __init__(self, time_treshold, len_steps, cv_folds, treshold):
        self.time_treshold = time_treshold
        self.len_steps = len_steps
        self.cv_folds = cv_folds
        self.treshold = treshold
        self.best = np.zeros(len_steps * cv_folds)
        self.reset()

    def report_time(self, time):
        if time > self.time_treshold:
            self.reset()
            raise optuna.TrialPruned()

    def report(self, metric, step, cv_fold):
        if self.best[cv_fold * self.len_steps + step] - self.treshold > metric:
            self.reset()
            raise optuna.TrialPruned()

        self.current[cv_fold * self.len_steps + step] = metric

        if self.current.prod() > 0 and self.current[-1] > self.best[-1]:
            self.best = self.current
            self.reset()

    def reset(self):
        self.current = np.zeros(self.len_steps * self.cv_folds)

if __name__ == '__main__':

    # Should prune trials 1, 4, 5

    cvp = CrossValidationPruner(5, 3, 5, .05)

    # Trial 1
    try:
        for cv_fold in range(5):
            cvp.report_time(6)
            cvp.report(.32, 0, cv_fold)
            cvp.report(.32, 1, cv_fold)
            cvp.report_time(4)
            cvp.report(.35, 2, cv_fold)
            cvp.report_time(4)
    except optuna.TrialPruned as e:
        print('trial 1 pruned')

    # Trial 2
    try:
        for cv_fold in range(5):
            for step in range(3):
                cvp.report(.32 + step * .2, step, cv_fold)
                cvp.report_time(4)
    except optuna.TrialPruned as e:
        print('trial 2 pruned')

    # Trial 3
    try:
        for cv_fold in range(5):
            for step in range(3):
                cvp.report(.32 + step * .3, step, cv_fold)
                cvp.report_time(4)
    except optuna.TrialPruned as e:
        print('trial 3 pruned')

    # Trial 4
    try:
        for cv_fold in range(5):
            for step in range(3):
                cvp.report(.32 + step * .2, step, cv_fold)
                cvp.report_time(4)
    except optuna.TrialPruned as e:
        print('trial 4 pruned')

    # Trial 5
    try:
        for cv_fold in range(5):
            for step in range(3):
                cvp.report(.1 + step * .5, step, cv_fold)
                cvp.report_time(4)
    except optuna.TrialPruned as e:
        print('trial 5 pruned')
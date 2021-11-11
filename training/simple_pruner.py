import optuna
from deep_hiv_ab_pred.training.cv_pruner import CrossValidationPruner

class SimplePruner(CrossValidationPruner):

    def __init__(self, time_treshold, len_steps, percentile):
        CrossValidationPruner.__init__(self, time_treshold, len_steps, 1, percentile)

    def report(self, metric, time, step):
        return CrossValidationPruner.report(self, metric, time, step, 0)

if __name__ == '__main__':

    # Should prune trials 1, 4, 5

    sp = SimplePruner(5, 3, 90)

    # Trial 1
    try:
        sp.report(.32, 6, 0)
    except optuna.TrialPruned as e:
        print('trial 1 pruned')

    # Trial 2
    try:
        for step in range(3):
            sp.report(.32 + step * .2, 4, step)
    except optuna.TrialPruned as e:
        print('trial 2 pruned')

    # Trial 3
    try:
        for step in range(3):
            sp.report(.32 + step * .3, 4, step)
    except optuna.TrialPruned as e:
        print('trial 3 pruned')

    # Trial 4
    try:
        for step in range(3):
            sp.report(.32 + step * .2, 4, step)
    except optuna.TrialPruned as e:
        print('trial 4 pruned')

    # Trial 5
    try:
        for step in range(3):
            sp.report(.1 + step * .5, 4, step)
    except optuna.TrialPruned as e:
        print('trial 5 pruned')
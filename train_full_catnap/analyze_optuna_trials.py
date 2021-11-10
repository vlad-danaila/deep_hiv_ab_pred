from typing import List
import optuna
import matplotlib.pyplot as plt

STUDY_NAME = 'ICERI2021_v2'
SELECTION_TRESHOLD = .48

def get_best_trials_from_study(study_name, selection_treshold):
    study = optuna.create_study(study_name = study_name, direction = 'maximize', storage = f'sqlite:///{study_name}.db', load_if_exists = True)
    trials = [t for t in study.get_trials() if t.state == optuna.trial.TrialState.COMPLETE and t.value > selection_treshold and t.duration.seconds < 1500]
    return trials

def plot_scatter_performance_vs_time(trials: List[optuna.trial.FrozenTrial]):
    perf = [t.value for t in trials]
    time = [t.duration.seconds for t in trials]
    plt.scatter(time, perf)
    for i, trial in enumerate(trials):
        plt.annotate(trial.number, (trial.duration.seconds, trial.value))

if __name__ == '__main__':
    trials = get_best_trials_from_study(STUDY_NAME, SELECTION_TRESHOLD)
    plot_scatter_performance_vs_time(trials)

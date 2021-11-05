import logging
import sys
import optuna

def setup_logging():
    optuna.logging.get_logger("optuna").addHandler(logging.FileHandler('optuna log'))

    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    rootLogger.addHandler(console)

    file = logging.FileHandler('main.log')
    file.setFormatter(formatter)
    rootLogger.addHandler(file)

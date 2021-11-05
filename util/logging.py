import logging

def setup_logging():
    rootLogger = logging.getLogger()
    rootLogger.addHandler(logging.FileHandler('logs.log'))
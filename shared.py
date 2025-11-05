import os
import json
import logging


def setup_logger(name):
    formatter = logging.Formatter(fmt="%(module)s %(levelname)s - %(message)s")

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    return logger

logger = setup_logger(__name__)

def read_config(path="config.json"):
    with open(path) as file:
        config = json.load(file)
    return config

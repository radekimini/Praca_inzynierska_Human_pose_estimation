import os
import json
import logging
from enum import Enum


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

class Joint(Enum):
    R_FOOT = 0
    R_KNEE = 1
    R_HIP = 2
    L_HIP = 3
    L_KNEE = 4
    L_FOOT = 5
    C_HIP = 6
    C_SHOULDER = 7
    NECK = 8
    HEAD = 9
    R_HAND = 10
    R_ELBOW = 11
    R_SHOULDER = 12
    L_SHOULDER = 13
    L_ELBOW = 14
    L_HAND = 15

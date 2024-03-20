import json
import logging
import os

import numpy as np


def setup_logger(logger_name: str, logging_level=logging.INFO):
    """
    Helper function for setting up logger both in stdout and file
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging_level)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # create file handler which logs even debug messages
    current_cwd = os.getcwd()
    log_dir = os.path.join(
        current_cwd,
        "logs/",
    )
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"{logger_name}.log")
    fh = logging.FileHandler(log_file_path, mode="w")
    fh.setLevel(logging.DEBUG)

    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger


def unusable(reason: str, date: str):
    """Means to highlight a function that should be halted if called during a specific development branch"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            raise NotImplementedError(
                f"This function was deprecated on {date}. Reason: {reason}"
            )

        return wrapper

    return decorator


def deprecated(reason: str, date: str):
    """This decorator disables the provided function."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            raise NotImplementedError(
                f"Function ({func.__name__}) was deprecated on {date}. Reason: {reason}"
            )

        return wrapper

    return decorator


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def get_statistics(alist: list):
    """Get summary statistics of a list"""
    iat = dict()

    if len(alist) > 1:
        iat["total"] = sum(alist)
        iat["max"] = max(alist)
        iat["min"] = min(alist)
        iat["mean"] = np.mean(alist)
        iat["std"] = np.sqrt(np.var(alist))
    else:
        iat["total"] = 0
        iat["max"] = 0
        iat["min"] = 0
        iat["mean"] = 0
        iat["std"] = 0

    return iat

import json
import logging
import os
import random
from datetime import datetime
from typing import Any, List, Union

import numpy as np
import pytz

FLAGS_TO_VAL = {
    "FIN": 0x01,
    "SYN": 0x02,
    "RST": 0x04,
    "PSH": 0x08,
    "ACK": 0x10,
    "URG": 0x20,
    "ECE": 0x40,
    "CWR": 0x80,
}


def pretty_print(dicto: dict, nesting_spacing=2) -> str:
    assert nesting_spacing >= 2, "Provice spacing of atleast 2 to pretty_print"
    pretty_str = ""
    for k, v in dicto.items():
        pretty_str += f"{k} : "
        if isinstance(v, dict):
            pretty_str += "\n" + pretty_print(v, nesting_spacing + 2)
        else:
            pretty_str += f"{v}\n"
    return pretty_str


def to_canadian(tstamp: float) -> str:
    # Convert to UTC-3
    utc_dt = datetime.utcfromtimestamp(tstamp)
    # Define the target timezone
    target_timezone = pytz.timezone("America/Belem")
    # Localize the naive datetime object to the target timezone
    localized_dt = pytz.utc.localize(utc_dt).astimezone(target_timezone)

    ## Return in HH:MM:SS format
    return localized_dt.strftime("%H:%M:%S")


def keychain_retrieve(nested_dict, keys) -> Union[None, Any]:
    current_dict_or_finval = nested_dict
    counter = 1
    for key in keys:
        if isinstance(current_dict_or_finval, dict):
            if key in current_dict_or_finval:
                current_dict_or_finval = current_dict_or_finval[key]
            else:
                current_dict_or_finval = None
        else:
            return current_dict_or_finval
        counter += 1
    return current_dict_or_finval


def get_keys_of_interest(source_dict: dict, keys_of_interest: List[List[str]]) -> dict:
    target_dict = {}
    for kc in keys_of_interest:
        val = keychain_retrieve(source_dict, kc)
        if val != None:
            target_dict[".".join(kc)] = val
    return target_dict


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")

class ColoredFormatter(logging.Formatter):
    def __init__(self, fmt, datefmt=None):
        super().__init__(fmt, datefmt)
        self.colors = {
            'DEBUG': '\033[94m',  # Blue
            'INFO': '\033[92m',   # Green
            'WARNING': '\033[93m', # Yellow
            'ERROR': '\033[91m',  # Red
            'CRITICAL': '\033[41m'  # Red background
        }

    def format(self, record: logging.LogRecord) -> str:
        color = self.colors.get(record.levelname, '\033[0m')
        # Format the prefix including date-time and log level
        prefix = f"{self.formatTime(record, self.datefmt)} {record.levelname}"
        colored_prefix = f"{color}{prefix}\033[0m"
        message = super().format(record)
        find_record = message.find(record.levelname)
        left_over = message[find_record+len(record.levelname):]
        message = colored_prefix + left_over

        # Replace the entire prefix with the colored version
        return message
def setup_logger(
    logger_name: str, logging_level=logging.INFO, overwrite=True
):
    """
    Helper function for setting up logger both in stdout and file
    """
    # Measures necessary to take for us to be able to do multiprocess logging
    localpid = os.getpid()
    logger_name_local = f"pid({localpid})" + logger_name

    # logger_name = "SUPADEBUG"
    logger = logging.getLogger(logger_name_local)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging_level)

    # create file handler which logs remaining debug messages
    current_cwd = os.getcwd()
    log_dir = os.path.join(
        current_cwd,
        "logs/",
    )
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"{logger_name_local}.log")
    mode = "w" if overwrite else "a"
    fh = logging.FileHandler(log_file_path, mode=mode)
    fh.setLevel(logging.DEBUG)

    formatter = ColoredFormatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger


def clamp(val: float, max_val: float, min_val: float):
    return max(min(val, max_val), min_val)


def within(val: float, min_val: float, max_val: float):
    return all([val <= max_val, val >= min_val])


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


def epoch_to_clean(epoch: float):
    return (
        datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M:%S") if epoch else "None"
    )


def set_all_seeds(seed):
    """
    Set the seed for all possible random sources to ensure reproducible results.

    Args:
    seed (int): The seed value to use for all random number generators.
    """
    # Python's built-in random module
    random.seed(seed)

    # NumPy's random module
    np.random.seed(seed)

    # For Python 3.7 and later, individual numpy generator
    rng = np.random.default_rng(seed)

    # TensorFlow (not using atm)
    # tf.random.set_seed(seed)

    # PyTorch (if you're using it)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    except ImportError:
        pass  # PyTorch is not installed
    # For any other libraries that you might be using, set their random seed here
    # Environment variables for some systems that use randomness
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_statistics(alist: list):
    """Get summary statistics of a list"""
    iat = {"total": 0, "max": 0, "min": 0, "mean": 0, "std": 0}

    # TODO: Feeling a bit funky about just setting things to 0 if the list is empty
    if len(alist) >= 1:
        iat["total"] = sum(alist)
        iat["max"] = max(alist)
        iat["min"] = min(alist)
        iat["mean"] = np.mean(alist)
    if len(alist) > 1:
        iat["std"] = np.sqrt(np.var(alist))

    return iat

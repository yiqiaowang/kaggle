import datetime
import pickle
from os import path as os_path

def get_timestamp():
    return datetime.datetime.now().strftime("%s")

def save_as_bin(path, arr):
    pickle.dump(arr, open(path, 'wb'))

def load_from_bin(path):
    return pickle.load(open(path, 'rb'))

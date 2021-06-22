import json
import pickle


def json_load(file_path):
    with open(file_path, 'r') as input_file:
        result = json.load(input_file)
    return result


def json_dump(obj, file_path):
    with open(file_path, 'w') as output_file:
        json.dump(obj, output_file)


def pickle_load(file_path):
    with open(file_path, 'rb') as input_file:
        result = pickle.load(input_file)
    return result


def pickle_dump(obj, file_path):
    with open(file_path, 'wb') as output_file:
        pickle.dump(obj, output_file)


def print_verbose(verbose):
    def f(msg):
        if verbose:
            print(msg, end='')
    return f
import numpy as np
import pandas as pd


def read_file(file_name, L2I):
    data_file = open(file_name, 'r')
    lines = data_file.readlines()
    parsed_lines = [line.split('\t') for line in lines[1:]]
    data = [((parsed_line[5], parsed_line[6]), parsed_line[0]) for parsed_line in parsed_lines if parsed_line[0] in L2I]
    return data

def read_glove(file_name):
    data = pd.read_csv(file_name, sep=' ', header=None).get_values()

    key2word = np.array([row[0] for row in data])
    W2I = {w: i for i, w in enumerate(key2word)}


    vecs = [row[1:].astype(np.float32) for row in data]
    return W2I, vecs

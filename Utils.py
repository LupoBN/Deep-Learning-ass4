import numpy as np


def read_file(file_name, L2I):
    data_file = open(file_name, 'r')
    lines = data_file.readlines()
    parsed_lines = [line.split('\t') for line in lines[1:]]
    data = [((parsed_line[5], parsed_line[6]), parsed_line[0]) for parsed_line in parsed_lines if parsed_line[0] in L2I]
    return data

def read_glove(file_name, words):
    glove_file = open(file_name, 'r')
    line = glove_file.readline()
    vecs = list()
    W2I = dict()
    while True:
        line = line.strip('\n')
        parsed_line = line.split(' ')
        if parsed_line[0] in words or parsed_line[0] == "UNK":
            W2I[parsed_line[0]] = len(W2I)
            vecs.append(np.array(parsed_line[1:]).astype(np.float32))
        line = glove_file.readline()
        if line == '':
            break
    if "UNK" not in W2I:
        print "No UNK on GLOVE, therefore Adding unk as random vector"
        W2I["UNK"] = len(W2I)
        vecs.append(np.random.rand(300))

    return W2I, np.array(vecs)

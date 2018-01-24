from collections import Counter

from Utils import *
from Models import EntailmentClassifier
from Helpers import train
import dynet as dy
import sys

EMBEDDING = 300
LSTM_DIMS = []
MLP_DIMS = []

if __name__ == "__main__":
    lstm_dim_str = sys.argv[3].split(',')
    mlp_dim_str = sys.argv[4].split(',')
    for dim in lstm_dim_str:
        LSTM_DIMS.append(int(dim))
    for dim in mlp_dim_str:
        MLP_DIMS.append(int(dim))
    MLP_DIMS.append(3)

    results_file = open(sys.argv[1], 'w')
    if '-m' in sys.argv:
        results_file.write("MLNI Dataset\n")
    else:
        results_file.write("SLNI Dataset\n")

    results_file.write("Embedding Size:\t" + str(EMBEDDING) + "\n")
    results_file.write("LSTM Layers:\t" + str(LSTM_DIMS) + "\n")
    results_file.write("MLP Layers:\t" + str(MLP_DIMS) + "\n")
    if '-relu' in sys.argv:
        results_file.write("Relu Activation\n")
    else:
        results_file.write("Tanh Activation\n")


    results_file.close()
    dyparams = dy.DynetParams()
    dyparams.set_autobatch(True)

    L2I = {"contradiction": 0, "neutral": 1, "entailment": 2}
    I2L = ["contradiction", "neutral", "entailment"]

    mtrain = read_file("Data/MLNI/multinli_1.0_train.txt", L2I)
    strain = read_file("Data/SNLI/snli_1.0_train.txt", L2I)
    word_counter = Counter()
    for example in strain:
        word_counter.update(example[0][0].split(' '))
        word_counter.update(example[0][1].split(' '))
    if '-m' in sys.argv:
        for example in mtrain:
            word_counter.update(example[0][0].split(' '))
            word_counter.update(example[0][1].split(' '))


    dev = read_file("Data/SNLI/snli_1.0_dev.txt", L2I)
    for example in dev:
        word_counter.update(example[0][0].split(' '))
        word_counter.update(example[0][1].split(' '))
    W2I, pre_lookup = read_glove("Glove.txt", word_counter)
    m = dy.ParameterCollection()


    classifier = EntailmentClassifier(EMBEDDING, LSTM_DIMS, W2I, m, pre_lookup, MLP_DIMS, L2I, I2L)
    trainer = dy.AdamTrainer(m)

    train_results, dev_results = train(strain, mtrain, dev, 50, trainer, classifier, sys.argv[2], sys.argv[1],
                                       batch_size=32)

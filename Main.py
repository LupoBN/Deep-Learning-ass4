import random

from Utils import *
from Models import EntailmentClassifier
from Helpers import train
import dynet as dy

EMBEDDING = 300
LSTM_DIMS = [512, 1024, 2048]
MLP_DIMS = [1600, 1600, 3]

if __name__ == "__main__":
    L2I = {"contradiction": 0, "neutral": 1, "entailment": 2}
    I2L = ["contradiction", "neutral", "entailment"]
    mtrain = read_file("Data/MLNI/multinli_1.0_train.txt", L2I)
    strain = read_file("Data/SNLI/snli_1.0_train.txt", L2I)
    dev = read_file("Data/SNLI/snli_1.0_dev.txt", L2I)
    W2I, pre_lookup = read_glove("Glove.txt")
    m = dy.ParameterCollection()

    classifier = EntailmentClassifier(EMBEDDING, LSTM_DIMS, W2I, m, pre_lookup, MLP_DIMS, L2I, I2L)
    trainer = dy.AdamTrainer(m)

    train_results, dev_results = train(strain, mtrain, dev, 50, trainer, classifier, "Model")


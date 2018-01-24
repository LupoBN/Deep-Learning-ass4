import dynet as dy
import numpy as np
import sys

class EntailmentClassifier(object):
    def __init__(self, embedding, lstm_dims, W2I, model, pre_lookup, mlp_dims, L2I, I2L):

        self._ssbilstm = ShortcutStackedBiLSTM(embedding, lstm_dims, W2I, model, pre_lookup)
        self._params = list()
        self._model = model
        self._I2L = I2L
        self._L2I = L2I

        last_dim = 2 * lstm_dims[-1]

        self._params.append(model.add_parameters((mlp_dims[0], last_dim + last_dim + last_dim + last_dim)))
        self._params.append(model.add_parameters(mlp_dims[0]))

        for i in range(1, len(mlp_dims)):
            self._params.append(model.add_parameters((mlp_dims[i], mlp_dims[i - 1])))
            self._params.append(model.add_parameters(mlp_dims[i]))

    def __call__(self, premise, hypothesis, dropout=0.0):
        precode = self._ssbilstm(premise)
        hypcode = self._ssbilstm(hypothesis)
        conc_pre_hyp = dy.concatenate([precode, hypcode])
        dist_pre_hyp = dy.abs(precode - hypcode)
        mult_pre_hyp = dy.cmult(precode, hypcode)
        x = dy.concatenate([conc_pre_hyp, dist_pre_hyp, mult_pre_hyp])
        expr = list()
        for exp in self._params:
            expr.append(dy.parameter(exp))
        for i in range(0, len(expr) - 2, 2):
            if '-relu' in sys.argv:
                x = dy.rectify((expr[i] * x) + expr[i + 1])
            else:
                x = dy.tanh((expr[i] * x) + expr[i + 1])

            if dropout != 0.0:
                x = dy.dropout(x, dropout)

        output = dy.softmax((expr[-2] * x) + expr[-1])
        return output

    def forward(self, premise, hypothesis, relation, dropout=0.0):

        out = self(premise.split(' '), hypothesis.split(' '), dropout=dropout)
        prediction = self._I2L[np.argmax(out.npvalue())]
        loss = -dy.log(dy.pick(out, self._L2I[relation]))
        return prediction, loss

    def predict(self, premise, hypothesis):
        out = self(premise, hypothesis)
        soft_out = dy.softmax(out)
        prediction = self._I2L[np.argmax(soft_out.npvalue())]
        return prediction

    def save_model(self, model_file):
        self._model.save(model_file)

    def load_model(self, model_file):
        self._model.populate(model_file)


class ShortcutStackedBiLSTM(object):
    def __init__(self, embedding_size, lstm_dims, W2I, model, pre_lookup):
        self._model = model
        self._stacks = list()
        self._E = model.add_lookup_parameters((len(W2I), embedding_size))
        self._E.init_from_array(pre_lookup)
        self._W2I = W2I
        self._stacks.append(BiLSTM(embedding_size, lstm_dims[0], model))

        current_dim = embedding_size + lstm_dims[0] * 2
        for i in range(1, len(lstm_dims)):
            self._stacks.append(BiLSTM(current_dim, lstm_dims[i], model))
            current_dim += lstm_dims[i] * 2

    def __call__(self, sequence):
        next_input = [dy.lookup(self._E, self._W2I[i]) if i in self._W2I else dy.lookup(self._E, self._W2I["UNK"])
                for i in sequence]
        for layer in self._stacks[0:-1]:
            output = layer(next_input)
            next_input = [dy.concatenate([next_input[i], output[i]]) for i in range(len(sequence))]
        output = layer(next_input)
        exp_output = dy.concatenate_cols(output)
        v = dy.kmax_pooling(exp_output, 1, d=1)

        return v


class BiLSTM(object):
    def __init__(self, in_dim, lstm_dim, model):
        self._model = model
        # the embedding paramaters

        self._fwd_RNN_first = dy.VanillaLSTMBuilder(1, in_dim, lstm_dim, model)
        self._bwd_RNN_first = dy.VanillaLSTMBuilder(1, in_dim, lstm_dim, model)

    def __call__(self, vecs):
        fwd = self._fwd_RNN_first.initial_state()
        bwd = self._bwd_RNN_first.initial_state()

        rnn_fwd = fwd.transduce(vecs)
        rnn_bwd = (bwd.transduce(vecs[::-1]))[::-1]

        output = [dy.concatenate([fwd_out, bwd_out]) for fwd_out, bwd_out in zip(rnn_fwd, rnn_bwd)]

        return output

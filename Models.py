import dynet as dy
import numpy as np


class EntailmentClassifier(object):
    def __init__(self, embedding, lstm_dims, W2I, model, pre_lookup, mlp_dims, I2L, L2I):
        self._ssbilstm = ShortcutStackedBiLSTM(embedding, lstm_dims, W2I, model, pre_lookup)
        self._params = list()
        self._model = model
        self._I2L = I2L
        self._L2I = L2I
        self._params.append(model.add_parameters((mlp_dims[0], 2 * lstm_dims[-1])))
        self._params.append(model.add_parameters(mlp_dims[0]))

        for i in range(1, len(mlp_dims)):
            self._params.append(model.add_parameters((mlp_dims[i - 1], mlp_dims[i])))
            self._params.append(model.add_parameters(mlp_dims[i]))
        pass

    def __call__(self, premise, hypothesis):
        precode = self._ssbilstm(premise)
        hypcode = self._ssbilstm(hypothesis)
        conc_pre_hyp = dy.concatenate(precode, hypothesis)
        dist_pre_hyp = precode - hypcode
        mult_pre_hyp = precode * hypcode
        x = dy.concatenate([conc_pre_hyp, dist_pre_hyp, mult_pre_hyp])
        expr = list()
        for exp in self._params:
            expr.append(dy.parameter(exp))
        for i in range(0, len(exp) - 2, 2):
            x = dy.dropout(dy.rectify((expr[i] * x) + expr[i]), 0.1)
        output = dy.softmax((exp[i] * x) + expr[i + 1])
        return output

    def forward(self, premise, hypothesis, relation):
        out = self(premise, hypothesis)
        prediction = np.argmax(self._I2L[out.npvalue()])
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
        self._E = model.lookup_parameters_from_numpy(pre_lookup)
        self._W2I = W2I
        self._stacks.append(BiLSTM(embedding_size, lstm_dims[0], model))
        current_dim = embedding_size + lstm_dims[0] * 2
        for i in range(2, len(lstm_dims)):
            self._stacks.append(BiLSTM(current_dim, lstm_dims[i], model))
            current_dim += lstm_dims[i] * 2

    """
    def _max_pool(self, vecs):
        return skimage.measure.block_reduce(vecs, (1, vecs.shape[0]), np.max).T
    """

    def __call__(self, sequence):
        vecs = [self._E[self._W2I[i]] if i in self._W2I else self._E[self._W2I["UUUNKKK"]]
                for i in sequence]
        next_input = vecs
        for layer in self._stacks:
            output = layer(next_input)
            next_input = dy.concatenate(next_input, output)
        output = dy.transpose(output)
        v = dy.maxpooling2d(output, [1, len(sequence)], [1, 1])
        # v = self._max_pool(vecs)
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

        output = dy.vecInput([dy.concatenate([fwd_out, bwd_out]) for fwd_out, bwd_out in zip(rnn_fwd, rnn_bwd)])

        return output

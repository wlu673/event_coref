import theano, theano.tensor as T, theano.tensor.shared_randomstreams
from collections import OrderedDict
import numpy as np


###########################################################################
# Optimization Functions

def adadelta(inputs, cost, names, parameters, gradients, lr, norm_lim, rho=0.95, eps=1e-6):
    zipped_grads = [theano.shared(p.get_value() * np.float32(0.), name='%s_grad' % k)
                    for k, p in zip(names, parameters)]
    running_up2 = [theano.shared(p.get_value() * np.float32(0.), name='%s_rup2' % k)
                   for k, p in zip(names, parameters)]
    running_grads2 = [theano.shared(p.get_value() * np.float32(0.), name='%s_rgrad2' % k)
                      for k, p in zip(names, parameters)]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, gradients)]
    rg2up = [(rg2, rho * rg2 + (1. - rho) * (g ** 2)) for rg2, g in zip(running_grads2, gradients)]
    f_grad_shared = theano.function(inputs, cost, updates=zgup + rg2up, on_unused_input='ignore')

    updir = [-T.sqrt(ru2 + eps) / T.sqrt(rg2 + eps) * zg
             for zg, ru2, rg2 in zip(zipped_grads, running_up2, running_grads2)]
    ru2up = [(ru2, rho * ru2 + (1. - rho) * (ud ** 2)) for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(parameters, updir)]

    if norm_lim > 0:
        param_up = clip_gradient(param_up, norm_lim, names)

    f_update_param = theano.function([lr], [], updates=ru2up + param_up, on_unused_input='ignore')

    return f_grad_shared, f_update_param


def sgd(ips, cost, names, parameters, gradients, lr, norm_lim):
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k) for k, p in zip(names, parameters)]
    gsup = [(gs, g) for gs, g in zip(gshared, gradients)]

    f_grad_shared = theano.function(ips, cost, updates=gsup, on_unused_input='ignore')

    pup = [(p, p - lr * g) for p, g in zip(parameters, gshared)]

    if norm_lim > 0:
        pup = clip_gradient(pup, norm_lim, names)

    f_param_update = theano.function([lr], [], updates=pup, on_unused_input='ignore')

    return f_grad_shared, f_param_update


def clip_gradient(updates, norm, names):
    id = -1
    res = []
    for p, g in updates:
        id += 1
        if not names[id].startswith('word') and 'multi' not in names[id] and p.get_value(borrow=True).ndim == 2:
            col_norms = T.sqrt(T.sum(T.sqr(g), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(norm))
            scale = desired_norms / (1e-7 + col_norms)
            g = g * scale

        res += [(p, g)]
    return res


###########################################################################
# Nonconsecutive CNN

def non_consecutive_cnn(model):
    X = get_concatenation(model.container['embeddings'],
                          model.container['vars'],
                          model.args['features'],
                          transpose=False)

    rep_cnn = non_consecutive_cnn_driver(X,
                                         model.args['cnn_filter_num'],
                                         model.args['cnn_filter_wins'],
                                         model.args['batch'] * model.args['max_inst_in_doc'],
                                         model.args['window'],
                                         model.container['fea_dim'],
                                         'non_consecutive_cnn',
                                         model.container['params'],
                                         model.container['names'])

    dim_cnn = model.args['cnn_filter_num'] * len(model.args['cnn_filter_wins'])

    return rep_cnn, dim_cnn


def get_concatenation(embeddings, vars, features, transpose=False):
    reps = []

    for fea in features:
        if features[fea] == 0:
            var = vars[fea] if not transpose else vars[fea].T
            reps += [embeddings[fea][var]]
        elif features[fea] == 1:
            if not transpose:
                reps += [vars[fea]]
            else:
                reps += [vars[fea].dimshuffle(1, 0, 2)]

    if len(reps) == 1:
        X = reps[0]
    else:
        X = T.cast(T.concatenate(reps, axis=2), dtype=theano.config.floatX)
    return X


def non_consecutive_cnn_driver(inputs, filter_num, filter_wins, batch, length, dim, prefix, params, names):
    X = inputs.dimshuffle(1, 0, 2)
    reps = []
    for win in filter_wins:
        Ws, b = prepare_params(win, filter_num, length, dim, prefix, params, names)
        rep_one = eval('non_consecutive_cnn_layer' + str(win))(X, filter_num, batch, Ws, b)
        reps += [rep_one]

    rep_cnn = T.cast(T.concatenate(reps, axis=1), dtype=theano.config.floatX)
    return rep_cnn


def prepare_params(window, filter_num, length, dim, prefix, params, names):
    fan_in = window * dim
    fan_out = filter_num * window * dim / length  # (length - window + 1)
    bound = np.sqrt(6. / (fan_in + fan_out))

    Ws = []
    for i in range(window):
        W = theano.shared(np.random.uniform(low=-bound,
                                            high=bound,
                                            size=(dim, filter_num)).astype(theano.config.floatX))
        Ws += [W]
        params += [W]
        names += [prefix + '_W_win' + str(window) + '_' + str(i)]

    b = theano.shared(np.zeros(filter_num, dtype=theano.config.floatX))
    params += [b]
    names += [prefix + '_b_win_' + str(window)]

    return Ws, b


def non_consecutive_cnn_layer2(inputs, filter_num, batch, Ws, b):
    def recurrence(_x, i_m1, i_m2):
        ati = T.dot(_x, Ws[0])
        _m1 = T.maximum(i_m1, ati)
        ati = i_m1 + T.dot(_x, Ws[1])
        _m2 = T.maximum(i_m2, ati)
        return [_m1, _m2]

    ret, _ = theano.scan(fn=recurrence,
                         sequences=[inputs],
                         outputs_info=[T.alloc(0., batch, filter_num), T.alloc(0., batch, filter_num)],
                         n_steps=inputs.shape[0])

    rep = T.tanh(ret[1][-1] + b[np.newaxis, :])
    return rep


def non_consecutive_cnn_layer3(inputs, filter_num, batch, Ws, b):
    def recurrence(_x, i_m1, i_m2, i_m3):
        ati = T.dot(_x, Ws[0])
        _m1 = T.maximum(i_m1, ati)
        ati = i_m1 + T.dot(_x, Ws[1])
        _m2 = T.maximum(i_m2, ati)
        ati = i_m2 + T.dot(_x, Ws[2])
        _m3 = T.maximum(i_m3, ati)
        return [_m1, _m2, _m3]

    ret, _ = theano.scan(fn=recurrence,
                         sequences=[inputs],
                         outputs_info=[T.alloc(0., batch, filter_num), T.alloc(0., batch, filter_num),
                                       T.alloc(0., batch, filter_num)],
                         n_steps=inputs.shape[0])

    rep = T.tanh(ret[2][-1] + b[np.newaxis, :])
    return rep


def non_consecutive_cnn_layer4(inputs, filter_num, batch, Ws, b):
    def recurrence(_x, i_m1, i_m2, i_m3, i_m4):
        ati = T.dot(_x, Ws[0])
        _m1 = T.maximum(i_m1, ati)
        ati = i_m1 + T.dot(_x, Ws[1])
        _m2 = T.maximum(i_m2, ati)
        ati = i_m2 + T.dot(_x, Ws[2])
        _m3 = T.maximum(i_m3, ati)
        ati = i_m3 + T.dot(_x, Ws[3])
        _m4 = T.maximum(i_m4, ati)
        return [_m1, _m2, _m3, _m4]

    ret, _ = theano.scan(fn=recurrence,
                         sequences=[inputs],
                         outputs_info=[T.alloc(0., batch, filter_num), T.alloc(0., batch, filter_num),
                                       T.alloc(0., batch, filter_num), T.alloc(0., batch, filter_num)],
                         n_steps=inputs.shape[0])

    rep = T.tanh(ret[3][-1] + b[np.newaxis, :])
    return rep


def non_consecutive_cnn_layer5(inputs, filter_num, batch, Ws, b):
    def recurrence(_x, i_m1, i_m2, i_m3, i_m4, i_m5):
        ati = T.dot(_x, Ws[0])
        _m1 = T.maximum(i_m1, ati)
        ati = i_m1 + T.dot(_x, Ws[1])
        _m2 = T.maximum(i_m2, ati)
        ati = i_m2 + T.dot(_x, Ws[2])
        _m3 = T.maximum(i_m3, ati)
        ati = i_m3 + T.dot(_x, Ws[3])
        _m4 = T.maximum(i_m4, ati)
        ati = i_m4 + T.dot(_x, Ws[4])
        _m5 = T.maximum(i_m5, ati)
        return [_m1, _m2, _m3, _m4, _m5]

    ret, _ = theano.scan(fn=recurrence,
                         sequences=[inputs],
                         outputs_info=[T.alloc(0., batch, filter_num), T.alloc(0., batch, filter_num),
                                       T.alloc(0., batch, filter_num), T.alloc(0., batch, filter_num),
                                       T.alloc(0., batch, filter_num)],
                         n_steps=inputs.shape[0])

    rep = T.tanh(ret[4][-1] + b[np.newaxis, :])
    return rep


###########################################################################
# Multi-Hidden Layer NN

def multi_hidden_layers(inputs, dim_hids, params, names, prefix):
    hidden_vector = inputs
    index = 0
    for dim_in, dim_out in zip(dim_hids, dim_hids[1:]):
        index += 1
        hidden_vector = hidden_layer(hidden_vector, dim_in, dim_out, params, names, prefix + '_layer' + str(index),)
    return hidden_vector


def hidden_layer(inputs, dim_in, dim_out, params, names, prefix):
    bound = np.sqrt(6. / (dim_in + dim_out))
    W = theano.shared(np.random.uniform(low=-bound, high=bound, size=(dim_in, dim_out)).astype(theano.config.floatX))
    b = theano.shared(np.zeros(dim_out, dtype=theano.config.floatX))
    res = []
    for x in inputs:
        out = T.nnet.sigmoid(T.dot(x, W) + b)
        res += [out]

    params += [W, b]
    names += [prefix + '_W', prefix + '_b']

    return res


###########################################################################
# RNN

def rnn_gru(inputs, dim_in, dim_hidden, mask, prefix, params, names):
    Uc = theano.shared(np.concatenate([ortho_weight(dim_hidden), ortho_weight(dim_hidden)], axis=1))
    Wc = theano.shared(np.concatenate([random_matrix(dim_in, dim_hidden), random_matrix(dim_in, dim_hidden)], axis=1))
    bc = theano.shared(np.zeros(2 * dim_hidden, dtype=theano.config.floatX))

    Ux = theano.shared(ortho_weight(dim_hidden))
    Wx = theano.shared(random_matrix(dim_in, dim_hidden))
    bx = theano.shared(np.zeros(dim_hidden, dtype=theano.config.floatX))

    params += [Wc, bc, Uc, Wx, Ux, bx]
    names += [prefix + '_Wc', prefix + '_bc', prefix + '_Uc', prefix + '_Wx', prefix + '_Ux', prefix + '_bx']

    def recurrence(x_t, m, h_tm1):
        h_tm1 = m * h_tm1
        preact = T.nnet.sigmoid(T.dot(h_tm1, Uc) + T.dot(x_t, Wc) + bc)

        r_t = _slice(preact, 0, dim_hidden)
        u_t = _slice(preact, 1, dim_hidden)

        h_t = T.tanh(T.dot(h_tm1, Ux) * r_t + T.dot(x_t, Wx) + bx)
        h_t = u_t * h_tm1 + (1. - u_t) * h_t

        return h_t

    h, _ = theano.scan(fn=recurrence,
                       sequences=[inputs, mask],
                       outputs_info=T.alloc(0., dim_hidden),
                       n_steps=inputs.shape[0])

    return h


def random_matrix(row, column, scale=0.2):
    # bound = np.sqrt(6. / (row + column))
    bound = 1.
    return scale * np.random.uniform(low=-bound, high=bound, size=(row, column)).astype(theano.config.floatX)


def ortho_weight(dim):
    W = np.random.randn(dim, dim)
    u, s, v = np.linalg.svd(W)
    return u.astype(theano.config.floatX)


def _slice(_x, n, dim):
    return _x[n*dim:(n+1)*dim]


###########################################################################
# Model Utilities

def trigger_contexts(model):
    wed_window = model.args['wed_window']
    extended_words = model.container['vars']['word']
    padding = T.zeros((extended_words.shape[0], wed_window), dtype='int32')
    extended_words = T.cast(T.concatenate([padding, extended_words, padding], axis=1), dtype='int32')

    def recurrence(words, position, emb):
        indices = words[position:(position + 2 * wed_window + 1)]
        rep = emb[indices].flatten()
        return [rep]

    rep_contexts, _ = theano.scan(fn=recurrence,
                                  sequences=[extended_words, model.container['anchor_position']],
                                  n_steps=extended_words.shape[0],
                                  non_sequences=[model.container['embeddings']['word']],
                                  outputs_info=[None])

    dim_contexts = (2 * wed_window + 1) * model.args['embeddings']['word'].shape[1]
    return rep_contexts, dim_contexts


def dropout_from_layer(rng, layers, p):
    srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))
    # p = 1-p because 1's indicate keep and p is prob of dropping
    res = []
    for layer in layers:
        mask = srng.binomial(n=1, p=1-p, size=layer.shape)
        # The cast is important because int * float32 = float64 which pulls things off the gpu
        output = layer * T.cast(mask, theano.config.floatX)
        res += [output]
    return res


###########################################################################


class MainModel(object):
    def __init__(self, args):
        self.args = args
        self.args['rng'] = np.random.RandomState(3435)
        self.args['dropout'] = args['dropout'] if args['dropout'] > 0. else 0.

        self.container = {}
        self.prepare_features()
        self.define_vars()

        rep_cnn, dim_cnn = self.get_cnn_rep()
        local_score = self.get_local_score(rep_cnn, dim_cnn)

        # self.f_grad_shared, self.f_update_param = self.build_train(rep_cnn, dim_cnn, local_score)
        # self.container['set_zero'] = OrderedDict()
        # self.container['zero_vecs'] = OrderedDict()
        # for ed in self.container['embeddings']:
        #     self.container['zero_vecs'][ed] = np.zeros(self.args['embeddings'][ed].shape[1], dtype='float32')
        #     self.container['set_zero'][ed] = \
        #         theano.function([self.container['zero_vector']],
        #                         updates=[(self.container['embeddings'][ed],
        #                                   T.set_subtensor(self.container['embeddings'][ed][0, :],
        #                                                   self.container['zero_vector']))])

        X = self.build_test(rep_cnn, local_score)

        inputs = [self.container['vars'][ed] for ed in self.args['features'] if self.args['features'][ed] >= 0]
        inputs += [self.container['anchor_position'],
                   self.container['prev_inst'],
                   self.container['test_current_hv'],
                   self.container['test_prev_inst_cluster'],
                   self.container['test_current_cluster']]

        self.F = theano.function(inputs, X, on_unused_input='warn')

    def prepare_features(self, header_width = 60):
        self.container['fea_dim'] = 0
        self.container['params'], self.container['names'] = [], []
        self.container['embeddings'], self.container['vars'] = OrderedDict(), OrderedDict()

        print 'Features'.center(header_width, '-')
        print 'Will update embeddings' if self.args['update_embs'] else 'Will not update embeddings'
        for fea in self.args['features']:
            if self.args['features'][fea] == 0:
                self.container['embeddings'][fea] = theano.shared(self.args['embeddings'][fea].astype(theano.config.floatX))

                if self.args['update_embs']:
                    self.container['params'] += [self.container['embeddings'][fea]]
                    self.container['names'] += [fea]

            if self.args['features'][fea] >= 0:
                dim_added = self.args['features_dim'][fea]
                self.container['fea_dim'] += dim_added
                self.container['vars'][fea] = T.imatrix() if self.args['features'][fea] == 0 else T.tensor3()
                print 'Using feature \'%s\' with dimension %d' % (fea, dim_added)

        print 'Total feature dimension:', self.container['fea_dim']
        print '-' * header_width

    def define_vars(self):
        self.container['cluster'] = T.ivector('cluster')
        self.container['mask_rnn'] = T.vector('mask_rnn')
        self.container['current_hv'] = T.imatrix('current_hv')
        self.container['prev_inst'] = T.imatrix('prev_inst')
        self.container['prev_inst_cluster'] = T.imatrix('prev_inst_cluster')
        self.container['prev_inst_cluster_gold'] = T.imatrix('prev_inst_cluster_gold')
        self.container['alpha'] = T.matrix('alpha')
        self.container['mask_cluster'] = T.vector('mask_cluster')
        self.container['anchor_position'] = T.ivector('anchor_position')
        self.container['lr'] = T.scalar('lr')
        self.container['zero_vector'] = T.vector('zero_vector')

        self.container['test_current_hv'] = T.tensor3('test_current_hv')
        self.container['test_prev_inst_cluster'] = T.imatrix('test_prev_inst_cluster')
        self.container['test_current_cluster'] = T.ivector('test_current_vector')

    def get_cnn_rep(self):
        rep_inter, dim_inter = non_consecutive_cnn(self)

        if self.args['wed_window'] > 0:
            rep_contexts, dim_contexts = trigger_contexts(self)
            rep_inter = T.concatenate([rep_inter, rep_contexts], axis=1)
            dim_inter += dim_contexts

        if self.args['dropout'] > 0:
            rep_inter = dropout_from_layer(self.args['rng'], [rep_inter], self.args['dropout'])[0]

        dim_hids = [dim_inter] + self.args['multilayer_nn']

        rep_cnn = multi_hidden_layers([rep_inter],
                                      dim_hids,
                                      self.container['params'],
                                      self.container['names'],
                                      'main_multi_nn')[0]
        dim_cnn = dim_hids[-1]

        return rep_cnn, dim_cnn

    def get_local_score(self, rep_cnn, dim_cnn):
        v = theano.shared(np.zeros([1, dim_cnn]).astype(theano.config.floatX))
        self.container['params'] += [v]
        self.container['names'] += ['v']
        padded = T.concatenate([v, rep_cnn, T.alloc(0., 1, dim_cnn)])
        prev_inst = padded[self.container['prev_inst']]
        local_score = T.batched_dot(rep_cnn, prev_inst.dimshuffle(0, 2, 1))
        return local_score

    def build_train(self, rep_cnn, dim_cnn, local_score):
        total_score = local_score + self.get_global_score(rep_cnn, dim_cnn)
        latent_score, alpha = self.get_latent(total_score)
        score = T.max(alpha * (1 + total_score - latent_score), axis=1)
        cost = T.sum(T.set_subtensor(score[T.nonzero(T.eq(score, -np.inf))], 0.))
        gradients = T.grad(cost, self.container['params'])

        inputs = [self.container['vars'][ed] for ed in self.args['features'] if self.args['features'][ed] >= 0]
        inputs += [self.container['anchor_position'],
                   self.container['prev_inst'],
                   self.container['cluster'],
                   self.container['mask_rnn'],
                   self.container['current_hv'],
                   self.container['prev_inst_cluster'],
                   self.container['prev_inst_cluster_gold'],
                   self.container['alpha'],
                   self.container['mask_cluster']]

        f_grad_shared, f_update_param = eval(self.args['optimizer'])(inputs,
                                                                     cost,
                                                                     self.container['names'],
                                                                     self.container['params'],
                                                                     gradients,
                                                                     self.container['lr'],
                                                                     self.args['norm_lim'])

        return f_grad_shared, f_update_param

    def get_global_score(self, rep_cnn, dim_cnn):
        padded = T.concatenate([rep_cnn, T.alloc(0., 1, dim_cnn)])
        X = padded[self.container['cluster']]
        rep_rnn = rnn_gru(X,
                          dim_cnn,
                          dim_cnn,
                          self.container['mask_rnn'],
                          'main_rnn',
                          self.container['params'],
                          self.container['names'])

        rep_rnn = T.concatenate([rep_rnn, T.alloc(0., 1, dim_cnn)])
        current_hv = rep_rnn[self.container['current_hv']]

        score_by_cluster = T.batched_dot(rep_cnn, current_hv.dimshuffle(0, 2, 1))
        score_nonana = T.batched_dot(rep_cnn, T.sum(current_hv, axis=1))
        score_by_cluster = T.concatenate([T.reshape(score_nonana, [self.args['batch'] * self.args['max_inst_in_doc'], 1]),
                                          score_by_cluster,
                                          T.alloc(0., self.args['batch'] * self.args['max_inst_in_doc'], 1)], axis=1)

        row_indices = np.array([[i] * self.args['max_inst_in_doc']
                                for i in np.arange(self.args['batch'] * self.args['max_inst_in_doc'])], dtype='int32')
        global_score = score_by_cluster[row_indices, self.container['prev_inst_cluster']]
        return global_score

    def get_latent(self, score):
        padded = T.concatenate([score, T.alloc(-np.inf, self.args['batch'] * self.args['max_inst_in_doc'], 1)], axis=1)
        row_indices = np.array([[i] * self.args['max_inst_in_doc']
                                for i in np.arange(self.args['batch'] * self.args['max_inst_in_doc'])], dtype='int32')
        ante_score = padded[row_indices, self.container['prev_inst_cluster_gold']]
        latent_score = T.max(ante_score, axis=1)
        latent_score = T.set_subtensor(latent_score[T.nonzero(T.eq(latent_score, -np.inf))], 0.)

        latent_inst = T.argmax(ante_score, axis=1)
        row_indices = np.array([i for i in np.arange(self.args['batch'] * self.args['max_inst_in_doc'])], dtype='int32')
        alpha = T.set_subtensor(self.container['alpha'][row_indices, latent_inst], self.container['mask_cluster'])

        return T.reshape(latent_score, [self.args['batch'] * self.args['max_inst_in_doc'], 1]), alpha

    def build_test(self, rep_cnn, local_score):
        def recurrence(curr_inst, current_hv, prev_inst_cluster, current_cluster):
            curr_indices = np.array([curr_inst + i * self.args['max_inst_in_doc']
                                     for i in np.arange(self.args['batch'])], dtype='int32')
            pic1 = T.set_subtensor(prev_inst_cluster[:, curr_inst], np.array([0] * self.args['batch'], dtype='int32'))

            curr_rep_cnn = rep_cnn[curr_indices]
            score_by_cluster = T.batched_dot(curr_rep_cnn, current_hv.transpose((0, 2, 1)))
            score_nonana = T.batched_dot(curr_rep_cnn, T.sum(current_hv, axis=1))
            score_by_cluster = T.concatenate([T.reshape(score_nonana, [self.args['batch'], 1]),
                                              score_by_cluster,
                                              T.alloc(-np.inf, self.args['batch'], 1)], axis=1)

            indices = np.array([[i] * self.args['max_inst_in_doc'] for i in np.arange(self.args['batch'])],
                               dtype='int32')
            global_score = score_by_cluster[indices, pic1]
            score = local_score[curr_indices] + global_score

            indices_single = np.array([i for i in np.arange(self.args['batch'])], dtype='int32')
            ante_cluster_raw = pic1[indices_single, T.argmax(score, axis=1)]
            indices_new_cluster = T.nonzero(T.eq(ante_cluster_raw, 0))
            ante_cluster = T.set_subtensor(ante_cluster_raw[indices_new_cluster], current_cluster[indices_new_cluster])
            cc1 = T.set_subtensor(current_cluster[indices_new_cluster], current_cluster[indices_new_cluster] + 1)
            pic2 = T.set_subtensor(prev_inst_cluster[:, curr_inst], ante_cluster)

            ante_hv = current_hv[indices_single, ante_cluster]

            return pic2

        return recurrence(0,
                          self.container['test_current_hv'],
                          self.container['test_prev_inst_cluster'],
                          self.container['test_current_cluster'])

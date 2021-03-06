import theano, theano.tensor as T, theano.tensor.shared_randomstreams
from collections import OrderedDict
import numpy as np
import cPickle


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
                          dim_num = 3,
                          transpose=False)

    rep_cnn = non_consecutive_cnn_driver(X,
                                         model.args['cnn_filter_num'],
                                         model.args['cnn_filter_wins'],
                                         model.args['batch'] * model.args['max_inst_in_doc'],
                                         model.args['window'],
                                         model.container['fea_dim'],
                                         'non_consecutive_cnn',
                                         model.container['params_local'],
                                         model.container['names_local'],
                                         model.args['kGivens'])

    dim_cnn = model.args['cnn_filter_num'] * len(model.args['cnn_filter_wins'])

    return rep_cnn, dim_cnn


def get_concatenation(embeddings, variables, features, dim_num=3, transpose=False):
    reps = []

    for fea in features:
        if features[fea] == 0:
            v = variables[fea]
            if not transpose:
                reps += [embeddings[fea][v]]
            else:
                reps += [embeddings[fea][v.T]] if dim_num == 3 else [embeddings[fea][v].dimshuffle(1, 0)]
        else:
            if not transpose:
                reps += [variables[fea]]
            else:
                reps += [variables[fea].dimshuffle(1, 0, 2)] if dim_num == 3 else [variables[fea].dimshuffle(1, 0)]

    if len(reps) == 1:
        X = reps[0]
    else:
        axis = 2 if dim_num == 3 else 1
        X = T.cast(T.concatenate(reps, axis=axis), dtype=theano.config.floatX)
    return X


def non_consecutive_cnn_driver(inputs, filter_num, filter_wins, batch, length, dim, prefix, params, names, kGivens):
    X = inputs.dimshuffle(1, 0, 2)
    reps = []
    for win in filter_wins:
        Ws, b = prepare_params(win, filter_num, length, dim, prefix, params, names, kGivens)
        rep_one = eval('non_consecutive_cnn_layer' + str(win))(X, filter_num, batch, Ws, b)
        reps += [rep_one]

    rep_cnn = T.cast(T.concatenate(reps, axis=1), dtype=theano.config.floatX)
    return rep_cnn


def prepare_params(window, filter_num, length, dim, prefix, params, names, kGivens):
    fan_in = window * dim
    fan_out = filter_num * window * dim / length  # (length - window + 1)
    bound = np.sqrt(6. / (fan_in + fan_out))

    Ws = []
    for i in range(window):
        name_W = prefix + '_W_win' + str(window) + '_' + str(i)
        W = create_shared(np.random.uniform(low=-bound,
                                            high=bound,
                                            size=(dim, filter_num)).astype(theano.config.floatX),
                          kGivens,
                          name_W)
        Ws += [W]
        params += [W]
        names += [name_W]

    name_b = prefix + '_b_win_' + str(window)
    b = create_shared(np.zeros(filter_num, dtype=theano.config.floatX), kGivens, name_b)
    params += [b]
    names += [name_b]

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

def multi_hidden_layers(inputs, dim_hids, params, names, prefix, kGivens):
    hidden_vector = inputs
    index = 0
    for dim_in, dim_out in zip(dim_hids, dim_hids[1:]):
        index += 1
        hidden_vector, _ = hidden_layer(hidden_vector, dim_in, dim_out, params, names, prefix + '_layer' + str(index), kGivens)
    return hidden_vector


def hidden_layer(inputs, dim_in, dim_out, params, names, prefix, kGivens, dropout=0.):
    bound = np.sqrt(6. / (dim_in + dim_out))
    W = create_shared(np.random.uniform(low=-bound, high=bound, size=(dim_in, dim_out)).astype(theano.config.floatX),
                      kGivens,
                      prefix + '_W')
    b = create_shared(np.zeros(dim_out, dtype=theano.config.floatX), kGivens, prefix + '_b')
    res = []
    for x in inputs:
        if dropout > 0.:
            out = T.nnet.sigmoid(T.dot(x, (1.0 - dropout) * W) + b)
        else:
            out = T.nnet.sigmoid(T.dot(x, W) + b)
        res += [out]

    params += [W, b]
    names += [prefix + '_W', prefix + '_b']

    return res, (W, b)


###########################################################################
# RNN

def rnn_gru(inputs, dim_in, dim_hidden, mask, prefix, params, names, kGivens):
    Uc = create_shared(np.concatenate([ortho_weight(dim_hidden), ortho_weight(dim_hidden)], axis=1),
                      kGivens,
                      prefix + '_Uc')
    Wc = create_shared(np.concatenate([random_matrix(dim_in, dim_hidden), random_matrix(dim_in, dim_hidden)], axis=1),
                      kGivens,
                      prefix + '_Wc')
    bc = create_shared(np.zeros(2 * dim_hidden, dtype=theano.config.floatX), kGivens, prefix + '_bc')

    Ux = create_shared(ortho_weight(dim_hidden), kGivens, prefix + '_Ux')
    Wx = create_shared(random_matrix(dim_in, dim_hidden), kGivens, prefix + '_Wx')
    bx = create_shared(np.zeros(dim_hidden, dtype=theano.config.floatX), kGivens, prefix + '_bx')

    gru_params = [Wc, bc, Uc, Wx, Ux, bx]
    params += gru_params
    names += [prefix + '_Wc', prefix + '_bc', prefix + '_Uc', prefix + '_Wx', prefix + '_Ux', prefix + '_bx']

    def _slice(_x, n):
        return _x[n * dim_hidden:(n + 1) * dim_hidden]

    def recurrence(x_t, m, h_tm1):
        h_tm1 = m * h_tm1
        preact = T.nnet.sigmoid(T.dot(h_tm1, Uc) + T.dot(x_t, Wc) + bc)

        r_t = _slice(preact, 0)
        u_t = _slice(preact, 1)

        h_t = T.tanh(T.dot(h_tm1, Ux) * r_t + T.dot(x_t, Wx) + bx)
        h_t = u_t * h_tm1 + (1. - u_t) * h_t

        return h_t

    h, _ = theano.scan(fn=recurrence,
                       sequences=[inputs, mask],
                       outputs_info=T.alloc(0., dim_hidden),
                       n_steps=inputs.shape[0])

    return h, gru_params


def random_matrix(row, column, scale=0.2):
    # bound = np.sqrt(6. / (row + column))
    bound = 1.
    return scale * np.random.uniform(low=-bound, high=bound, size=(row, column)).astype(theano.config.floatX)


def ortho_weight(dim):
    W = np.random.randn(dim, dim)
    u, s, v = np.linalg.svd(W)
    return u.astype(theano.config.floatX)


###########################################################################
# Model Utilities


def create_shared(random, kGivens, name):
    if name in kGivens:
        if name == 'ffnn_b' or kGivens[name].shape == random.shape:
            print '>>> Using given', name
            return theano.shared(kGivens[name])
        else:
            print '>>> Dimension mismatch with given', name, ': Given:', kGivens[name].shape, ', Actual:', random.shape
    return theano.shared(random)


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
        rep_pairwise, dim_pairwise = self.get_pairwise_rep(rep_cnn, dim_cnn)
        if 'combined' in self.args['model_config']:
            local_score = self.get_local_score_nn(rep_cnn, dim_cnn, rep_pairwise, dim_pairwise)
            global_score, gru_params, ffnn_params = self.get_global_score_nn(rep_cnn, dim_cnn)
            # self.f_grad_shared = self.build_train_combined(local_score, global_score)
            self.f_grad_shared, self.f_update_param = self.build_train_combined(local_score, global_score)
            self.predict = self.build_predict_combined_nn(rep_cnn, dim_cnn, local_score, gru_params, ffnn_params)
        else:
            score = self.get_pairwise_score_nn(rep_pairwise, dim_pairwise)
            self.f_grad_shared, self.f_update_param = self.build_train_pair_nn(score)
            self.predict = self.build_predict_pairwise_nn(score)

        self.container['set_zero'] = OrderedDict()
        self.container['zero_vecs'] = OrderedDict()
        for ed in self.container['embeddings']:
            if ed == 'realis':
                continue
            self.container['zero_vecs'][ed] = np.zeros(self.args['embeddings'][ed].shape[1]).astype(theano.config.floatX)
            self.container['set_zero'][ed] = \
                theano.function([self.container['zero_vector']],
                                updates=[(self.container['embeddings'][ed],
                                          T.set_subtensor(self.container['embeddings'][ed][0, :],
                                                          self.container['zero_vector']))])

    def save(self, path_out):
        storer = dict()
        params_all = self.container['params_local']
        names_all = self.container['names_local']
        if 'combined' in self.args['model_config']:
            params_all += self.container['params_global']
            names_all += self.container['names_global']
        for param, name in zip(params_all, names_all):
            storer[name] = param.get_value()
        cPickle.dump(storer, open(path_out, 'w'))

    def prepare_features(self, header_width=80):
        self.container['fea_dim'] = 0
        self.container['params_local'], self.container['params_global'],\
        self.container['names_local'], self.container['names_global'] = [], [], [], []
        self.container['embeddings'], self.container['vars'] = OrderedDict(), OrderedDict()

        print ' Features '.center(header_width, '-')
        print 'Will update embeddings' if self.args['update_embs'] else 'Will not update embeddings'
        for fea in self.args['features']:
            if self.args['features'][fea] == 0:
                self.container['embeddings'][fea] = create_shared(
                    self.args['embeddings'][fea].astype(theano.config.floatX),
                    self.args['kGivens'],
                    fea)

                if self.args['update_embs']:
                    self.container['params_local'] += [self.container['embeddings'][fea]]
                    self.container['names_local'] += [fea]

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
        self.container['prev_inst_hv'] = T.ivector('prev_inst_hv')
        self.container['prev_inst'] = T.imatrix('prev_inst')
        self.container['prev_inst_cluster'] = T.imatrix('prev_inst_cluster')
        self.container['anchor_position'] = T.ivector('anchor_position')
        self.container['pairwise_fea'] = T.matrix('pairwise_fea')
        self.container['pairwise_coref'] = T.ivector('pairwise_coref')
        self.container['prev_inst_coref'] = T.imatrix('prev_inst_coref')
        self.container['lr'] = T.scalar('lr')
        self.container['zero_vector'] = T.vector('zero_vector')

    def get_cnn_rep(self):
        rep_inter, dim_inter = non_consecutive_cnn(self)

        if self.args['wed_window'] > 0:
            rep_contexts, dim_contexts = trigger_contexts(self)
            rep_inter = T.concatenate([rep_inter, rep_contexts], axis=1)
            dim_inter += dim_contexts

        if self.args['dropout'] > 0.:
            rep_inter = dropout_from_layer(self.args['rng'], [rep_inter], self.args['dropout'])[0]

        dim_hids = [dim_inter] + self.args['multilayer_nn_cnn']

        rep_cnn = multi_hidden_layers([rep_inter],
                                      dim_hids,
                                      self.container['params_local'],
                                      self.container['names_local'],
                                      'main_multi_nn_cnn',
                                      self.args['kGivens'])[0]
        dim_cnn = dim_hids[-1]

        return rep_cnn, dim_cnn

    def get_pairwise_rep(self, rep_cnn, dim_cnn):
        indices_pair = np.array([[i, (i - i % self.args['max_inst_in_doc']) + j]
                                 for i in range(self.args['max_inst_in_doc'] * self.args['batch'])
                                 for j in range(i % self.args['max_inst_in_doc'])], dtype='int32')
        rep_inter = T.reshape(rep_cnn[indices_pair], newshape=(indices_pair.shape[0], 2 * dim_cnn))
        rep_inter = T.concatenate([rep_inter, self.container['pairwise_fea']], axis=1)
        dim_inter = 2 * dim_cnn + self.args['features_dim']['pairwise_fea']

        dim_hids = [dim_inter] + self.args['multilayer_nn_pair']
        rep_pairwise = multi_hidden_layers([rep_inter],
                                           dim_hids,
                                           self.container['params_local'],
                                           self.container['names_local'],
                                           'main_multi_nn_pair',
                                           self.args['kGivens'])[0]
        dim_pairwise = dim_hids[-1]
        return rep_pairwise, dim_pairwise

    def get_pairwise_score_nn(self, rep_pairwise, dim_pairwise):
        res, _ = hidden_layer(inputs=[rep_pairwise],
                                dim_in=dim_pairwise,
                                dim_out=2,
                                params=self.container['params_local'],
                                names=self.container['names_local'],
                                prefix='ffnn',
                                kGivens=self.args['kGivens'],
                                dropout=self.args['dropout'])
        # return T.tanh(score + pairwise_score + b)
        # return T.nnet.nnet.sigmoid(score + pairwise_score + b)
        return res[0]

    def build_train_pair_nn(self, score):
        cost = -T.mean(T.log(score)[T.arange(self.container['pairwise_coref'].shape[0]), self.container['pairwise_coref']])

        gradients = T.grad(cost, self.container['params_local'])
        # updates = [(p, p - (self.container['lr'] * g)) for p, g in zip(self.container['params'], gradients)]

        inputs = [self.container['vars'][ed] for ed in self.args['features']]
        inputs += [self.container['anchor_position'],
                   self.container['pairwise_fea'],
                   self.container['pairwise_coref']]

        # return theano.function(inputs, cost, on_unused_input='ignore')
        f_grad_shared, f_update_param = eval(self.args['optimizer'])(inputs,
                                                                     cost,
                                                                     self.container['names_local'],
                                                                     self.container['params_local'],
                                                                     gradients,
                                                                     self.container['lr'],
                                                                     self.args['norm_lim'])
        return f_grad_shared, f_update_param

    def get_local_score_nn(self, rep_cnn, dim_cnn, rep_pairwise, dim_pairwise):
        res, _ = hidden_layer(inputs=[rep_pairwise],
                              dim_in=dim_pairwise,
                              dim_out=1,
                              params=self.container['params_local'],
                              names=self.container['names_local'],
                              prefix='ffnn_local_ana',
                              kGivens=self.args['kGivens'],
                              dropout=self.args['dropout'])
        num_pairs = self.args['max_inst_in_doc'] * (self.args['max_inst_in_doc'] - 1) / 2
        score_ana = T.reshape(res[0], newshape=(self.args['batch'] * num_pairs,))
        res, _ = hidden_layer(inputs=[rep_cnn],
                              dim_in=dim_cnn,
                              dim_out=1,
                              params=self.container['params_local'],
                              names=self.container['names_local'],
                              prefix='ffnn_local_nonana',
                              kGivens=self.args['kGivens'],
                              dropout=self.args['dropout'])
        score_nonana = T.reshape(res[0], newshape=(self.args['batch'] * self.args['max_inst_in_doc'],))

        score = T.concatenate([np.array([0.], dtype=theano.config.floatX), score_ana, score_nonana])
        indices_single = np.array(
            [[1 + row * (row - 1) / 2 + col for col in range(row)] + [-1] * (self.args['max_inst_in_doc'] - row)
             for row in np.arange(self.args['max_inst_in_doc'])], dtype='int32')
        offset = 1 + self.args['max_inst_in_doc'] * (self.args['max_inst_in_doc'] - 1) / 2 * self.args['batch']
        np.fill_diagonal(indices_single, offset + np.arange(self.args['max_inst_in_doc']))
        indices = np.concatenate([indices_single + i * self.args['max_inst_in_doc']
                                  for i in np.arange(self.args['batch'])])
        mask_single = np.not_equal(indices_single, -1)
        mask = np.concatenate([mask_single] * self.args['batch'])
        return score[indices * mask]

    def get_global_score_nn(self, rep_cnn, dim_cnn):
        padded = T.concatenate([rep_cnn, T.alloc(0., 1, dim_cnn)])
        X = padded[self.container['cluster']]
        rep_rnn, gru_params = rnn_gru(X,
                                      dim_cnn,
                                      dim_cnn,
                                      self.container['mask_rnn'],
                                      'main_rnn',
                                      self.container['params_global'],
                                      self.container['names_global'],
                                      self.args['kGivens'])

        rep_rnn = T.concatenate([rep_rnn, T.alloc(0., 1, dim_cnn)])
        current_hv = rep_rnn[self.container['current_hv']]

        rep_inter1 = T.concatenate([T.sum(current_hv, axis=1, keepdims=True), current_hv], axis=1)
        rep_inter2 = T.stack([rep_cnn] * (self.args['max_cluster_in_doc'] + 1), axis=1)

        rep_inter = T.concatenate([rep_inter1, rep_inter2], axis=2)
        dim_inter = dim_cnn + dim_cnn

        res, ffnn_params = hidden_layer(inputs=[rep_inter],
                                        dim_in=dim_inter,
                                        dim_out=1,
                                        params=self.container['params_global'],
                                        names=self.container['names_global'],
                                        prefix='ffnn_global',
                                        kGivens=self.args['kGivens'],
                                        dropout=self.args['dropout'])

        score_cluster = T.reshape(res[0], newshape=(self.args['batch'] * self.args['max_inst_in_doc'],
                                                           self.args['max_cluster_in_doc'] + 1))
        score_cluster = T.concatenate([score_cluster,
                                       T.alloc(0., self.args['batch'] * self.args['max_inst_in_doc'], 1)], axis=1)
        row_indices = np.array([[i] * self.args['max_inst_in_doc']
                                for i in np.arange(self.args['batch'] * self.args['max_inst_in_doc'])], dtype='int32')
        score = score_cluster[row_indices, self.container['prev_inst_cluster']]
        return score, gru_params, ffnn_params

    def build_train_combined(self, local_score, global_score):
        score = local_score + global_score
        ante_score, ante = T.max_and_argmax(score, axis=1)
        score_gold = score * self.container['prev_inst_coref']
        ante_score_gold, ante_gold = T.max_and_argmax(score_gold, axis=1)
        cost = T.mean((1. + ante_score - ante_score_gold) * T.neq(ante, ante_gold))

        # total_score = total_score / T.sum(total_score, axis=1, keepdims=True)
        # total_score = T.nnet.softmax(local_score + global_score)

        # ante = T.argmax(total_score, axis=1)
        # row_indices = T.arange(self.args['batch'] * self.args['max_inst_in_doc'], dtype='int32')
        # ante_prob = total_score[row_indices, ante]
        # y = self.container['prev_inst_coref'][row_indices, ante]
        # cost = T.nnet.binary_crossentropy(ante_prob, y).mean()

        params_all = self.container['params_local'] + self.container['params_global']
        names_all = self.container['names_local'] + self.container['names_global']
        gradients = T.grad(cost, params_all)

        inputs = [self.container['vars'][ed] for ed in self.args['features']]
        inputs += [self.container['pairwise_fea'],
                   self.container['prev_inst'],
                   self.container['cluster'],
                   self.container['current_hv'],
                   self.container['mask_rnn'],
                   self.container['prev_inst_cluster'],
                   self.container['anchor_position'],
                   self.container['prev_inst_coref']]

        # return theano.function(inputs, outputs=[latent_inst, alpha], on_unused_input='warn')
        # return theano.function(inputs, gradients, on_unused_input='warn')

        f_grad_shared, f_update_param = eval(self.args['optimizer'])(inputs,
                                                                     cost,
                                                                     names_all,
                                                                     params_all,
                                                                     gradients,
                                                                     self.container['lr'],
                                                                     self.args['norm_lim'])
        return f_grad_shared, f_update_param

    def build_predict_pairwise_nn(self, score):
        inputs = [self.container['vars'][ed] for ed in self.args['features']]
        inputs += [self.container['anchor_position'],
                   self.container['pairwise_fea']]
        return theano.function(inputs, T.argmax(score, axis=1), on_unused_input='ignore')
        # return theano.function(inputs, score > 0.5, on_unused_input='ignore')

    def build_predict_combined_nn(self, rep_cnn, dim_cnn, local_score, gru_params, ffnn_params):
        Wc, bc, Uc, Wx, Ux, bx = gru_params
        W_global, b_global = ffnn_params

        def _slice(_x, n):
            return _x[:, n * dim_cnn:(n + 1) * dim_cnn]

        def gru_step(x_t, h_tm1):
            preact = T.nnet.sigmoid(T.dot(h_tm1, Uc) + T.dot(x_t, Wc) + bc)

            r_t = _slice(preact, 0)
            u_t = _slice(preact, 1)

            h_t = T.tanh(T.dot(h_tm1, Ux) * r_t + T.dot(x_t, Wx) + bx)
            h_t = u_t * h_tm1 + (1. - u_t) * h_t

            return h_t

        offsets = np.array([i * self.args['max_inst_in_doc'] for i in np.arange(self.args['batch'])], dtype='int32')

        def recurrence(curr_inst, current_hv, prev_inst_cluster, current_cluster):
            curr_indices = offsets + curr_inst
            prev_inst_cluster = T.set_subtensor(prev_inst_cluster[:, curr_inst], 0)
            # prev_inst_cluster[:, curr_inst] = 0

            curr_rep_cnn = rep_cnn[curr_indices]

            rep_inter1 = T.concatenate([T.sum(current_hv, axis=1, keepdims=True), current_hv], axis=1)
            rep_inter2 = T.stack([curr_rep_cnn] * (self.args['max_inst_in_doc'] + 1), axis=1)
            rep_inter = T.concatenate([rep_inter1, rep_inter2], axis=2)

            score_cluster = T.nnet.sigmoid(T.dot(rep_inter, W_global) + b_global)
            score_cluster = T.reshape(score_cluster, newshape=(self.args['batch'], self.args['max_inst_in_doc'] + 1))
            score_cluster = T.concatenate([score_cluster, T.alloc(-np.inf, self.args['batch'], 1)], axis=1)

            row_indices = np.array([[i] * self.args['max_inst_in_doc']
                                    for i in np.arange(self.args['batch'])], dtype='int32')
            global_score = score_cluster[row_indices, prev_inst_cluster]
            score = local_score[curr_indices] + global_score

            indices_single = T.arange(self.args['batch'], dtype='int32')
            ante_cluster_raw = prev_inst_cluster[indices_single, T.argmax(score, axis=1)]
            indices_new_cluster = T.nonzero(T.eq(ante_cluster_raw, 0))
            ante_cluster = T.set_subtensor(ante_cluster_raw[indices_new_cluster], current_cluster[indices_new_cluster])

            current_cluster = T.set_subtensor(current_cluster[indices_new_cluster], current_cluster[indices_new_cluster]+1)
            prev_inst_cluster = T.set_subtensor(prev_inst_cluster[:, curr_inst], ante_cluster)

            ante_hv = current_hv[indices_single, ante_cluster-1]
            current_hv = T.set_subtensor(current_hv[indices_single, ante_cluster-1], gru_step(curr_rep_cnn, ante_hv))

            return current_hv, prev_inst_cluster, current_cluster

        ini_current_hv = np.array([[[0.] * dim_cnn] * self.args['max_inst_in_doc']]
                                  * self.args['batch']).astype(theano.config.floatX)
        ini_prev_inst_cluster = np.array([[-1] * self.args['max_inst_in_doc']] * self.args['batch'], dtype='int32')
        ini_current_cluster = np.array([1] * self.args['batch'], dtype='int32')

        ret, _ = theano.scan(fn=recurrence,
                             sequences=T.arange(self.args['max_inst_in_doc'], dtype='int32'),
                             outputs_info=[ini_current_hv, ini_prev_inst_cluster, ini_current_cluster])

        inputs = [self.container['vars'][ed] for ed in self.args['features']]
        inputs += [self.container['prev_inst'],
                   self.container['pairwise_fea'],
                   self.container['anchor_position']]
        return theano.function(inputs, ret[1][-1], on_unused_input='ignore')

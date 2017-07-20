from collections import OrderedDict
import theano, theano.tensor as T
import numpy as np


########################## Optimization Functions ##########################

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

    f_param_update = theano.function([lr], [], updates=ru2up + param_up, on_unused_input='ignore')

    return f_grad_shared, f_param_update


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


################################## Models #################################


def non_consecutive_cnn(model):
    X = get_concatenation(model.container['embeddings'],
                          model.container['vars'],
                          model.args['features'],
                          transpose=False)

    rep_cnn = non_consecutive_cnn_driver(X,
                                         model.args['cnn_filter_num'],
                                         model.args['cnn_filter_win'],
                                         model.args['batch'],
                                         model.args['window'],
                                         model.container['fea_dim'],
                                         'non_consecutive_cnn',
                                         model.container['params'],
                                         model.container['names'])

    dim_cnn = model.args['cnn_filter_num'] * len(model.args['cnn_filter_win'])

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
                reps += [vars[fea].dimshuffle(1,0,2)]

    if len(reps) == 1:
        X = reps[0]
    else:
        X = T.cast(T.concatenate(reps, axis=2), dtype=theano.config.floatX)
    return X


def non_consecutive_cnn_driver(inputs, feature_map, cnn_windows, batch, length, dim, prefix, params, names):
    X = inputs.dimshuffle(1, 0, 2)
    reps = []
    for win in cnn_windows:
        Ws, b = prepare_params(win, feature_map, length, dim, prefix, params, names)
        rep_one = eval('non_consecutive_cnn_layer' + str(win))(X, feature_map, batch, Ws, b)
        reps += [rep_one]

    rep_cnn = T.cast(T.concatenate(reps, axis=1), dtype=theano.config.floatX)
    return rep_cnn


def prepare_params(window, feature_map, length, dim, prefix, params, names):
    fan_in = window * dim
    fan_out = feature_map * window * dim / length  # (length - window + 1)
    bound = np.sqrt(6. / (fan_in + fan_out))

    Ws = []
    for i in range(window):
        W = theano.shared(np.random.uniform(low=-bound,
                                            high=bound,
                                            size=(dim, feature_map)).astype(theano.config.floatX))
        Ws += [W]
        params += [W]
        names += [prefix + '_W_win' + str(window) + '_' + str(i)]

    b = theano.shared(np.zeros(feature_map, dtype=theano.config.floatX))
    params += [b]
    names += [prefix + '_b_win_' + str(window)]

    return Ws, b


def non_consecutive_cnn_layer2(inputs, feature_map, batch, Ws, b):
    def recurrence(_x, i_m1, i_m2):
        ati = T.dot(_x, Ws[0])
        _m1 = T.maximum(i_m1, ati)
        ati = i_m1 + T.dot(_x, Ws[1])
        _m2 = T.maximum(i_m2, ati)
        return [_m1, _m2]

    ret, _ = theano.scan(fn=recurrence,
                        sequences=[inputs],
                        outputs_info=[T.alloc(0., batch, feature_map), T.alloc(0., batch, feature_map)],
                        n_steps=inputs.shape[0])

    rep = T.tanh(ret[1][-1] + b[np.newaxis, :])
    return rep


def non_consecutive_cnn_layer3(inputs, feature_map, batch, Ws, b):
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
                        outputs_info=[T.alloc(0., batch, feature_map),
                                      T.alloc(0., batch, feature_map),
                                      T.alloc(0., batch, feature_map)],
                        n_steps=inputs.shape[0])

    rep = T.tanh(ret[2][-1] + b[np.newaxis, :])
    return rep


###########################################################################


class BaseModel(object):
    def __init__(self, args):
        self.container = {}
        self.header_width = 60

        self.args = args
        self.args['rng'] = np.random.RandomState(3435)
        self.args['dropout'] = args['dropout'] if args['dropout'] > 0. else 0.

        self.container['params'], self.container['names'] = [], []
        self.container['embeddings'], self.container['vars'] = OrderedDict(), OrderedDict()
        self.container['fea_dim'] = 0

        self.prepare_features()

        self.container['cluster'] = T.lvector('cluster')
        self.container['mask_rnn'] = T.lvector('mask_rnn')
        self.container['current_hv'] = T.lmatrix('current_hv')
        self.container['prev_inst'] = T.lmatrix('prev_inst')
        self.container['row_indices'] = T.lmatrix('row_indices')
        self.container['prev_inst_cluster'] = T.lmatrix('prev_inst_cluster')
        self.container['prev_inst_cluster_gold'] = T.lmatrix('prev_inst_cluster_gold')
        self.container['alphas'] = T.matrix('alphas')

        self.container['lr'] = T.scalar('lr')
        self.container['zero_vector'] = T.vector('zero_vector')

    def prepare_features(self):
        print 'Features'.center(self.header_width, '-')
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
        print '-' * self.header_width

    def build_functions(self, score, score_dropout, pred_ante):

        cost = T.sum(score) if self.args['dropout'] == 0. else T.sum(score_dropout)

        # if self.args['regularizer'] > 0.:
        #     for p, n in zip(self.container['params'], self.container['names']):
        #         if 'multi' in n:
        #             cost += self.args['regularizer'] * (p ** 2).sum()

        gradients = T.grad(cost, self.container['params'])

        classify_inputs = [self.container['vars'][ed] for ed in self.args['features'] if self.args['features'][ed] >= 0]
        classify_inputs += [self.container['prev_inst']]
        self.classify = theano.function(inputs=classify_inputs, outputs=pred_ante, on_unused_input='ignore')

        train_inputs = classify_inputs + [self.container[item] for item in ['cluster',
                                                                            'mask_rnn',
                                                                            'current_hv',
                                                                            'row_indices',
                                                                            'prev_inst_cluster',
                                                                            'prev_inst_cluster_gold',
                                                                            'alphas']]

        self.f_grad_shared, self.f_update_param = eval(self.args['optimizer'])(train_inputs,
                                                                               cost,
                                                                               self.container['names'],
                                                                               self.container['params'],
                                                                               gradients,
                                                                               self.container['lr'],
                                                                               self.args['norm_lim'])

        # Set the embedding vectors for placeholders to zero
        self.container['set_zero'] = OrderedDict()
        self.container['zero_vecs'] = OrderedDict()
        for ed in self.container['embeddings']:
            self.container['zero_vecs'][ed] = np.zeros(self.args['embeddings'][ed].shape[1], dtype='float32')
            self.container['set_zero'][ed] = theano.function([self.container['zero_vector']],
                                                             updates=[(self.container['embeddings'][ed],
                                                                       T.set_subtensor(self.container['embeddings'][ed][0, :],
                                                                                       self.container['zero_vector']))])


class MainModel(BaseModel):
    def __init__(self, args):
        BaseModel.__init__(self, args)

        X, _ = non_consecutive_cnn(self)

        inputs = [self.container['vars'][ed] for ed in self.args['features'] if self.args['features'][ed] >= 0]
        self.F = theano.function(inputs=inputs, outputs=X, on_unused_input='warn')

        # if self.args['wedWindow'] > 0:
        #     rep, dim_rep = localWordEmbeddingsTrigger(self)
        #     fetre = T.concatenate([fetre, rep], axis=1)
        #     dim_inter += dim_rep
        #
        # fetre_dropout = _dropout_from_layer(self.args['rng'], [fetre], self.args['dropout'])
        # fetre_dropout = fetre_dropout[0]
        #
        # hids = [dim_inter] + self.args['multilayerNN1']
        #
        # mul = MultiHiddenLayers([fetre, fetre_dropout], hids, self.container['params'], self.container['names'],
        #                         'multiMainModel', kGivens=self.args['kGivens'])
        #
        # fetre, fetre_dropout = mul[0], mul[1]
        #
        # dim_inter = hids[len(hids) - 1]
        #
        # fW = theano.shared(
        #     createMatrix(randomMatrix(dim_inter, self.args['nc']), self.args['kGivens'], 'sofmaxMainModel_W'))
        # fb = theano.shared(createMatrix(numpy.zeros(self.args['nc'], dtype=theano.config.floatX), self.args['kGivens'],
        #                                 'sofmaxMainModel_b'))
        #
        # self.container['params'] += [fW, fb]
        # self.container['names'] += ['sofmaxMainModel_W', 'sofmaxMainModel_b']
        #
        # p_y_given_x_dropout = T.nnet.softmax(T.dot(fetre_dropout, fW) + fb)
        #
        # p_y_given_x = T.nnet.softmax(T.dot(fetre, (1.0 - self.args['dropout']) * fW) + fb)
        #
        # self.buildFunctions(p_y_given_x, p_y_given_x_dropout)

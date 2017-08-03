from collections import OrderedDict, defaultdict
import numpy as np
import cPickle
import theano
from model import *


np.set_printoptions(threshold=np.nan)


def prepare_word_embeddings(with_word_embs, embeddings):
    if not with_word_embs:
        print 'Using random word embeddings'
        word_vecs = embeddings['word_random']
    else:
        print 'Using pre-trained word embeddings'
        word_vecs = embeddings['word']
    del embeddings['word']
    del embeddings['word_random']
    embeddings['word'] = word_vecs


def prepare_features(expected_features):
    if expected_features['dep'] >= 0:
        expected_features['dep'] = 1
    if expected_features['possibleTypes'] >= 0:
        expected_features['possibleTypes'] = 1

    features = OrderedDict([('word', 0)])

    for fea in expected_features:
        features[fea] = expected_features[fea]
        if expected_features[fea] == 0:
            print 'Using feature: %-10s - embeddings' % fea
        elif expected_features[fea] == 1:
            print 'Using feature: %-10s - binary' % fea
    return features


def get_dim_mapping(embeddings, map_fea_to_index, features):
    map_dim_emb, map_dim_bin = {}, {}
    for fea in features:
        if not fea == 'dep' and not fea == 'possibleTypes':
            map_dim_emb[fea] = embeddings[fea].shape[1]
        if fea == 'word':
            continue
        if fea == 'anchor':
            binary_dim = embeddings['anchor'].shape[0] - 1
        elif fea == 'possibleTypes' or fea == 'dep':
            binary_dim = len(map_fea_to_index[fea])
        else:
            binary_dim = len(map_fea_to_index[fea]) - 1
        map_dim_bin[fea] = binary_dim
    return map_dim_emb, map_dim_bin


def prepare_data(max_lengths, corpora, map_fea_to_index, features, map_dim_bin, alphas):
    data_sets = {}
    for corpus in corpora:
        data_sets[corpus] = []
        for doc in corpora[corpus]:
            if len(corpora[corpus][doc]['instances']) == 0:
                pass

            data_doc = defaultdict(list)
            inst_in_doc = corpora[corpus][doc]['instances']
            for inst in inst_in_doc:
                ret = add_instance(data_doc, inst, map_fea_to_index, features, map_dim_bin)
                if not ret == True:
                    print 'Error in %s corpus in document %s: cannot find index for word %s\n', corpus, doc, ret
                    exit(0)
            num_placeholder = max_lengths['instance'] - len(inst_in_doc)
            window = len(inst_in_doc[0]['word'])
            for i in range(num_placeholder):
                add_instance_placeholder(data_doc, features, map_fea_to_index, map_dim_bin, window)

            prev_inst = [0] + [-1] * (max_lengths['instance'] - 1)
            mask_prev_inst = [0] * max_lengths['instance']
            for i in range(1, len(inst_in_doc)):
                data_doc['prev_inst'] += [prev_inst[:]]
                data_doc['mask_prev_inst'] += [mask_prev_inst[:]]
                prev_inst[i - 1] = i
                prev_inst[i] = 0
                mask_prev_inst[i - 1] = 1
            data_doc['prev_inst'] += [prev_inst[:]] + [[-1] * max_lengths['instance']] * num_placeholder
            data_doc['mask_prev_inst'] += [mask_prev_inst[:]] + [[0] * max_lengths['instance']] * num_placeholder

            data_doc['prev_inst'] = np.array(data_doc['prev_inst'], dtype='int32')
            data_doc['mask_prev_inst'] = np.array(data_doc['mask_prev_inst'], dtype='int32')

            process_cluster(data_doc, max_lengths, corpora[corpus][doc]['coreference'], len(inst_in_doc), num_placeholder, alphas)

            data_doc['doc_id'] = doc
            data_doc['inst_id_to_index'] = corpora[corpus][doc]['inst_id_to_index']
            data_sets[corpus] += [data_doc]

    return data_sets


def add_instance(data_doc, inst, map_fea_to_index, features, map_dim_bin):
    num_possible_types = len(map_fea_to_index['possibleTypes'])
    num_dep = len(map_fea_to_index['dep'])

    data_inst = defaultdict(list)
    for index in range(len(inst['word'])):
        for fea in features:
            if fea == 'word':
                word = inst['word'][index]
                if word not in map_fea_to_index['word']:
                    return word
                data_inst['word'] += [map_fea_to_index['word'][word]]
                continue

            if fea == 'anchor':
                is_placeholder = True if inst['word'][index] == '######' else False
                anchor_scalar = index + 1 if not is_placeholder else 0
                anchor_vector = [0] * map_dim_bin['anchor']
                if not is_placeholder:
                    anchor_vector[index] = 1
                data_inst['anchor'].append(anchor_vector if features['anchor'] == 1 else anchor_scalar)
                continue

            if fea == 'possibleTypes' or fea == 'dep':
                fea_vector = [0] * (num_dep if fea == 'dep' else num_possible_types)
                for fea_id in inst[fea][index]:
                    fea_vector[fea_id] = 1
                data_inst[fea].append(fea_vector)
                continue

            fea_scalar = inst[fea][index]
            fea_vector = [0] * map_dim_bin[fea]
            if fea_scalar > 0:
                fea_vector[0] = fea_scalar - 1
            data_inst[fea].append(fea_vector if features[fea] == 1 else fea_scalar)

    for fea in data_inst:
        data_doc[fea] += [data_inst[fea]]
    data_doc['anchor_position'] += [inst['anchor']]

    return True


def add_instance_placeholder(data_doc, features, map_fea_to_index, map_dim_bin, window):
    num_possible_types = len(map_fea_to_index['possibleTypes'])
    num_dep = len(map_fea_to_index['dep'])

    data_inst = defaultdict(list)
    for index in range(window):
        for fea in features:
            if fea == 'word':
                data_inst['word'] += [0]
                continue

            if fea == 'anchor':
                anchor_scalar = 0
                anchor_vector = [0] * map_dim_bin['anchor']
                data_inst['anchor'].append(anchor_vector if features['anchor'] == 1 else anchor_scalar)
                continue

            if fea == 'possibleTypes' or fea == 'dep':
                fea_vector = [0] * (num_dep if fea == 'dep' else num_possible_types)
                data_inst[fea].append(fea_vector)
                continue

            fea_scalar = 0
            fea_vector = [0] * map_dim_bin[fea]
            data_inst[fea].append(fea_vector if features[fea] == 1 else fea_scalar)

    for fea in data_inst:
        data_doc[fea] += [data_inst[fea]]
    data_doc['anchor_position'] += [0]


def process_cluster(data_doc, max_lengths, coref, num_inst, num_placeholder, alphas):
    map_inst_to_cluster = {}
    cluster_offset = 0
    starting_hv = [0] * len(coref) + [-1] * (max_lengths['cluster'] - len(coref))
    inst_init = [0] * num_inst
    index = 0
    for chain in coref:
        inst_init[chain[0]] = 1
        for inst in chain:
            map_inst_to_cluster[inst] = index

        data_doc['cluster'] += chain
        mask = [0] + [1] * (len(chain) - 1)
        data_doc['mask_rnn'] += mask

        starting_hv[index] = cluster_offset
        cluster_offset += len(chain)
        index += 1

    current_hv = [-1] * max_lengths['cluster']
    mask_current_hv = [0] * max_lengths['cluster']
    for i in range(num_inst):
        data_doc['current_hv'] += [current_hv[:]]
        data_doc['mask_current_hv'] += [mask_current_hv[:]]
        cluster_index = map_inst_to_cluster[i]
        mask_current_hv[cluster_index] = 1
        if current_hv[cluster_index] == -1:
            current_hv[cluster_index] = starting_hv[cluster_index]
        else:
            current_hv[cluster_index] += 1

    prev_inst_cluster = [0] + [-1] * (max_lengths['instance'] - 1)
    for i in range(1, num_inst):
        data_doc['prev_inst_cluster'] += [prev_inst_cluster[:]]
        prev_inst_cluster[i] = 0
        prev_inst_cluster[i - 1] = map_inst_to_cluster[i - 1] + 1
    data_doc['prev_inst_cluster'] += [prev_inst_cluster[:]]

    data_doc['prev_inst_cluster_gold'] = np.array([[-1] * max_lengths['instance']] * max_lengths['instance'],
                                                  dtype='int32')
    for inst_curr in range(num_inst):
        chain = coref[map_inst_to_cluster[inst_curr]]
        if chain[0] == inst_curr:
            data_doc['prev_inst_cluster_gold'][inst_curr][inst_curr] = inst_curr
            continue
        for inst_prev in chain:
            if inst_prev >= inst_curr:
                break
            data_doc['prev_inst_cluster_gold'][inst_curr][inst_prev] = inst_prev

    data_doc['cluster'] += [-1] * num_placeholder
    data_doc['mask_rnn'] += [0] * num_placeholder
    data_doc['mask_cluster'] = [1] * num_inst + [0] * num_placeholder
    data_doc['current_hv'] += [[-1] * max_lengths['cluster']] * num_placeholder
    data_doc['mask_current_hv'] += [[0] * max_lengths['cluster']] * num_placeholder
    data_doc['prev_inst_cluster'] += [[-1] * max_lengths['instance']] * (max_lengths['instance'] - num_inst)

    data_doc['alpha'] = get_penalty_rates(inst_init, max_lengths, alphas)

    for item in ['cluster', 'mask_cluster', 'current_hv', 'mask_current_hv']:
        data_doc[item] = np.array(data_doc[item], dtype='int32')


def get_penalty_rates(inst_init, max_lengths, alphas):
    penalty_rates = np.zeros(shape=(max_lengths['instance'], max_lengths['instance'])).astype(theano.config.floatX)
    for i in range(len(inst_init)):
        if inst_init[i] == 1:
            penalty_rates[i, i] = 0
            penalty_rates[i, 0:i] = alphas[0]
        else:
            penalty_rates[i, i] = alphas[1]
            penalty_rates[i, 0:i] = alphas[2]
    return penalty_rates


def main(dataset_path='/scratch/wl1191/event_coref/data/nugget.pkl',
         window=31,
         wed_window=2,
         expected_features=OrderedDict([('anchor', 0),
                                        ('pos', -1),
                                        ('chunk', -1),
                                        ('possibleTypes', -1),
                                        ('dep', 1),
                                        ('nonref', -1),
                                        ('title', -1),
                                        ('eligible', -1)]),
         with_word_embs=True,
         update_embs=True,
         cnn_filter_num=300,
         cnn_filter_wins=[2, 3, 4, 5],
         dropout=0.,
         multilayer_nn=[300],
         dim_cnn=300,
         optimizer='adadelta',
         lr=0.01,
         norm_lim=0.,
         alphas=(0.5, 1.2, 1),
         batch=3):

    print 'Loading dataset:', dataset_path, '...'
    max_lengths, corpora, embeddings, map_fea_to_index = cPickle.load(open(dataset_path, 'rb'))

    prepare_word_embeddings(with_word_embs, embeddings)
    features = prepare_features(expected_features)
    map_dim_emb, map_dim_bin = get_dim_mapping(embeddings, map_fea_to_index, features)
    data_sets = prepare_data(max_lengths, corpora, map_fea_to_index, features, map_dim_bin, alphas)
    data_train = data_sets['train']

    features_dim = OrderedDict([('word', map_dim_emb['word'])])
    for fea in expected_features:
        fea_dim = 0
        if expected_features[fea] == 1:
            fea_dim = map_dim_bin[fea]
        elif expected_features[fea] == 0:
            fea_dim = map_dim_emb[fea]
        features_dim[fea] = fea_dim

    params = {'embeddings': embeddings,
              'features': features,
              'features_dim': features_dim,
              'window': window,
              'update_embs': update_embs,
              'wed_window': wed_window,
              'cnn_filter_num': cnn_filter_num,
              'cnn_filter_wins': cnn_filter_wins,
              'dropout': dropout,
              'multilayer_nn': multilayer_nn,
              'dim_cnn': dim_cnn,
              'optimizer': optimizer,
              'lr': lr,
              'norm_lim': norm_lim,
              'batch': batch,
              'max_inst_in_doc': max_lengths['instance'],
              'max_cluster_in_doc': max_lengths['cluster']}

    features_batch, inputs_batch = defaultdict(list), defaultdict(list)
    for i, doc in enumerate(data_train):
        for fea in features:
            features_batch[fea] += doc[fea]
        for item in ['prev_inst', 'cluster', 'current_hv']:
            inputs_batch[item] += [doc[item] + doc['mask_' + item] * batch * i]
        for item in ['mask_rnn', 'prev_inst_cluster', 'anchor_position']:
            inputs_batch[item] += doc[item]
        inputs_batch['prev_inst_cluster_gold'] += [doc['prev_inst_cluster_gold']]
        inputs_batch['alpha'] += [doc['alpha']]

    inputs_test = []
    for fea in features:
        if features[fea] == 0:
            inputs_test += [np.array(features_batch[fea], dtype='int32')]
        elif features[fea] == 1:
            inputs_test += [np.array(features_batch[fea]).astype(theano.config.floatX)]
    inputs_test += [np.concatenate(inputs_batch['prev_inst'])]

    inputs_train = inputs_test[:]
    inputs_test += [np.array(inputs_batch['anchor_position'], dtype='int32')]

    for item in ['cluster', 'current_hv']:
        inputs_train += [np.concatenate(inputs_batch[item])]
    inputs_train += [np.array(inputs_batch['mask_rnn'], dtype=theano.config.floatX)]
    for item in ['prev_inst_cluster', 'anchor_position']:
        inputs_train += [np.array(inputs_batch[item], dtype='int32')]
    inputs_train += [np.concatenate(inputs_batch['prev_inst_cluster_gold'])]
    inputs_train += [np.concatenate(inputs_batch['alpha'])]

    # inputs_names = ['word']
    # for fea in expected_features:
    #     if expected_features[fea] >= 0:
    #         inputs_names += [fea]
    # inputs_names += ['prev_inst', 'cluster', 'current_hv', 'mask_rnn', 'prev_inst_cluster', 'anchor_position', 'prev_inst_cluster_gold', 'alpha']
    #
    # print '\n', ' Shapes '.center(120, '='), '\n'
    # for name, var in zip(inputs_names, inputs_train):
    #     print name, ':', var.shape
    # print '\n', ' Embeddings Dim '.center(120, '='), '\n'
    # print map_dim_emb
    # print '\n', ' Binary Dim '.center(120, '='), '\n'
    # print map_dim_bin
    # print '\n', ' Max Lengths '.center(120, '='), '\n'
    # print 'Instance in doc =', max_lengths['instance']
    # print 'Cluster in doc =', max_lengths['cluster']

    print '\nBuilding model ...\n'
    model = MainModel(params)

    # print model.train(*inputs_train)

    print '\nTraining ...\n'
    for i in range(300):
        cost = model.f_grad_shared(*inputs_train)
        model.f_update_param(params['lr'])

        for fea in model.container['embeddings']:
            model.container['set_zero'][fea](model.container['zero_vecs'][fea])
        print '>>> Epoch', i, ': cost = ', cost

        if cost == 0.:
            break

    print model.test(*inputs_test)

if __name__ == '__main__':
    main()

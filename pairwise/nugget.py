from collections import OrderedDict, defaultdict
import numpy as np
import cPickle
import subprocess
import random
import time
import copy
import sys
import theano
from model import *


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


def prepare_features(expected_features, embeddings, map_fea_to_index):
    if expected_features['dep'] >= 0:
        expected_features['dep'] = 1
    if expected_features['possibleTypes'] >= 0:
        expected_features['possibleTypes'] = 1

    features = OrderedDict([('word', 0)])
    features_dim = {'word': embeddings['word'].shape[1]}

    for fea in expected_features:
        if expected_features[fea] >= 0:
            features[fea] = expected_features[fea]
            if expected_features[fea] == 0:
                features_dim[fea] = embeddings[fea].shape[1]
                print 'Using feature: %-10s - embeddings' % fea
            elif expected_features[fea] == 1:
                if fea == 'dep' or fea == 'dep':
                    features_dim[fea] = len(map_fea_to_index[fea])
                elif fea == 'anchor':
                    features_dim[fea] = embeddings[fea].shape[0] - 1
                else:
                    features_dim[fea] = len(map_fea_to_index[fea]) - 1
                print 'Using feature: %-10s - binary' % fea
    return features, features_dim


def prepare_data(corpora, prefix, map_fea_to_index, features, features_dim):
    data_sets = {}
    map_sent_dist_index = {}
    coref_features_all = {}
    for corpus in corpora:
        if corpus == 'test':
            continue
        data_sets[corpus] = {'doc_info': {}}
        inst_pairs = defaultdict(list)
        coref_features = defaultdict(list)
        for doc_id in corpora[corpus]:
            doc = dict()
            for item in ['instances', 'coreference', 'inst_id_to_index', 'missing_inst']:
                doc[item] = corpora[corpus][doc_id][prefix + item]
            if len(doc['instances']) == 0:
                pass

            data_doc = defaultdict(list)
            inst_in_doc = doc['instances']
            for inst in inst_in_doc:
                ret = add_instance(data_doc, inst, map_fea_to_index, features, features_dim)
                if not ret == True:
                    print 'Error in %s corpus in document %s: cannot find index for word %s\n', corpus, doc, ret
                    exit(0)

            create_pairs(inst_pairs, data_doc, inst_in_doc, doc['coreference'], map_sent_dist_index, coref_features)
            inst_pairs['doc_id'] += [doc_id] * (len(inst_in_doc) * (len(inst_in_doc)-1) / 2)

            data_sets[corpus]['doc_info'][doc_id] = {'inst_id_to_index': doc['inst_id_to_index'],
                                                     'missing_inst': doc['missing_inst']}
        coref_features_all[corpus] = coref_features
        data_sets[corpus]['inst_pairs'] = inst_pairs

    for corpus in corpora:
        if corpus == 'test':
            continue
        for index, sent_dist_index in enumerate(coref_features_all[corpus]['sent_dist']):
            sent_dist_vector = [0] * len(map_sent_dist_index)
            sent_dist_vector[sent_dist_index] = 1
            match_type = coref_features_all[corpus]['match_type'][index]
            match_subtype = coref_features_all[corpus]['match_subtype'][index]
            data_sets[corpus]['inst_pairs']['coref_features'] += [sent_dist_vector + [match_type] + [match_subtype]]
    features_dim['coref_features'] = len(map_sent_dist_index) + 2

    return data_sets


def add_instance(data_doc, inst, map_fea_to_index, features, features_dim):
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
                anchor_vector = [0] * features_dim['anchor']
                if not is_placeholder:
                    anchor_vector[index] = 1
                data_inst['anchor'].append(anchor_vector if features['anchor'] == 1 else anchor_scalar)
                continue

            if fea == 'possibleTypes' or fea == 'dep':
                fea_vector = [0] * (features_dim[fea] if fea == 'dep' else features_dim['possibleType'])
                for fea_id in inst[fea][index]:
                    fea_vector[fea_id] = 1
                data_inst[fea].append(fea_vector)
                continue

            fea_scalar = inst[fea][index]
            fea_vector = [0] * features_dim[fea]
            if fea_scalar > 0:
                fea_vector[fea_scalar - 1] = 1
            data_inst[fea].append(fea_vector if features[fea] == 1 else fea_scalar)

    for fea in data_inst:
        data_doc[fea] += [data_inst[fea]]
    data_doc['anchor_position'] += [inst['anchor']]

    return True


def create_pairs(inst_pairs, data_doc, inst_in_doc, coref, map_sent_dist_index, coref_features):
    map_inst_to_cluster = dict()
    for index, chain in enumerate(coref):
        for inst in chain:
            map_inst_to_cluster[inst] = index

    for i in range(len(data_doc['word'])):
        for j in range(i):
            for fea in data_doc:
                inst_pairs[fea + '1'] += [data_doc[fea][i]]
                inst_pairs[fea + '2'] += [data_doc[fea][j]]
            inst_pairs['inst_id1'] += [i]
            inst_pairs['inst_id2'] += [j]
            if map_inst_to_cluster[i] == map_inst_to_cluster[j]:
                inst_pairs['label'] += [1]
            else:
                inst_pairs['label'] += [0]

            inst1 = inst_in_doc[i]
            inst2 = inst_in_doc[j]
            coref_features['match_type'] += [1 if inst1['type'] == inst2['type'] else 0]
            coref_features['match_subtype'] += [1 if inst1['subtype'] == inst2['subtype'] else 0]
            sent_dist = abs(inst1['sentenceId'] - inst2['sentenceId'])
            if sent_dist not in map_sent_dist_index:
                map_sent_dist_index[sent_dist] = len(map_sent_dist_index)
            coref_features['sent_dist'] += [map_sent_dist_index[sent_dist]]


def fit_data_to_batch(data, batch):
    data_fitted = dict()
    if len(data['word1']) % batch > 0:
        num_to_add = batch - len(data['word1']) % batch
        np.random.seed(3435)
        indices_to_add = np.random.permutation(len(data['word1']))[:num_to_add]
        for item in data:
            data_fitted[item] = copy.copy(data[item])
            for i in indices_to_add:
                data_fitted[item] += [data[item][i]]
    else:
        num_to_add = 0
        data_fitted = data
    return data_fitted, num_to_add


def prepare_realis_output(path_realis, path_golden, data_eval, pipeline):
    path_fin = path_realis + data_eval + '.realis' if pipeline else path_realis + '../golden/' + data_eval
    realis_ouput = dict()
    with open(path_fin, 'r') as fin:
        current_doc = ''
        body = []
        for line in fin:
            if not line:
                continue
            if line.startswith('#BeginOfDocument'):
                current_doc = line.rstrip('\n').split()[1]
            elif line.startswith('#EndOfDocument'):
                realis_ouput[current_doc] = body
                body = []
            elif line.startswith('@Coreference'):
                continue
            else:
                body += [line]
    return realis_ouput


def get_batch_inputs(data, features, batch, batch_index):
    inputs = []
    start = batch_index * batch
    end = start + batch
    for fea in features:
        data1 = data[fea + '1'][start: end]
        data2 = data[fea + '2'][start: end]
        if features[fea] == 0:
            inputs += [np.array(data1, dtype='int32')]
            inputs += [np.array(data2, dtype='int32')]
        else:
            inputs += [np.array(data1, dtype=theano.config.floatX)]
            inputs += [np.array(data2, dtype=theano.config.floatX)]
    inputs += [np.array(data['anchor_position1'][start: end], dtype='int32')]
    inputs += [np.array(data['anchor_position2'][start: end], dtype='int32')]
    inputs += [np.array(data['coref_features'][start: end], dtype=theano.config.floatX)]
    labels = np.array(data['label'][start: end], dtype='int32')
    return inputs, labels


def train(model, data, params, epoch, features, batch, num_batch, verbose):
    total_cost = 0
    print (' Training in epoch %d ' % epoch).center(80, '-')
    time_start = time.time()
    for index, batch_index in enumerate(np.random.permutation(range(num_batch))):
        inputs, labels = get_batch_inputs(data, features, batch, batch_index)
        inputs += [labels]
        total_cost += model.f_grad_shared(*inputs)
        model.f_update_param(params['lr'])
        for fea in model.container['embeddings']:
            if fea == 'realis':
                continue
            model.container['set_zero'][fea](model.container['zero_vecs'][fea])
    if verbose:
        print 'Completed in %.2f seconds\nCost = %.5f' % (time.time() - time_start, total_cost)
    return total_cost


def predict(model, data, features, batch):
    num_batch = len(data['word1']) / batch
    predictions = []
    for batch_index in range(num_batch):
        inputs, _ = get_batch_inputs(data, features, batch, batch_index)
        predictions += [model.predict(*inputs)]
    return np.concatenate(predictions)


def get_coref_chain(data, predictions):
    coref_chain = dict()
    for inst1, inst2, doc_id, label in zip(data['inst_pairs']['inst_id1'],
                                           data['inst_pairs']['inst_id2'],
                                           data['inst_pairs']['doc_id'],
                                           predictions):
        if doc_id not in coref_chain:
            if label == 1:
                coref_chain[doc_id] = {'map_inst_to_cluster': {inst1: 0, inst2: 0},
                                       'curr_cluster': defaultdict(list)}
                coref_chain[doc_id]['curr_cluster'][0] += [inst1, inst2]
            else:
                coref_chain[doc_id] = {'map_inst_to_cluster': {inst1: 0, inst2: 1},
                                       'curr_cluster': defaultdict(list)}
                coref_chain[doc_id]['curr_cluster'][0] += [inst1]
                coref_chain[doc_id]['curr_cluster'][1] += [inst2]
        else:
            if label == 0:
                if inst1 not in coref_chain[doc_id]['map_inst_to_cluster']:
                    cluster_index = len(coref_chain[doc_id]['curr_cluster'])
                    coref_chain[doc_id]['map_inst_to_cluster'][inst1] = cluster_index
                    coref_chain[doc_id]['curr_cluster'][cluster_index] += [inst1]
                if inst2 not in coref_chain[doc_id]['map_inst_to_cluster']:
                    cluster_index = len(coref_chain[doc_id]['curr_cluster'])
                    coref_chain[doc_id]['map_inst_to_cluster'][inst2] = cluster_index
                    coref_chain[doc_id]['curr_cluster'][cluster_index] += [inst2]
            else:
                if inst1 not in coref_chain[doc_id]['map_inst_to_cluster'] and inst2 not in coref_chain[doc_id]['map_inst_to_cluster']:
                    cluster_index = len(coref_chain[doc_id]['curr_cluster'])
                    coref_chain[doc_id]['map_inst_to_cluster'][inst1] = cluster_index
                    coref_chain[doc_id]['map_inst_to_cluster'][inst2] = cluster_index
                    coref_chain[doc_id]['curr_cluster'][cluster_index] += [inst1, inst2]
                elif inst1 in coref_chain[doc_id]['map_inst_to_cluster'] and inst2 not in coref_chain[doc_id]['map_inst_to_cluster']:
                    cluster_index = coref_chain[doc_id]['map_inst_to_cluster'][inst1]
                    coref_chain[doc_id]['map_inst_to_cluster'][inst2] = cluster_index
                    coref_chain[doc_id]['curr_cluster'][cluster_index] += [inst2]
                elif inst1 not in coref_chain[doc_id]['map_inst_to_cluster'] and inst2 in coref_chain[doc_id]['map_inst_to_cluster']:
                    cluster_index = coref_chain[doc_id]['map_inst_to_cluster'][inst2]
                    coref_chain[doc_id]['map_inst_to_cluster'][inst1] = cluster_index
                    coref_chain[doc_id]['curr_cluster'][cluster_index] += [inst1]
                else:
                    cluster_index1 = coref_chain[doc_id]['map_inst_to_cluster'][inst1]
                    cluster_index2 = coref_chain[doc_id]['map_inst_to_cluster'][inst2]
                    if not cluster_index1 == cluster_index2:
                        coref_chain[doc_id]['curr_cluster'][cluster_index1] += coref_chain[doc_id]['curr_cluster'][cluster_index2]
                        for inst in coref_chain[doc_id]['curr_cluster'][cluster_index2]:
                            coref_chain[doc_id]['map_inst_to_cluster'][inst] = cluster_index1
                        coref_chain[doc_id]['curr_cluster'][cluster_index2] = []

    coref_chain_str = dict()
    for doc_id in coref_chain:
        coref_chain_str[doc_id] = defaultdict(list)
        map_inst_to_cluster = dict()
        curr_cluster_index = 0
        inst_index_to_id = dict((k, v) for v, k in data['doc_info'][doc_id]['inst_id_to_index'].iteritems())
        for cluster_index in coref_chain[doc_id]['curr_cluster']:
            if len(coref_chain[doc_id]['curr_cluster'][cluster_index]) == 0:
                continue
            for inst in coref_chain[doc_id]['curr_cluster'][cluster_index]:
                coref_chain_str[doc_id][curr_cluster_index] += [inst_index_to_id[inst]]
                map_inst_to_cluster[inst] = curr_cluster_index
            curr_cluster_index += 1
        for cluster in data['doc_info'][doc_id]['missing_inst']:
            cluster_index = len(coref_chain_str[doc_id])
            for missing_inst in cluster:
                if cluster[missing_inst] is None:
                    coref_chain_str[doc_id][cluster_index] += [missing_inst]
                else:
                    cluster_index = map_inst_to_cluster[cluster[missing_inst]]
                    coref_chain_str[doc_id][cluster_index] += [missing_inst]
        coref_chain_str[doc_id] = coref_chain_str[doc_id].values()

    return coref_chain_str


def write_out(epoch, corpus_name, coref_chain, realis_output, path_out):
    with open(path_out + corpus_name + '.coref.pred' + str(epoch), 'w') as fout:
        for doc_id, chains in coref_chain.iteritems():
            fout.write('#BeginOfDocument ' + doc_id + '\n')
            for line in realis_output[doc_id]:
                fout.write(line)
            for cluster_index, cluster in enumerate(chains):
                fout.write('@Coreference\tC%d\t' % cluster_index)
                chain_str = ''
                for inst in cluster:
                    chain_str += 'E' + inst.split('-')[1] + ','
                fout.write(chain_str[:-1] + '\n')
            fout.write('#EndOfDocument\n')


def get_score(path_golden, path_output, path_scorer, path_token, path_conllTemp):
    proc = subprocess.Popen(
        ["python", path_scorer, "-g", path_golden, "-s", path_output, "-t", path_token, "-c", path_conllTemp],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE)
    ous, _ = proc.communicate()

    spanP, spanR, spanF1 = 0.0, 0.0, 0.0
    subtypeP, subtypeR, subtypeF1 = 0.0, 0.0, 0.0
    realisP, realisR, realisF1 = 0.0, 0.0, 0.0
    realisAndTypeP, realisAndTypeR, realisAndTypeF1 = 0.0, 0.0, 0.0
    bcub, ceafe, ceafm, muc, blanc, averageCoref = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    startMentionScoring = False
    startCoreferenceScoring = False
    for line in ous.split('\n'):
        line = line.strip()
        if line == '=======Final Mention Detection Results=========':
            startMentionScoring = True
            startCoreferenceScoring = False
            continue
        if line == '=======Final Mention Coreference Results=========':
            startMentionScoring = False
            startCoreferenceScoring = True
            continue
        if not startMentionScoring and not startCoreferenceScoring: continue
        if startMentionScoring and line.startswith('plain'):
            els = line.split('\t')
            spanP, spanR, spanF1 = float(els[1]), float(els[2]), float(els[3])
            continue
        if startMentionScoring and line.startswith('mention_type') and not line.startswith(
                'mention_type+realis_status'):
            els = line.split('\t')
            subtypeP, subtypeR, subtypeF1 = float(els[1]), float(els[2]), float(els[3])
            continue
        if startMentionScoring and line.startswith('realis_status'):
            els = line.split('\t')
            realisP, realisR, realisF1 = float(els[1]), float(els[2]), float(els[3])
            continue
        if startMentionScoring and line.startswith('mention_type+realis_status'):
            els = line.split('\t')
            realisAndTypeP, realisAndTypeR, realisAndTypeF1 = float(els[1]), float(els[2]), float(els[3])
            continue
        if startCoreferenceScoring and 'bcub' in line:
            els = line.split()
            bcub = float(els[4])
            continue
        if startCoreferenceScoring and 'ceafe' in line:
            els = line.split()
            ceafe = float(els[4])
            continue
        if startCoreferenceScoring and 'ceafm' in line:
            els = line.split()
            ceafm = float(els[4])
            continue
        if startCoreferenceScoring and 'muc' in line:
            els = line.split()
            muc = float(els[4])
            continue
        if startCoreferenceScoring and 'blanc' in line:
            els = line.split()
            blanc = float(els[4])
            continue
        if startCoreferenceScoring and 'Overall Average CoNLL score' in line:
            els = line.split()
            averageCoref = float(els[4])
            continue

    return OrderedDict({'spanP': spanP, 'spanR': spanR, 'spanF1': spanF1,
                        'typeP': subtypeP, 'typeR': subtypeR, 'typeF1': subtypeF1,
                        'subtypeP': subtypeP, 'subtypeR': subtypeR, 'subtypeF1': subtypeF1,
                        'realisP': realisP, 'realisR': realisR, 'realisF1': realisF1,
                        'realisAndTypeP': realisAndTypeP, 'realisAndTypeR': realisAndTypeR,
                        'realisAndTypeF1': realisAndTypeF1,
                        'bcub': bcub, 'ceafe': ceafe, 'ceafm': ceafm, 'muc': muc, 'blanc': blanc,
                        'averageCoref': averageCoref})


def print_perf(performance, msg):
    print (' ' + msg + ' ').center(80, '-')

    print 'plain: ', str(performance['spanP']) + '\t' + str(performance['spanR']) + '\t' + str(performance['spanF1'])
    print 'mention_type: ', str(performance['typeP']) + '\t' + str(performance['typeR']) + '\t' + str(
        performance['typeF1'])
    print 'mention_subtype: ', str(performance['subtypeP']) + '\t' + str(performance['subtypeR']) + '\t' + str(
        performance['subtypeF1'])
    print 'realis_status: ', str(performance['realisP']) + '\t' + str(performance['realisR']) + '\t' + str(
        performance['realisF1'])
    print 'mention_type+realis_status: ', str(performance['realisAndTypeP']) + '\t' + str(
        performance['realisAndTypeR']) + '\t' + str(performance['realisAndTypeF1'])
    print 'bcub: ', performance['bcub']
    print 'ceafe: ', performance['ceafe']
    print 'ceafm: ', performance['ceafm']
    print 'muc: ', performance['muc']
    print 'blanc: ', performance['blanc']
    print 'averageCoref: ', performance['averageCoref']

    print '-' * 80


def run(path_dataset='/scratch/wl1191/event_coref/data/nugget.pkl',
        path_realis='/scratch/wl1191/event_coref/data/realis/',
        path_golden='/scratch/wl1191/event_coref/officialScorer/hopper/eval.tbf',
        path_token='/scratch/wl1191/event_coref/officialScorer/hopper/tkn/',
        path_scorer='/scratch/wl1191/event_coref/officialScorer/scorer_v1.7.py',
        path_conllTemp='/scratch/wl1191/event_coref/data/coref/conllTempFile_Coreference.txt',
        path_out='/scratch/wl1191/event_coref/out/',
        path_kGivens='/scratch/wl1191/event_coref/out/params29.pkl',
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
        pipeline=False,
        with_word_embs=True,
        update_embs=True,
        cnn_filter_num=300,
        cnn_filter_wins=[2, 3, 4, 5],
        dropout=0.5,
        multilayer_nn=[600, 300],
        optimizer='adadelta',
        lr=0.05,
        lr_decay=False,
        norm_lim=0,
        batch=200,
        nepochs=30,
        seed=3435,
        verbose=True):
    print '\nLoading dataset:', path_dataset, '...\n'
    _, corpora, embeddings, map_fea_to_index = cPickle.load(open(path_dataset, 'rb'))

    prepare_word_embeddings(with_word_embs, embeddings)
    features, features_dim = prepare_features(expected_features, embeddings, map_fea_to_index)

    prefix = 'pipe_' if pipeline else 'gold_'
    data_sets = prepare_data(corpora, prefix, map_fea_to_index, features, features_dim)

    kGivens = dict()
    if path_kGivens is not None:
        kGivens = cPickle.load(open(path_kGivens, 'r'))

    params = {'features': features,
              'features_dim': features_dim,
              'window': window,
              'update_embs': update_embs,
              'wed_window': wed_window,
              'cnn_filter_num': cnn_filter_num,
              'cnn_filter_wins': cnn_filter_wins,
              'dropout': dropout,
              'multilayer_nn': multilayer_nn,
              'optimizer': optimizer,
              'lr': lr,
              'norm_lim': norm_lim,
              'batch': batch,
              'kGivens': kGivens}

    params_all = params.copy()
    params['embeddings'] = embeddings
    params_all.update({'pipeline': pipeline,
                       'with_word_embs': with_word_embs,
                       'update_embs': update_embs})

    # print 'Saving model configuration ...'
    # cPickle.dump(params_all, open(path_out + 'model_config.pkl', 'w'))

    data_train, _ = fit_data_to_batch(data_sets['train']['inst_pairs'], batch)
    num_batch = len(data_train['word1']) / batch
    print 'Number of batches:', num_batch, '\n'

    print 'Loading realis outputs ...'
    # realis_outputs = {'valid': prepare_realis_output(path_realis, path_golden, 'valid', pipeline),
    #                   'test': prepare_realis_output(path_realis, path_golden, 'test', pipeline)}
    realis_outputs = {'valid': prepare_realis_output(path_realis, path_golden, 'valid', pipeline)}

    print '\nBuilding model ...\n'
    np.random.seed(seed)
    random.seed(seed)
    model = MainModel(params)

    # for i in range(1000):
    #     if train(model, data_train, params, i, features, batch, num_batch, verbose) == 0.:
    #         break
    #
    # print 'Saving parameters ...'
    # model.save(path_out + 'params' + '.pkl')
    #
    # print '\nTesting ...'
    # data_valid, num_added = fit_data_to_batch(data_sets['valid']['inst_pairs'], batch)
    # print 'Number of batches to eval:', len(data_valid['word1']) / batch
    # predictions = predict(model, data_valid, features, batch)
    # if num_added > 0:
    #     predictions = predictions[:-num_added]
    # cPickle.dump(predictions, open(path_out + 'predictions.pkl', 'w'))
    # predictions = cPickle.load(open(path_out + 'predictions.pkl', 'r'))
    # coref_chain = get_coref_chain(data_sets['valid'], predictions)
    # print 'Writing out ...'
    # write_out(0, 'valid', coref_chain, realis_outputs['valid'], path_out)

    # data_sets_eval = OrderedDict([('valid', fit_data_to_batch(data_sets['valid'], batch)),
    #                               ('test', fit_data_to_batch(data_sets['test'], batch))])
    data_sets_eval = OrderedDict([('valid', fit_data_to_batch(data_sets['valid']['inst_pairs'], batch))])
    print '\nNumber of batches to eval:', len(data_sets_eval['valid'][0]['word1']) / batch
    predictions = OrderedDict()
    best_f1 = -np.inf
    best_performance = None
    best_epoch = -1
    curr_lr = lr
    print '\nTraining ...\n'
    sys.stdout.flush()
    for epoch in xrange(30, 30 + nepochs):
        train(model, data_train, params, epoch, features, batch, num_batch, verbose)
        sys.stdout.flush()
        if (epoch + 1) % 1 == 0:
            print (' Evaluating in epoch %d ' % epoch).center(80, '-')
            sys.stdout.flush()
            for data_eval in data_sets_eval:
                data, num_added = data_sets_eval[data_eval]
                predictions[data_eval] = predict(model, data, features, batch)
                if num_added > 0:
                    predictions[data_eval] = predictions[data_eval][:-num_added]
                cPickle.dump(predictions[data_eval], open(path_out + 'predictions' + str(epoch) + '.pkl', 'w'))
                coref_chain = get_coref_chain(data_sets[data_eval], predictions[data_eval])
                write_out(epoch, 'valid', coref_chain, realis_outputs['valid'], path_out)

            path_output = path_out + 'valid.coref.pred' + str(epoch)
            performance = get_score(path_golden, path_output, path_scorer, path_token, path_conllTemp)

            print 'Saving parameters'
            model.save(path_out + 'params' + str(epoch) + '.pkl')

            if performance['averageCoref'] > best_f1:
                best_f1 = performance['averageCoref']
                best_performance = performance
                best_epoch = epoch
                print 'NEW BEST: Epoch', epoch
            if verbose:
                print_perf(performance, 'Current Performance')

            # learning rate decay if no improvement in 10 epochs
            if lr_decay and abs(best_epoch - epoch) >= 10:
                curr_lr *= 0.5
            if curr_lr < 1e-5:
                break

    print '\n', '=' * 80, '\n'
    print 'BEST RESULT: Epoch', best_epoch
    print_perf(best_performance, 'Best Performance')


if __name__ == '__main__':
    run()

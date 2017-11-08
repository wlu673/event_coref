from collections import OrderedDict, defaultdict
import numpy as np
import cPickle
import subprocess
import random
import time
import sys
import theano
from model import *


np.set_printoptions(threshold=np.nan)


def prepare_realis_output(path_realis, data_eval, pipeline):
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


def fit_data_to_batch(data, batch):
    if len(data) % batch > 0:
        num_to_add = batch - len(data) % batch
        np.random.seed(3435)
        data_fitted = np.concatenate([data, np.random.permutation(data)[: num_to_add]])
    else:
        num_to_add = 0
        data_fitted = data
    return data_fitted, num_to_add


def get_batch_inputs(data, features, batch):
    features_batch, inputs_batch = defaultdict(list), defaultdict(list)
    for i, doc in enumerate(data):
        for fea in features:
            features_batch[fea] += doc[fea]
        for item in ['prev_inst', 'cluster', 'current_hv']:
            inputs_batch[item] += [doc[item] + doc['mask_' + item] * batch * i]
        for item in ['mask_rnn', 'prev_inst_cluster', 'anchor_position']:
            inputs_batch[item] += doc[item]
        for item in ['prev_inst_coref', 'pairwise_fea', 'pairwise_coref']:
            inputs_batch[item] += [doc[item]]
    return features_batch, inputs_batch


def get_train_inputs(data, features, batch, model_config):
    features_batch, inputs_batch = get_batch_inputs(data, features, batch)

    inputs = []
    for fea in features:
        if features[fea] == 0:
            inputs += [np.array(features_batch[fea], dtype='int32')]
        else:
            inputs += [np.array(features_batch[fea]).astype(theano.config.floatX)]

    prev_inst = np.concatenate(inputs_batch['prev_inst'])
    if 'combined' in model_config:
        inputs += [np.concatenate(inputs_batch['pairwise_fea']), prev_inst]
        for item in ['cluster', 'current_hv']:
            inputs += [np.concatenate(inputs_batch[item])]
        inputs += [np.array(inputs_batch['mask_rnn'], dtype=theano.config.floatX)]
        for item in ['prev_inst_cluster', 'anchor_position']:
            inputs += [np.array(inputs_batch[item], dtype='int32')]
        inputs += [np.concatenate(inputs_batch['prev_inst_coref'])]
    else:
        inputs += [np.array(inputs_batch['anchor_position'], dtype='int32')]
        for item in ['pairwise_fea', 'pairwise_coref']:
            inputs += [np.concatenate(inputs_batch[item])]

    return inputs


def get_pred_inputs(data, features, batch, model_config):
    features_batch, inputs_batch = get_batch_inputs(data, features, batch)

    inputs = []
    for fea in features:
        if features[fea] == 0:
            inputs += [np.array(features_batch[fea], dtype='int32')]
        else:
            inputs += [np.array(features_batch[fea]).astype(theano.config.floatX)]
    if 'combined' in model_config:
        inputs += [np.concatenate(inputs_batch['prev_inst'])]
    inputs += [np.concatenate(inputs_batch['pairwise_fea']),
               np.array(inputs_batch['anchor_position'], dtype='int32')]

    return inputs


def train(model, data, params, epoch, features, batch, num_batch, model_config, verbose):
    total_cost = 0
    print (' Training in epoch %d ' % epoch).center(80, '-')
    time_start = time.time()
    for index, batch_index in enumerate(np.random.permutation(range(num_batch))):
        inputs_train = get_train_inputs(data[batch_index * batch: (batch_index + 1) * batch],
                                        features,
                                        batch,
                                        model_config)
        total_cost += model.f_grad_shared(*inputs_train)
        model.f_update_param(params['lr'])
        for fea in model.container['embeddings']:
            if fea == 'realis':
                continue
            model.container['set_zero'][fea](model.container['zero_vecs'][fea])
    if verbose:
        print 'Completed in %.2f seconds\nCost = %.5f' % (time.time() - time_start, total_cost)
    return total_cost


def predict(model, data, features, batch, max_inst_in_doc, model_config):
    num_batch = len(data) / batch
    predictions = []
    for batch_index in range(num_batch):
        inputs_pred = get_pred_inputs(data[batch_index * batch: (batch_index + 1) * batch],
                                      features,
                                      batch,
                                      model_config)
        if 'combined' in model_config:
            predict_combined(model, data, batch, batch_index, inputs_pred, predictions)
        else:
            predict_local(model, data, batch, batch_index, max_inst_in_doc, inputs_pred, predictions)

    return predictions


def predict_combined(model, data, batch, batch_index, inputs_pred, predictions):
    cluster_batch = model.predict(*inputs_pred)
    for doc_index in range(batch):
        doc = data[batch_index * batch + doc_index]
        inst_index_to_id = dict((k, v) for v, k in doc['inst_id_to_index'].iteritems())
        coref = defaultdict(list)
        for inst_index in range(len(inst_index_to_id)):
            coref[cluster_batch[doc_index][inst_index]] += [inst_index_to_id[inst_index]]
        for chain in doc['missing_inst']:
            cluster_new = len(coref) + 1
            for inst in chain:
                if chain[inst] is None:
                    coref[cluster_new] += [inst]
                else:
                    cluster_index = cluster_batch[doc_index][chain[inst]]
                    coref[cluster_index] += [inst]
        predictions += [coref.values()]


def predict_local(model, data, batch, batch_index, max_inst_in_doc, inputs_pred, predictions):
    prediction_batch_pairs = model.predict(*inputs_pred)
    prediction_batch = np.zeros((batch * max_inst_in_doc, max_inst_in_doc), dtype='int32')
    index = 0
    for i in range(batch * max_inst_in_doc):
        for j in range(i % max_inst_in_doc):
            if prediction_batch_pairs[index] == 1:
                prediction_batch[i, j] = 1
            index += 1

    for doc_index in range(batch):
        doc = data[batch_index * batch + doc_index]
        inst_index_to_id = dict((k, v) for v, k in doc['inst_id_to_index'].iteritems())
        coref = defaultdict(list)
        offset = doc_index * max_inst_in_doc
        map_inst_to_cluster = dict()
        clusters_curr = defaultdict(list)
        for inst_curr in range(len(inst_index_to_id)):
            clusters_ante = []
            for inst_ante in range(inst_curr):
                if prediction_batch[offset + inst_curr][inst_ante] == 1 and \
                                map_inst_to_cluster[inst_ante] not in clusters_ante:
                        clusters_ante += [map_inst_to_cluster[inst_ante]]
            if len(clusters_ante) == 0:
                map_inst_to_cluster[inst_curr] = len(clusters_curr)
                clusters_curr[len(clusters_curr)] += [inst_curr]
            elif len(clusters_ante) == 1:
                map_inst_to_cluster[inst_curr] = clusters_ante[0]
                clusters_curr[clusters_ante[0]] += [inst_curr]
            else:
                clusters_to_merge = []
                clusters_new = defaultdict(list)
                for cluster_index, chain in clusters_curr.iteritems():
                    if cluster_index in clusters_ante:
                        clusters_to_merge += chain
                    else:
                        for inst_index in chain:
                            map_inst_to_cluster[inst_index] = len(clusters_new)
                        clusters_new[len(clusters_new)] = chain
                clusters_to_merge += [inst_curr]
                for inst_index in clusters_to_merge:
                    map_inst_to_cluster[inst_index] = len(clusters_new)
                clusters_new[len(clusters_new)] = clusters_to_merge
                clusters_curr = clusters_new
        for cluster_index, chain in clusters_curr.iteritems():
            coref[cluster_index] = [inst_index_to_id[inst_index] for inst_index in chain]
        for chain in doc['missing_inst']:
            cluster_new = len(coref) + 1
            for inst in chain:
                if chain[inst] is None:
                    coref[cluster_new] += [inst]
                else:
                    cluster_index = map_inst_to_cluster[chain[inst]]
                    coref[cluster_index] += [inst]
        predictions += [coref.values()]


def write_out(epoch, data_eval, data, predictions, realis_output, path_out):
    with open(path_out + data_eval + '.coref.pred' + str(epoch), 'w') as fout:
        for doc, coref in zip(data, predictions):
            fout.write('#BeginOfDocument ' + doc['doc_id'] + '\n')
            for line in realis_output[doc['doc_id']]:
                fout.write(line)
            for cluster_index, chain in enumerate(coref):
                fout.write('@Coreference\tC%d\t' % cluster_index)
                chain_str = ''
                for inst in chain:
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


def run(path_data='/scratch/wl1191/event_coref/data/sample/data.pkl',
        path_realis='/scratch/wl1191/event_coref/data/sample/realis/',
        path_golden='/scratch/wl1191/event_coref/officialScorer/hopper/eval.tbf',
        path_token='/scratch/wl1191/event_coref/officialScorer/hopper/tkn/',
        path_scorer='/scratch/wl1191/event_coref/officialScorer/scorer_v1.7.py',
        path_conllTemp='/scratch/wl1191/event_coref/data/sample/coref/conllTempFile_Coreference.txt',
        path_out='/scratch/wl1191/event_coref/data/sample/out/',
        path_kGivens=None, # '/scratch/wl1191/event_coref/params49.pkl',
        model_config='combined',
        window=31,
        wed_window=2,
        pipeline=False,
        with_word_embs=True,
        update_embs=True,
        cnn_filter_num=40, # 300,
        cnn_filter_wins=[2, 3], # [2, 3, 4, 5],
        dropout=0., # 0.5,
        multilayer_nn_cnn=[80, 40], # [600, 300],
        multilayer_nn_pair=[80, 40], # [600, 300],
        optimizer='adadelta',
        lr=0.05,
        lr_decay=False,
        norm_lim=0,
        batch=3, # 2,
        nepochs=300,
        seed=3435,
        verbose=True):

    print '\nLoading dataset:', path_data, '...\n'
    data_sets, features, features_dim, embeddings, max_lengths = cPickle.load(open(path_data, 'r'))

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
              'multilayer_nn_cnn': multilayer_nn_cnn,
              'multilayer_nn_pair': multilayer_nn_pair,
              'optimizer': optimizer,
              'lr': lr,
              'norm_lim': norm_lim,
              'batch': batch,
              'max_inst_in_doc': max_lengths['instance'],
              'max_cluster_in_doc': max_lengths['cluster'],
              'kGivens': kGivens,
              'model_config': model_config}

    params_all = params.copy()
    params['embeddings'] = embeddings
    params_all.update({'pipeline': pipeline,
                       'with_word_embs': with_word_embs,
                       'update_embs': update_embs})

    print 'Saving model configuration ...'
    cPickle.dump(params_all, open(path_out + 'model_config.pkl', 'w'))

    data_train, _ = fit_data_to_batch(data_sets['train'], batch)
    num_batch = len(data_train) / batch
    print 'Number of batches:', num_batch, '\n'

    print 'Loading realis outputs ...'
    # realis_outputs = {'valid': prepare_realis_output(path_realis, path_golden, 'valid', pipeline),
    #                   'test': prepare_realis_output(path_realis, 'test', pipeline)}
    realis_outputs = {'valid': prepare_realis_output(path_realis, 'valid', pipeline)}

    print '\nBuilding model ...\n'
    np.random.seed(seed)
    random.seed(seed)
    model = MainModel(params)

    # inputs_train = get_train_inputs(data_train[0:3],
    #                                 features,
    #                                 batch,
    #                                 max_lengths['instance'],
    #                                 model_config)
    #
    # print model.train(*inputs_train)

    # print train(model, data_train, params, 0, features, batch, num_batch, max_lengths['instance'], model_config, verbose)
    # predictions = predict(model, data_train, features, batch, max_lengths['instance'], model_config)
    # print predictions

    for i in range(1000):
        train(model, data_train, params, i, features, batch, num_batch, model_config, verbose)

    print 'Saving parameters ...'
    model.save(path_out + 'params' + '.pkl')

    print '\nTesting ...'
    data, num_added = fit_data_to_batch(data_sets['valid'], batch)
    predictions = predict(model, data, features, batch, max_lengths['instance'], model_config)
    if num_added > 0:
        predictions = predictions[:-num_added]
    print 'Writing out ...'
    write_out(0, 'valid', data, predictions, realis_outputs['valid'], path_out)

    # # data_sets_eval = OrderedDict([('valid', fit_data_to_batch(data_sets['valid'], batch)),
    # #                               ('test', fit_data_to_batch(data_sets['test'], batch))])
    # data_sets_eval = OrderedDict([('valid', fit_data_to_batch(data_sets['valid'], batch))])
    # predictions = OrderedDict()
    # best_f1 = -np.inf
    # best_performance = None
    # best_epoch = -1
    # curr_lr = lr
    # print '\nTraining ...\n'
    # for epoch in xrange(50, 50 + nepochs):
    #     train(model, data_train, params, epoch, features, batch, num_batch, model_config, verbose)
    #
    #     if (epoch + 1) % 5 == 0:
    #     # if epoch >= 0:
    #         print (' Evaluating in epoch %d ' % epoch).center(80, '-')
    #         for data_eval in data_sets_eval:
    #             data, num_added = data_sets_eval[data_eval]
    #             predictions[data_eval] = predict(model, data, features, batch, max_lengths['instance'], model_config)
    #             if num_added > 0:
    #                 predictions[data_eval] = predictions[data_eval][:-num_added]
    #             write_out(epoch, data_eval, data, predictions[data_eval], realis_outputs[data_eval], path_out)
    #
    #         path_output = path_out + 'valid.coref.pred' + str(epoch)
    #         performance = get_score(path_golden, path_output, path_scorer, path_token, path_conllTemp)
    #
    #         print 'Saving parameters'
    #         model.save(path_out + 'params' + str(epoch) + '.pkl')
    #
    #         if performance['averageCoref'] > best_f1:
    #             best_f1 = performance['averageCoref']
    #             best_performance = performance
    #             best_epoch = epoch
    #             print 'NEW BEST: Epoch', epoch
    #         if verbose:
    #             print_perf(performance, 'Current Performance')
    #
    #         # learning rate decay if no improvement in 10 epochs
    #         if lr_decay and abs(best_epoch - epoch) >= 10:
    #             curr_lr *= 0.5
    #         if curr_lr < 1e-5:
    #             break
    #
    #     sys.stdout.flush()
    #
    # print '\n', '=' * 80, '\n'
    # print 'BEST RESULT: Epoch', best_epoch
    # print_perf(best_performance, 'Best Performance')


if __name__ == '__main__':
    run()

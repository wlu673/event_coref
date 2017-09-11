from collections import OrderedDict, defaultdict
import numpy as np
import cPickle
import subprocess
import random
import time
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


def prepare_features(expected_features, expected_features_event):
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

    features_event = OrderedDict()
    for fea in expected_features_event:
        features_event[fea] = expected_features_event[fea]
        if expected_features_event[fea] == 0:
            print 'Using feature: %-10s - embeddings' % fea
        elif expected_features_event[fea] == 1:
            print 'Using feature: %-10s - binary' % fea

    return features, features_event


def get_dim_mapping(embeddings, map_fea_to_index, features, features_event):
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

    for fea in features_event:
        map_dim_bin[fea] = len(map_fea_to_index[fea]) if fea == 'realis' else len(map_fea_to_index[fea])-1
        map_dim_emb[fea] = embeddings[fea].shape[1]

    return map_dim_emb, map_dim_bin


def prepare_data(max_lengths, corpora, prefix, map_fea_to_index, features, features_event, map_dim_bin, alphas):
    data_sets = {}
    for corpus in corpora:
        data_sets[corpus] = []
        for doc_id in corpora[corpus]:
            doc = dict()
            for item in ['instances', 'coreference', 'inst_id_to_index', 'missing_inst']:
                doc[item] = corpora[corpus][doc_id][prefix + item]

            if len(doc['instances']) == 0:
                pass

            data_doc = defaultdict(list)
            inst_in_doc = doc['instances']
            for inst in inst_in_doc:
                ret = add_instance(data_doc, inst, map_fea_to_index, features, features_event, map_dim_bin)
                if not ret == True:
                    print 'Error in %s corpus in document %s: cannot find index for word %s\n', corpus, doc, ret
                    exit(0)
            num_placeholder = max_lengths['instance'] - len(inst_in_doc)
            window = len(inst_in_doc[0]['word'])
            for i in range(num_placeholder):
                add_instance_placeholder(data_doc, features, features_event, map_fea_to_index, map_dim_bin, window)

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

            process_cluster(data_doc, max_lengths, doc['coreference'], len(inst_in_doc), num_placeholder, alphas)

            data_doc['doc_id'] = doc_id
            data_doc['inst_id_to_index'] = doc['inst_id_to_index']
            data_sets[corpus] += [data_doc]

    return data_sets


def add_instance(data_doc, inst, map_fea_to_index, features, features_event, map_dim_bin):
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
                fea_vector[fea_scalar - 1] = 1
            data_inst[fea].append(fea_vector if features[fea] == 1 else fea_scalar)

    for fea in data_inst:
        data_doc[fea] += [data_inst[fea]]

    for fea in features_event:
        fea_scalar = inst[fea]
        fea_vector = [0] * map_dim_bin[fea]
        if fea == 'realis':
            fea_vector[fea_scalar] = 1
        else:
            fea_vector[fea_scalar-1] = 1
        data_doc[fea].append(fea_vector if features_event[fea] == 1 else fea_scalar)

    data_doc['anchor_position'] += [inst['anchor']]

    return True


def add_instance_placeholder(data_doc, features, features_event, map_fea_to_index, map_dim_bin, window):
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
            fea_vector = [0] * (map_dim_bin[fea]-1)
            data_inst[fea].append(fea_vector if features[fea] == 1 else fea_scalar)

    for fea in data_inst:
        data_doc[fea] += [data_inst[fea]]

    for fea in features_event:
        fea_scalar = 0
        fea_vector = [0] * map_dim_bin[fea]
        data_doc[fea].append(fea_vector if features_event[fea] == 1 else fea_scalar)

    data_doc['anchor_position'] += [0]


def process_cluster(data_doc, max_lengths, coref, num_inst, num_placeholder, alphas):
    map_inst_to_cluster = dict()
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


def get_features_dim(expected_features, expected_features_event, map_dim_bin, map_dim_emb):
    features_dim = OrderedDict([('word', map_dim_emb['word'])])
    for fea in expected_features:
        fea_dim = 0
        if expected_features[fea] == 1:
            fea_dim = map_dim_bin[fea]
        elif expected_features[fea] == 0:
            fea_dim = map_dim_emb[fea]
        features_dim[fea] = fea_dim
    for fea in expected_features_event:
        fea_dim = 0
        if expected_features_event[fea] == 1:
            fea_dim = map_dim_bin[fea]
        elif expected_features_event[fea] == 0:
            fea_dim = map_dim_emb[fea]
        features_dim[fea] = fea_dim
    return features_dim


def prepare_realis_output(path_realis, data_eval):
    realis_ouput = dict()
    with open(path_realis + data_eval + '.realis', 'r') as fin:
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


def get_batch_inputs(data, features, features_event, batch):
    features_batch, inputs_batch = defaultdict(list), defaultdict(list)
    for i, doc in enumerate(data):
        for fea in features:
            features_batch[fea] += doc[fea]
        for fea in features_event:
            features_batch[fea] += doc[fea]
        for item in ['prev_inst', 'cluster', 'current_hv']:
            inputs_batch[item] += [doc[item] + doc['mask_' + item] * batch * i]
        for item in ['mask_rnn', 'prev_inst_cluster', 'anchor_position']:
            inputs_batch[item] += doc[item]
        inputs_batch['prev_inst_cluster_gold'] += [doc['prev_inst_cluster_gold']]
        inputs_batch['alpha'] += [doc['alpha']]

    return features_batch, inputs_batch


def get_train_inputs(data, features, features_event, batch):
    features_batch, inputs_batch = get_batch_inputs(data, features, features_event, batch)

    inputs = []
    for fea in features:
        if features[fea] == 0:
            inputs += [np.array(features_batch[fea], dtype='int32')]
        elif features[fea] == 1:
            inputs += [np.array(features_batch[fea]).astype(theano.config.floatX)]
    for fea in features_event:
        if features_event[fea] == 0:
            inputs += [np.array(features_batch[fea], dtype='int32')]
        elif features_event[fea] == 1:
            inputs += [np.array(features_batch[fea]).astype(theano.config.floatX)]
    inputs += [np.concatenate(inputs_batch['prev_inst'])]

    for item in ['cluster', 'current_hv']:
        inputs += [np.concatenate(inputs_batch[item])]
    inputs += [np.array(inputs_batch['mask_rnn'], dtype=theano.config.floatX)]
    for item in ['prev_inst_cluster', 'anchor_position']:
        inputs += [np.array(inputs_batch[item], dtype='int32')]
    inputs += [np.concatenate(inputs_batch['prev_inst_cluster_gold'])]
    inputs += [np.concatenate(inputs_batch['alpha'])]

    return inputs


def get_pred_inputs(data, features, features_event, batch):
    features_batch, inputs_batch = get_batch_inputs(data, features, features_event, batch)

    inputs = []
    for fea in features:
        if features[fea] == 0:
            inputs += [np.array(features_batch[fea], dtype='int32')]
        elif features[fea] == 1:
            inputs += [np.array(features_batch[fea]).astype(theano.config.floatX)]
    for fea in features_event:
        if features_event[fea] == 0:
            inputs += [np.array(features_batch[fea], dtype='int32')]
        elif features_event[fea] == 1:
            inputs += [np.array(features_batch[fea]).astype(theano.config.floatX)]
    inputs += [np.concatenate(inputs_batch['prev_inst'])]
    inputs += [np.array(inputs_batch['anchor_position'], dtype='int32')]

    return inputs


def train(model, data, params, epoch, features, features_event, batch, num_batch, verbose):
    total_cost = 0
    print (' Training in epoch %d ' % epoch).center(80, '-')
    time_start = time.time()
    for index, batch_index in enumerate(np.random.permutation(range(num_batch))):
        inputs_train = get_train_inputs(data[batch_index * batch: (batch_index + 1) * batch],
                                        features,
                                        features_event,
                                        batch)
        total_cost += model.f_grad_shared(*inputs_train)
        model.f_update_param(params['lr'])
        for fea in model.container['embeddings']:
            if fea == 'realis':
                continue
            model.container['set_zero'][fea](model.container['zero_vecs'][fea])
    if verbose:
        print 'Completed in %.2f seconds\nCost = %.2f' % (time.time() - time_start, total_cost)


def predict(model, data, features, features_event, batch, model_config):
    num_batch = len(data) / batch
    predictions = []
    for batch_index in range(num_batch):
        inputs_pred = get_pred_inputs(data[batch_index * batch: (batch_index + 1) * batch],
                                      features,
                                      features_event,
                                      batch)
        cluster_batch = model.predict(*inputs_pred)
        for doc_index in range(cluster_batch.shape[0]):
                doc = data[batch_index * batch + doc_index]
                inst_index_to_id = dict((k, v) for v, k in doc['inst_id_to_index'].iteritems())
                coref = defaultdict(list)
                if 'global' in model_config:
                    for inst_index in range(len(inst_index_to_id)):
                        coref[cluster_batch[doc_index][inst_index]] += [inst_index_to_id[inst_index]]
                else:
                    prev_inst_cluster = [0]
                    curr_num_cluster = 1
                    for inst_index in range(1, len(inst_index_to_id)):
                        if cluster_batch[doc_index][inst_index] == inst_index:
                            curr_cluster = curr_num_cluster
                            curr_num_cluster += 1
                        else:
                            curr_cluster = prev_inst_cluster[cluster_batch[doc_index][inst_index]]
                        prev_inst_cluster += [curr_cluster]
                        coref[curr_cluster] += [inst_index_to_id[inst_index]]
                for inst in doc['missing_inst']:
                    if doc['missing_inst'][inst] is None:
                        coref[len(coref) + 1] += [inst]
                    else:
                        cluster_index = cluster_batch[doc_index][doc['missing_inst'][inst]]
                        coref[cluster_index] += [inst]
                predictions += [coref.values()]

    return predictions


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


def main(path_dataset='/scratch/wl1191/event_coref/data/nugget.pkl',
         path_realis='/scratch/wl1191/event_coref/data/realis/',
         path_golden='/scratch/wl1191/event_coref/officialScorer/hopper/eval.tbf',
         path_token='/scratch/wl1191/event_coref/officialScorer/hopper/tkn/',
         path_scorer='/scratch/wl1191/event_coref/officialScorer/scorer_v1.7.py',
         path_conllTemp='/scratch/wl1191/event_coref/data/coref/conllTempFile_Coreference.txt',
         path_out='/scratch/wl1191/event_coref/out/',
         path_kGivens=None,
         model_config='local',
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
         expected_features_event=OrderedDict([('type', 0), ('subtype', 0), ('realis', -1)]),
         pipeline=True,
         with_word_embs=True,
         update_embs=True,
         cnn_filter_num=300,
         cnn_filter_wins=[2, 3, 4, 5],
         dropout=0.5,
         multilayer_nn=[600, 300],
         dim_cnn=300,
         optimizer='adadelta',
         lr=0.01,
         lr_decay=False,
         norm_lim=9.0,
         alphas=(0.5, 1.2, 1),
         batch=1,
         nepochs=400,
         seed=3435,
         verbose=True):

    print '\nLoading dataset:', path_dataset, '...\n'
    max_lengths, corpora, embeddings, map_fea_to_index = cPickle.load(open(path_dataset, 'rb'))

    prepare_word_embeddings(with_word_embs, embeddings)
    features, features_event = prepare_features(expected_features, expected_features_event)
    map_dim_emb, map_dim_bin = get_dim_mapping(embeddings, map_fea_to_index, features, features_event)

    prefix = 'pipe_' if pipeline else 'gold_'
    for item in ['instance', 'cluster']:
        max_lengths[item] = max_lengths[prefix + item]
        del max_lengths['gold_' + item]
        del max_lengths['pipe_' + item]
    data_sets = prepare_data(max_lengths, corpora, prefix, map_fea_to_index, features, features_event, map_dim_bin, alphas)
    features_dim = get_features_dim(expected_features, expected_features_event, map_dim_bin, map_dim_emb)

    kGivens = dict()
    if path_kGivens is not None:
        kGivens = cPickle.load(open(path_kGivens, 'r'))

    params = {'features': features,
              'features_event': features_event,
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
              'max_cluster_in_doc': max_lengths['cluster'],
              'kGivens': kGivens,
              'model_config': model_config}

    print 'Saving model configuration ...'
    cPickle.dump(params, open(path_out + 'model_config.pkl', 'w'))
    params['embeddings'] = embeddings

    data_train, _ = fit_data_to_batch(data_sets['train'], batch)
    num_batch = len(data_train) / batch
    print 'Number of batches:', num_batch, '\n'

    print 'Loading realis outputs ...'
    # realis_outputs = {'valid': prepare_realis_output(path_realis, path_golden, 'valid', pipeline),
    #                   'test': prepare_realis_output(path_realis, path_golden, 'test', pipeline)}
    realis_outputs = {'valid': prepare_realis_output(path_realis, 'valid')}

    print '\nBuilding model ...\n'
    np.random.seed(seed)
    random.seed(seed)
    model = MainModel(params)

    # data_sets_eval = OrderedDict([('valid', fit_data_to_batch(data_sets['valid'], batch)),
    #                               ('test', fit_data_to_batch(data_sets['test'], batch))])
    data_sets_eval = OrderedDict([('valid', fit_data_to_batch(data_sets['valid'], batch))])
    predictions = OrderedDict()

    best_f1 = -np.inf
    best_performance = None
    best_epoch = -1
    curr_lr = lr
    print '\nTraining ...\n'
    for epoch in xrange(nepochs):
        train(model, data_train, params, epoch, features, features_event, batch, num_batch, verbose)

        if (epoch + 1) % 20 == 0:
            print (' Evaluating in epoch %d ' % epoch).center(80, '-')
            for data_eval in data_sets_eval:
                data, num_added = data_sets_eval[data_eval]
                predictions[data_eval] = predict(model, data, features, features_event, batch, model_config)
                if num_added > 0:
                    predictions[data_eval] = predictions[data_eval][:-num_added]
                write_out(epoch, data_eval, data, predictions[data_eval], realis_outputs[data_eval], path_out)

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

        sys.stdout.flush()

    print '\n', '=' * 80, '\n'
    print 'BEST RESULT: Epoch', best_epoch
    print_perf(best_performance, 'Best Performance')


if __name__ == '__main__':
    main()

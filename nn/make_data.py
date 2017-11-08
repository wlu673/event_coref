from collections import OrderedDict, defaultdict
import numpy as np
import theano
import cPickle


def prepare_word_embeddings(use_pretrained_emb, embeddings):
    if not use_pretrained_emb:
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


def prepare_data(corpora, prefix, map_fea_to_index, features, features_dim, max_lengths):
    data_sets = {}
    map_sent_dist_index = {}
    for corpus in corpora:
        data_sets[corpus] = prepare_data_in_corpus(corpus,
                                                   corpora[corpus],
                                                   prefix,
                                                   map_fea_to_index,
                                                   features,
                                                   features_dim,
                                                   max_lengths,
                                                   map_sent_dist_index)

    dim_pairwise_fea = len(map_sent_dist_index) + 3
    dim_row = max_lengths['instance'] * (max_lengths['instance'] - 1) / 2
    for corpus in data_sets:
        for data_doc in data_sets[corpus]:
            pairwise_fea = data_doc['pairwise_fea']
            data_doc['pairwise_fea'] = np.zeros((dim_row, dim_pairwise_fea), dtype=theano.config.floatX)
            num_inst = len(data_doc['inst_id_to_index'])
            index = 0
            for i in range(num_inst):
                for j in range(i):
                    sent_dist = pairwise_fea[index, 3]
                    sent_dist_vector = np.array([0] * len(map_sent_dist_index), dtype='int32')
                    sent_dist_vector[map_sent_dist_index[sent_dist]] = 1
                    data_doc['pairwise_fea'][index] = np.concatenate([pairwise_fea[index][:3], sent_dist_vector])
                    index += 1
    features_dim['pairwise_fea'] = dim_pairwise_fea

    return data_sets


def prepare_data_in_corpus(corpus_name, corpus_, prefix, map_fea_to_index, features, features_dim, max_lengths, map_sent_dist_index):
    data_corpus = []
    for doc_id in corpus_:
        doc = dict()
        for item in ['instances', 'coreference', 'inst_id_to_index', 'missing_inst']:
            doc[item] = corpus_[doc_id][prefix + item]

        if len(doc['instances']) == 0:
            continue

        data_doc = defaultdict(list)
        data_doc['missing_inst'] = doc['missing_inst']

        process_instance_in_doc(corpus_name, doc, data_doc, map_fea_to_index, features, features_dim, max_lengths)

        data_doc['pairwise_fea'], data_doc['y'] = process_pairs(doc['instances'],
                                                                doc['coreference'],
                                                                map_sent_dist_index,
                                                                max_lengths['instance'])

        process_cluster(data_doc, max_lengths, doc['coreference'], len(doc['instances']))

        data_doc['doc_id'] = doc_id
        data_doc['inst_id_to_index'] = doc['inst_id_to_index']

        data_corpus += [data_doc]

    return data_corpus


def process_instance_in_doc(corpus_name, doc, data_doc, map_fea_to_index, features, features_dim, max_lengths):
    inst_in_doc = doc['instances']
    for inst in inst_in_doc:
        ret = add_instance(data_doc, inst, map_fea_to_index, features, features_dim)
        if not ret == True:
            print 'Error in %s corpus in document %s: cannot find index for word %s\n' % (corpus_name, doc, ret)
            exit(0)

    num_placeholder = max_lengths['instance'] - len(inst_in_doc)
    window = len(inst_in_doc[0]['word'])
    for i in range(num_placeholder):
        add_instance_placeholder(data_doc, features, map_fea_to_index, features_dim, window)


def add_instance(data_doc, inst, map_fea_to_index, features, features_dim):
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
                anchor_vector = [0] * features_dim['anchor']
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
            fea_vector = [0] * features_dim[fea]
            if fea_scalar > 0:
                fea_vector[fea_scalar - 1] = 1
            data_inst[fea].append(fea_vector if features[fea] == 1 else fea_scalar)

    for fea in data_inst:
        data_doc[fea] += [data_inst[fea]]
    data_doc['anchor_position'] += [inst['anchor']]

    return True


def add_instance_placeholder(data_doc, features, map_fea_to_index, features_dim, window):
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
                anchor_vector = [0] * features_dim['anchor']
                data_inst['anchor'].append(anchor_vector if features['anchor'] == 1 else anchor_scalar)
                continue

            if fea == 'possibleTypes' or fea == 'dep':
                fea_vector = [0] * (num_dep if fea == 'dep' else num_possible_types)
                data_inst[fea].append(fea_vector)
                continue

            fea_scalar = 0
            fea_vector = [0] * features_dim[fea]
            data_inst[fea].append(fea_vector if features[fea] == 1 else fea_scalar)

    for fea in data_inst:
        data_doc[fea] += [data_inst[fea]]
    data_doc['anchor_position'] += [0]


def process_pairs(inst_in_doc, coref, map_sent_dist_index, max_inst):
    map_inst_to_cluster = dict()
    for index, chain in enumerate(coref):
        for inst in chain:
            map_inst_to_cluster[inst] = index
    dim_row = max_inst * (max_inst - 1) / 2
    pairwise_features = np.zeros((dim_row, 4), dtype='int32')
    y = np.zeros(dim_row, dtype='int32')
    index = 0
    for i in range(len(inst_in_doc)):
        for j in range(i):
            inst1 = inst_in_doc[i]
            inst2 = inst_in_doc[j]

            pairwise_features[index, 0] = 1 if inst1['type'] == inst2['type'] else 0
            pairwise_features[index, 1] = 1 if inst1['subtype'] == inst2['subtype'] else 0
            pairwise_features[index, 2] = 1 if inst1['realis'] == inst2['realis'] else 0
            sent_dist = abs(inst1['sentenceId'] - inst2['sentenceId'])
            if sent_dist not in map_sent_dist_index:
                map_sent_dist_index[sent_dist] = len(map_sent_dist_index)
            pairwise_features[index, 3] = map_sent_dist_index[sent_dist]

            if map_inst_to_cluster[i] == map_inst_to_cluster[j]:
                y[index] = 1
            index += 1

    return pairwise_features, y


def generate_inputs(doc, data_doc, max_lengths):
    num_placeholder = max_lengths['instance'] - len(doc['instances'])
    prev_inst = [0] + [-1] * (max_lengths['instance'] - 1)
    mask_prev_inst = [0] * max_lengths['instance']
    for i in range(1, len(doc['instances'])):
        data_doc['prev_inst'] += [prev_inst[:]]
        data_doc['mask_prev_inst'] += [mask_prev_inst[:]]
        prev_inst[i - 1] = i
        prev_inst[i] = 0
        mask_prev_inst[i - 1] = 1
    data_doc['prev_inst'] += [prev_inst[:]] + [[-1] * max_lengths['instance']] * num_placeholder
    data_doc['mask_prev_inst'] += [mask_prev_inst[:]] + [[0] * max_lengths['instance']] * num_placeholder

    data_doc['prev_inst'] = np.array(data_doc['prev_inst'], dtype='int32')
    data_doc['mask_prev_inst'] = np.array(data_doc['mask_prev_inst'], dtype='int32')


def process_cluster(data_doc, max_lengths, coref, num_inst):
    num_placeholder = max_lengths['instance'] - num_inst
    map_inst_to_cluster = dict()
    cluster_offset = 0
    starting_hv = [0] * len(coref) + [-1] * (max_lengths['cluster'] - len(coref))
    inst_init = [0] * num_inst

    for cluster_index, chain in enumerate(coref):
        inst_init[chain[0]] = 1

        for inst in chain:
            map_inst_to_cluster[inst] = cluster_index

        data_doc['cluster'] += chain
        mask = [0] + [1] * (len(chain) - 1)
        data_doc['mask_rnn'] += mask

        starting_hv[cluster_index] = cluster_offset
        cluster_offset += len(chain)

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
    data_doc['prev_inst_coref'] = np.zeros(max_lengths['instance'] * max_lengths['instance'], dtype='int32')

    for inst_curr in range(num_inst):
        chain = coref[map_inst_to_cluster[inst_curr]]
        if chain[0] == inst_curr:
            data_doc['prev_inst_cluster_gold'][inst_curr][inst_curr] = inst_curr
            data_doc['prev_inst_coref'][inst_curr][inst_curr] = 1
            continue
        for inst_prev in chain:
            if inst_prev >= inst_curr:
                break
            data_doc['prev_inst_cluster_gold'][inst_curr][inst_prev] = inst_prev
            data_doc['prev_inst_coref'][inst_curr][inst_prev] = 1

    data_doc['cluster'] += [-1] * num_placeholder
    data_doc['mask_rnn'] += [0] * num_placeholder
    data_doc['mask_cluster'] = [1] * num_inst + [0] * num_placeholder
    data_doc['current_hv'] += [[-1] * max_lengths['cluster']] * num_placeholder
    data_doc['mask_current_hv'] += [[0] * max_lengths['cluster']] * num_placeholder
    data_doc['prev_inst_cluster'] += [[-1] * max_lengths['instance']] * (max_lengths['instance'] - num_inst)

    for item in ['cluster', 'mask_cluster', 'current_hv', 'mask_current_hv']:
        data_doc[item] = np.array(data_doc[item], dtype='int32')


def main(path_dataset='/scratch/wl1191/event_coref/data/sample/nugget.pkl',
         expected_features=OrderedDict([('anchor', 0),
                                       ('pos', -1),
                                       ('chunk', -1),
                                       ('possibleTypes', -1),
                                       ('dep', 1),
                                       ('nonref', -1),
                                       ('title', -1),
                                       ('eligible', -1)]),
         use_pretrained_emb=True,
         pipeline=False,
         path_out='/scratch/wl1191/event_coref/data/sample/'):
    print '\nLoading dataset:', path_dataset, '...\n'
    max_lengths, corpora, embeddings, map_fea_to_index = cPickle.load(open(path_dataset, 'rb'))

    prepare_word_embeddings(use_pretrained_emb, embeddings)
    features, features_dim = prepare_features(expected_features, embeddings, map_fea_to_index)
    prefix = 'pipe_' if pipeline else 'gold_'
    data_sets = prepare_data(corpora, prefix, map_fea_to_index, features, features_dim, max_lengths)
    cPickle.dump(data_sets, open(path_out + 'data_sets.pkl', 'w'))


if __name__ == '__main__':
    main()
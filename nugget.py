from collections import OrderedDict, defaultdict
import numpy as np
import cPickle


np.set_printoptions(threshold=np.nan)


def prepare_word_embeddings(with_word_embs, embeddings):
    if not with_word_embs:
        word_vecs = embeddings['word_random']
    else:
        print 'Using word embeddings to initialize the network ...'
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
            print 'Using feature:', fea, ': embeddings'
        elif expected_features[fea] == 1:
            print 'Using feature:', fea, ': binary'
    return features


def get_dim_mapping(embeddings, map_fea_to_index, features):
    map_binary_dim = {}
    for fea in features:
        if fea == 'word':
            continue
        if fea == 'anchor':
            binary_dim = embeddings['anchor'].shape[0] - 1
        elif fea == 'possibleTypes' or fea == 'dep':
            binary_dim = len(map_fea_to_index[fea])
        else:
            binary_dim = len(map_fea_to_index[fea]) - 1
        map_binary_dim[fea] = binary_dim
    return map_binary_dim


def prepare_data(max_lengths, corpora, embeddings, map_fea_to_index, features, map_binary_dim, alphas):
    data_sets = {}
    for corpus in corpora:
        data_sets[corpus] = []
        for doc in corpora[corpus]:
            data_doc = defaultdict(list)
            inst_in_doc = corpora[corpus][doc]['instances']
            for inst in inst_in_doc:
                ret = add_instance(data_doc, inst, map_fea_to_index, features, map_binary_dim)
                if not ret == True:
                    print 'Error in %s corpus in document %s: cannot find index for word %s\n', corpus, doc, ret
                    exit(0)
            num_placeholder = max_lengths['instance'] - len(inst_in_doc)
            window = len(inst_in_doc[0]['word'])
            for i in range(num_placeholder):
                add_instance_placeholder(data_doc, features, map_fea_to_index, map_binary_dim, window)

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

            process_cluster(data_doc, max_lengths, corpora[corpus][doc]['coreference'], len(inst_in_doc), num_placeholder, alphas)

            data_doc['doc_id'] = doc
            data_doc['inst_id_to_index'] = corpora[corpus][doc]['inst_id_to_index']
            data_sets[corpus] += [data_doc]

    return data_sets


def add_instance(data_doc, inst, map_fea_to_index, features, map_binary_dim):
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
                anchor_scalar = index + 1 if is_placeholder else 0
                anchor_vector = [0] * map_binary_dim['anchor']
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
            fea_vector = [0] * map_binary_dim[fea]
            if fea_scalar > 0:
                fea_vector[0] = fea_scalar - 1
            data_inst[fea].append(fea_vector if features[fea] == 1 else fea_scalar)

    for fea in data_inst:
        data_doc[fea] += [data_inst[fea]]

    return True


def add_instance_placeholder(data_doc, features, map_fea_to_index, map_binary_dim, window):
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
                anchor_vector = [0] * map_binary_dim['anchor']
                data_inst['anchor'].append(anchor_vector if features['anchor'] == 1 else anchor_scalar)
                continue

            if fea == 'possibleTypes' or fea == 'dep':
                fea_vector = [0] * (num_dep if fea == 'dep' else num_possible_types)
                data_inst[fea].append(fea_vector)
                continue

            fea_scalar = 0
            fea_vector = [0] * map_binary_dim[fea]
            data_inst[fea].append(fea_vector if features[fea] == 1 else fea_scalar)

    for fea in data_inst:
        data_doc[fea] += [data_inst[fea]]


def process_cluster(data_doc, max_lengths, coref, num_inst, num_placeholder, alphas):
    map_inst_to_cluster = {}
    cluster_offset = 0
    current_hv = [0] * len(coref) + [-1] * (max_lengths['cluster'] - len(coref))
    inst_init = [0] * num_inst
    index = 0
    for chain in coref:
        inst_init[chain[0]] = 1
        for inst in chain:
            map_inst_to_cluster[inst] = index

        data_doc['cluster'] += chain
        mask = [0] + [1] * (len(chain) - 1)
        data_doc['mask_rnn'] += mask

        current_hv[index] = cluster_offset
        cluster_offset += len(chain)
        index += 1

    for i in range(num_inst):
        data_doc['current_hv'] += [current_hv[:]]
        current_hv[map_inst_to_cluster[i]] += 1
    mask_current_hv = [[1] * len(coref) + [0] * (max_lengths['cluster'] - len(coref))] * num_inst

    prev_inst_cluster = [0] + [-1] * (max_lengths['instance'] - 1)
    for i in range(1, num_inst):
        data_doc['prev_inst_cluster'] += [prev_inst_cluster[:]]
        prev_inst_cluster[i] = 0
        prev_inst_cluster[i - 1] = map_inst_to_cluster[i - 1] + 1
    data_doc['prev_inst_cluster'] += [prev_inst_cluster[:]]

    data_doc['prev_inst_cluster_gold'] = np.array([[-1] * max_lengths['instance']] * max_lengths['instance'], dtype='int32')
    for inst_curr in range(num_inst):
        chain = coref[map_inst_to_cluster[inst_curr]]
        for inst_prev in chain:
            if inst_prev > inst_curr:
                break
            data_doc['prev_inst_cluster_gold'][inst_curr][inst_prev] = inst_prev

    data_doc['cluster'] += [-1] * num_placeholder
    data_doc['mask_rnn'] += [0] * num_placeholder
    data_doc['mask_cluster'] = [1] * num_inst + [0] * num_placeholder
    data_doc['current_hv'] += [[-1] * max_lengths['cluster']] * num_placeholder
    data_doc['mask_current_hv'] = mask_current_hv + [[0] * max_lengths['cluster']] * num_placeholder
    data_doc['prev_inst_cluster'] += [[0] * max_lengths['instance']] * (max_lengths['instance'] - num_inst)

    data_doc['alpha'] = get_penalty_rates(inst_init, max_lengths, alphas)


def get_penalty_rates(inst_init, max_lengths, alphas):
    penalty_rates = np.array([[-np.inf] * max_lengths['instance']] * len(inst_init), dtype='float32')
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
         model='non_consecutive_cnn',
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
         dropout=0.5,
         lr=0.01,
         norm_lim=9.0,
         alphas=(0.5, 1.2, 1),
         batch=2):

    print 'Loading dataset:', dataset_path, '...'
    max_lengths, corpora, embeddings, map_fea_to_index = cPickle.load(open(dataset_path, 'rb'))

    prepare_word_embeddings(with_word_embs, embeddings)
    features = prepare_features(expected_features)
    map_binary_dim = get_dim_mapping(embeddings, map_fea_to_index, features)
    data_sets = prepare_data(max_lengths, corpora, embeddings, map_fea_to_index, features, map_binary_dim, alphas)
    print data_sets['train'][0]['alpha']
    print corpora['train'][data_sets['train'][0]['doc_id']]['coreference']


if __name__ == '__main__':
    main()

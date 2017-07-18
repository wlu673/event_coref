from collections import OrderedDict, defaultdict
import numpy as np
import cPickle


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


def prepare_data(max_lengths, corpora, embeddings, map_fea_to_index, features, map_binary_dim):
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

            process_cluster(data_doc, max_lengths, corpora[corpus][doc]['coreference'])

            data_doc['id'] = doc
            data_doc['event_id'] = corpora[corpus][doc]['event_id']
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
    data_doc['binaryFeatures'] += [inst['binaryFeatures']]

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
    data_doc['binaryFeatures'] += [[]]


def process_cluster(data_doc, max_lengths, coref):
    pass


def main(dataset_path='/scratch/wl1191/event_coref/data/nugget.pkl',
         window=31,
         expected_features=OrderedDict([('anchor', 0),
                                        ('pos', -1),
                                        ('chunk', -1),
                                        ('possibleTypes', -1),
                                        ('dep', 1),
                                        ('nonref', -1),
                                        ('title', -1),
                                        ('eligible', -1)]),
         with_word_embs=True,
         batch=2):

    print 'Loading dataset:', dataset_path, '...'
    max_lengths, corpora, embeddings, map_fea_to_index = cPickle.load(open(dataset_path, 'rb'))

    prepare_word_embeddings(with_word_embs, embeddings)
    features = prepare_features(expected_features)
    map_binary_dim = get_dim_mapping(embeddings, map_fea_to_index, features)
    prepare_data(max_lengths, corpora, embeddings, map_fea_to_index, features, map_binary_dim)


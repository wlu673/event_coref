from collections import OrderedDict
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


def prepare_data(max_lengths, corpora, embeddings, map_fea_to_index, features, window):
    map_binary_dim, map_emb_index_dim = {}, {}
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

        emb_index_dim = 1
        if fea == 'possibleTypes':
            emb_index_dim = max_lengths['possible_types']
        if fea == 'dep':
            emb_index_dim = max_lengths['dep']
        map_emb_index_dim[fea] = emb_index_dim




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
    prepare_data(max_lengths, corpora, embeddings, map_fea_to_index, features, window)


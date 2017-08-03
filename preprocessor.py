from collections import defaultdict
import numpy as np
import cPickle


def build_data(src_dir, corpus_type, window):
    corpora = {}
    max_lengths = {'sentence': -1,
                   'instance': -1,
                   'cluster': -1,
                   'possible_types': -1,
                   'dep': -1}
    vocab = defaultdict(int)
    counters = {'sent_length': defaultdict(int),
                'num_of_inst': defaultdict(lambda: defaultdict(int)),
                'num_of_cluster': defaultdict(lambda: defaultdict(int)),
                'type': defaultdict(lambda: defaultdict(int)),
                'subtype': defaultdict(lambda: defaultdict(int)),
                'realis': defaultdict(lambda: defaultdict(int))}
    map_fea_to_index = {'pos': {'######': 0},
                        'chunk': {'######': 0, 'O': 1},
                        'possibleTypes': {'NONE': 0},
                        'dep': {'NONE': 0},
                        'nonref': {'######': 0, 'false': 1},
                        'title': {'######': 0, 'false': 1},
                        'eligible': {'######': 0, '0': 1},
                        'type': {'NONE': 0},
                        'subtype': {'NONE': 0},
                        'realis': {'other': 0}}
    fea_placeholder = {}

    current_doc = ''
    current_sent = defaultdict(list)
    sent_id = -1
    inst_in_doc = 0
    cluster_in_doc = 0
    for type_ in corpus_type:
        corpora[type_] = {}
        print 'Processing %s data' % type_
        with open(src_dir + '/raw/sample/' + type_ + '.txt', 'r') as fin:
            for line in fin:
                line = line.strip()

                if line.startswith('#BeginOfDocument'):
                    current_doc = line[(line.find(' ') + 1):]
                    corpora[type_][current_doc] = {'instances': [], 'coreference': [], 'inst_id_to_index': {}}
                    continue

                if line == '#EndOfDocument':
                    counters['num_of_inst'][type_][inst_in_doc] += 1
                    if max_lengths['instance'] < inst_in_doc:
                        max_lengths['instance'] = inst_in_doc
                    counters['num_of_cluster'][type_][cluster_in_doc] += 1
                    if max_lengths['cluster'] < cluster_in_doc:
                        max_lengths['cluster'] = cluster_in_doc

                    corpora[type_][current_doc]['coreference'] = sorted(corpora[type_][current_doc]['coreference'])

                    current_doc = ''
                    sent_id = -1
                    inst_in_doc = 0
                    cluster_in_doc = 0
                    continue

                if not line and not current_doc:
                    continue

                if not line:
                    sent_id += 1
                    sent_length = len(current_sent['word'])
                    counters['sent_length'][sent_length] += 1
                    if sent_length > max_lengths['sentence']:
                        max_lengths['sentence'] = sent_length
                    for anchor_index in range(len(current_sent['eventId'])):
                        event_id = current_sent['eventId'][anchor_index]
                        if not event_id == 'NONE':
                            inst = parse_inst(current_sent, fea_placeholder, anchor_index, window, sent_id)
                            corpora[type_][current_doc]['instances'] += [inst]
                            update_counters(type_, inst, counters, max_lengths)
                            corpora[type_][current_doc]['inst_id_to_index'][event_id] = inst_in_doc
                            inst_in_doc += 1
                    current_sent = defaultdict(list)
                    continue

                if line.startswith('@Coreference'):
                    chain = parse_coreference(line, corpora[type_][current_doc]['inst_id_to_index'])
                    if chain == 'Error':
                        print 'Incorrect coreference format in %s data:\nDocument: %s\n%s' % (type_, current_doc, line)
                        exit(0)
                    corpora[type_][current_doc]['coreference'] += [chain]
                    cluster_in_doc += 1
                    continue

                if not parse_line(line, current_sent, fea_placeholder, map_fea_to_index, vocab):
                    print 'Incorrect line format in %s data:\nDocument: %s\n%s' % (type_, current_doc, line)
                    exit(0)

    write_stats(src_dir, corpus_type, counters, map_fea_to_index, max_lengths)

    return max_lengths, corpora, map_fea_to_index, vocab


def parse_line(line, current_sent, fea_placeholder, map_fea_to_index, vocab):
    def update_map(msg, feature, feature_map):
        if feature not in feature_map:
            index = len(feature_map)
            feature_map[feature] = index
            # if msg:
            #     print '%s: %s --> id = %d' % (msg, feature, feature_map[feature])

    entries = line.split('\t')
    if len(entries) != 21:
        return False

    wordStart = int(entries[1])
    current_sent['wordStart'] += [wordStart]
    wordEnd = int(entries[2]) + 1
    current_sent['wordEnd'] += [wordEnd]

    word = entries[3].lower()
    current_sent['word'] += [word]
    vocab[word] += 1
    if '-' in word:
        for sub_word in word.split('-'):
            vocab[sub_word] += 1
    if 'word' not in fea_placeholder:
        fea_placeholder['word'] = '######'

    pos = entries[5]
    update_map('POS', pos, map_fea_to_index['pos'])
    current_sent['pos'] += [map_fea_to_index['pos'][pos]]
    if 'pos' not in fea_placeholder:
        fea_placeholder['pos'] = 0

    chunk = entries[6]
    update_map('CHUNK', chunk, map_fea_to_index['chunk'])
    current_sent['chunk'] += [map_fea_to_index['chunk'][chunk]]
    if 'chunk' not in fea_placeholder:
        fea_placeholder['chunk'] = 0

    possible_types = entries[9].split()
    for type_ in possible_types:
        update_map('POSSIBLE TYPE', type_, map_fea_to_index['possibleTypes'])
    current_sent['possibleTypes'] += [[map_fea_to_index['possibleTypes'][type_] for type_ in possible_types]]
    if 'possibleTypes' not in fea_placeholder: fea_placeholder['possibleTypes'] = []

    dep_paths = entries[12].split()
    for dep in dep_paths:
        update_map('DEP', dep, map_fea_to_index['dep'])
    current_sent['dep'] += [[map_fea_to_index['dep'][dep] for dep in dep_paths]]
    if 'dep' not in fea_placeholder:
        fea_placeholder['dep'] = []

    nonref = entries[13]
    update_map('NONREF', nonref, map_fea_to_index['nonref'])
    current_sent['nonref'] += [map_fea_to_index['nonref'][nonref]]
    if 'nonref' not in fea_placeholder:
        fea_placeholder['nonref'] = 0

    title = entries[14]
    update_map('TITLE', title, map_fea_to_index['title'])
    current_sent['title'] += [map_fea_to_index['title'][title]]
    if 'title' not in fea_placeholder:
        fea_placeholder['title'] = 0

    eligible = entries[15]
    update_map('ELIGIBLE', eligible, map_fea_to_index['eligible'])
    current_sent['eligible'] += [map_fea_to_index['eligible'][eligible]]
    if 'eligible' not in fea_placeholder:
        fea_placeholder['eligible'] = 0

    binary_features = entries[16].split()
    current_sent['binaryFeatures'] += [binary_features]

    event_type = entries[17]
    update_map('EVENT TYPE', event_type, map_fea_to_index['type'])
    current_sent['type'] += [map_fea_to_index['type'][event_type]]

    event_subtype = entries[18]
    update_map('EVENT SUBTYPE', event_subtype, map_fea_to_index['subtype'])
    current_sent['subtype'] += [map_fea_to_index['subtype'][event_subtype]]

    event_realis = entries[19]
    if event_realis == 'NONE':
        current_sent['realis'] += [-1]
    else:
        event_realis = event_realis.lower()
        update_map('EVENT REALIS', event_realis, map_fea_to_index['realis'])
        current_sent['realis'] += [map_fea_to_index['realis'][event_realis]]

    event_event_id = entries[20]
    current_sent['eventId'] += [event_event_id]

    return True


def parse_inst(current_sent, fea_placeholder, anchor_index, window, sent_id):
    sent_length = len(current_sent['word'])
    inst = {}

    lower = anchor_index - window / 2
    anchor_index_new = anchor_index - lower

    for i in range(window):
        id = i + lower
        for key in fea_placeholder:
            addent = fea_placeholder[key]
            if 0 <= id < sent_length:
                addent = current_sent[key][id]
            if key not in inst:
                inst[key] = []
            inst[key] += [addent]

    for key in current_sent:
        if key not in fea_placeholder:
            inst[key] = current_sent[key][anchor_index]

    inst['anchor'] = anchor_index_new
    inst['sentenceId'] = sent_id

    return inst


def update_counters(type_, inst, counters, max_lengths):
    event_type = inst['type']
    event_subtype = inst['subtype']
    event_realis = inst['realis']

    counters['type'][type_][event_type] += 1
    counters['subtype'][type_][event_subtype] += 1
    counters['realis'][type_][event_realis] += 1

    max_possible_types = max([len(item) for item in inst['possibleTypes']])
    if max_possible_types > max_lengths['possible_types']:
        max_lengths['possible_types'] = max_possible_types
    max_dep = max([len(item) for item in inst['dep']])
    if max_dep > max_lengths['dep']:
        max_lengths['dep'] = max_dep


def parse_coreference(line, inst_id_to_index):
    entries = line.split('\t')

    if len(entries) != 3:
        return 'Error'

    chain = []
    for event_id in entries[2].split(','):
        chain += [inst_id_to_index[event_id]]
    return sorted(chain)


def write_stats(src_dir, corpus_type, counters, map_fea_to_index, max_lengths):
    with open(src_dir + '/' + 'statistics.txt', 'w') as fout:
        header_width = 60
        print >> fout, '\n', 'Stats'.center(header_width, '='), '\n'
        print >> fout, 'Distribution of number of instances in the corpora:'
        for type_ in corpus_type:
            print >> fout, type_.center(header_width, '-')
            print >> fout, counters['num_of_inst'][type_]
            print >> fout, 'Total: ', sum(counters['num_of_inst'][type_].keys())
        print >> fout, '-' * header_width
        print >> fout, 'Max number of instances in one doc: ', max_lengths['instance']
        print >> fout, '\n', '=' * header_width, '\n'
        print >> fout, 'Distribution of number of clusters in the corpora:'
        for type_ in corpus_type:
            print >> fout, type_.center(header_width, '-')
            print >> fout, counters['num_of_cluster'][type_]
            print >> fout, 'Total: ', sum(counters['num_of_cluster'][type_].keys())
        print >> fout, '-' * header_width
        print >> fout, 'Max number of clusters in one doc: ', max_lengths['cluster']
        print >> fout, '\n', '=' * header_width, '\n'
        print >> fout, 'Distribution of sentence lengths in the corpora:'
        print >> fout, counters['sent_length']
        print >> fout, '-' * header_width
        print >> fout, 'Max length of a sentence: ', max_lengths['sentence']
        print >> fout, '\n', '=' * header_width, '\n'

        def print_feature_stats(fea):
            map_index_to_fea = {}
            for item in map_fea_to_index[fea]:
                map_index_to_fea[map_fea_to_index[fea][item]] = item
            print >> fout, 'Distribution of event %s in the corpora:' % fea
            for type_ in corpus_type:
                print >> fout, type_.center(header_width, '-')
                for item_index in map_index_to_fea:
                    print >> fout, '#', item_index, ' : ', counters[fea][type_][item_index]
            print >> fout, '\n', '=' * header_width, '\n'

        for fea in ['type', 'subtype', 'realis']:
            print_feature_stats(fea)


def create_word_embeddings(src_dir, w2v_file, vocab):
    print "Vocab size: " + str(len(vocab))
    print "Loading word embeddings..."
    dim_word_vecs, word_vecs = load_bin_vec(src_dir, w2v_file, vocab)
    print "Word embeddings loaded!"
    print "Number of words already in word embeddings: " + str(len(word_vecs))
    add_unknown_words(word_vecs, vocab, 1, dim_word_vecs)
    W_trained, word_index_map = get_W(word_vecs, dim_word_vecs)

    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab, 1, dim_word_vecs)
    W_random, _ = get_W(rand_vecs, dim_word_vecs)

    return W_trained, word_index_map, W_random


def load_bin_vec(src_dir, w2v_file, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    dim = 0
    with open(src_dir + '/' + w2v_file, 'rb') as fin:
        header = fin.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = fin.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
                word_vecs[word] = np.fromstring(fin.read(binary_len), dtype='float32')
                dim = word_vecs[word].shape[0]
            else:
                fin.read(binary_len)
    print 'Dim: ', dim
    return dim, word_vecs


def add_unknown_words(word_vecs, vocab, min_df=1, dim_word_vecs=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            added = False
            if '-' in word:
                for pword in word.split('-')[::-1]:
                    if pword in word_vecs:
                        word_vecs[word] = np.copy(word_vecs[pword])
                        added = True
            if not added:
                word_vecs[word] = np.random.uniform(-0.25, 0.25, dim_word_vecs)


def get_W(word_vecs, dim_word_vecs=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, dim_word_vecs))
    W[0] = np.zeros(dim_word_vecs)
    word_idx_map['######'] = 0
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map


def create_feature_embeddings(map_fea_to_index, embeddings, window):
    # Distance to mention position
    size_dist = window
    dim_dist = 50
    D = np.random.uniform(-0.25, 0.25, (size_dist + 1, dim_dist))
    D[0] = np.zeros(dim_dist)
    embeddings['anchor'] = D

    dim_fea = {'pos': 50,
               'chunk': 50,
               'nonref': 50,
               'title': 50,
               'eligible': 50}

    def create_embedding(fea):
        dim = dim_fea[fea]
        W = np.random.uniform(-0.25, 0.25, (len(map_fea_to_index[fea]), dim))
        W[0] = np.zeros(dim)
        embeddings[fea] = W

    for fea in dim_fea.keys():
        create_embedding(fea)

    for fea in map_fea_to_index:
        print 'Size of', fea, ': ', len(map_fea_to_index[fea])


def main():
    np.random.seed(8989)

    w2v_file = 'GoogleNews-vectors-negative300.bin'
    src_dir = '/scratch/wl1191/event_coref/data'
    corpus_type = ["train", "valid", "test"]
    window = 31

    print "\nLoading data..."
    max_lengths, corpora, map_fea_to_index, vocab = build_data(src_dir, corpus_type, window)
    print "Data loaded!"

    W_trained, word_index_map, W_random = create_word_embeddings(src_dir, w2v_file, vocab)
    map_fea_to_index['word'] = word_index_map
    embeddings = {'word': W_trained, 'word_random': W_random}

    create_feature_embeddings(map_fea_to_index, embeddings, window)

    print 'Dumping ...'
    cPickle.dump([max_lengths, corpora, embeddings, map_fea_to_index],
                 open(src_dir + '/' + 'nugget.pkl', 'wb'))
    # cPickle.dump([max_lengths, corpora], open(src_dir + '/' + 'corpora.pkl', 'wb'))
    print 'Dataset created!'


if __name__ == '__main__':
    main()

from collections import defaultdict
import numpy as np
import cPickle


###########################################################################
# Create General Data Sets

def create_general_data_sets(path_src, path_realis, path_w2v_bin, path_w2v_text, emb_type, corpus_type, window):
    print "\nLoading raw data..."
    max_lengths, corpora, map_fea_to_index, vocab = make_data_general(path_src, path_realis, corpus_type, window)
    print "Raw data loaded!"

    W_trained, word_index_map, W_random = create_word_embeddings(path_w2v_bin, path_w2v_text, emb_type, vocab)
    map_fea_to_index['word'] = word_index_map
    embeddings = {'word': W_trained, 'word_random': W_random}

    create_feature_embeddings(map_fea_to_index, embeddings, window)

    print 'Dumping ...'
    cPickle.dump([max_lengths, corpora, embeddings, map_fea_to_index], open(path_src + 'nugget.pkl', 'w'))
    cPickle.dump(corpora, open(path_src + 'corpora.pkl', 'w'))
    print 'General datasets created!'


def make_data_general(path_src, path_realis, corpus_type, window):
    corpora = {}
    max_lengths = {'sentence': -1,
                   'gold_instance': -1,
                   'pipe_instance': -1,
                   'gold_cluster': -1,
                   'pipe_cluster': -1,
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
    gold_inst_in_doc = 0
    pipe_inst_in_doc = 0
    gold_cluster_in_doc = 0

    for type_ in corpus_type:
        corpora[type_] = {}
        realis_output = load_realis_output(path_src, path_realis, type_)
        print 'Processing %s data' % type_
        with open(path_src + 'raw/' + type_ + '.txt', 'r') as fin:
            for line in fin:
                line = line.strip()

                if line.startswith('#BeginOfDocument'):
                    current_doc = line[(line.find(' ') + 1):]
                    corpora[type_][current_doc] = {'gold_instances': [],
                                                   'gold_coreference': [],
                                                   'gold_missing_inst': [],
                                                   'gold_inst_id_to_index': {},
                                                   'pipe_instances': [],
                                                   'pipe_coreference': [],
                                                   'pipe_missing_inst': [],
                                                   'pipe_inst_id_to_index': {}}
                    continue

                if line == '#EndOfDocument':
                    pipe_cluster_in_doc = 0
                    for coref_line in realis_output[current_doc]['coref']:
                        chain, missing_inst = parse_coreference(coref_line,
                                                                corpora[type_][current_doc]['pipe_inst_id_to_index'])
                        if chain == 'Error':
                            print 'Incorrect coreference format in realis output for %s:\nDocument: %s\n%s' % (
                            type_, current_doc, line)
                            exit(0)
                        if len(chain) > 0:
                            corpora[type_][current_doc]['pipe_coreference'] += [chain]
                        corpora[type_][current_doc]['pipe_missing_inst'] += [missing_inst]
                        pipe_cluster_in_doc += 1

                    counters['num_of_inst'][type_][gold_inst_in_doc] += 1
                    if max_lengths['gold_instance'] < gold_inst_in_doc:
                        max_lengths['gold_instance'] = gold_inst_in_doc
                    if max_lengths['pipe_instance'] < pipe_inst_in_doc:
                        max_lengths['pipe_instance'] = pipe_inst_in_doc
                    counters['num_of_cluster'][type_][gold_cluster_in_doc] += 1
                    if max_lengths['gold_cluster'] < gold_cluster_in_doc:
                        max_lengths['gold_cluster'] = gold_cluster_in_doc
                    if max_lengths['pipe_cluster'] < pipe_cluster_in_doc:
                        max_lengths['pipe_cluster'] = pipe_cluster_in_doc

                    corpora[type_][current_doc]['gold_coreference'] = sorted(
                        corpora[type_][current_doc]['gold_coreference'])
                    corpora[type_][current_doc]['pipe_coreference'] = sorted(
                        corpora[type_][current_doc]['pipe_coreference'])

                    current_doc = ''
                    sent_id = -1
                    gold_inst_in_doc = 0
                    pipe_inst_in_doc = 0
                    gold_cluster_in_doc = 0
                    continue

                if not line and not current_doc:
                    continue

                if not line:
                    sent_id += 1
                    sent_length = len(current_sent['word'])
                    counters['sent_length'][sent_length] += 1
                    if sent_length > max_lengths['sentence']:
                        max_lengths['sentence'] = sent_length
                    for anchor_index in range(len(current_sent['gold_eventId'])):
                        gold_event_id = current_sent['gold_eventId'][anchor_index]
                        if not gold_event_id == 'NONE':
                            gold_inst = parse_inst(current_sent, fea_placeholder, anchor_index, window, sent_id, 'gold')
                            corpora[type_][current_doc]['gold_instances'] += [gold_inst]
                            update_counters(type_, gold_inst, counters, max_lengths)
                            corpora[type_][current_doc]['gold_inst_id_to_index'][gold_event_id] = gold_inst_in_doc
                            gold_inst_in_doc += 1
                        pipe_event_id = current_sent['pipe_eventId'][anchor_index]
                        if not pipe_event_id == 'None':
                            pipe_inst = parse_inst(current_sent, fea_placeholder, anchor_index, window, sent_id, 'pipe')
                            corpora[type_][current_doc]['pipe_instances'] += [pipe_inst]
                            corpora[type_][current_doc]['pipe_inst_id_to_index'][pipe_event_id] = pipe_inst_in_doc
                            pipe_inst_in_doc += 1
                    current_sent = defaultdict(list)
                    continue

                if line.startswith('@Coreference'):
                    chain, missing_inst = parse_coreference(line, corpora[type_][current_doc]['gold_inst_id_to_index'])
                    if chain == 'Error':
                        print 'Incorrect coreference format in %s data:\nDocument: %s\n%s' % (type_, current_doc, line)
                        exit(0)
                    if len(chain) > 0:
                        corpora[type_][current_doc]['gold_coreference'] += [chain]
                    corpora[type_][current_doc]['gold_missing_inst'] += [missing_inst]
                    gold_cluster_in_doc += 1
                    continue

                if not parse_line(line,
                                  realis_output[current_doc]['instances'],
                                  current_sent,
                                  fea_placeholder,
                                  map_fea_to_index,
                                  vocab):
                    print 'Incorrect line format in %s data:\nDocument: %s\n%s' % (type_, current_doc, line)
                    exit(0)

    write_stats(path_src, corpus_type, counters, map_fea_to_index, max_lengths)

    return max_lengths, corpora, map_fea_to_index, vocab


def load_realis_output(path_src, path_realis, type_):
    realis_output = dict()
    with open(path_src + path_realis + type_ + '.realis', 'r') as fin:
        current_doc = ''
        instances = dict()
        coref = []
        for line in fin:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#BeginOfDocument'):
                current_doc = line.rstrip('\n').split()[1]
            elif line.startswith('#EndOfDocument'):
                realis_output[current_doc] = {'instances': instances, 'coref': coref}
                instances = dict()
                coref = []
            elif line.startswith('@Coreference'):
                coref += [line]
            else:
                entries = line.split('\t')
                event_id = entries[2].replace('E', 'em-')
                start_end = entries[3]
                event_type, event_subtype = entries[5].split('_')
                realis = entries[6]
                instances[start_end] = {'eventId': event_id,
                                        'type': event_type,
                                        'subtype': event_subtype,
                                        'realis': realis}
    return realis_output


def parse_line(line, realis_output_doc, current_sent, fea_placeholder, map_fea_to_index, vocab):
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
    current_sent['gold_type'] += [map_fea_to_index['type'][event_type]]

    event_subtype = entries[18]
    update_map('EVENT SUBTYPE', event_subtype, map_fea_to_index['subtype'])
    current_sent['gold_subtype'] += [map_fea_to_index['subtype'][event_subtype]]

    event_realis = entries[19]
    if event_realis == 'NONE':
        current_sent['gold_realis'] += [-1]
    else:
        event_realis = event_realis.lower()
        update_map('EVENT REALIS', event_realis, map_fea_to_index['realis'])
        current_sent['gold_realis'] += [map_fea_to_index['realis'][event_realis]]

    event_id = entries[20]
    current_sent['gold_eventId'] += [event_id]

    key = str(wordStart) + ',' + str(wordEnd)
    if key in realis_output_doc:
        inst = realis_output_doc[key]
        for msg, item in zip(['EVENT TYPE', 'EVENT SUBTYPE', 'EVENT REALIS'], ['type', 'subtype', 'realis']):
            update_map(msg, inst[item], map_fea_to_index[item])
            current_sent['pipe_' + item] += [map_fea_to_index[item][inst[item]]]
        current_sent['pipe_eventId'] += [inst['eventId']]
    else:
        for item in ['type', 'subtype', 'realis', 'eventId']:
            current_sent['pipe_' + item] += ['None']

    return True


def parse_inst(current_sent, fea_placeholder, anchor_index, window, sent_id, prefix):
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
            if '_' in key:
                entries = key.split('_')
                if entries[0] == prefix:
                    inst[entries[1]] = current_sent[key][anchor_index]
            else:
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
    missing_inst = {}
    for event_id in entries[2].split(','):
        event_id = event_id.replace('E', 'em-')
        if event_id in inst_id_to_index:
            chain += [inst_id_to_index[event_id]]
        else:
            missing_inst[event_id] = None
    if len(chain) > 0:
        for event_id in missing_inst:
            missing_inst[event_id] = chain[0]
    return sorted(chain), missing_inst


def write_stats(path_src, corpus_type, counters, map_fea_to_index, max_lengths):
    with open(path_src + 'statistics.txt', 'w') as fout:
        header_width = 60
        print >> fout, '\n', 'Stats'.center(header_width, '='), '\n'
        print >> fout, 'Distribution of number of instances in the corpora:'
        for type_ in corpus_type:
            print >> fout, type_.center(header_width, '-')
            print >> fout, counters['num_of_inst'][type_]
            print >> fout, 'Total: ', sum(counters['num_of_inst'][type_].keys())
        print >> fout, '-' * header_width
        print >> fout, 'Max number of golden instances in one doc: ', max_lengths['gold_instance']
        print >> fout, 'Max number of pipeline instances in one doc: ', max_lengths['pipe_instance']
        print >> fout, '\n', '=' * header_width, '\n'
        print >> fout, 'Distribution of number of clusters in the corpora:'
        for type_ in corpus_type:
            print >> fout, type_.center(header_width, '-')
            print >> fout, counters['num_of_cluster'][type_]
            print >> fout, 'Total: ', sum(counters['num_of_cluster'][type_].keys())
        print >> fout, '-' * header_width
        print >> fout, 'Max number of golden clusters in one doc: ', max_lengths['gold_cluster']
        print >> fout, 'Max number of pipelien clusters in one doc: ', max_lengths['pipe_cluster']
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


def create_word_embeddings(path_w2v_bin, path_w2v_text, emb_type, vocab):
    print "Vocab size: " + str(len(vocab))
    print "Loading word embeddings..."
    if emb_type == 'word2vec':
        dim_word_vecs, word_vecs = load_bin_vec(path_w2v_bin, vocab)
    else:
        dim_word_vecs, word_vecs = load_text_vec(path_w2v_text, vocab)
    print "Word embeddings loaded!"
    print "Number of words already in word embeddings: " + str(len(word_vecs))
    add_unknown_words(word_vecs, vocab, 1, dim_word_vecs)
    W_trained, word_index_map = get_W(word_vecs, dim_word_vecs)

    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab, 1, dim_word_vecs)
    W_random, _ = get_W(rand_vecs, dim_word_vecs)

    return W_trained, word_index_map, W_random


def load_bin_vec(w2v_file, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    dim = 0
    with open(w2v_file, 'rb') as fin:
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
    print 'Word embedding dim:', dim
    return dim, word_vecs


def load_text_vec(w2v_file, vocab):
    word_vecs = {}
    count = 0
    dim = 0
    with open(w2v_file, 'r') as fin:
        for line in fin:
            count += 1
            line = line.strip()
            if count == 1:
                if len(line.split()) < 10:
                    dim = int(line.split()[1])
                    print 'Word embedding dim:', dim
                    continue
                else:
                    dim = len(line.split()) - 1
                    print 'Word embedding dim:', dim
            word = line.split()[0]
            em_str = line[(line.find(' ') + 1):]
            if word in vocab:
                word_vecs[word] = np.fromstring(em_str, dtype='float32', sep=' ')
                if word_vecs[word].shape[0] != dim:
                    print 'Found a word with mismatched dimension:', dim, word_vecs[word].shape[0]
                    exit()
    print 'Word embedding dim:', dim
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
               'eligible': 50,
               'type': 50,
               'subtype': 50,
               'realis': 50}

    def create_embedding(fea):
        dim = dim_fea[fea]
        W = np.random.uniform(-0.25, 0.25, (len(map_fea_to_index[fea]), dim))
        W[0] = np.zeros(dim)
        embeddings[fea] = W

    for fea in dim_fea.keys():
        create_embedding(fea)

    for fea in map_fea_to_index:
        print 'Size of', fea, ': ', len(map_fea_to_index[fea])


def main(path_src='/scratch/wl1191/event_coref/data/',
         path_realis='realis/',
         path_w2v_bin='/scratch/wl1191/event_coref/data/GoogleNews-vectors-negative300.bin',
         path_w2v_text='/scratch/wl1191/event_coref/data/concatEmbeddings.txt',
         emb_type='text',
         corpus_type=['train', 'test', 'valid'],
         window=31):
    np.random.seed(8989)
    create_general_data_sets(path_src, path_realis, path_w2v_bin, path_w2v_text, emb_type, corpus_type, window)


if __name__ == '__main__':
    main()

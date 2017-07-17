from collections import defaultdict
import numpy as np
import random


def build_data(src_dir, corpus_type, window):
    corpra = {}
    max_inst_in_doc = -1
    max_cluster_in_doc = -1
    max_sent_length = -1
    vocab = defaultdict(int)
    counters = {'sent_length': defaultdict(int),
                'num_of_inst': defaultdict(int),
                'num_of_cluster': defaultdict(int),
                'event_type': defaultdict(lambda: defaultdict(int)),
                'event_subtype': defaultdict(lambda: defaultdict(int)),
                'realis': defaultdict(lambda: defaultdict(int))}
    fea_to_index_map = {'pos': {'######': 0},
                        'chunck': {'######': 0, 'O': 1},
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
        corpra[type_] = {}
        print 'processing %s data' % type_
        with open(src_dir + '/' + type_ + '.txt', 'r') as fin:
            for line in fin:
                line = line.strip()

                if line.startswith('#BeginOfDocument'):
                    current_doc = line[(line.find(' ') + 1):]
                    corpra[type_][current_doc] = {'instances': [], 'coreference': []}
                    continue

                if line == '#EndOfDocument':
                    counters['num_of_inst'][inst_in_doc] += 1
                    if max_inst_in_doc < inst_in_doc:
                        max_inst_in_doc = inst_in_doc
                    counters['num_of_cluster'][cluster_in_doc] += 1
                    if max_cluster_in_doc < cluster_in_doc:
                        max_cluster_in_doc = cluster_in_doc

                    currentDoc = ''
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
                    if sent_length > max_sent_length:
                        max_sent_length = sent_length
                    for anchor_index in range(current_sent['eventId']):
                        event_id = current_sent['eventId'][anchor_index]
                        if not event_id == 'NONE':
                            inst = parse_inst(current_sent, fea_placeholder, anchor_index, window, sent_id)
                            corpra[type_][current_doc]['instances'] += [inst]
                            corpra[type_][current_doc]['event_id'] += [event_id]
                            update_counters(type_, inst, counters)
                            inst_in_doc += 1
                    current_sent = defaultdict(list)
                    continue

                if line.startswith('@Coreference'):
                    chain = parse_coreference(line)
                    if chain == 'Error':
                        print 'Incorrect coreference format in %s data:\nDocument: %s\n%s' % (type_, current_doc, line)
                        exit(0)
                    corpra[type_][current_doc]['coreference'] += [chain]
                    cluster_in_doc += 1
                    continue

                if not parse_line(line, current_sent, fea_placeholder, fea_to_index_map, vocab):
                    print 'Incorrect line format in %s data:\nDocument: %s\n%s' % (type_, current_doc, line)
                    exit(0)


def update_counters(type_, inst, counters):
    event_type = inst['type']
    event_subtype = inst['subtype']
    event_realis = inst['realis']

    counters['event_type'][type_][event_type] += 1
    counters['event_subtype'][type_][event_subtype] += 1
    counters['realis'][type_][event_realis] += 1

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


def parse_coreference(line):
    entries = line.split('\t')

    if len(entries) != 3:
        return 'Error'

    chain = entries[2].split(',')
    return chain


def parse_line(line, current_sent, fea_placeholder, fea_to_index_map, vocab):
    def update_map(msg, feature, feature_map):
        if feature not in feature_map:
            index = len(feature_map)
            feature_map[feature] = index
            if msg:
                print '%s: %s --> id = %d' % (msg, feature, feature_map[index])

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
    update_map('POS', pos, fea_to_index_map['pos'])
    current_sent['pos'] += [fea_to_index_map['pos'][pos]]
    if 'pos' not in fea_placeholder:
        fea_placeholder['pos'] = 0

    chunk = entries[6]
    update_map('CHUNK', chunk, fea_to_index_map['chunk'])
    current_sent['chunk'] += [fea_to_index_map['chunk'][chunk]]
    if 'chunk' not in fea_placeholder:
        fea_placeholder['chunk'] = 0

    possible_types = entries[9].split()
    for type_ in possible_types:
        update_map('POSSIBLE TYPE', type_, fea_to_index_map['possibleTypes'])
    current_sent['possibleTypes'] += [[fea_to_index_map['possibleTypes'][type_] for type_ in possible_types]]
    if 'possibleTypes' not in fea_placeholder: fea_placeholder['possibleTypes'] = []

    dep_paths = entries[12].split()
    for dep in dep_paths:
        update_map('DEP', dep, fea_to_index_map['dep'])
    current_sent['dep'] += [[fea_to_index_map['dep'][dep] for dep in dep_paths]]
    if 'dep' not in fea_placeholder:
        fea_placeholder['dep'] = []

    nonref = entries[13]
    update_map('NONREF', nonref, fea_to_index_map['nonref'])
    current_sent['nonref'] += [fea_to_index_map['nonref'][nonref]]
    if 'nonref' not in fea_placeholder:
        fea_placeholder['nonref'] = 0

    title = entries[14]
    update_map('TITLE', title, fea_to_index_map['title'])
    current_sent['title'] += [fea_to_index_map['title'][title]]
    if 'title' not in fea_placeholder:
        fea_placeholder['title'] = 0

    eligible = entries[15]
    update_map('ELIGIBLE', eligible, fea_to_index_map['eligible'])
    current_sent['eligible'] += [fea_to_index_map['eligible'][eligible]]
    if 'eligible' not in fea_placeholder:
        fea_placeholder['eligible'] = 0

    binary_features = entries[16].split()
    current_sent['binaryFeatures'] += [binary_features]

    event_type = entries[17]
    update_map('EVENT TYPE', event_type, fea_to_index_map['type'])
    current_sent['type'] += [fea_to_index_map['type'][event_type]]

    event_subtype = entries[18]
    update_map('EVENT SUBTYPE', event_subtype, fea_to_index_map['subtype'])
    current_sent['subtype'] += [fea_to_index_map['subtype'][event_subtype]]

    event_realis = entries[19]
    if event_realis == 'NONE':
        current_sent['realis'] += [-1]
    else:
        event_realis = event_realis.lower()
        update_map('EVENT REALIS', event_realis, fea_to_index_map['realis'])
        current_sent['realis'] += [fea_to_index_map['realis'][event_realis]]

    event_event_id = entries[20]
    current_sent['eventId'] += [event_event_id]

    return True


def main():
    np.random.seed(8989)
    random.seed(8989)
    # embType = sys.argv[1]
    # w2v_file = sys.argv[2]
    # srcDir = sys.argv[3]
    emb_type = 'word2vec'
    w2v_file = 'GoogleNews-vectors-negative300.bin'
    src_dir = '.'

    corpus_type = ["train", "valid", "test"]
    window = 31
    print "loading data...\n"
    # revs, fea_to_index_map, vocab, nodeFetCounter = build_data(src_dir, corpus_type, window)


if __name__ == '__main__':
    main()

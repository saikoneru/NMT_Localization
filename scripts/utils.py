import os, sys, argparse, regex as re, numpy as np
from sklearn.utils import shuffle
from pathlib import Path



def test_apps(cnts, test_len):
    curr_cnt = 0
    ids = np.arange(len(cnts))
    rand_ids = shuffle(ids, random_state=0)
    iter_cnt = 0
    test_appids = []
    while curr_cnt < test_len:
        cur_idx = rand_ids[iter_cnt]
        if( (cnts[cur_idx] > 100 ) and (cnts[cur_idx] < 300)):
            test_appids.append(rand_ids[iter_cnt])
            curr_cnt += cnts[rand_ids[iter_cnt]]
            if curr_cnt >= test_len / 2:
                break
        iter_cnt += 1
    
    test_appids.pop()
    test_appids.sort()
    test_applens = [cnts[x] for x in test_appids]
    print(test_applens)
    test_ids = test_sents(cnts, test_appids)
    valid_appids = []
    while curr_cnt < test_len:
        cur_idx = rand_ids[iter_cnt]
        if( (cnts[cur_idx] > 100 ) and (cnts[cur_idx] < 300)):
            valid_appids.append(rand_ids[iter_cnt])
            curr_cnt += cnts[rand_ids[iter_cnt]]
        iter_cnt += 1

    valid_appids.pop()
    valid_appids.sort()
    valid_ids = test_sents(cnts, valid_appids)
    valid_applens = [cnts[x] for x in valid_appids]
    print(valid_applens)
    return (valid_ids, test_ids)


def test_sents(cnts, test_appids):
    test_ids = []
    sum_cnts = part_sums(cnts)
    for appid in test_appids:
        test_ids.extend(list(range(sum_cnts[appid], sum_cnts[(appid + 1)])))

    return test_ids


def part_sums(ls):
    result = ls + [0]
    for i in range(len(result) - 1, 0, -1):
        result[i - 1] = result[i] + result[(i - 1)]

    result = [-(x - result[0]) for x in result]
    return result

def get_left(idx, lst, n):
    start = idx - n
    return [lst[(start + x) % len(lst)] for x in range(n)]

def get_right(idx, lst, n):
    start = idx + 1
    return [lst[(start + x) % len(lst)] for x in range(n)]

def add_side(data, window):
    data = [str(x) for x in data]
    cxt = [' '.join(get_left(idx,data,window)) +  ' <tag> ' + ' '.join(get_right(idx,data,window)) + ' <tag> ' for idx in range(len(data))]
    return cxt


def clean_data(sents):
    clean_sents = []
    for sent in sents:
        sent_newlines = re.sub('\\s{2,}', ' ', sent.replace('\n', ''))
        sent_quotes = re.sub('\\s{2,}', ' ', sent_newlines.replace('"', ''))
        sent_break = sent_quotes.replace('_', '')
        clean_sents.append(sent_break)

    return clean_sents


def read_po_file(po_file):
    data = open(po_file, 'r').read()
    entries = re.findall('#:([^#]*)', data, re.DOTALL)
    context = []
    src = []
    tgt = []
    prefix_context = ''
    for entry in entries:
        if not ('msgid' in entry and 'msgstr' in entry):
            prefix_context += entry.split('\n')[0]
        else:
            entry_cxt = prefix_context + entry.split('\n')[0]
            context.append(entry_cxt.strip())
            entry_src = re.findall('msgid.*?msgstr', entry, re.DOTALL)[0]
            entry_src = entry_src[len('msgid'):len(entry_src) - len('\nmsgstr')]
            src.append(entry_src)
            entry_tgt = re.findall('(?<=msgstr).*', entry, re.DOTALL)[0]
            tgt.append(entry_tgt)
            prefix_context = ''

    clean_context = clean_data(context)
    clean_src = clean_data(src)
    clean_tgt = clean_data(tgt)
    empty_id = get_empty(clean_src)
    empty_id.extend(get_empty(clean_tgt))
    empty_id.extend(get_empty(clean_context))
    new_src = filter_data(empty_id, clean_src)
    new_tgt = filter_data(empty_id, clean_tgt)
    new_cxt = filter_data(empty_id, clean_context)
    return (
     new_cxt, new_src, new_tgt)

def read_saved_po(dirpath, basepath, src_lang, tgt_lang):
    saved_src = read_data(dirpath + basepath + ".full." + src_lang)
    saved_tgt = read_data(dirpath + basepath + ".full." + tgt_lang)
    saved_cxt = read_data(dirpath + basepath + ".full.cxt")
    saved_name = read_data(dirpath + basepath + ".full.name")
    return saved_src,saved_tgt,saved_cxt,saved_name

def write_data(src, tgt, output_dir, split, src_lang, tgt_lang):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(output_dir + '//' + split + '.' + src_lang, 'w') as (f):
        for item in src:
            f.write('%s\n' % item)

    with open(output_dir + '//' + split + '.' + tgt_lang, 'w') as (f):
        for item in tgt:
            f.write('%s\n' % item)



def write_cxt_data(df, output_dir, split, src_lang, tgt_lang,append_src):
    src = list(df['src'])
    tgt = list(df['tgt'])
    src_raw = list(df['src_raw'])
    tgt_raw = list(df['tgt_raw'])
    
    if append_src == 'neighbour':
        src_add = list(df['cxt_src'])
        tgt_add = list(df['cxt_tgt'])
    else:
        src_add = list(df['cxt'])
        tgt_add = list(df['cxt'])

    if append_src != "none":
        src_cxt = [m + " " + n for m,n in zip(src_add,src)]
        tgt_cxt = [m + " " + n for m,n in zip(tgt_add,tgt)]

    src_tgt_output = output_dir + '/' + src_lang + '-' + tgt_lang
    tgt_src_output = output_dir + '/' + tgt_lang + '-' + src_lang
    if append_src == "none":
        write_data(src, tgt, src_tgt_output, split, src_lang, tgt_lang)
        write_data(src_raw, tgt_raw, src_tgt_output + "/raw/", split, src_lang, tgt_lang)
        write_data(src, tgt, tgt_src_output, split, src_lang, tgt_lang)
        write_data(src_raw, tgt_raw, tgt_src_output + "/raw/", split, src_lang, tgt_lang)
    else:
        write_data(src_cxt, tgt, src_tgt_output, split, src_lang, tgt_lang)
        write_data(src, tgt_cxt, tgt_src_output, split, src_lang, tgt_lang)
    return

def filter_long(df, max_size, append_src):
    if append_src != "neighbour":
        df.loc[:,"cxt_src_lens"] = df["cxt"] + " " + df["src"]
        df.loc[:,"cxt_tgt_lens"] = df["cxt"] + " " + df["tgt"]
    else:
        df.loc[:,"cxt_src_lens"] = df["cxt_src"] + " " + df["src"]
        df.loc[:,"cxt_tgt_lens"] = df["cxt_tgt"] + " " + df["tgt"]
    df.loc[:,'cxt_src_lens'] = df['cxt_src_lens'].map(lambda x: len(x.split(" ")))
    df.loc[:,'cxt_tgt_lens'] = df['cxt_tgt_lens'].map(lambda x: len(x.split(" ")))
    mask = (df['cxt_src_lens'] < max_size ) & (df['cxt_tgt_lens'] < max_size)
    df = df.loc[mask]
    return df

def write_list(data, output_path):
    with open(output_path, 'w') as (f):
        for item in data:
            f.write('%s\n' % item)


def read_data(filepath):
    data = open(filepath, mode='r', encoding='utf-8', newline='\n').readlines()
    data = [x.rstrip() for x in data]
    return data


def read_eval_data(folderpath, src, tgt):
    valid_src = read_data(folderpath + '/valid.' + src)
    valid_tgt = read_data(folderpath + '/valid.' + tgt)
    test_src = read_data(folderpath + '/test.' + src)
    test_tgt = read_data(folderpath + '/test.' + tgt)
    return (valid_src, valid_tgt, test_src, test_tgt)


def overlap_ids(train, valid, test):
    i = 0
    overlaps = []
    valid_dict = {}
    test_dict = {}
    for v in valid:
        valid_dict[v] = v

    for t in test:
        test_dict[t] = t

    for example in train:
        if example in valid_dict or example in test_dict:
            overlaps.append(i)
        i += 1

    return overlaps


def remove_overlap(train_src, valid_src, test_src, train_tgt, train_cxt):
    i = 0
    overlaps = []
    for example in train_src:
        if example in valid_src or example in test_src:
            overlaps.append(i)
        i += 1

    return (
     filter_data(overlaps, train_src), filter_data(overlaps, train_tgt), filter_data(overlaps, train_cxt))


def filter_data(ids, data):
    ids_dict = {}
    for i in ids:
        ids_dict[i] = i

    return [data[i] for i in range(len(data)) if i not in ids_dict]


def select_data(src, tgt, ids):
    new_src = [src[i] for i in ids]
    new_tgt = [tgt[i] for i in ids]
    return (
     new_src, new_tgt)


def split_data(data, train, valid, test):
    train_data = [data[x] for x in train]
    valid_data = [data[x] for x in valid]
    test_data = [data[x] for x in test]
    return (train_data, valid_data, test_data)


def get_empty(data):
    ids = []
    data = [x.strip() for x in data]
    for i in range(len(data)):
        if data[i] == '' or len(data[i]) > 200:
            ids.append(i)
        if data[i] == 'translator-credits':
            ids.append(i)

    return ids


def find_pairs(src_full, tgt_full, src_part, tgt_part):
    full = [i + ' ' + j for i, j in zip(src_full, tgt_full)]
    part = [i + ' ' + j for i, j in zip(src_part, tgt_part)]
    ids = []
    for entry in part:
        ids.append(full.index(entry))

    return ids

def find_unique(src,tgt):
    ids = []
    unique_src = []
    unique_tgt = []
    dict_ids = {}
    
    pair_tuples = list(zip(src,tgt))
    for i in range(len(pair_tuples)):
        dict_ids[pair_tuples[i]] = i

    unique_tuples = list(dict.fromkeys(pair_tuples))

    for unique in unique_tuples:
        idx = dict_ids[unique]
        ids.append(idx)
        unique_src.append(src[idx])
        unique_tgt.append(tgt[idx])
    return ids, unique_src, unique_tgt

def create_eval(eval_src, eval_tgt, eval_path, eval_srclang, eval_tgtlang):
    total = len(eval_src)
    eval_ids = np.arange(total)
    valid = np.random.choice(eval_ids, (int(len(eval_ids) * 0.5)), replace=False)
    test = np.setdiff1d(eval_ids, valid)
    valid_src, valid_tgt = select_data(eval_src, eval_tgt, valid)
    test_src, test_tgt = select_data(eval_src, eval_tgt, test)
    write_data(valid_src, valid_tgt, eval_path, 'valid', eval_srclang, eval_tgtlang)
    write_data(test_src, test_tgt, eval_path, 'test', eval_srclang, eval_tgtlang)

def strip_elem(x):
    return x.rstrip().replace("\n", " ")

def replace_punct(cxt):
    cxt = str(cxt)
    cxt = re.sub('[,.;@#?!&$/-]+  # Accept one or more copies of punctuation\n                \\ *           # plus zero or more copies of a space, ',
      ' ', cxt,
      flags=(re.VERBOSE))
    return cxt.replace(':', ' ')


def return_appdata(repo, full_src, full_tgt, full_name, full_filename):
    index_pos = [i for i in range(len(full_name)) if full_name[i] == repo]
    app_name = [full_name[i] for i in index_pos]
    app_src = [full_src[i] for i in index_pos]
    app_tgt = [full_tgt[i] for i in index_pos]
    app_filename = [full_filename[i] for i in index_pos]
    return (app_src, app_tgt, app_name, app_filename)


def read_evalset(input_path, src, tgt):
    valid_src = open(params.input_path + '/valid.' + src)
    valid_tgt = open(params.input_path + '/valid.' + tgt)
    test_src = open(params.input_path + '/test.' + src)
    test_tgt = open(params.input_path + '/test.' + tgt)
    return (valid_src, valid_tgt, test_src, test_tgt)


def extract_cxtsents(cxt_src, cxt_tgt, only_src, only_tgt, valid_src, valid_tgt, test_src, test_tgt, app_names):
    valid_ids = find_pairs(only_src, only_tgt, valid_src, valid_tgt)
    test_ids = find_pairs(only_src, only_tgt, test_src, test_tgt)
    cxt_valid_src, cxt_valid_tgt = select_data(cxt_src, cxt_tgt, valid_ids)
    cxt_test_src, cxt_test_tgt = select_data(cxt_src, cxt_tgt, test_ids)
    valid_apps, _ = select_data(app_names, app_names, valid_ids)
    test_apps, _ = select_data(app_names, app_names, test_ids)
    return (cxt_valid_src, cxt_valid_tgt, cxt_test_src, cxt_test_tgt, valid_apps, test_apps)


def doc_format(app_names, data):
    apps = list(dict.fromkeys(app_names))
    doc_dict = {}
    doc_data = ['<d>']
    for app in apps:
        doc_dict[app] = []

    for i in range(len(data)):
        doc_dict[app_names[i]].append(data[i])

    for app in apps:
        doc_data.extend(doc_dict[app])
        doc_data.append('<d>')

    return doc_data
# okay decompiling utils.cpython-36.pyc

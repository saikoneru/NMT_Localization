from pathlib import Path
import numpy as np
import utils
import multiple_extract
import logging
import polib
import pandas as pd
import timeit
from multiprocessing import Pool
import functools
import tqdm
from sacremoses import MosesTruecaser, MosesTokenizer
import argparse
import os
import sys

def get_parser():
    """
    Generate a parameter parser
    """

    #Path configuration
    parser = argparse.ArgumentParser(description="Extract and create test sets from po files")

    parser.add_argument("--file_paths", type=str, default="./po_data/filepaths.txt",
                                    help="Filepaths to po files")
    parser.add_argument("--output_path", type=str, default="./gitdata_parsed/",
                                     help="Output path to store the dataset")

    parser.add_argument("--num_workers", type=int, default=20)
    parser.add_argument("--src_lang", type=str, default="en")
    parser.add_argument("--tgt_lang", type=str, default="de")
     
    return parser


def parse_po(po_file):
    return polib.pofile(po_file).translated_entries()


def read_po_file(po_file, keep_empty=False):
    ## Extract data from the po files and append source sentence with the desired information 
    src = []
    tgt = []
    cxt = []
    app_name = []

    try:
        entries = parse_po(po_file)
    except Exception as e:
        logging.warning("Unable to parse: {} \n Check if the file is available on git \n Exception {}".format(po_file,str(e)))
        return
    valid_entries = [e for e in entries if not e.obsolete]
    for entry in valid_entries:
        #entry_short = (len(entry.msgid.split(" ")) < 3)
        #if ( ((detect(entry.msgid) == params.src_lang) and (detect(entry.msgstr) == params.tgt_lang)) or entry_short):
        if ((entry.msgid !="" and entry.msgstr !="") or (keep_empty)):
            src.append(entry.msgid.rstrip().replace('\n', ' '))
            tgt.append(entry.msgstr.rstrip().replace('\n', ' '))
            tmp = ""
            for occur in entry.occurrences:
                tmp = occur[0] + "<cxt>" + tmp
                break
            cxt.append(tmp)
    #else:
     #   print(entry.msgid,entry.msgstr)
     #   exit()
     #   logging.info("Cannot detect language for pair {} --- {}".format(entry.msgid,entry.msgstr))
    assert len(src) == len(tgt)
    name = [po_file] * len(src)
    app_name.extend(name)
    
    output_path = os.path.normpath(os.path.abspath(po_file) + os.sep + os.pardir) + "/"
    basepath = os.path.basename(po_file).rstrip()
    utils.write_data(src,tgt,output_path, basepath + ".full", params.src_lang, params.tgt_lang)
    utils.write_list(cxt,output_path + basepath + ".full.cxt") 
    utils.write_list(app_name,output_path + basepath + ".full.name")
    logging.info("Finished parsing: {}".format(po_file))


def join_stats_data(filepaths):
    cnts = []
    full_src = []
    full_tgt = []
    full_cxt = []
    full_name = []
    filepaths_holder = []
    for filepath in filepaths:
        output_path = os.path.normpath(os.path.abspath(filepath) + os.sep + os.pardir) + "/"
        basepath =os.path.basename(filepath).rstrip()
        try:
            src_lines = open(output_path + basepath + ".full." + params.src_lang , mode = 'r', encoding = 'utf-8', newline = "\n").readlines()
            tgt_lines = open(output_path + basepath + ".full." + params.tgt_lang , mode = 'r', encoding = 'utf-8', newline = "\n").readlines()
            cxt_lines = open(output_path + basepath + ".full.cxt", mode = 'r', encoding = 'utf-8').readlines()
            name_lines = open(output_path + basepath + ".full.name", mode = 'r', encoding = 'utf-8').readlines()

            src_lines = [x.rstrip() for x in src_lines]
            tgt_lines = [x.rstrip() for x in tgt_lines]
            cxt_lines = [x.rstrip() for x in cxt_lines]
            name_lines = [x.rstrip() for x in name_lines]
            
            try:
                assert len(src_lines) == len(tgt_lines)
                filtered_filepaths.append(filepath.rstrip())
            except Exception as e:
                logging.debug("Src and target lines not equal: {}".format(filepath))
                sys.exit(1)

            full_src.extend(src_lines)
            full_tgt.extend(tgt_lines)
            full_cxt.extend(cxt_lines)
            full_name.extend(name_lines)
            filepaths_holder.extend([filepath] * len(src_lines))
            cnts.append(len(src_lines))
        except Exception as e:
            logging.info("Not adding data for: {} \n Because {}".format(filepath,str(e)))
    return full_src,full_tgt,full_cxt,full_name,cnts, filepaths_holder


def write_local(df):
    grouped_df = df.groupby('filepath')
    for filepath, frame in grouped_df:
        src = list(frame['src'])
        tgt = list(frame['tgt'])
        cxt = list(frame['cxt'])
        name = list(frame['name'])
        
        assert len(src) == len(tgt)
        
        dirpath = os.path.normpath(os.path.abspath(filepath) + os.sep + os.pardir) + "/"
        basepath = os.path.basename(filepath).rstrip()
        utils.write_data(src,tgt,dirpath, basepath + ".full", params.src_lang, params.tgt_lang)
        utils.write_list(cxt,dirpath + basepath + ".full.cxt") 
        utils.write_list(name,dirpath + basepath + ".full.name")
        logging.info("Finished saving file without duplicates for: {}".format(filepath))


def main(params):
    start_time = timeit.default_timer()
    fp = open(params.file_paths,mode ='r',encoding ='utf-8').readlines()
    po_files = [x.rstrip() for x in fp]
    logging.basicConfig(filename='extract_git_mp.log', level=logging.INFO)
    logging.info("Total {} files found".format(len(po_files)))

    #Store filepaths that actually downloaded po files
    global filtered_filepaths
    filtered_filepaths = []

    #Data without any context to filter overlaps between training and testing
    src = []
    tgt = []
    cxt = [] # In this script it refers to filename in po file but not the appended context in NMT
    app_name = []
    with Pool(params.num_workers) as pool:
        r = list(tqdm.tqdm(pool.imap(functools.partial(read_po_file),po_files),total = len(po_files)))
    read_time = timeit.default_timer()
    print("The time for reading files is :", read_time - start_time)
    
    full_src,full_tgt,full_cxt,full_name,cnts,fp_unique  = join_stats_data(fp)
    join_time = timeit.default_timer()
    print("The time for joining files is :", join_time - read_time)
    
    #Filter duplicates takes a long time, have to improve
    #unique_ids, unique_src, unique_tgt = utils.find_unique(full_src,full_tgt)
    #unique_name, unique_cxt = utils.select_data(full_name,full_cxt,ids)
    df_data = pd.DataFrame(list(zip(full_src, full_tgt, full_cxt, full_name, fp_unique)),
                           columns =['src', 'tgt', 'cxt', 'name', 'filepath'], dtype=str)
    df_data = df_data.drop_duplicates(subset = ['src','tgt'], keep = 'first')
    df_data['src'] = df_data['src'].astype('str')
    df_data['tgt'] = df_data['tgt'].astype('str')
    mask = (df_data['src'].str.len() < 200) |  (df_data['tgt'].str.len() < 200)
    df_data = df_data.loc[mask]
    drop_time = timeit.default_timer()
    print("The time for dropping duplicates and texts with more than 200 character length is :", drop_time - join_time)
    #Write data without duplication before creating context, overwrite previous files
    #mtok_src = MosesTokenizer(lang=params.src_lang)
    #mtok_tgt = MosesTokenizer(lang=params.tgt_lang)
    
    #df_data['src'] = df_data['src'].apply(lambda x: mtok_src.tokenize(x, return_str=True))
    #df_data['tgt'] = df_data['tgt'].apply(lambda x: mtok_tgt.tokenize(x, return_str=True))
    tok_time = timeit.default_timer()
    #print("The time for tokenizing lines :", tok_time - drop_time)
    #write_local(df_data)
    
    write_local_time = timeit.default_timer()
    print("The time for writing without duplicates locally is :", write_local_time - tok_time)
    unique_src = list(df_data['src'])
    unique_tgt = list(df_data['tgt'])
    unique_cxt = list(df_data['cxt'])
    unique_name = list(df_data['name'])
    unique_fp = list(df_data['filepath'])
    unique_fp = [x.rstrip() for x in unique_fp]
    
    utils.write_data(unique_src,unique_tgt,params.output_path,"full", params.src_lang, params.tgt_lang) 
    utils.write_list(unique_cxt,params.output_path + "full.cxt") 
    utils.write_list(unique_name,params.output_path + "full.name")
    utils.write_list(unique_fp,params.output_path + "full.filepath")
    write_global_time = timeit.default_timer()
    utils.write_list(df_data['filepath'].value_counts().tolist(),"cnts.txt")
    print("The time for writing without duplicates globally is :", write_global_time - write_local_time)
    #Write data and stats
    #df_info = pd.DataFrame(list(zip(sent_cnts)), columns = [ 'Number of sentences'])
    #df_info.to_csv(params.output_path + "stats.csv",index=False)
    df_data.to_csv(params.output_path + "/data.csv")
    exit()


if __name__ == '__main__':

    parser = get_parser()
    params = parser.parse_args()

    main(params)

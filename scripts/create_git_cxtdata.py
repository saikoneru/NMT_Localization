import os
import sys
from pathlib import Path
import argparse
import regex as re
import numpy as np
import utils
#import multiple_extract
import logging
import pandas as pd

def get_parser():
    """
    Generate a parameter parser
    """

    #Path configuration
    parser = argparse.ArgumentParser(description="Extract and create test sets from po files")

    parser.add_argument("--data_path", type=str, default="./gitdata_parsed/data.csv",
                                     help="Input path to store the dataset")
    #parser.add_argument("--gnome_eval", type=str, default="./gnome_final/v2/cxt_data/none_bpe/full_eval/",
     #                                help="Remove eval data from training")
    parser.add_argument("--output_path", type=str, default="./gitdata_parsed/v1/",
                                     help="Output path to store the dataset")
    parser.add_argument("--root_path", type=str, default="",
                                     help="Root path to properly find the dataset")
    parser.add_argument("--bpe_data_path", type=str, default="./gitdata_parsed/",
                                     help="Input path to store the dataset")

    parser.add_argument("--src_lang", type=str, default="en")
    parser.add_argument("--tgt_lang", type=str, default="de")
    parser.add_argument('--max_size', default=100, type=int,  help="Max lenght for sentence (words) after adding context")
    
    #Split configuration
    parser.add_argument('--append_src', default='none', const='none', nargs='?', choices=['none', 'neighbour', 'filename','sent_cluster', 'domain'], help='Different ways to append the source sentence')
    parser.add_argument('--neighbour_size', default=1, type=int,  help="How many left and right sentences to add?")
    parser.add_argument('--num_cluster', default=30, type=int,  help="How many clusters for sentence clustering")
    parser.add_argument('--train_split', default=0.9, type=float,  help="Amount of data portion to be kept in the training split, remaining is divided by half into validation and testing")
    
    
    return parser

def main(params):

    logging.basicConfig(level=logging.DEBUG)
    cnts = [] # Number of examples in each application
    
    #Placeholder for cxt - domain,filename,id,etc
    cxt = []
    
    #Placeholder for varying cxt based on src and tgt
    cxt_src = []
    cxt_tgt = []

    #Data without any context to filter overlaps between training and testing
    only_src = []
    only_tgt = []
    only_name = [] # The repository name, can be same for different po files
    
    repo_data = {} # Keeping track of repos with multiple po files
    ## Read Raw, Tokenized and BPE source files, Load to dataframe
    bpe_src = utils.read_data(params.bpe_data_path + "/full.tok." + params.src_lang + ".bpe") 
    bpe_tgt = utils.read_data(params.bpe_data_path + "/full.tok." + params.tgt_lang + ".bpe") 
    bpe_cxt = utils.read_data(params.bpe_data_path + "/full.tok.cxt.bpe") 
    
    src_raw = utils.read_data(params.bpe_data_path + "/full." + params.src_lang) 
    tgt_raw = utils.read_data(params.bpe_data_path + "/full." + params.tgt_lang)

    df_data = pd.read_csv(params.data_path, dtype='str', lineterminator='\n')
    df_data['src'] = bpe_src
    df_data['tgt'] = bpe_tgt
    df_data['cxt'] = bpe_cxt
    df_data['src_raw'] = src_raw
    df_data['tgt_raw'] = tgt_raw

    #Reinitialize raw to store in the correct order after creating splits
    src_raw = []
    tgt_raw = []

    ## Extract data from the po files and append source sentence with the desired information
     
    idx = 0
    po_paths = list(df_data['filepath'])
    repos = [x.split("/")[2] for x in po_paths]
    df_data['repos'] = repos
    
    #df_paths = pd.DataFrame(list(zip(po_paths,repos)), columns = ["paths", "repos"], dtype=str)
    repo_frames = df_data.groupby('repos') # Create a dict with repo name as key and list of paths as values
    for repo,repo_frame in repo_frames:
        tmp_cnt = 0
        #po_paths = list(repo_frame['filepath'])
        app_frames = repo_frame.groupby('filepath')
        for po,app_frame in app_frames:
            dirpath = os.path.normpath(os.path.abspath(params.root_path + "/" + po) + os.sep + os.pardir) + "/"
            basepath =os.path.basename(po).rstrip()
            #app_src,app_tgt,app_filename,app_name = utils.read_saved_po(dirpath, basepath, params.src_lang, params.tgt_lang)
            app_src = list(app_frame['src'])
            app_tgt = list(app_frame['tgt'])
            app_filename = list(app_frame['cxt'])
            app_name = list(app_frame['name'])
            app_src_raw = list(app_frame['src_raw'])
            app_tgt_raw = list(app_frame['tgt_raw'])
            idx+=1
            try:
                assert len(app_src) == len(app_tgt)
            except:
                logging.debug('Unequal source and target lines for {}'.format(po))
                sys.exit(1)
            only_src.extend(app_src)
            src_raw.extend(app_src_raw)
            only_tgt.extend(app_tgt)
            tgt_raw.extend(app_tgt_raw)
            if params.append_src == 'domain' or params.append_src == 'none':
                cxt.extend([repo]*len(app_name))
            if params.append_src == 'neighbour': 
                app_src = utils.add_side(app_src,params.neighbour_size)
                app_tgt = utils.add_side(app_tgt,params.neighbour_size)
                cxt_src.extend(app_src)
                cxt_tgt.extend(app_tgt)

            if params.append_src == 'filename':
                app_cxt = [utils.replace_punct(x) for x in app_filename]
                cxt.extend(app_cxt) 

            tmp_cnt += len(app_name)
        cnts.append(tmp_cnt)
   
    #Modify to recieve only cxt and not cxt_src cxt_tgt
    #if params.append_src == 'sent_cluster': 
    #    cxt_src, cxt_tgt = utils.cluster_sent(cxt_src,cxt_tgt,params.num_cluster)
    
    ## Split the data randomly or leave few applications out for test data
    logging.debug('Processed input data of total length {} and attached context type {}'.format(len(only_src),str(params.append_src)))
    total = len(only_src)
    train_cnt = int(params.train_split * total) # Total training examples
    idx = np.arange(total)
    np.random.seed(11)

    logging.debug('Splitting context appended data based on ids for unknown apps')
    valid, test = utils.test_apps(cnts, total - train_cnt)
    eval_ids = np.asarray(valid + test)
    train = np.setdiff1d(idx,eval_ids) # Remove evaluation ids from total data


    # Create evaluation set for known apps
    logging.debug('Creating random eval ids')
    np.random.seed(33)
    valid_rnd = np.random.choice(train, int((total - train_cnt)/2), replace = False)
    traintest_rnd = np.setdiff1d(train,valid_rnd)
    np.random.seed(33)
    logging.debug('Splitting context appended data based on ids for known apps')
    test_rnd = np.random.choice(traintest_rnd, int((total - train_cnt)/2), replace = False)
    
    # Create dataframes for faster selection and deletion
    if params.append_src == 'neighbour':
        df_cxt = pd.DataFrame(list(zip(cxt_src,cxt_tgt,only_src,only_tgt,src_raw,tgt_raw)), columns = ['cxt_src', 'cxt_tgt', 'src', 'tgt','src_raw', 'tgt_raw'])
    else:
        if params.append_src == 'domain':
            cxt = [x + "##" for x in cxt]
        df_cxt = pd.DataFrame(list(zip(cxt,only_src,only_tgt,src_raw,tgt_raw)), columns = ['cxt', 'src', 'tgt', 'src_raw', 'tgt_raw'])
    df_cxt_rnd_valid = df_cxt.iloc[valid_rnd]
    df_cxt_rnd_test = df_cxt.iloc[test_rnd]
    
    df_cxt_train = df_cxt.iloc[train]
    df_cxt_valid = df_cxt.iloc[valid]
    df_cxt_test = df_cxt.iloc[test]
    
    eval_df = pd.concat([df_cxt_valid, df_cxt_test, df_cxt_rnd_valid, df_cxt_rnd_test], ignore_index=False, sort=False)
    #gnome_eval_src = utils.read_data(params.gnome_eval + '/eval.' + params.src_lang)
    #gnome_eval_tgt = utils.read_data(params.gnome_eval + '/eval.' + params.tgt_lang)
    
    df_cxt_train = df_cxt_train[~df_cxt_train['src'].isin(eval_df['src'])]
    df_cxt_train = df_cxt_train[~df_cxt_train['tgt'].isin(eval_df['tgt'])]
    train_cnt_overlap = df_cxt_train.shape[0]
    #df_cxt_train = df_cxt_train[~df_cxt_train['src'].isin(gnome_eval_src)]
    #df_cxt_train = df_cxt_train[~df_cxt_train['tgt'].isin(gnome_eval_tgt)]
    #train_cnt = df_cxt_train.shape[0]
    #logging.debug('Removed total {} sentences using gnome eval data'.format(train_cnt_overlap - train_cnt))
    logging.debug('Writing context appended train, valid and test data')
    #Initialize dataframes into lists for writing
    
    
    
    # Write training, valid and test data split 
    utils.write_cxt_data(df_cxt_train,params.output_path,"train", params.src_lang, params.tgt_lang, params.append_src)
    utils.write_cxt_data(df_cxt_valid,params.output_path,"valid", params.src_lang, params.tgt_lang, params.append_src)
    utils.write_cxt_data(df_cxt_test,params.output_path,"test", params.src_lang, params.tgt_lang, params.append_src)
    

    utils.write_cxt_data(df_cxt_rnd_valid,params.output_path + "/rnd/","valid", params.src_lang, params.tgt_lang, params.append_src)
    utils.write_cxt_data(df_cxt_rnd_test,params.output_path + "/rnd/","test", params.src_lang, params.tgt_lang, params.append_src)
   
    exit()

if __name__ == '__main__':

    parser = get_parser()
    params = parser.parse_args()

    main(params)

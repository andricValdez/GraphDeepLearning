

import os
import sys
import joblib
import time 
import numpy as np
import pandas as pd 
import logging
import traceback 
import glob
from tqdm import tqdm 
import torch
import networkx as nx
import scipy as sp
from scipy.sparse import coo_array 
import gensim
from torch_geometric.data import DataLoader, Data
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import warnings
import torch.utils.data as data_utils
from torch.nn.modules.module import Module
import networkx as nx
import gc
from sklearn.metrics import f1_score, accuracy_score
from datasets import load_dataset

from polyglot.detect import Detector
from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

import utils
import baselines
import gnn
import text2graph
import node_feat_init

#************************************* CONFIGS
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s; - %(levelname)s; - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


#************************************* MAIN

def main():
    ...

 

def extract_embeddings_subtask1():
    # NOTA: el prefijo "autext_" se quedó para todos los dataset, no solo para el de autexttification

    # ****************************** READ DATASET SEMEVAL 2024

    dataset_name = 'semeval24'
    autext_train_set = utils.read_json(dir_path=utils.DATASET_DIR + 'semeval2024/subtask1/subtaskA_train_monolingual.jsonl')
    autext_val_set = utils.read_json(dir_path=utils.DATASET_DIR + 'semeval2024/subtask1/subtaskA_dev_monolingual.jsonl')
    autext_test_set = utils.read_json(dir_path=utils.DATASET_DIR + 'semeval2024/subtask1/subtaskA_test_monolingual.jsonl')
    print(autext_train_set.info())
    
    autext_train_set['word_len'] = autext_train_set['text'].str.split().str.len()
    autext_train_set = autext_train_set[autext_train_set['word_len'] <= 1500]
    autext_train_set = autext_train_set[autext_train_set['word_len'] >= 10]
    #autext_train_set, autext_val_set = train_test_split(autext_train_set, test_size=0.3)
    print(autext_train_set.value_counts('label'))
    print(autext_train_set.nlargest(5, ['word_len']) )
    #autext_val_set = pd.concat([autext_val_set, autext_val_set_2], axis=0)

    print(autext_train_set.info())
    print(autext_val_set.info())
    print(autext_test_set.info())
    print(autext_val_set['model'].value_counts())
    
    '''
    autext_train_set = utils.read_json(dir_path=utils.DATASET_DIR + 'semeval2024/original/subtaskA_train_monolingual.jsonl')
    autext_val_set = utils.read_json(dir_path=utils.DATASET_DIR + 'semeval2024/original/subtaskA_dev_monolingual.jsonl')
    autext_test_set = utils.read_json(dir_path=utils.DATASET_DIR + 'semeval2024/original/subtaskA_test_monolingual.jsonl')
    autext_train_set = shuffle(autext_train_set)
    autext_val_set = shuffle(autext_val_set)
    autext_test_set = shuffle(autext_test_set)

    autext_train_set = autext_train_set.to_dict('records')
    autext_val_set = autext_val_set.to_dict('records')
    autext_test_set = autext_test_set.to_dict('records')
    utils.save_jsonl(autext_train_set, file_path=utils.DATASET_DIR + 'semeval2024/subtask1/subtaskA_train_monolingual.jsonl')
    utils.save_jsonl(autext_val_set, file_path=utils.DATASET_DIR + 'semeval2024/subtask1/subtaskA_dev_monolingual.jsonl')
    utils.save_jsonl(autext_test_set, file_path=utils.DATASET_DIR + 'semeval2024/subtask1/subtaskA_test_monolingual.jsonl')
    return
    '''
    # ****************************** READ DATASET AUTEXT 2023
    '''
    dataset_name = 'autext23' # autext23, autext23_s2
    subtask = 'subtask1' # subtask1, subtask2
    
    autext_train_set = utils.read_csv(file_path=f'{utils.DATASET_DIR}autext2023/{subtask}/train_set.csv') 
    autext_val_set = utils.read_csv(file_path=f'{utils.DATASET_DIR}autext2023/{subtask}/val_set.csv') 
    autext_test_set = utils.read_csv(file_path=f'{utils.DATASET_DIR}autext2023/{subtask}/test_set.csv') 
    print(autext_train_set.info())
    print(autext_val_set.info())
    print(autext_test_set.info())
    print(autext_train_set['label'].value_counts())
    '''
    '''
    
    print(40*'*', 'Dataset Distro-Partition')
    subtask = 'subtask1' # subtask1, subtask2
    dataset = load_dataset("symanto/autextification2023", 'attribution_en') # ['detection_en', 'attribution_en', 'detection_es', 'attribution_es']
    train_set = pd.DataFrame(dataset['train'])
    autext_test_set = pd.DataFrame(dataset['test'])
    autext_train_set, autext_val_set = train_test_split(train_set, test_size=0.3)
    print("autext_train_set: ", autext_train_set.info())
    print("autext_val_set:  ", autext_val_set.info())
    autext_train_set.to_csv(f'{utils.DATASET_DIR}autext2023/{subtask}/train_set.csv')
    autext_val_set.to_csv(f'{utils.DATASET_DIR}autext2023/{subtask}/val_set.csv')
    autext_test_set.to_csv(f'{utils.DATASET_DIR}autext2023/{subtask}/test_set.csv')
    return
    '''
    # ****************************** READ DATASET AUTEXT 2024
    '''
    dataset_name = 'autext24'
    # ********** TRAIN
    autext_train_set = utils.read_json(dir_path=utils.DATASET_DIR + 'subtask_1/train_set.jsonl') 
    autext_train_set['label'] = np.where(autext_train_set['label'] == 'human', 1, 0)
    print(autext_train_set.info())
    print(autext_train_set['label'].value_counts())

    # ********** VAL
    autext_val_set = utils.read_json(dir_path=utils.DATASET_DIR + 'subtask_1/val_set.jsonl') 
    autext_val_set['label'] = np.where(autext_val_set['label'] == 'human', 1, 0)
    print(autext_val_set.info())
    print(autext_val_set['label'].value_counts())

    # ********** TEST
    autext_test_set = utils.read_json(dir_path=utils.DATASET_DIR + 'subtask_1/test_set_original.jsonl') 
    
    '''

    # ****************************** FINE TUNE LLM
    '''
    for llm in ['FacebookAI/roberta-base']:
        node_feat_init.llm_fine_tuning(
            model_name = 'semeval24',  # autext23, autext24, semeval24
            train_set_df = autext_train_set, 
            val_set_df = autext_test_set, # autext_val_set, autext_test_set
            device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu"),
            llm_to_finetune = llm,
            num_labels = 2
        )
    return
    # google-bert/bert-base-uncased
    # FacebookAI/roberta-base
    # andricValdez/bert-base-uncased-finetuned-autext23_sub2
    # andricValdez/roberta-base-finetuned-autext23_sub2
    # andricValdez/bert-base-uncased-finetuned-semeval24
    # andricValdez/roberta-base-finetuned-semeval24 
    '''
    # ****************************** PROCESS AUTEXT DATASET && CUTOF
    cut_off_dataset = 10
    train_text_docs = utils.process_dataset(autext_train_set)
    val_text_docs = utils.process_dataset(autext_val_set)
    #test_text_docs = utils.process_dataset(autext_test_set)

    cut_dataset_train = len(train_text_docs) * (int(cut_off_dataset) / 100)
    train_text_docs = train_text_docs[:int(cut_dataset_train)]

    #cut_dataset_val = len(val_text_docs) * (int(cut_off_dataset) / 100)
    cut_dataset_val = len(val_text_docs) * (int(100) / 100)
    val_text_docs = val_text_docs[:int(cut_dataset_val)]

    #cut_dataset_test = len(test_text_docs) * (int(cut_off_dataset) / 100)
    #test_text_docs = test_text_docs[:int(cut_dataset_test)]

    # validation class balance for cut_off_dataset  and get some stats
    cnt_0, cnt_1 = 0, 0
    for instance in train_text_docs: # train_text_docs, val_text_docs, test_text_docs
        if instance['context']['target'] == 0:
            cnt_0 += 1
        else:
            cnt_1 += 1

    print("cnt_0: ", cnt_0)
    print("cnt_1: ", cnt_1)
    utils.text_metrics(autext_train_set)
    #print(pd.DataFrame(train_text_docs).value_counts('model'))
    #print("*** first_doc: \n", train_text_docs[0]['doc'])
    #print("*** last_doc: \n", train_text_docs[-1]['doc'])
    #return
    
    # ****************************** BASELINES
    '''
    print(40*'*', 'Train and Test ML baseline models')
    models = ['LinearSVC','MultinomialNB','LogisticRegression','SGDClassifier','xgboost']
    #models = ['xgboost']
    for model in models:
        print(20*'*', 'model: ', model)
        baselines.main(
            train_set=autext_train_set[ : int(len(train_text_docs) * (int(50) / 100))], 
            val_set=autext_val_set[ : int(len(val_text_docs) * (int(100) / 100))], 
            test_set=autext_test_set[:], 
            algo_ml=model,
            target_names=['human', 'generated'],
            #target_names=['A', 'B', 'C', 'D', 'E', 'F'],
        )
        print('\n')
    return
    '''
    # ****************************** GRAPH NEURAL NETWORK - RUN EXPERIMENTS IN BATCHES
    '''
    num_classes = 2
    exp_file_name = 'experiments_cooccurrence_20241019131147_llm' 
    experiments_path = f'{utils.OUTPUT_DIR_PATH}batch_experiments/'
    experiments_path_dir = f'{experiments_path}{exp_file_name}/'
    experiments_path_file = f'{experiments_path}{exp_file_name}.csv'
    utils.create_expriment_dirs(experiments_path_dir)
    gnn.graph_neural_network_batch(
        autext_train_set, 
        autext_val_set, 
        autext_test_set, 
        experiments_path_dir, 
        experiments_path_file, 
        num_classes=num_classes
    )
    return
    '''
    # ****************************** GRAPH NEURAL NETWORK - ONE RUNNING

    lang = 'en' #es, en, fr 
    t2g_instance = text2graph.Text2Graph(
        graph_type = 'DiGraph', 
            window_size = 3,
            apply_prep = True, 
            steps_preprocessing = {
                "to_lowercase": True, 
                "handle_blank_spaces": True,
                "handle_html_tags": True,
                "handle_special_chars": True,
                "handle_stop_words": False,
            },
            language = lang, #es, en, fr
    )

    exp_file_name = "test"
    dataset_partition = f'{dataset_name}_{cut_off_dataset}perc' # perc | perc_go_cls | perc_go_e5
    exp_file_path = f'{utils.OUTPUT_DIR_PATH}{exp_file_name}_{dataset_partition}/'
    utils.create_expriment_dirs(exp_file_path)
    
    cuda_num = 0
    num_classes = 2 # num output classes
    num_features = 256 # llm: 768 | w2v: 768, 512, 256
    batch_size_gnn = 64 # 16 -> semeval | 64 -> autext
    edge_features = False
    nfi = 'w2v' # llm, w2v, fasttext, random
    llm_model_name = 'andricValdez/bert-base-uncased-finetuned-semeval24'
    
    device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")
    
    gnn.graph_neural_network( 
        exp_file_name = 'test', 
        dataset_partition = dataset_partition,
        exp_file_path = exp_file_path,
        graph_trans = True, # True, False, None
        nfi = nfi,
        cut_off_dataset = cut_off_dataset, 
        t2g_instance = t2g_instance,
        train_text_docs = train_text_docs, # train_text_docs[ : int(len(train_text_docs) * (int(100) / 100))], 
        val_text_docs = val_text_docs, # val_text_docs[ : int(len(val_text_docs) * (int(100) / 100))],
        test_text_docs = None,
        device = device,
        edge_features = edge_features,
        edge_dim = 2,
        num_features = num_features, 
        batch_size_gnn = batch_size_gnn,
        build_dataset = True, # True, False
        save_data = True,
        llm_finetuned_name = llm_model_name,
        num_classes = num_classes,
    )
    # google-bert/bert-base-uncased
    # FacebookAI/roberta-base

    # *** AUTEXT 2023 Subtask1
    # andricValdez/bert-base-uncased-finetuned-autext23
    # andricValdez/roberta-base-finetuned-autext23

    # *** AUTEXT 2023 Subtask2
    # andricValdez/bert-base-uncased-finetuned-autext23_sub2
    # andricValdez/roberta-base-finetuned-autext23_sub2

    # *** SEMEVAL 2024
    # andricValdez/bert-base-uncased-finetuned-semeval24
    # andricValdez/roberta-base-finetuned-semeval24 
    
    torch.cuda.empty_cache()
    gc.collect()
    

    #******************* GET stylo feat
    #utils.get_stylo_feat(exp_file_path=exp_file_path, text_data=train_text_docs, subset='train') # train, train_all
    #utils.get_stylo_feat(exp_file_path=exp_file_path, text_data=val_text_docs, subset='val')
     
    #******************* GET llm_get_embbedings
    '''
    utils.llm_get_embbedings(
        text_data=train_text_docs[ : ], exp_file_path=exp_file_path+'embeddings_cls_llm_1/', 
        subset='train', emb_type='llm_cls', device=device, save_emb=True,
        llm_finetuned_name=llm_model_name, num_labels=num_classes)
    utils.llm_get_embbedings(
        text_data=val_text_docs[ : ], exp_file_path=exp_file_path+'embeddings_cls_llm_1/', 
        subset='val', emb_type='llm_cls', device=device, save_emb=True,
        llm_finetuned_name=llm_model_name, num_labels=num_classes)
    '''

def train_clf_model_batch_subtask():
    dataset_name = 'semeval24' # autext23, autext23_s2, semeval24
    cuda_num = 0
    cut_off_dataset = 10

    train_set_mode = 'train' # train | train_all
    # test_autext24_all_100perc, subtask2/test_autext24_subtask2_all_100perc
    #exp_file_path = utils.OUTPUT_DIR_PATH + f'subtask2/test_autext24_subtask2_all_100perc'
    exp_file_path = utils.OUTPUT_DIR_PATH + f'test_{dataset_name}_{cut_off_dataset}perc'

    # delete
    feat_types = [
        #'embedding_llm',
        #'embedding_llm_stylo',
        'embedding_gnn',
        #'embedding_gnn_stylo',
        #'embedding_gnn_llm',
        #'embedding_all',
    ]
    
    clf_models_dict = {
        'LinearSVC': LinearSVC,
        'LogisticRegression': LogisticRegression,
        #'RandomForestClassifier': RandomForestClassifier,
        'SGDClassifier': SGDClassifier,
        'XGBClassifier': XGBClassifier,
        #'RRNN_Dense_Clf': gnn.NeuralNetwork
    } 

    emb_train_merge_df, emb_val_merge_df = get_features(exp_file_path)
    #emb_train_merge_df = utils.read_json(dir_path=f'{exp_file_path}/embeddings_cls_llm_1/autext_train_embeddings.jsonl')
    #emb_val_merge_df = utils.read_json(dir_path=f'{exp_file_path}/embeddings_cls_llm_1/autext_val_embeddings.jsonl')
    
    for feat_type in feat_types:
        print('\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> feat_type: ', feat_type)
        for model in clf_models_dict:
            if model == 'RRNN_Dense_Clf':
                device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")
            else:
                device = 'cpu'

            train_clf_model(
                exp_file_path=exp_file_path,
                feat_type=feat_type,
                clf_model=model,
                clf_models_dict=clf_models_dict,
                train_set_mode=train_set_mode,
                emb_train_merge_df=emb_train_merge_df, 
                emb_val_merge_df=emb_val_merge_df,
                device = device
            )


def get_features(exp_file_path):
     # ----------------------------------- Embeddings GNN
    emb_train_gnn_files = glob.glob(f'{exp_file_path}/embeddings_gnn/autext_train_emb_batch_*.jsonl')
    emb_val_gnn_files = glob.glob(f'{exp_file_path}/embeddings_gnn/autext_val_emb_batch_*.jsonl')
    emb_train_gnn_lst_df = [utils.read_json(file) for file in emb_train_gnn_files]
    emb_val_gnn_lst_df = [utils.read_json(file) for file in emb_val_gnn_files]
    emb_train_gnn_df = pd.concat(emb_train_gnn_lst_df)
    emb_val_gnn_df = pd.concat(emb_val_gnn_lst_df)
    print(emb_train_gnn_df.info())
    print(emb_val_gnn_df.info())

    # ----------------------------------- Embeddings LLM CLS
    #emb_train_llm_df = utils.read_json(dir_path=f'{exp_file_path}/embeddings_cls_llm_1/autext_train_embeddings.jsonl')
    #emb_val_llm_df = utils.read_json(dir_path=f'{exp_file_path}/embeddings_cls_llm_1/autext_val_embeddings.jsonl')
    #print(emb_train_llm_df.info())
    #print(emb_val_llm_df.info())

    # ----------------------------------- Features Stylometrics
    #stylo_train_feat = utils.read_json(f'{exp_file_path}/stylometry_train_feat.json')
    #stylo_val_feat = utils.read_json(f'{exp_file_path}/stylometry_val_feat.json')
    #print(stylo_train_feat.info())
    #print(stylo_val_feat.info())
    
    # ----------------------------------- Merge/concat vectors
    #emb_train_merge_df = emb_train_gnn_df.merge(emb_train_llm_df, on='doc_id', how='inner')
    #emb_train_merge_df = emb_train_merge_df.rename(columns={'label_x': 'label_gnn', 'label_y': 'label_llm', 'embedding_x': 'embedding_gnn', 'embedding_y': 'embedding_llm'})
    #emb_train_merge_df = emb_train_merge_df.merge(stylo_train_feat, on='doc_id', how='inner')
    #emb_train_merge_df['embedding_gnn_llm'] = emb_train_merge_df['embedding_gnn'] + emb_train_merge_df['embedding_llm']
    #emb_train_merge_df['embedding_gnn_stylo'] = emb_train_merge_df['embedding_gnn'] + emb_train_merge_df['stylo_feat']
    #emb_train_merge_df['embedding_llm_stylo'] = emb_train_merge_df['embedding_llm'] + emb_train_merge_df['stylo_feat']
    #emb_train_merge_df['embedding_all'] =  emb_train_merge_df['embedding_llm']  + emb_train_merge_df['embedding_gnn'] #+ emb_train_merge_df['stylo_feat']
    #print(emb_train_merge_df.info())

    #emb_val_merge_df = emb_val_gnn_df.merge(emb_val_llm_df, on='doc_id', how='inner')
    #emb_val_merge_df = emb_val_merge_df.rename(columns={'label_x': 'label_gnn', 'label_y': 'label_llm', 'embedding_x': 'embedding_gnn', 'embedding_y': 'embedding_llm'})
    #emb_val_merge_df = emb_val_merge_df.merge(stylo_val_feat, on='doc_id', how='inner')
    #emb_val_merge_df['embedding_gnn_llm'] = emb_val_merge_df['embedding_gnn'] + emb_val_merge_df['embedding_llm']
    #emb_val_merge_df['embedding_gnn_stylo'] = emb_val_merge_df['embedding_gnn'] + emb_val_merge_df['stylo_feat']
    #emb_val_merge_df['embedding_llm_stylo'] = emb_val_merge_df['embedding_llm'] + emb_val_merge_df['stylo_feat']
    #emb_val_merge_df['embedding_all'] =  emb_val_merge_df['embedding_llm']  + emb_val_merge_df['embedding_gnn'] #+ emb_val_merge_df['stylo_feat']
    #print(emb_val_merge_df.info())

    # TMP - FOR ONLY USE GNN
    emb_train_merge_df = emb_train_gnn_df.rename(columns={'label': 'label_gnn', 'embedding': 'embedding_gnn'})
    emb_val_merge_df = emb_val_gnn_df.rename(columns={'label': 'label_gnn', 'embedding': 'embedding_gnn'})

    return emb_train_merge_df, emb_val_merge_df


def train_clf_model(exp_file_path, feat_type, clf_model, clf_models_dict, train_set_mode, emb_train_merge_df, emb_val_merge_df, device='cpu'):

    # ----------------------------------- Train CLF Model
    if train_set_mode == 'train_all':
        emb_train_merge_df = pd.concat([emb_train_merge_df, emb_val_merge_df])
        print(emb_train_merge_df.info())
        
    ## TRAIN SET
    #train_data = [torch.tensor(np.asarray(emb), dtype=torch.float, device="cpu") for emb in emb_train_merge_df[feat_type]]
    #train_data = torch.vstack(train_data)
    #train_labels = [torch.tensor(np.asarray(label), dtype=torch.int64, device="cpu") for label in emb_train_merge_df['label_gnn']]
    #train_labels = torch.vstack(train_labels)
    #train = data_utils.TensorDataset(train_data, train_labels)
    #train_loader = data_utils.DataLoader(train, batch_size=64, shuffle=True)
    ### VAL SET
    #val_data = [torch.tensor(np.asarray(emb), dtype=torch.float, device="cpu") for emb in emb_val_merge_df[feat_type]]
    #val_data = torch.vstack(val_data)
    #val_labels = [torch.tensor(np.asarray(label), dtype=torch.int64, device="cpu") for label in emb_val_merge_df['label_gnn']]
    #val_labels = torch.vstack(val_labels)
    #val = data_utils.TensorDataset(val_data, val_labels)
    #val_loader = data_utils.DataLoader(val, batch_size=64, shuffle=True)

    # TRAIN SET
    train_data = emb_train_merge_df[feat_type].values.tolist()
    train_labels = emb_train_merge_df['label_gnn'].values.tolist()
    # VAL SET
    val_data = emb_val_merge_df[feat_type].values.tolist()
    val_labels = emb_val_merge_df['label_gnn'].values.tolist()

    print(' ****** clf_model: ', clf_model, ' | feat_train_len: ', len(train_data[0]), ' | train_instances: ', len(train_data), ' | feat_val_len: ', len(val_data[0]), ' | val_labels: ', val_labels[0])
    if clf_model == 'RRNN_Dense_Clf':
        dense_model = gnn.NeuralNetwork(
            in_channels = len(train_data[0]),
            nhid = 256, 
            out_ch = 2, 
            layers_num = 3
        )
        traind_model = gnn.train_dense_rrnn_clf_model(dense_model, train_loader, val_data, val_labels, device="cpu")
        torch.save(traind_model, f'{exp_file_path}clf_models/{clf_model}_{train_set_mode}_{feat_type}.pt')
    else:
        traind_model = gnn.train_ml_clf_model(clf_models_dict[clf_model], train_data, train_labels, val_data, val_labels)
        utils.save_data(traind_model, path=f'{exp_file_path}/clf_models/', file_name=f'{clf_model}_{train_set_mode}_{feat_type}')


def test_eval_subtask():
    print('*** test_eval_subtask')
    dataset = 'semeval2024' # autext2023, semeval2024
    dataset_name = 'semeval24' # autext23, autext23_s2, semeval24
    subtask = 'subtask_1' # subtask_1, subtask_2
    feat_type = 'embedding_' + 'gnn' # embedding_all | embedding_gnn_llm | embedding_gnn |embedding_gnn_stylo | embedding_llm_stylo | embedding_llm_stylo
    
    extract_embeddings = True
    cuda_num = 0
    num_labels = 2
    num_features = 256
    edge_features = False
    nfi='w2v' # llm, w2v, fasttext, random
    llm_finetuned_name = 'andricValdez/bert-base-uncased-finetuned-semeval24'

    cut_off_dataset = 10
    cut_off_test_dataset = 100
    exp_file_name = "test"

    device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")

    if subtask == 'subtask_1':
        clf_model_name_ml = f'XGBClassifier_train_{feat_type}' # XGBClassifier_train_ | LogisticRegression_train_ | SGDClassifier_train_ | LinearSVC_train_
        dataset_partition = f'{dataset_name}_{cut_off_dataset}perc'
        exp_file_path = f'{utils.OUTPUT_DIR_PATH}{exp_file_name}_{dataset_partition}/'
        #autext_test_set = utils.read_csv(file_path=f'{utils.DATASET_DIR}{dataset}/subtask1/test_set.csv')   
        autext_test_set = utils.read_json(dir_path=utils.DATASET_DIR + 'semeval2024/subtask1/subtaskA_test_monolingual.jsonl')

    if subtask == 'subtask_2':
        clf_model_name_ml = f'XGBClassifier_train_{feat_type}' # XGBClassifier_train_ | LogisticRegression_train_ | SGDClassifier_train_ | LinearSVC_train_
        dataset_partition = f'{dataset_name}_s2_{cut_off_dataset}perc'
        exp_file_path = f'{utils.OUTPUT_DIR_PATH}{exp_file_name}_{dataset_partition}/'
        autext_test_set = utils.read_csv(file_path=f'{utils.DATASET_DIR}{dataset}/subtask2/test_set.csv') 


    # ******************** Read TEST set
    #print(autext_test_set.info())
    autext_test_set = autext_test_set.sort_values('id')

    test_text_docs = []
    corpus_text_docs_dict = autext_test_set.to_dict('records')
    for instance in corpus_text_docs_dict:
        doc = {
            "id": instance['id'], 
            "doc": instance['text'][:], 
            "context": {"id": instance['id'], "target": instance['label']}
        }
        test_text_docs.append(doc)

    cut_dataset_test = len(test_text_docs) * (int(cut_off_test_dataset) / 100)
    test_text_docs = test_text_docs[:int(cut_dataset_test)]

    t2g_instance = text2graph.Text2Graph(
        graph_type = 'DiGraph',
            window_size = 3, 
            apply_prep = True, 
            steps_preprocessing = {
                "to_lowercase": True,
                "handle_blank_spaces": True,
                "handle_html_tags": True,
                "handle_special_chars":True,
                "handle_stop_words": False,
            },
            language = 'en', #es, en, fr
    )
    
    #gnn.graph_neural_network_test_eval(test_text_docs, t2g_instance, nfi, exp_file_path, dataset_partition, llm_finetuned_name, device) 
    #return


    # ******************** get Features/Embeddings    
    if extract_embeddings:
        #  get GNN Emb
        gnn.graph_neural_network_test_eval(test_text_docs, t2g_instance, nfi, exp_file_path, dataset_partition, llm_finetuned_name, edge_features, num_features, device) 
  
        #  get LLM Emb 1
        #utils.llm_get_embbedings(
        #    text_data=test_text_docs, 
        #    exp_file_path=exp_file_path+'embeddings_cls_llm_1/', subset='test', 
        #    emb_type='llm_cls', device=device, save_emb=True, 
        #    llm_finetuned_name=llm_finetuned_name, 
        #    num_labels=num_labels
        #)

        # get Stylo Feat
        #utils.get_stylo_feat(exp_file_path=exp_file_path, text_data=test_text_docs, subset='test')
        
    # ******************** Load feat
    # emb_test_gnn
    emb_test_gnn_files = glob.glob(f'{exp_file_path}/embeddings_gnn/autext_test_emb_batch_*.jsonl')
    emb_test_gnn_lst_df = [utils.read_json(file) for file in emb_test_gnn_files]
    gnn_test_embeddings = pd.concat(emb_test_gnn_lst_df)
    # emb_test_llm
    #llm_1_test_embeddings = utils.read_json(dir_path=f'{exp_file_path}/embeddings_cls_llm_1/autext_test_embeddings.jsonl')

    # stylo_test_feat
    #feat_test_stylo = utils.read_json(f'{exp_file_path}/stylometry_test_feat.json')
    
    # ******************** Concat feat  
    #merge_test_embeddings = gnn_test_embeddings.merge(llm_1_test_embeddings, on='doc_id', how='left')
    #merge_test_embeddings = merge_test_embeddings.rename(columns={'label_x': 'label_gnn', 'label_y': 'label_llm', 'embedding_x': 'embedding_gnn', 'embedding_y': 'embedding_llm'})   
    #merge_test_embeddings = merge_test_embeddings.merge(feat_test_stylo, on='doc_id', how='left')

    #merge_test_embeddings['embedding_gnn_stylo'] = merge_test_embeddings['embedding_gnn'] + merge_test_embeddings['stylo_feat']
    #merge_test_embeddings['embedding_llm_stylo'] = merge_test_embeddings['embedding_llm'] + merge_test_embeddings['stylo_feat']
    #merge_test_embeddings['embedding_gnn_llm'] = merge_test_embeddings['embedding_gnn'] + merge_test_embeddings['embedding_llm']
    #merge_test_embeddings['embedding_all'] = merge_test_embeddings['embedding_gnn'] + merge_test_embeddings['embedding_llm'] #+ merge_test_embeddings['stylo_feat'] 
    #print(merge_test_embeddings.info()) 


    # TMP - FOR ONLY USE GNN
    merge_test_embeddings = gnn_test_embeddings.rename(columns={'label': 'label_gnn', 'embedding': 'embedding_gnn'})
    
    
    # ******************** final clf model
    device = 'cpu'
    print("device: ", device)
    #test_feat_data_pt = [torch.tensor(np.asarray(emb), dtype=torch.float, device=device) for emb in merge_test_embeddings[feat_type]]
    #test_feat_data_pt = torch.vstack(test_feat_data_pt)
    test_feat_data_pt = merge_test_embeddings[feat_type].values.tolist()
    
    clf_model = utils.load_data(path=f'{exp_file_path}clf_models/', file_name=f'{clf_model_name_ml}') 
    y_pred = clf_model.predict(test_feat_data_pt)
    
    merge_test_embeddings['y_pred'] = y_pred
    y_true = merge_test_embeddings['label_gnn'].values.tolist()
    y_pred = merge_test_embeddings['y_pred'].values.tolist()
    #print(merge_test_embeddings.info()) 
    print(merge_test_embeddings['label_gnn'].value_counts())
    print(merge_test_embeddings['y_pred'].value_counts())

    print('\t Accuracy:', accuracy_score(y_true, y_pred))
    print('\t F1Score:', f1_score(y_true, y_pred , average='macro'))

    correct = 0
    for idx in range(0, len(merge_test_embeddings)):
        if y_pred[idx] == y_true[idx]:
            correct += 1

    print(correct, correct/len(merge_test_embeddings))

    return

 



if __name__ == '__main__':
    
    #main()
    extract_embeddings_subtask1() 
    train_clf_model_batch_subtask()
    test_eval_subtask()


# ********* CMDs
# python main.py
# nohup bash main.sh >> logs/xxx.log &
# nohup python main.py >> logs/text_to_graph_transform_small.log &
# ps -ef | grep python | grep avaldez
# tail -f logs/experiments_cooccurrence_20240502062849.log 




















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

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from datasets import load_dataset
import mlflow
from mlflow import MlflowClient

from polyglot.detect import Detector
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.naive_bayes import GaussianNB, MultinomialNB
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

# Set our tracking server uri for logging
mlflow.set_tracking_uri(uri="http://localhost:5000")
#mlflow.autolog()

client = MlflowClient()
experiment_id = "0"
run = client.create_run(experiment_id)


cuda_num = 0
cut_off_dataset = 100
cut_off_test_dataset = 100
graph_type = 'cooc' # cooc, hetero
dataset_name = 'semeval24' # semeval24, coling24, autext23, autext23_s2
num_classes = 2 # num output classes
num_features = 768 # llm: 768 | w2v: 128, 256, 512, 768
batch_size_gnn = 256 # 16 -> semeval | 64 -> autext
edge_features = False

nfi = 'llm' # llm, w2v, fasttext, random

# google-bert/bert-base-uncased
# FacebookAI/roberta-base
# andricValdez/bert-base-uncased-finetuned-autext23
# andricValdez/roberta-base-finetuned-autext23
# andricValdez/bert-base-uncased-finetuned-autext23_sub2
# andricValdez/roberta-base-finetuned-autext23_sub2
# andricValdez/bert-base-uncased-finetuned-semeval24
# andricValdez/roberta-base-finetuned-semeval24
# andricValdez/roberta-base-finetuned-coling24
llm_model_name = 'andricValdez/roberta-base-finetuned-autext23_sub2' 

device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("device: ", device)

graph_params = {
    'graph_type': 'DiGraph',  
    'window_size': 5,
    'apply_prep': True, 
    'steps_preprocessing': {
        "to_lowercase": True, 
        "handle_blank_spaces": True,
        "handle_html_tags": True,
        "handle_special_chars": False,
        "handle_stop_words": False,
    },
    'language': 'en', #es, en, fr
}

mlflow.set_experiment(f"GNN - {dataset_name}")
run_description = f"""Run experiment for GNN Classification Task using dataset {dataset_name} using {cut_off_dataset} % of the dataset and a {graph_type} graph type"""
run_tags = {
    'mlflow.note.content': run_description,
    'mlflow.source.type': "LOCAL",
    #'mlflow.runName': "test_1perc"
}

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
    autext_train_set = autext_train_set.sample(frac=1).reset_index(drop=True)
    autext_val_set = autext_val_set.sample(frac=1).reset_index(drop=True)
    autext_test_set = autext_test_set.sample(frac=1).reset_index(drop=True)
    print(autext_train_set.info())
    print("label_distro_train_val_test: ", autext_train_set.value_counts('label'), autext_val_set.value_counts('label'), autext_test_set.value_counts('label'))
    
    autext_train_set['word_len'] = autext_train_set['text'].str.split().str.len()
    autext_val_set['word_len'] = autext_val_set['text'].str.split().str.len()
    autext_test_set['word_len'] = autext_test_set['text'].str.split().str.len()
    print("\n min_max_avg_token Train: ", autext_train_set['word_len'].min(), autext_train_set['word_len'].max(), int(autext_train_set['word_len'].mean()))
    print("min_max_avg_token Val:   ", autext_val_set['word_len'].min(), autext_val_set['word_len'].max(),  int(autext_val_set['word_len'].mean()))
    print("min_max_avg_token Test:  ", autext_test_set['word_len'].min(), autext_test_set['word_len'].max(), int(autext_test_set['word_len'].mean()))

    autext_train_set = autext_train_set[(autext_train_set['word_len'] >= 10) & (autext_train_set['word_len'] <= 1500)]
    autext_val_set = autext_val_set[(autext_val_set['word_len'] >= 10) & (autext_val_set['word_len'] <= 1500)]
    #autext_train_set, autext_val_set = train_test_split(autext_train_set, test_size=0.3)
    print("label_distro_train_val_test: ", autext_train_set.value_counts('label'), autext_val_set.value_counts('label'), autext_test_set.value_counts('label'))
    print(autext_train_set.nlargest(5, ['word_len']) )
    #autext_val_set = pd.concat([autext_val_set, autext_val_set_2], axis=0)

    print(autext_train_set.info())
    print(autext_val_set.info())
    print(autext_test_set.info())
    print(autext_train_set['model'].value_counts())
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
    # ****************************** READ DATASET COLING 2024
    '''
    dataset_name = 'coling24' 
    autext_train_set = utils.read_json(dir_path=f'{utils.DATASET_DIR}coling2024/en_train.jsonl')
    autext_val_set = utils.read_json(dir_path=f'{utils.DATASET_DIR}coling2024/en_dev.jsonl')
    autext_test_set = utils.read_json(dir_path=f'{utils.DATASET_DIR}coling2024/test_set_en_with_label.jsonl')
    autext_train_set = autext_train_set.sample(frac=1).reset_index(drop=True)
    autext_val_set = autext_val_set.sample(frac=1).reset_index(drop=True)
    autext_test_set = autext_test_set.sample(frac=1).reset_index(drop=True)
    print(autext_train_set.info())
    autext_test_set = autext_test_set[['testset_id', 'label', 'text']]
    print("distro_train_val_test: ", autext_train_set.shape, autext_val_set.shape, autext_test_set.shape)
    
    autext_train_set['word_len'] = autext_train_set['text'].str.split().str.len()
    autext_val_set['word_len'] = autext_val_set['text'].str.split().str.len()
    autext_test_set['word_len'] = autext_test_set['text'].str.split().str.len()
    print("min_max_avg_token Train: ", autext_train_set['word_len'].min(), autext_train_set['word_len'].max(), int(autext_train_set['word_len'].mean()))
    print("min_max_avg_token Val:   ", autext_val_set['word_len'].min(), autext_val_set['word_len'].max(),  int(autext_val_set['word_len'].mean()))
    print("min_max_avg_token Test:  ", autext_test_set['word_len'].min(), autext_test_set['word_len'].max(), int(autext_test_set['word_len'].mean()))

    min_token_text = 10
    max_token_text = 1500
    autext_train_set = autext_train_set[(autext_train_set['word_len'] >= min_token_text) & (autext_train_set['word_len'] <= max_token_text)]
    autext_val_set = autext_val_set[(autext_val_set['word_len'] >= min_token_text) & (autext_val_set['word_len'] <= max_token_text)]
    print("label_distro_train_val_test: ", autext_train_set.value_counts('label'), autext_val_set.value_counts('label'), autext_test_set.value_counts('label'))
    #print(autext_train_set.nlargest(5, ['word_len']) )

    print("distro_train_val_test: ", autext_train_set.shape, autext_val_set.shape, autext_test_set.shape)
    #print(autext_train_set['model'].value_counts())
    #print(autext_val_set['model'].value_counts())
    '''
    
    # ****************************** READ DATASET AUTEXT 2023
    '''
    dataset_name = 'autext23' # autext23, autext23_s2
    subtask = 'subtask1' # subtask1, subtask2
    
    autext_train_set = utils.read_csv(file_path=f'{utils.DATASET_DIR}autext2023/{subtask}/train_set.csv') 
    autext_val_set = utils.read_csv(file_path=f'{utils.DATASET_DIR}autext2023/{subtask}/val_set.csv') 
    autext_test_set = utils.read_csv(file_path=f'{utils.DATASET_DIR}autext2023/{subtask}/test_set.csv') 
    print(autext_train_set.info())
    print("distro_train_val_test: ", autext_train_set.shape, autext_val_set.shape, autext_test_set.shape)
    print("label_distro_train_val_test: ", autext_train_set.value_counts('label'), autext_val_set.value_counts('label'), autext_test_set.value_counts('label'))
    
    autext_train_set['word_len'] = autext_train_set['text'].str.split().str.len()
    autext_val_set['word_len'] = autext_val_set['text'].str.split().str.len()
    autext_test_set['word_len'] = autext_test_set['text'].str.split().str.len()
    print("min_max_avg_token Train: ", autext_train_set['word_len'].min(), autext_train_set['word_len'].max(), int(autext_train_set['word_len'].mean()))
    print("min_max_avg_token Val:   ", autext_val_set['word_len'].min(), autext_val_set['word_len'].max(),  int(autext_val_set['word_len'].mean()))
    print("min_max_avg_token Test:  ", autext_test_set['word_len'].min(), autext_test_set['word_len'].max(), int(autext_test_set['word_len'].mean()))
    '''
    '''
    print(40*'*', 'Dataset Distro-Partition')
    subtask = 'subtask1' # subtask1, subtask2
    dataset = load_dataset("symanto/autextification2023", 'detection_en') # ['detection_en', 'attribution_en', 'detection_es', 'attribution_es']
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
    for llm in ['andricValdez/bert-base-uncased-finetuned-autext23', 'andricValdez/roberta-base-finetuned-autext23']:
        node_feat_init.llm_fine_tuning(
            model_name = 'autext23',  # autext23, autext24, semeval24, coling24
            train_set_df = autext_train_set, 
            val_set_df = autext_test_set, # autext_val_set, autext_test_set
            device = device,
            llm_to_finetune = llm,
            num_labels = 2,
            mode = 'inference' # finetune, inference
        )
    return
    # google-bert/bert-base-uncased
    # FacebookAI/roberta-base
    # andricValdez/bert-base-uncased-finetuned-autext23
    # andricValdez/roberta-base-finetuned-autext23
    # andricValdez/bert-base-uncased-finetuned-autext23_sub2
    # andricValdez/roberta-base-finetuned-autext23_sub2
    # andricValdez/bert-base-uncased-finetuned-semeval24
    # andricValdez/roberta-base-finetuned-semeval24 
    # andricValdez/roberta-base-finetuned-coling24
    '''
    # ****************************** PROCESS AUTEXT DATASET && CUTOF
    # *** TRAIN
    cut_dataset_train = len(autext_train_set) * (int(cut_off_dataset) / 100)
    autext_train_set = autext_train_set[:int(cut_dataset_train)]
    # *** VAL
    cut_dataset_val = len(autext_val_set) * (int(cut_off_dataset) / 100)
    #cut_dataset_val = len(autext_val_set) * (int(100) / 100)
    autext_val_set = autext_val_set[:int(cut_dataset_val)]
    # *** TEST
    cut_dataset_test = len(autext_test_set) * (int(cut_off_dataset) / 100)
    autext_test_set = autext_test_set[:int(cut_dataset_test)]

    print("label_distro_train_val_test: ", autext_train_set.value_counts('label'), autext_val_set.value_counts('label'), autext_test_set.value_counts('label'))

    # cooc
    train_text_docs = utils.process_dataset(autext_train_set)
    val_text_docs = utils.process_dataset(autext_val_set)
    test_text_docs = utils.process_dataset(autext_test_set)

    # hetero
    #all_text_docs = utils.process_dataset(pd.concat([autext_train_set, autext_val_set, autext_test_set], axis=0))

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
            train_set=autext_train_set[ : ], 
            val_set=autext_val_set[ : ], 
            test_set=autext_test_set[ : ], 
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
    built_graph_dataset = False
    exp_file_name = 'experiments_cooccurrence_20241121192844' 
    experiments_path = f'{utils.OUTPUT_DIR_PATH}batch_experiments/'
    experiments_path_dir = f'{experiments_path}{exp_file_name}/'
    experiments_path_file = f'{experiments_path}{exp_file_name}.csv'

    dataset_partition = f'{dataset_name}_{cut_off_dataset}perc' # perc | perc_go_cls | perc_go_e5
    exp_file_path = f'{utils.OUTPUT_DIR_PATH}test_{dataset_partition}/'

    utils.create_expriment_dirs(experiments_path_dir)
    gnn.graph_neural_network_batch(
        autext_train_set, 
        autext_val_set, 
        autext_test_set, 
        experiments_path_dir, 
        experiments_path_file, 
        exp_file_path=exp_file_path,
        built_graph_dataset=built_graph_dataset,
        num_classes=num_classes
    )
    return
    '''
    # ****************************** GRAPH NEURAL NETWORK - ONE RUNNING
    if graph_type == 'cooc':
        t2g_instance = text2graph.Text2CoocGraph(
            graph_type = graph_params['graph_type'], window_size = graph_params['window_size'], apply_prep = graph_params['apply_prep'], 
            steps_preprocessing = graph_params['steps_preprocessing'], language = graph_params['language'],
        )
    if graph_type == 'hetero':
        ...


    # ****************************** BASELINES W2v
    '''
    #*** GET ground truth
    y_train = np.asarray(autext_train_set['label'].to_list()[:], dtype=np.float32).reshape(-1, 1)
    y_val = np.asarray(autext_val_set['label'].to_list()[:], dtype=np.float32).reshape(-1, 1)
    y_test = np.asarray(autext_test_set['label'].to_list()[:], dtype=np.float32).reshape(-1, 1)
    print("y-train-val-test: ", y_train.shape, y_val.shape, y_test.shape)

    #*** GET w2v doc embeddings
    vector_size = 128
    model_w2v, X_train_vect_avg, X_val_vect_avg, X_test_vect_avg = node_feat_init.w2v_train_v2(train_text_docs[:], val_text_docs, test_text_docs, num_features=vector_size)
    X_train = np.asarray(X_train_vect_avg)
    X_val = np.asarray(X_val_vect_avg)
    X_test = np.asarray(X_test_vect_avg)
    print("X-train-val-test: ", X_train.shape, X_val.shape, X_test.shape)

    #*** TRAIN classifier
    #clf = LinearSVC()
    clf = xgb.XGBClassifier(n_jobs=-1)
    clf_model = clf.fit(X_train, y_train)

    #*** PREDICT val-test
    y_val_pred = clf_model.predict(X_val)
    y_test_pred = clf_model.predict(X_test)

    #*** GET metrics
    print("*** VAL")
    print('Precision: {} / Recall: {} / Accuracy: {} / F1-Score Macro: {}'.format(
        round(precision_score(y_val, y_val_pred), 3), round(recall_score(y_val, y_val_pred), 3), round(accuracy_score(y_val, y_val_pred), 3), round(f1_score(y_val, y_val_pred, average='macro'), 3)))

    print("*** TEST")
    print('Precision: {} / Recall: {} / Accuracy: {} / F1-Score: {}'.format(
        round(precision_score(y_test, y_test_pred), 3), round(recall_score(y_test, y_test_pred), 3), round(accuracy_score(y_test, y_test_pred), 3), round(f1_score(y_test, y_test_pred, average='macro'), 3)))

    return
    '''
    exp_file_name = "test"
    dataset_partition = f'{dataset_name}_{cut_off_dataset}perc' # perc | perc_go_cls | perc_go_e5
    exp_file_path = f'{utils.OUTPUT_DIR_PATH}{exp_file_name}_{dataset_partition}/'
    #utils.delete_dir_files(exp_file_path)
    utils.create_expriment_dirs(exp_file_path)

    # ML Flow Setting
    mlflow.set_tag("mlflow.runName", f"run_{cut_off_dataset}perc_{graph_type}_{nfi}")
    mlflow.log_param('graph_params', graph_params)
    mlflow.log_param('exp_file_path', exp_file_path)
    mlflow.log_param('graph_type', graph_type)
    mlflow.log_param('graph_token', 'lemma') # lemma, text
    mlflow.set_tags({"dataset": dataset_name, "graph_type": graph_type, "cut_off": cut_off_dataset, "nfi": nfi })
    if nfi == 'llm':
        mlflow.log_param('llm_model_name', llm_model_name)
    
    gnn.graph_neural_network( 
        exp_file_name = 'test', 
        dataset_partition = dataset_partition,
        exp_file_path = exp_file_path,
        graph_trans = None, # True, False, None
        nfi = nfi,
        cut_off_dataset = cut_off_dataset, 
        t2g_instance = t2g_instance,
        train_text_docs =  train_text_docs[ : int(len(train_text_docs) * (int(100) / 100))], #train_text_docs
        val_text_docs = val_text_docs[ : int(len(val_text_docs) * (int(100) / 100))], # val_text_docs
        test_text_docs = test_text_docs[ : int(len(test_text_docs) * (int(100) / 100))], # test_text_docs
        #all_text_docs = all_text_docs,
        device = device,
        edge_features = edge_features,
        edge_dim = 2,
        num_features = num_features, 
        batch_size_gnn = batch_size_gnn,
        build_dataset = False, # True, False
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
    #dataset_name = 'semeval24' # autext23, autext23_s2, semeval24
    #cuda_num = 0
    #cut_off_dataset = 40

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
        #'MultinomialNB': MultinomialNB,
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
    print("label_distro_train_val: ", emb_val_gnn_df.value_counts('label'), emb_val_gnn_df.value_counts('label'))


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
    #dataset = 'semeval2024' # autext2023, semeval2024
    #dataset_name = 'semeval24' # autext23, autext23_s2, semeval24
    subtask = 'subtask_1' # subtask_1, subtask_2
    feat_type = 'embedding_' + 'gnn' # embedding_all | embedding_gnn_llm | embedding_gnn |embedding_gnn_stylo | embedding_llm_stylo | embedding_llm_stylo
    
    extract_embeddings = True
    #cuda_num = 0
    #num_labels = 2
    #num_features = 256 # 256, 512, 768
    #edge_features = True
    #nfi='llm' # llm, w2v, fasttext, random
    #llm_model_name = 'andricValdez/bert-base-uncased-finetuned-semeval24'

    #cut_off_dataset = 40
    #cut_off_test_dataset = 33
    exp_file_name = "test"

    device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")

    if subtask == 'subtask_1':
        dataset_partition = f'{dataset_name}_{cut_off_dataset}perc'
        exp_file_path = f'{utils.OUTPUT_DIR_PATH}{exp_file_name}_{dataset_partition}/'
        if dataset_name == 'semeval24': 
            autext_test_set = utils.read_json(dir_path=utils.DATASET_DIR + 'semeval2024/subtask1/subtaskA_test_monolingual.jsonl')
        if dataset_name == 'autext23': 
            autext_test_set = utils.read_csv(file_path=f'{utils.DATASET_DIR}autext2023/subtask1/test_set.csv') 
        if dataset_name == 'coling24': 
            autext_test_set = utils.read_json(dir_path=utils.DATASET_DIR + 'coling2024/test_set_en_with_label.jsonl')

    if subtask == 'subtask_2':
        clf_model_name_ml = f'XGBClassifier_train_{feat_type}' # XGBClassifier_train_ | LogisticRegression_train_ | SGDClassifier_train_ | LinearSVC_train_
        dataset_partition = f'{dataset_name}_s2_{cut_off_dataset}perc'
        exp_file_path = f'{utils.OUTPUT_DIR_PATH}{exp_file_name}_{dataset_partition}/'
        autext_test_set = utils.read_csv(file_path=f'{utils.DATASET_DIR}{dataset}/subtask2/test_set.csv') 


    # ******************** Read TEST set
    #print(autext_test_set.info())
    #autext_test_set = autext_test_set.sort_values('id')
    ''' tmp
    test_text_docs = []
    corpus_text_docs_dict = autext_test_set.to_dict('records')
    for idx, instance in enumerate(corpus_text_docs_dict):
        doc = {
            "id": idx+1, 
            "doc": instance['text'][:], 
            "context": {"id": idx+1, "target": instance['label']}
        }
        test_text_docs.append(doc)

    cut_dataset_test = len(test_text_docs) * (int(cut_off_test_dataset) / 100)
    test_text_docs = test_text_docs[:int(cut_dataset_test)]

    t2g_instance = text2graph.Text2CoocGraph(
        graph_type = graph_params['graph_type'], window_size = graph_params['window_size'], apply_prep = graph_params['apply_prep'], 
        steps_preprocessing = graph_params['steps_preprocessing'], language = graph_params['language'],
    )
    '''
    #gnn.graph_neural_network_test_eval(test_text_docs, t2g_instance, nfi, exp_file_path, dataset_partition, llm_model_name, device) 
    #return


    # ******************** get Features/Embeddings    
    if extract_embeddings:
        #  get GNN Emb
        test_text_docs = None
        t2g_instance = None
        gnn.graph_neural_network_test_eval(test_text_docs, t2g_instance, nfi, exp_file_path, dataset_partition, llm_model_name, edge_features, num_features, batch_size_gnn, device) 
  
        #  get LLM Emb 1
        #utils.llm_get_embbedings(
        #    text_data=test_text_docs, 
        #    exp_file_path=exp_file_path+'embeddings_cls_llm_1/', subset='test', 
        #    emb_type='llm_cls', device=device, save_emb=True, 
        #    llm_finetuned_name=llm_model_name, 
        #    num_labels=num_classes
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

    
    classifiers = ['XGBClassifier_train_', 'LinearSVC_train_', 'SGDClassifier_train_', 'LogisticRegression_train_']
    for clf in classifiers:
        #print("****************************** ", clf)
        clf_model_name_ml = f'{clf}{feat_type}' # XGBClassifier_train_ | LogisticRegression_train_ | SGDClassifier_train_ | LinearSVC_train_

        clf_model = utils.load_data(path=f'{exp_file_path}clf_models/', file_name=f'{clf_model_name_ml}') 
        y_pred = clf_model.predict(test_feat_data_pt)
        
        merge_test_embeddings['y_pred'] = y_pred
        y_true = merge_test_embeddings['label_gnn'].values.tolist()
        y_pred = merge_test_embeddings['y_pred'].values.tolist()
        #print(merge_test_embeddings.info()) 
        print(merge_test_embeddings['label_gnn'].value_counts())
        print(merge_test_embeddings['y_pred'].value_counts())

        acc_score = accuracy_score(y_true, y_pred)
        mf1_score = f1_score(y_true, y_pred , average='macro')
        print('\t Accuracy:', acc_score)
        print('\t F1Score:', mf1_score)

        correct = 0
        for idx in range(0, len(merge_test_embeddings)):
            if y_pred[idx] == y_true[idx]:
                correct += 1

        print(correct, correct/len(merge_test_embeddings))

        #mlflow.log_metric(key=f"F1Score-test", value=float(mf1_score))
        #mlflow.log_metric(key=f"Accuracy-test", value=float(acc_score))


    return

 

if __name__ == '__main__':
    with mlflow.start_run(tags=run_tags) as run:
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


















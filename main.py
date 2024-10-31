

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

cuda_num = 0
device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")

cut_off_dataset = 100
dataset_name = 'autext23'
num_classes = 2 # num output classes
num_features = 256 # llm: 768 | w2v: 256, 512, 768
batch_size_gnn = 32 # 16 -> semeval | 64 -> autext
edge_features = True
nfi = 'w2v' # llm, w2v, fasttext, random

# google-bert/bert-base-uncased
# FacebookAI/roberta-base
# andricValdez/bert-base-uncased-finetuned-semeval24
llm_model_name = 'andricValdez/bert-base-uncased-finetuned-semeval24'


def main():
    # NOTA: el prefijo "autext_" se quedó para todos los dataset, no solo para el de autexttification

    # ****************************** READ DATASET AUTEXT 2023
    dataset_name = 'autext23' # autext23, autext23_s2
    subtask = 'subtask1' # subtask1, subtask2
    
    autext_train_set = utils.read_csv(file_path=f'{utils.DATASET_DIR}autext2023/{subtask}/train_set.csv') 
    autext_val_set = utils.read_csv(file_path=f'{utils.DATASET_DIR}autext2023/{subtask}/val_set.csv') 
    autext_test_set = utils.read_csv(file_path=f'{utils.DATASET_DIR}autext2023/{subtask}/test_set.csv') 
    print(autext_train_set.info())
    print(autext_val_set.info())
    print(autext_test_set.info())
    print(autext_train_set['label'].value_counts())

    # ****************************** PROCESS AUTEXT DATASET && CUTOF
    train_text_docs = utils.process_dataset(autext_train_set)
    val_text_docs = utils.process_dataset(autext_val_set)
    test_text_docs = utils.process_dataset(autext_test_set)

    # *** TRAIN
    cut_dataset_train = len(train_text_docs) * (int(cut_off_dataset) / 100)
    train_text_docs = train_text_docs[:int(cut_dataset_train)]
    # *** VAL
    cut_dataset_val = len(val_text_docs) * (int(cut_off_dataset) / 100)
    val_text_docs = val_text_docs[:int(cut_dataset_val)]
    # *** TEST
    cut_dataset_test = len(test_text_docs) * (int(cut_off_dataset) / 100)
    test_text_docs = test_text_docs[:int(cut_dataset_test)]


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

    # ****************************** GRAPH NEURAL NETWORK - ONE RUNNING
    graph_params = {
        'graph_type': 'DiGraph', 
        'window_size': 3,
        'apply_prep': True, 
        'steps_preprocessing': {
            "to_lowercase": True, 
            "handle_blank_spaces": True,
            "handle_html_tags": True,
            "handle_special_chars": True,
            "handle_stop_words": True,
        },
        'language': 'en', #es, en, fr
    }

    t2g_instance = text2graph.Text2Graph(
        graph_type = graph_params['graph_type'], window_size = graph_params['window_size'], apply_prep = graph_params['apply_prep'], 
        steps_preprocessing = graph_params['steps_preprocessing'], language = graph_params['language'],
    )

    exp_file_name = "test"
    dataset_partition = f'{dataset_name}_{cut_off_dataset}perc' # perc | perc_go_cls | perc_go_e5
    exp_file_path = f'{utils.OUTPUT_DIR_PATH}{exp_file_name}_{dataset_partition}/'
    utils.create_expriment_dirs(exp_file_path)
    
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
        build_dataset = True, # False, False
        save_data = True,
        llm_finetuned_name = llm_model_name,
        num_classes = num_classes,
    )
  
    torch.cuda.empty_cache()
    gc.collect()
    




if __name__ == '__main__':
    main()


# ********* CMDs
# python main.py
# nohup bash main.sh >> logs/xxx.log &
# nohup python main.py >> logs/text_to_graph_transform_small.log &
# ps -ef | grep python | grep avaldez
# tail -f logs/experiments_cooccurrence_20240502062849.log 


















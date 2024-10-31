

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

# google-bert/bert-base-uncased
# FacebookAI/roberta-base
llm = 'google-bert/bert-base-uncased'


def main():

    # ****************************** READ DATASET 

    dataset_name = 'autext23' # autext23, autext23_s2
    subtask = 'subtask1' # subtask1, subtask2
    
    autext_train_set = utils.read_csv(file_path=f'{utils.DATASET_DIR}autext2023/{subtask}/train_set.csv') 
    autext_val_set = utils.read_csv(file_path=f'{utils.DATASET_DIR}autext2023/{subtask}/val_set.csv') 
    autext_test_set = utils.read_csv(file_path=f'{utils.DATASET_DIR}autext2023/{subtask}/test_set.csv') 
    print(autext_train_set.info())
    print(autext_val_set.info())
    print(autext_test_set.info())
    print(autext_train_set['label'].value_counts())


    # ****************************** Fine Tunning

    node_feat_init.llm_fine_tuning(
        model_name = 'autext23',  # autext23, autext24, semeval24
        train_set_df = autext_train_set, 
        val_set_df = autext_val_set, # autext_val_set, autext_test_set
        device = device,
        llm_to_finetune = llm,
        num_labels = 2
    )
    
    
    


if __name__ == '__main__':
    main()















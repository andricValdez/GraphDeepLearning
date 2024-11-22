
import os
import sys
import joblib
import time
import numpy as np
import pandas as pd 
import logging
import traceback  
import math 
from tqdm import tqdm
import torch
import torch.nn as nn
import networkx as nx 
import scipy as sp
import scipy.sparse as sp
import gensim

from scipy.sparse import coo_array 
from sklearn.datasets import fetch_20newsgroups
import gc
import glob
import torch.nn.functional as F
from torch_geometric.data import DataLoader, Data
from collections import OrderedDict
import warnings
from transformers import logging as transform_loggin

import utils
import node_feat_init
from stylometric import StyloCorpus
import text2graph


#************************************* CONFIGS
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s; - %(levelname)s; - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
warnings.filterwarnings("ignore")
transform_loggin.set_verbosity_error()

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


class BuildDataset():
    def __init__(self, graphs_data, subset, device,  edge_features=False, nfi='llm', llm_finetuned_name=node_feat_init.LLM_HF_FINETUNED_NAME, exp_file_path=utils.OUTPUT_DIR_PATH, num_labels=2, dataset_partition=None, num_features=None):
        self.graphs_data = graphs_data
        self.nfi_model = None
        self.device = device
        self.subset = subset
        self.nfi = nfi
        self.edge_features = edge_features
        self.llm_finetuned_name = llm_finetuned_name
        self.exp_file_path = exp_file_path
        self.num_labels = num_labels
        self.dataset_partition = dataset_partition
        self.num_features = num_features

    def process_dataset(self):

        if self.subset == 'train':
            if self.nfi == 'w2v':
                self.nfi_model = node_feat_init.w2v_train(graph_data=self.graphs_data, num_features=self.num_features)
                utils.save_data(self.nfi_model, path=f'{utils.OUTPUT_DIR_PATH}w2v_models/', file_name=f'model_w2v_{self.dataset_partition}')
            if self.nfi == 'fasttext':
                self.nfi_model = node_feat_init.fasttext_train(graph_data=self.graphs_data, num_features=self.num_features)
                utils.save_data(self.nfi_model, path=f'{utils.OUTPUT_DIR_PATH}fasttext_models/', file_name=f'model_fasttext_{self.dataset_partition}')

        else:
            if self.nfi == 'w2v':
                self.nfi_model = utils.load_data(path=f'{utils.OUTPUT_DIR_PATH}w2v_models/', file_name=f'model_w2v_{self.dataset_partition}')
            if self.nfi == 'fasttext':
                self.nfi_model = utils.load_data(path=f'{utils.OUTPUT_DIR_PATH}fasttext_models/', file_name=f'model_fasttext_{self.dataset_partition}')
                        
        # self.nfi == 'llm' is handle below in the batch graph process (due to memory)
        
        block = 1
        batch_size = utils.LLM_GET_EMB_BATCH_SIZE_DATALOADER
        num_batches = math.ceil(len(self.graphs_data) / batch_size)
        oov_cnt_total = 0

        for index_out_batch, _ in enumerate(tqdm(range(num_batches))):
            data_list = []
            graphs_data_batch = self.graphs_data[batch_size*(block-1) : batch_size*block]
            if self.nfi == 'llm':
                # Data storage/saved
                #self.llm_model = utils.read_json(file_path=path_file + f'pan24_{self.subset}_emb_batch_{block-1}.json')
                #self.llm_model = self.llm_model.to_dict('records')
                # Data In memory
                dataset = [{'id': d['context']['id'], 'label': d['context']['target'], 'text': " ".join(list(d['graph'].nodes))} for d in graphs_data_batch]
                self.nfi_model = node_feat_init.llm_get_embbedings_2(dataset, subset=self.subset, emb_type='llm_word', device=self.device, save_emb=False, llm_finetuned_name=self.llm_finetuned_name, num_labels=self.num_labels)
                # Data in bacthes
                '''
                cnt_tokens = 1
                unique_nodes, lst_nodes, dataset = [], [], []
                for g in self.graphs_data:
                    unique_nodes.extend([n for n in list(g['graph'].nodes)])
                unique_nodes = set(unique_nodes)
                for idx, node in enumerate(unique_nodes): 
                    lst_nodes.append(str(node))
                    if cnt_tokens == 100:
                        dataset.append({'text': " ".join(lst_nodes)})
                        lst_nodes = []
                        cnt_tokens = 0
                    cnt_tokens += 1
                self.nfi_model = node_feat_init.llm_get_embbedings_2(dataset, subset=self.subset, emb_type='llm_word', device=self.device, save_emb=False, llm_finetuned_name=self.llm_finetuned_name, num_labels=self.num_labels)
                #print("unique_nodes: ", len(unique_nodes), " | nfi_model_keys: ", len(self.nfi_model.keys()) , ' | perc: ', (len(self.nfi_model.keys())/len(unique_nodes))*100)
                '''
            for index_in_batch, g in enumerate(graphs_data_batch):
                #print(g['graph'])
                try:
                    # Get node features
                    node_feats, oov_cnt = self.get_node_features(g, nfi_type=self.nfi)
                    # Get edge features
                    edge_attr = self.get_edge_features(g['graph'])
                    # Get adjacency info
                    edge_index = self.get_adjacency_info(g['graph'])
                    # Get labels info
                    label = self.get_labels(g["context"]["target"])
                    
                    #print(node_feats.shape, edge_index.shape, label.shape)
                    data = Data(
                        x = node_feats,
                        edge_index = edge_index,
                        edge_attr = edge_attr,
                        y = label,
                        pred = '',
                        context = g["context"]
                    )
                    data_list.append(data)
                except Exception as e:
                    ...
                    logger.error('Error: %s', str(e))
                    #print(g)
                    #print("traceback: ", str(traceback.format_exc()))
                else:
                    ...
                    oov_cnt_total += oov_cnt
                    #print(len(g['graph'].nodes), " | oov_cnt: ", oov_cnt)

            torch.save(data_list, f'{self.exp_file_path}embeddings_word_llm/data_{self.subset}_{index_out_batch}.pt')
            block += 1
            #del self.llm_model
        print("oov_cnt_total: ", oov_cnt_total)
        return data_list
        

    def get_node_features(self, g, nfi_type='w2v'):
        oov_cnt = 0
        graph_node_feat = []

        if nfi_type == 'ohe':
            for node in list(g['graph'].nodes):
                vector = np.zeros(len(self.vocab))
                vector[self.node_to_index[node]] = 1
                graph_node_feat.append(vector)
        
        elif nfi_type in ['w2v', 'fasttext']:
            for node in list(g['graph'].nodes):
                try:
                    w_emb = self.nfi_model.wv[node]
                    graph_node_feat.append(w_emb)
                except Exception as e:
                    g['graph'].remove_node(node)
                    #logger.error('Error: %s', str(e))
                    #graph_node_feat.append(self.get_random_emb(emb_dim=self.num_features))
                    oov_cnt += 1

        elif nfi_type == 'llm':
            #print(str(g['doc_id']), len(g['graph'].nodes))
            for n in list(g['graph'].nodes):
                if str(n) in self.nfi_model[str(g['doc_id'])]['embedding'].keys(): 
                    graph_node_feat.append(self.nfi_model[str(g['doc_id'])]['embedding'][n]) 
                #if str(n) in self.nfi_model.keys():
                #    graph_node_feat.append(self.nfi_model[str(n)])
                else:
                    g['graph'].remove_node(n)
                    #graph_node_feat.append(self.get_random_emb(emb_dim=self.num_features))
                    oov_cnt += 1
            #print(str(g['doc_id']), len(g['graph'].nodes))

        else: # random init

            # initialise an Embedding layer from Torch
            word_nodes = list(g['graph'].nodes)
            encoded_word_nodes = [indx for indx, word_node in enumerate(word_nodes)]
            emb = nn.Embedding(len(word_nodes), self.num_features)
            word_vectors_emb = emb(torch.tensor(encoded_word_nodes))
            graph_node_feat = word_vectors_emb.detach().numpy()

        graph_node_feat = np.asarray(graph_node_feat)
        return torch.tensor(graph_node_feat, dtype=torch.float), oov_cnt


    def get_adjacency_info(self, g):
        adj_tmp = nx.to_scipy_sparse_array(g,  weight='pmi', dtype=np.cfloat) # weight= weight, pmi, tfidf
        #adj_tmp = adj_tmp + adj_tmp.T.multiply(adj_tmp.T > adj_tmp) - adj_tmp.multiply(adj_tmp.T > adj_tmp)
        #adj_normalized = self.normalize_adj(adj_tmp + sp.eye(adj_tmp.shape[0]))
        #return self.sparse_mx_to_torch_sparse_tensor(adj_normalized)['indices']
        
        adj_coo = sp.coo_array(adj_tmp)
        edge_indices = []
        for index in range(len(g.edges)):
            edge_indices += [[adj_coo.row[index], adj_coo.col[index]]]

        edge_indices = torch.tensor(edge_indices) 
        t = edge_indices.t().to(torch.long).view(2, -1)  
        #print("edge_index:", t.shape)
        return edge_indices.t().to(torch.long).view(2, -1)
        

    def get_edge_features(self, g):
        if self.edge_features:
            all_edge_feats = []
            for edge in g.edges(data=True):
                feats = edge[2]
                edge_feats = []
                # Feature 1: freq
                edge_feats.append(feats['freq'])
                # Feature 2: pmi
                edge_feats.append(feats['pmi'])
                # Append node features to matrix (twice, per direction)
                edge_feats = np.asarray(edge_feats)
                edge_feats = edge_feats/np.linalg.norm(edge_feats)
                all_edge_feats += [edge_feats]

            all_edge_feats = np.asarray(all_edge_feats)
            all_edge_feats = torch.tensor(all_edge_feats, dtype=torch.float)
            #print("edge_feat :", all_edge_feats.shape)
            return all_edge_feats
        else:
            return None


    def get_labels(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)

    def get_random_emb(self, emb_dim):
        emb = nn.Embedding(1, emb_dim)
        word_emb = emb(torch.tensor([0]))
        return word_emb.detach().numpy()[0]

    def sparse_mx_to_torch_sparse_tensor(self, adj):
        """Convierta una matriz esparcida a un tensor esparcido."""
        adj = adj.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((adj.row, adj.col)).astype(np.int64))
        values = torch.from_numpy(adj.data)
        shape = torch.Size(adj.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def normalize_adj(self, adj):
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
        
        
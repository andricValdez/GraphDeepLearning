
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
from transformers import AutoTokenizer, AutoModel, Trainer, AutoModelForSequenceClassification, TrainingArguments

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
    def __init__(self, graphs_data, text_docs, subset, device,  edge_features=False, 
                 nfi='llm', llm_finetuned_name=node_feat_init.LLM_HF_FINETUNED_NAME, 
                 exp_file_path=utils.OUTPUT_DIR_PATH, num_labels=2, dataset_partition=None, 
                 num_features=None, text2graph_type='cooc', avg_llm_found_tokens=True,
                 avg_llm_not_found_tokens=True, doc_embs_hetero=[], set_idxs={}
                ):
        
        self.graphs_data = graphs_data
        self.doc_embs_hetero = doc_embs_hetero 
        self.text_docs = text_docs
        self.nfi_model = {}
        self.device = device
        self.subset = subset
        self.nfi = nfi
        self.edge_features = edge_features
        self.llm_finetuned_name = llm_finetuned_name
        self.exp_file_path = exp_file_path
        self.num_labels = num_labels
        self.dataset_partition = dataset_partition
        self.num_features = num_features
        self.text2graph_type = text2graph_type
        self.avg_llm_found_tokens = avg_llm_found_tokens
        self.avg_llm_not_found_tokens = avg_llm_not_found_tokens
        self.set_idxs = set_idxs

    def process_dataset(self):
        if self.text2graph_type == 'cooc':
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

            text_docs_cnt = 0
            tokenizer = AutoTokenizer.from_pretrained(self.llm_finetuned_name, trust_remote_code=True)
            model = AutoModel.from_pretrained(self.llm_finetuned_name, trust_remote_code=True, output_hidden_states=True)
            model = model.to(self.device)

            for index_out_batch, _ in enumerate(tqdm(range(num_batches))):
                data_list = []
                graphs_data_batch = self.graphs_data[batch_size*(block-1) : batch_size*block]
                if self.nfi == 'llm':
                    # Data storage/saved
                    #self.llm_model = utils.read_json(file_path=path_file + f'pan24_{self.subset}_emb_batch_{block-1}.json')
                    #self.llm_model = self.llm_model.to_dict('records')
                    # Data In memory
                    # ************************************************************* Prev implementation
                    #dataset = [{'id': d['context']['id'], 'label': d['context']['target'], 'text': " ".join(list(d['graph'].nodes))} for d in graphs_data_batch]
                    #self.nfi_model = node_feat_init.llm_get_embbedings_2(dataset, subset=self.subset, emb_type='llm_word', device=self.device, save_emb=False, llm_finetuned_name=self.llm_finetuned_name, num_labels=self.num_labels)
                    
                    # ************************************************** NEW implementation (20/01/2025)
                    self.nfi_model = {}
                    for idx, g in enumerate(graphs_data_batch): 
                        embeddings_word_dict = node_feat_init.llm_get_embbedings_3([{'doc': self.text_docs[text_docs_cnt]['doc']}], self.device, model, tokenizer, self.avg_llm_found_tokens, self.text2graph_type)
                        #print('\n')
                        #print("doc_id: ", self.text_docs[text_docs_cnt]['id'])
                        #print(len(list(g['graph'].nodes)), list(g['graph'].nodes))
                        #print(len(embeddings_word_dict.keys()), embeddings_word_dict.keys())
                        self.nfi_model[str(graphs_data_batch[idx]['doc_id'])] = node_feat_init.get_emb_word_nodes(embeddings_word_dict, g, tokenizer, self.avg_llm_not_found_tokens, self.text2graph_type)
                        text_docs_cnt += 1
                
                for index_in_batch, g in enumerate(graphs_data_batch):
                    #print(g['graph'])
                    try:
                        # Get node features
                        node_feats, oov_cnt = self.get_node_cooc_features(g, nfi_type=self.nfi)
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
                        print(g)
                        #print("traceback: ", str(traceback.format_exc()))
                    else:
                        ...
                        oov_cnt_total += oov_cnt
                        #print(len(g['graph'].nodes), " | oov_cnt: ", oov_cnt)

                torch.save(data_list, f'{self.exp_file_path}embeddings_word_llm/data_{self.subset}_{index_out_batch}.pt')
                block += 1
                #del self.llm_model
            print("oov_cnt_total: ", oov_cnt_total)        


        if self.text2graph_type == 'hetero':
            data_list = []
            graph_data = self.graphs_data[0]
            if self.nfi == 'llm':
                # google/bigbird-roberta-base
                # bert-base-cased
                tokenizer = AutoTokenizer.from_pretrained(self.llm_finetuned_name)
                model = AutoModel.from_pretrained(self.llm_finetuned_name, output_hidden_states=True)
                model = model.to(self.device)

                if self.nfi == 'llm':
                    embeddings_word_dict = node_feat_init.llm_get_embbedings_3(self.text_docs, self.device, model, tokenizer, self.avg_llm_found_tokens, self.text2graph_type)
                    self.nfi_model = node_feat_init.get_emb_word_nodes(embeddings_word_dict, graph_data, tokenizer, self.avg_llm_not_found_tokens, self.text2graph_type)
                    print("nfi_model.keys: ", len(self.nfi_model.keys()))
            else:
                self.nfi_model = node_feat_init.w2v_train(graph_data=graph_data, num_features=self.num_features)

            try:
                # Get node features
                node_feats, oov_cnt, _ = self.get_node_hetero_features(graph_data, nfi_type=self.nfi)
                train_mask, val_mask, test_mask, y_mask = utils.get_masks_hetero_graph(graph_data, self.text_docs, self.set_idxs)
                # Get edge features
                edge_attr = self.get_edge_features(graph_data['graph'])
                # Get adjacency info
                edge_index = self.get_adjacency_info(graph_data['graph'])

                data = Data(
                    x = node_feats, 
                    edge_index = edge_index,
                    edge_attr = edge_attr,
                    y = torch.tensor(np.asarray(y_mask), dtype=torch.int64),
                    train_mask = torch.tensor(np.asarray(train_mask), dtype=torch.bool),
                    val_mask = torch.tensor(np.asarray(val_mask), dtype=torch.bool),
                    test_mask = torch.tensor(np.asarray(test_mask), dtype=torch.bool),
                )
                data_list.append(data)
            except Exception as e:
                ...
                print("error: ", e)
                print("traceback: ", str(traceback.format_exc()))
            else:
                print("oov_cnt_total: ", oov_cnt)
        
            torch.save(data_list, f'{self.exp_file_path}embeddings_word_llm/data_hetero_all.pt')

        del self.nfi_model
        embeddings_word_dict = []
        return data_list


    def get_node_cooc_features(self, g, nfi_type='w2v'):
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
            #print(str(g['doc_id']), len(g['graph'].nodes), self.nfi_model[str(g['doc_id'])].keys())
            #print(self.nfi_model[str(g['doc_id'])].keys())
            for n in list(g['graph'].nodes):
                #if str(n) in self.nfi_model[str(g['doc_id'])]['embedding'].keys(): 
                #    graph_node_feat.append(self.nfi_model[str(g['doc_id'])]['embedding'][n]) 
                if str(n) in self.nfi_model[str(g['doc_id'])].keys():
                    if len(self.nfi_model[str(g['doc_id'])][str(n)]) != 768:
                        #print('error in not_found_tokens, removing node from graph: ', str(g['doc_id']), n, len(self.nfi_model[str(g['doc_id'])][str(n)]), self.nfi_model[str(g['doc_id'])][str(n)])
                        g['graph'].remove_node(str(n))
                        continue
                    graph_node_feat.append(self.nfi_model[str(g['doc_id'])][str(n)])
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

    
    def get_node_hetero_features(self, g, nfi_type='w2v'):
        oov_cnt, no_oov_cnt = 0, 0
        graph_node_feat = []

        if nfi_type == 'ohe':
            for node in list(g['graph'].nodes):
                vector = np.zeros(len(self.vocab))
                vector[self.node_to_index[node]] = 1
                graph_node_feat.append(vector)

        elif nfi_type in ['w2v', 'fasttext']:
          words = set(self.nfi_model_w2v.wv.index_to_key)
          doc_idx = 0
          for node in list(g['graph'].nodes):
              try:
                if str(node).startswith("D-"):
                  d_emb = self.doc_embs[doc_idx]
                  graph_node_feat.append(d_emb)
                  doc_idx += 1
                else:
                  if str(node) in words:
                    w_emb = self.nfi_model_w2v.wv[node]
                    graph_node_feat.append(w_emb)
                  else:
                    oov_cnt += 1
                    if self.oov_feat_type == 'remove':
                      g['graph'].remove_node(node)
                    elif self.oov_feat_type == 'random':
                      graph_node_feat.append(self.get_random_emb(emb_dim=self.num_features))
                    elif self.oov_feat_type == 'zeros':
                      graph_node_feat.append(torch.zeros(self.num_features))
                    elif self.oov_feat_type == 'ones':
                      graph_node_feat.append(torch.ones(self.num_features))
                    else:
                      graph_node_feat.append(self.get_random_emb(emb_dim=self.num_features))
              except Exception as e:
                print('Error: %s', str(e))

        elif nfi_type == 'llm':
          doc_idx = 0
          for node in list(g['graph'].nodes):
            try:
              if str(node).startswith("D-"):
                d_emb = self.doc_embs_hetero[doc_idx]
                graph_node_feat.append(d_emb)
                doc_idx += 1
              else:
                if str(node) in self.nfi_model.keys():
                    if len(self.nfi_model[str(node)]) != 768:
                      #print('error in not_found_tokens, removing node from graph: ', node, len(self.nfi_model[str(node)]), self.nfi_model[str(node)])
                      g['graph'].remove_node(str(node))
                      continue
                    graph_node_feat.append(self.nfi_model[str(node)])
                else:
                    oov_cnt += 1
                    if self.oov_feat_type == 'remove':
                      g['graph'].remove_node(node)
                    elif self.oov_feat_type == 'random':
                      graph_node_feat.append(self.get_random_emb(emb_dim=self.num_features))
                    elif self.oov_feat_type == 'zeros':
                      graph_node_feat.append(torch.zeros(self.num_features))
                    elif self.oov_feat_type == 'ones':
                      graph_node_feat.append(torch.ones(self.num_features))
                    else:
                      graph_node_feat.append(self.get_random_emb(emb_dim=self.num_features))

            except Exception as e:
              print('Error: %s', str(e))
              graph_node_feat.append(self.get_random_emb(emb_dim=self.num_features))

        elif nfi_type == 'ones': # generate a vect-emb of 1s
            word_nodes = list(g['graph'].nodes)
            graph_node_feat = [torch.ones(self.num_features) for indx, word_node in enumerate(word_nodes)]

        elif nfi_type == 'identity': # generate a vect-emb of 0s
            graph_node_feat = np.eye(len(list(g['graph'].nodes)))

        else: # random init
            # initialise an Embedding layer from Torch
            word_nodes = list(g['graph'].nodes)
            encoded_word_nodes = [indx for indx, word_node in enumerate(word_nodes)]
            emb = nn.Embedding(len(word_nodes), self.num_features)
            word_vectors_emb = emb(torch.tensor(encoded_word_nodes))
            graph_node_feat = word_vectors_emb.detach().numpy()

        #for i in list(graph_node_feat):
        #  if len(list(i)) != 768:
        #    print('< 768')
        #    print(len(list(i)), i)

        graph_node_feat = np.asarray(graph_node_feat)
        return torch.tensor(graph_node_feat, dtype=torch.float), oov_cnt, no_oov_cnt


    def get_adjacency_info(self, g):
        adj_tmp = nx.to_scipy_sparse_array(g,  weight='weight', dtype=np.cfloat)
        adj_coo = sp.coo_array(adj_tmp)
        edge_index = torch.sparse_coo_tensor([adj_coo.row, adj_coo.col], adj_coo.data, (2, len(adj_coo.col)))
        return edge_index.coalesce().indices().to(torch.long) 

        '''
        adj_tmp = nx.to_scipy_sparse_array(g,  weight='weight', dtype=np.cfloat) # weight= weight, pmi, tfidf
        adj_coo = sp.coo_array(adj_tmp)
        edge_indices = []
        for index in range(len(g.edges)):
            edge_indices += [[adj_coo.row[index], adj_coo.col[index]]]

        edge_indices = torch.tensor(edge_indices) 
        t = edge_indices.t().to(torch.long).view(2, -1)  
        return t
        '''


    def get_edge_features(self, g):
        if self.edge_features:
            all_edge_feats = []
            adj_coo = sp.coo_array(nx.to_scipy_sparse_array(g,  weight='weight', dtype=np.cfloat))
            adj_sparse_coo = torch.sparse_coo_tensor([adj_coo.row, adj_coo.col], adj_coo.data, (2, len(adj_coo.col)))
            all_edge_feats = [[ 1 / np.linalg.norm(feat.numpy()) ] for feat in adj_sparse_coo.coalesce().values()]
            all_edge_feats = np.asarray(all_edge_feats)
            all_edge_feats = torch.tensor(all_edge_feats, dtype=torch.float)
            return all_edge_feats
        else:
            return None
    
        '''
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
        '''


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
        
        

import pprint
import torch
import numpy as np
from torch_geometric.data import Data, DataLoader
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
from gensim.models import Word2Vec
from itertools import combinations
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import os
import scipy as sp
from scipy.sparse import coo_array
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import gensim
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
from sklearn.utils import shuffle
from collections import Counter, defaultdict
from datasets import load_dataset
import joblib
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
import networkx as nx
import networkx
import sys, traceback, time
from joblib import Parallel, delayed
import warnings
import nltk
import re, string, math
import codecs
import multiprocessing
from spacy.tokens import Doc
import spacy
from spacy.lang.xx import MultiLanguage
from spacy.cli import download
from spacy.tokenizer import Tokenizer
from spacy.util import compile_prefix_regex, compile_infix_regex, compile_suffix_regex
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import itertools
from math import log
from sklearn.metrics import accuracy_score, f1_score
from transformers import logging
from transformers import AutoTokenizer, AutoModel, Trainer, AutoModelForSequenceClassification, TrainingArguments
from transformers import BertForSequenceClassification, RobertaForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import get_scheduler
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
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
import copy
from tqdm import tqdm
import random
from scipy.sparse import coo_array
import gc
import glob
import torch.nn.functional as F
from torch_geometric.data import DataLoader, Data
from collections import OrderedDict
import warnings
from transformers import logging as transform_loggin
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, TransformerConv, TopKPooling, GraphConv, SAGPooling, GENConv, GINConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch.nn import Linear, BatchNorm1d, ModuleList, LayerNorm
import torch
torch.cuda.is_available()

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import contractions
nltk.download('stopwords')
nltk.download('punkt_tab')

import test_utils
import utils

#************************************* CONFIGS
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s; - %(levelname)s; - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
warnings.filterwarnings("ignore")


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        #print(validation_loss, self.min_validation_loss, self.counter)
        if validation_loss <= self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, dense_hidden_dim, output_dim, dropout, num_layers, edge_attr=False, gnn_type='GCNConv', heads=1, task='node'):
        super(GNN, self).__init__()
        self.task = task
        self.heads = heads
        self.gnn_type = gnn_type
        self.edge_attr = edge_attr
        self.conv1 = self.build_conv_model(input_dim, hidden_dim, self.heads)
        self.norm1 = nn.LayerNorm(hidden_dim * heads)
        self.convs = nn.ModuleList()
        #self.convs.append(self.build_conv_model(input_dim, hidden_dim, self.heads))
        self.lns = nn.ModuleList()
        for l in range(num_layers):
            self.convs.append(self.build_conv_model(hidden_dim * heads, hidden_dim, self.heads))
            self.lns.append(nn.LayerNorm(hidden_dim * heads))

        # Post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim * heads, dense_hidden_dim),
            #nn.Dropout(dropout),
            #nn.LayerNorm(dense_hidden_dim),
            nn.Linear(dense_hidden_dim, int(dense_hidden_dim // 2)),
            #nn.Dropout(dropout),
            #nn.LayerNorm(int(dense_hidden_dim // 2)),
            nn.Linear(int(dense_hidden_dim // 2), output_dim),
            #nn.Linear(int(dense_hidden_dim // 4), output_dim)
        )

        self.dropout = dropout
        self.num_layers = num_layers

    def build_conv_model(self, input_dim, hidden_dim, heads):
        if self.gnn_type == 'GCNConv':
            return GCNConv(input_dim, hidden_dim)
        if self.gnn_type == 'GINConv':
            return GCNConv(input_dim, hidden_dim)
        elif self.gnn_type == 'GATConv':
            return GATConv(input_dim, hidden_dim, heads=heads)
        elif self.gnn_type == 'TransformerConv':
            if self.edge_attr:
                return TransformerConv(input_dim, hidden_dim, heads=heads, edge_dim=2)
            else:    
                return TransformerConv(input_dim, hidden_dim, heads=heads)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if self.edge_attr:
            x = self.conv1(x, edge_index, edge_attr)
        else:
            x = self.conv1(x, edge_index)
        emb = x
        x = F.relu(x)
        x = self.norm1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        for i in range(self.num_layers):
            if self.edge_attr:
                x = self.convs[i](x, edge_index, edge_attr)
            else:
                x = self.convs[i](x, edge_index)
            emb = x
            x = F.relu(x)
            x = self.lns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.task == 'graph':
            x = global_mean_pool(x, batch)

        x = self.post_mp(x)
        # F.log_softmax(x, dim=1)
        # self.sigmoid(x)
        return emb, None, F.log_softmax(x, dim=1)

def train_cooc(model, loader, device, optimizer, criterion):
        model.train()
        train_loss = 0.0
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            emb, _, out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = criterion(out, data.y)
            loss.backward(retain_graph=True)
            optimizer.step()
            train_loss += loss.item()
        return train_loss / len(loader)

def test_cooc(loader, model, device, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    all_preds = []
    all_labels = []
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            emb, _, out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        pred = out.argmax(dim=1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(data.y.cpu().numpy())
        correct += int((pred == data.y).sum())
        loss = criterion(out, data.y)
        test_loss += loss.item()

    f1_macro = f1_score(all_labels, all_preds, average='macro')
    accuracy = correct / len(loader.dataset)
    return accuracy, f1_macro, test_loss / len(loader), all_preds, all_labels

def extract_doc_edges(text_tokenized, label, word_features, vocab, window_size):
    try:
        # Get unique words in the document
        unique_words = list(set(text_tokenized))
        unique_words = [word for word in unique_words if (word in vocab and word in word_features.keys())]  # Filter out OOV words

        # Create node features for this graph (only words in the document)
        #node_features = torch.stack([word_features[word] for word in unique_words])
        node_features = torch.stack([word_features[word] for word in unique_words if word in word_features])

        # Create a local word-to-index mapping for this graph
        local_word_to_index = {word: idx for idx, word in enumerate(unique_words)}

        # Calculate word frequencies in the document
        word_freq = Counter(text_tokenized)

        # Create word-word edges (co-occurrence within a window size) and calculate PMI
        word_word_edges = set()  # Use a set to avoid duplicate edges
        co_occurrence_matrix = defaultdict(int)
        total_pairs = 0

        for i in range(len(text_tokenized)):
            window = text_tokenized[i:i + window_size + 1]
            for word1, word2 in combinations(window, 2):
                if word1 in local_word_to_index and word2 in local_word_to_index:
                    word1_id = local_word_to_index[word1]
                    word2_id = local_word_to_index[word2]
                    word_word_edges.add((word1_id, word2_id))  # Add edge as a tuple to the set
                    co_occurrence_matrix[(word1, word2)] += 1
                    total_pairs += 1

        # Calculate PMI for each edge
        pmi_matrix = calculate_pmi(co_occurrence_matrix, word_freq, total_pairs)

        # Add PMI and frequency as edge attributes
        edge_attr_pmi = []
        edge_attr_freq = []
        edges = []
        for (word1, word2), pmi in pmi_matrix.items():
            word1_id = local_word_to_index[word1]
            word2_id = local_word_to_index[word2]
            edges.append((word1_id, word2_id))  # Add edge to the list
            edge_attr_pmi.append(pmi)
            edge_attr_freq.append(word_freq[word1] + word_freq[word2])  # Sum of frequencies of both words

        # Combine all edges and edge attributes
        edges = torch.tensor(edges, dtype=torch.long).t()  # Shape: [2, num_edges]
        edge_attr = torch.tensor([edge_attr_pmi, edge_attr_freq], dtype=torch.float).t()  # Shape: [num_edges, 2]

        # Debug: Check shapes
        #print(f"edges shape: {edges.shape}, edge_attr shape: {edge_attr.shape}")

        # Normalize edge attributes
        edge_attr = min_max_normalize(edge_attr)

        # Create the PyG Data object
        data = Data(x=node_features, edge_index=edges, edge_attr=edge_attr, y=label, unique_words=unique_words)
        return data
    except Exception as e:
        print('Error: %s', str(e))

def extract_doc_edges2(text_tokenized, label, word_features, word_to_index, vocab, window_size):
    try:
        # Get unique words in the document
        unique_words = list(set(text_tokenized))
        unique_words = [word for word in unique_words if (word in vocab)]  # Filter out OOV words
        #print("\n unique_words: ", len(unique_words), unique_words)

        # Create node features for this graph (only words in the document)
        #node_features = torch.stack([word_features[word_to_index[word]] for word in unique_words])

        # Create a local word-to-index mapping for this graph
        local_word_to_index = {word: idx for idx, word in enumerate(unique_words)}
        #print(local_word_to_index)

        # Create word-word edges (co-occurrence within a window size)
        word_word_edges = set()
        for i in range(len(text_tokenized)):
            window = text_tokenized[i:i + window_size + 1]
            for word1, word2 in combinations(window, 2):
                if word1 in local_word_to_index and word2 in local_word_to_index:
                    word1_id = local_word_to_index[word1]
                    word2_id = local_word_to_index[word2]
                    word_word_edges.add((word1_id, word2_id))
                    word_word_edges.add((word2_id, word1_id))

        # Combine all edges
        edges = torch.tensor(list(word_word_edges), dtype=torch.long).t()
        # Create the PyG Data object
        data = Data(x=[], edge_index=edges, y=label, unique_words=unique_words)
        return data
    except Exception as e:
        print('Error: %s', str(e))

def extract_node_features(data, word_features, word_to_index): 
    try:   
        #def extract_node_features(data, word_features):
        # Create node features for this graph (only words in the document)
        node_features = torch.stack([word_features[word_to_index[word]] for word in data.unique_words])
        #node_features = torch.stack([word_features['tokens'][word] for word in data.unique_words])

        # set node_features for documents
        data.x = node_features
        return data
    except Exception as e:
        print('Error: %s', str(e))

def get_word_embeddings(word, tokenizer, language_model, device):
    # Tokenize the word and convert to tensor
    inputs = tokenizer(word, return_tensors='pt', truncation=True, padding=True).to(device)
    # Get BERT embeddings
    with torch.no_grad():
        outputs = language_model(**inputs)
    # Use the embeddings from the last hidden state (CLS token or average pooling)
    word_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return word_embedding

def get_word_embeddings2(texts_set, texts_tokenized, tokenizer, language_model, vocab, device, reduction_layer, set_corpus='train', not_found_tokens='avg'):
    doc_words_embeddings_dict = {}

    for idx, text in enumerate(tqdm(texts_set, desc=f"Extracting {set_corpus} word embeddings")):
        doc_tokens_freq = defaultdict(int)
        doc_words_embeddings_dict[idx] = {'tokens': {}, 'not_found_tokens': []}

        try:
            encoded_text = tokenizer.encode_plus(text, return_tensors="pt", padding=True, truncation=True)
            encoded_text.to(device)

            with torch.no_grad():
                outputs_model = language_model(**encoded_text, output_hidden_states=True)
            last_hidden_state = outputs_model.hidden_states[-1]

            # Store embeddings before reduction
            token_embeddings = []
            token_list = []

            for i in range(len(last_hidden_state)):
                raw_tokens = [tokenizer.decode([token_id]) for token_id in encoded_text['input_ids'][i]]
                for token, embedding in zip(raw_tokens, last_hidden_state[i]):
                    token = str(token).strip()
                    if token not in vocab:
                        continue
                    doc_tokens_freq[token] += 1
                    token_embeddings.append(embedding.detach().cpu())  # FIX: Detach tensor
                    token_list.append(token)

            if token_embeddings:
                # Convert list to tensor
                token_embeddings_tensor = torch.stack(token_embeddings).to(device)

                # Reduce embedding size
                reduced_embeddings = reduce_dimension_linear(token_embeddings_tensor, reduction_layer, device)

                # Assign reduced embeddings back
                for token, reduced_emb in zip(token_list, reduced_embeddings):
                    if token not in doc_words_embeddings_dict[idx]['tokens']:
                        doc_words_embeddings_dict[idx]['tokens'][token] = reduced_emb
                    else:
                        doc_words_embeddings_dict[idx]['tokens'][token] = np.add.reduce([
                            doc_words_embeddings_dict[idx]['tokens'][token], reduced_emb.detach().cpu().numpy()  # FIX: Detach before numpy
                        ])

            # Normalize embeddings based on frequency
            for token, freq in doc_tokens_freq.items():
                doc_words_embeddings_dict[idx]['tokens'][token] = torch.tensor(
                    np.divide(doc_words_embeddings_dict[idx]['tokens'][token], freq).tolist(), dtype=torch.float
                )

        except Exception as e:
            print('Error:', str(e))

    # Handling missing tokens
    for idx, doc_tokens in enumerate(tqdm(texts_tokenized)): 
        try:
            merge_tokens = list(set(vocab) & (set(doc_tokens) - set(doc_words_embeddings_dict[idx]['tokens'].keys())))
            for token in merge_tokens:
                not_found_tokens_dict = {}

                if not_found_tokens == 'ones':
                    doc_words_embeddings_dict[idx]['tokens'][token] = torch.ones(128, dtype=torch.float)
                elif not_found_tokens == 'zeros':
                    doc_words_embeddings_dict[idx]['tokens'][token] = torch.zeros(128, dtype=torch.float)
                else:  # avg
                    node_tokens = tokenizer.encode_plus(token, return_tensors="pt", padding=True, truncation=True)
                    raw_tokens = [tokenizer.decode([token_id]) for token_id in node_tokens['input_ids'][0]]
                    not_found_tokens_dict[token] = raw_tokens[1:-1]
                    
                    avg_emb, emb_list = [], []
                    for token, subtokens in not_found_tokens_dict.items():
                        if token in doc_words_embeddings_dict[idx]['tokens']:
                            continue
                        for subtoken in subtokens:
                            if subtoken in doc_words_embeddings_dict[idx]['tokens']:
                                emb_list.append(doc_words_embeddings_dict[idx]['tokens'][subtoken].detach().cpu())  # FIX: Detach tensor
                    
                    if len(emb_list) == 0:
                        avg_emb = torch.zeros(128, dtype=torch.float)
                    else:
                        avg_emb = torch.mean(torch.stack(emb_list), axis=0)
                    
                    doc_words_embeddings_dict[idx]['tokens'][token] = avg_emb
        except Exception as e:
            print('Error:', str(e))

    return doc_words_embeddings_dict

def get_word_embeddings3(corpus, tokenizer, language_model, vocab, device, not_found_tokens='avg'):
    embeddings_word_dict = {}
    token_freq = defaultdict(int)

    for idx, text in enumerate(tqdm(corpus, desc="Extracting word embeddings")): # hetero
        #text = utils.text_normalize_v2(text['doc'])
        with torch.no_grad():
            encoded_text = tokenizer.encode_plus(text, return_tensors="pt", padding=True, truncation=True)
            encoded_text.to(device)
            outputs_model = language_model(**encoded_text, output_hidden_states=True)
            last_hidden_state = outputs_model.hidden_states[-1]

            for i in range(0, len(last_hidden_state)):
                raw_tokens = [tokenizer.decode([token_id]) for token_id in encoded_text['input_ids'][i]]
                for token, embedding in zip(raw_tokens, last_hidden_state[i]):
                    token = str(token).strip()
                    token_freq[token] += 1
                    current_emb = embedding.cpu().detach().numpy().tolist()

                    if token not in embeddings_word_dict.keys():
                        embeddings_word_dict[token] = current_emb
                    else:
                        embeddings_word_dict[token] = np.add.reduce([embeddings_word_dict[token], current_emb])

    for token, freq in token_freq.items():
        if freq > 1:
            embeddings_word_dict[token] = np.divide(embeddings_word_dict[token], freq).tolist()

    #print("len_embeddings_word_dict", len(embeddings_word_dict.keys()), embeddings_word_dict.keys())
    # Step 3: Create a list of embeddings in the order of the vocabulary
    cnt_found = 0
    cnt_not_found = 0
    word_embeddings = []
    not_found_tokens_dict = {}
    for word in tqdm(vocab, desc="Extracting word embeddings2"):
        if word in embeddings_word_dict:
            cnt_found+=1
            # Average embeddings for repeated words
            word_embeddings.append(embeddings_word_dict[word])
        else:
            cnt_not_found+=1
            # Handle words not in the embeddings (e.g., assign a random vector)
            if not_found_tokens == 'avg':
                node_tokens = tokenizer.encode_plus(word, return_tensors="pt", padding=True, truncation=True)
                raw_tokens = [tokenizer.decode([token_id]) for token_id in node_tokens['input_ids'][0]]
                not_found_tokens_dict[word] = raw_tokens[1:-1]
                avg_emb, emb_list = [], []
                for token, subtokens in not_found_tokens_dict.items():
                    if token in embeddings_word_dict:
                        continue
                    for subtoken in subtokens:
                        if subtoken in embeddings_word_dict:
                            emb_list.append(embeddings_word_dict[subtoken])
                if len(emb_list) == 0:
                    avg_emb = np.ones(language_model.config.hidden_size)
                else:
                    avg_emb = np.mean(emb_list, axis=0).flatten().tolist()
                word_embeddings.append(avg_emb)
                embeddings_word_dict[token] = avg_emb
            if not_found_tokens == 'ones':
                word_embeddings.append(np.ones(language_model.config.hidden_size))
            if not_found_tokens == 'zeros':
                word_embeddings.append(np.zeros(language_model.config.hidden_size))
            else:
                ... # remove¿?

    print("emb_words cnt_found: ", cnt_found)
    print("emb_words cnt_not_found: ", cnt_not_found)
    print("not_found_tokens_dict: ", len(not_found_tokens_dict))
    return word_embeddings

def normalize_text(texts, tokenize_pattern, special_chars=False, stop_words=False, set='train'):
    texts_norm = []
    tokenized_corpus = []
    for text in tqdm(texts, desc=f"normalizing {set} corpus"):
        text_norm = test_utils.text_normalize(text, special_chars, stop_words) 
        texts_norm.append(text_norm)
        text_doc_tokens = re.findall(tokenize_pattern, text_norm)
        tokenized_corpus.append(text_doc_tokens)
    return texts_norm, tokenized_corpus

def create_vocab(texts, set='all', min_df=1, max_df=0.9, max_features=5000):
    # Create a vocabulary
    #vectorizer = CountVectorizer()
    vectorizer = CountVectorizer(min_df=min_df, max_df=max_df, max_features=max_features)
    vectorizer.fit_transform(texts)
    vocab = vectorizer.get_feature_names_out()
    # Create a word-to-index dictionary for fast lookups
    word_to_index = {word: idx for idx, word in enumerate(vocab)}
    print(f'vocab {set} set: ', len(vocab))
    return vocab, word_to_index

def calculate_pmi(co_occurrence_matrix, word_freq, total_pairs):
    pmi_matrix = {}
    for (word1, word2), count in co_occurrence_matrix.items():
        pmi = log((count * total_pairs) / (word_freq[word1] * word_freq[word2]))
        pmi_matrix[(word1, word2)] = pmi
    return pmi_matrix

def min_max_normalize(tensor):
    min_vals = tensor.min(dim=0).values  # Min values for each feature
    max_vals = tensor.max(dim=0).values  # Max values for each feature
    edge_attr_normalized = (tensor - min_vals) / (max_vals - min_vals + 1e-8)  # Add small epsilon to avoid division by zero
    return edge_attr_normalized

def calculate_graph_metrics(edges, node_features):
    try:
        # Create a NetworkX graph from edge_index
        G = nx.Graph()
        G.add_edges_from(edges.t().tolist())

        # Get the number of nodes
        num_nodes = node_features.size(0)

        # Calculate node-level metrics
        degree_centrality = torch.tensor(list(nx.degree_centrality(G).values()), dtype=torch.float).view(-1, 1)
        # Calculate graph metrics using approximation algorithms
        betweenness_centrality = min_max_normalize(test_utils.approximate_betweenness_centrality(edges, num_nodes=num_nodes, k=100))
        eigenvector_centrality = min_max_normalize(test_utils.approximate_eigenvector_centrality(edges, num_nodes=num_nodes, max_iter=50))
        pagerank = min_max_normalize(test_utils.approximate_pagerank(edges, num_nodes=num_nodes, max_iter=50))
        clustering_coefficient = min_max_normalize(test_utils.approximate_clustering_coefficient(edges, num_nodes=num_nodes, k=100))
        #closeness_centrality = min_max_normalize(test_utils.approximate_closeness_centrality(edges, num_nodes=node_features.size(0), k=100))

        # Verify that all metrics have the same number of nodes
        if not all(metric.size(0) == num_nodes for metric in [degree_centrality, betweenness_centrality, eigenvector_centrality, pagerank, clustering_coefficient]):
            raise ValueError("Inconsistent number of nodes in graph metrics")

        # Combine all metrics into a single tensor
        graph_metrics = torch.cat([
            degree_centrality,
            betweenness_centrality,
            eigenvector_centrality,
            pagerank,
            clustering_coefficient,
            #closeness_centrality,
        ], dim=-1)

        return graph_metrics
    except Exception as e:
        # Log the error and return None
        logger.error(f"Error calculating graph metrics: {str(e)}")
        return None
    
def generate_random_embedding(dim):
    """Generate a random embedding vector for a word."""
    return torch.tensor([random.uniform(-1, 1) for _ in range(dim)], dtype=torch.float)

def reduce_dimension_linear(x, reduction_layer, device, batch_size=128):
    reduced_x_list = []
    for i in range(0, x.shape[0], batch_size):
        batch = x[i:i + batch_size]  
        batch = batch.to(device)
        reduced_batch = reduction_layer(batch).detach().cpu()  # FIX: Detach before moving to CPU
        reduced_x_list.append(reduced_batch)

    return torch.cat(reduced_x_list, dim=0)



def main(build_graph, dataset_name, cut_off_dataset,  experiment_data, experiments_path_file, cuda_num, output_dim):    

    config = {
        'build_graph': build_graph,
        'dataset_name': dataset_name, # autext23, semeval24, coling24, autext23_s2, semeval24_s2
        'cut_off_dataset': cut_off_dataset, # train-val-test
        'cuda_num': cuda_num,
        
        'window_size': 10,
        'graph_direction': 'undirected', # undirected | directed 
        'special_chars': False,
        'stop_words': False,
        'min_df': 2, # 1->autext | 5->semeval | 5-coling
        'max_df': 0.9,
        'max_features': None, # None -> all | 5000, 10000
        'embed_reduction': True, # speacially for llm to reduce emb_size from 768 -> 128 
        'not_found_tokens': 'avg', # avg, remove, zeros, ones
        'add_edge_attr': False,
        'add_graph_metric': False,

        "gnn_type": 'TransformerConv', # GCNConv, GINConv, GATConv, TransformerConv
        "dropout": 0.8,
        "patience": 5, # 5-autext23 | 10-semeval | 10-coling
        "learnin_rate": 2e-05, # autext23_s2 -> llm: 0.0002 | autext23 -> llm: 0.00001 | semeval -> llm: 0.000005  | coling -> llm: 0.0001 
        "batch_size": 32 * 1,
        
        "hidden_dim": 100, # 300 autext_s2, 100 others
        "dense_hidden_dim": 32, # 64-autext23 | 32-semeval | 64-coling
        "num_layers": 1,
        "heads": 1,
        "output_dim": 2, # 2-bin | 6-multi 
        "weight_decay": 0.00001, 
        'input_dim': 768,
        'epochs': 200,
        "llm_name": 'microsoft/deberta-v3-base'
    }
    ## google-bert/bert-base-uncased
    ## FacebookAI/roberta-base
    ## microsoft/deberta-v3-base
    
    file_name_data = f"cooc_data_{config['dataset_name']}_{config['cut_off_dataset']}perc"
    #output_dir = f'{utils.OUTPUT_DIR_PATH}test_graph/{config["llm_name"].split("/")[1]}/'
    #output_dir = f'{utils.OUTPUT_DIR_PATH}test_graph/'
    output_dir = f'{test_utils.EXTERNAL_DISK_PATH}cooc_graph/{config["llm_name"].split("/")[1]}/'

    device = torch.device(f"cuda:{config['cuda_num']}" if torch.cuda.is_available() else "cpu")
    pprint.pprint(config)
    
    if config['build_graph'] == True:
        start = time.time()
        # ****************************** PROCESS AUTEXT DATASET && CUTOF
        # Load and preprocess dataset
        train_text_set, val_text_set, test_text_set = test_utils.read_dataset(config['dataset_name'])

        # Cut off datasets
        cut_off_train = int(config['cut_off_dataset'].split('-')[0])
        cut_off_val = int(config['cut_off_dataset'].split('-')[1])
        cut_off_test = int(config['cut_off_dataset'].split('-')[2])

        train_set = train_text_set[:int(len(train_text_set) * (cut_off_train / 100))][:]
        val_set = val_text_set[:int(len(val_text_set) * (cut_off_val / 100))][:]
        test_set = test_text_set[:int(len(test_text_set) * (cut_off_test / 100))][:]

        print("distro_train_val_test: ", len(train_set), len(val_set), len(test_set))
        print("label_distro_train_val_test: ", train_set.value_counts('label'), val_set.value_counts('label'), test_set.value_counts('label'))

        # Example text data (split into train, validation, and test sets)
        train_texts = list(train_set['text'])[:]
        val_texts = list(val_set['text'])[:]
        test_texts = list(test_set['text'])[:]

        # Labels (binary classification: 0 or 1)
        train_labels = list(train_set['label'])[:]
        val_labels = list(val_set['label'])[:]
        test_labels = list(test_set['label'])[:]

        # Normalize and Tokenize the corpus 
        tokenize_pattern = "[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+"
        train_texts_norm, train_texts_tokenized = normalize_text(train_texts, tokenize_pattern, special_chars=config['special_chars'], stop_words=config['stop_words'], set='train')
        val_texts_norm, val_texts_tokenized = normalize_text(val_texts, tokenize_pattern, special_chars=config['special_chars'], stop_words=config['stop_words'], set='val')
        test_texts_norm, test_texts_tokenized = normalize_text(test_texts, tokenize_pattern, special_chars=config['special_chars'], stop_words=config['stop_words'], set='test')

        # create a vocabulary
        all_texts_norm = train_texts_norm + val_texts_norm + test_texts_norm
        vocab, word_to_index = create_vocab(all_texts_norm, min_df=config['min_df'], max_df=config['max_df'], max_features=config['max_features'])
        print("not_found_tokens approach: ", config['not_found_tokens'])
        # LLM        
        # Extract embedding from language model
        tokenizer = AutoTokenizer.from_pretrained(config['llm_name'], model_max_length=512)
        language_model = AutoModel.from_pretrained(config['llm_name'], output_hidden_states=True).to(device)

        # *** Generate embeddings method 2  - LLM 
        embedding_reduction = nn.Linear(768, 128).to(device)
        train_words_emb = get_word_embeddings2(train_texts_norm, train_texts_tokenized, tokenizer, language_model, vocab, device, embedding_reduction, set_corpus='train', not_found_tokens=config['not_found_tokens'])
        val_words_emb = get_word_embeddings2(val_texts_norm, val_texts_tokenized, tokenizer, language_model, vocab, device, embedding_reduction, set_corpus='val', not_found_tokens=config['not_found_tokens'])
        test_words_emb = get_word_embeddings2(test_texts_norm, test_texts_tokenized, tokenizer, language_model, vocab, device, embedding_reduction, set_corpus='test', not_found_tokens=config['not_found_tokens'])
        '''
        # Train WORD2VEC model on the corpus (or load a pre-trained model)
        w2v_model = Word2Vec(sentences=train_texts_tokenized + val_texts_tokenized + test_texts_tokenized, vector_size=config['input_dim'], window=5, min_count=1, workers=4)
        word_features = {word: torch.tensor(w2v_model.wv[word], dtype=torch.float) for word in w2v_model.wv.index_to_key}
        train_words_emb = {idx: {'tokens': {word: word_features[word] for word in text if word in word_features}} for idx, text in enumerate(train_texts_tokenized)}
        val_words_emb = {idx: {'tokens': {word: word_features[word] for word in text if word in word_features}} for idx, text in enumerate(val_texts_tokenized)}
        test_words_emb = {idx: {'tokens': {word: word_features[word] for word in text if word in word_features}} for idx, text in enumerate(test_texts_tokenized)}
        '''
        '''
        # Use RANDOM embeddings for train, val, and test data
        word_features = {word: generate_random_embedding(200) for word in vocab}  # Random embeddings for all words
        train_words_emb = {idx: {'tokens': {word: word_features[word] for word in text if word in word_features}} for idx, text in enumerate(train_texts_tokenized)}
        val_words_emb = {idx: {'tokens': {word: word_features[word] for word in text if word in word_features}} for idx, text in enumerate(val_texts_tokenized)}
        test_words_emb = {idx: {'tokens': {word: word_features[word] for word in text if word in word_features}} for idx, text in enumerate(test_texts_tokenized)}
        '''

        # extract doc edges
        train_data, val_data, test_data = [], [], []
        for idx, (text_tokenized, label) in enumerate(zip(tqdm(train_texts_tokenized, desc="Extracting doc train edges"), train_labels)):
            doc_edges = extract_doc_edges(text_tokenized, label, train_words_emb[idx]['tokens'], vocab, config['window_size'])
            if doc_edges:
                train_data.append(doc_edges)
        for idx, (text_tokenized, label) in enumerate(zip(tqdm(val_texts_tokenized, desc="Extracting doc val edges"), val_labels)):
            doc_edges = extract_doc_edges(text_tokenized, label, val_words_emb[idx]['tokens'], vocab, config['window_size'])
            if doc_edges:
                val_data.append(doc_edges)
        for idx, (text_tokenized, label) in enumerate(zip(tqdm(test_texts_tokenized, desc="Extracting doc test edges"), test_labels)):
            doc_edges = extract_doc_edges(text_tokenized, label, test_words_emb[idx]['tokens'], vocab, config['window_size'])
            if doc_edges:
                test_data.append(doc_edges)

        if config['add_graph_metric']:
            for data in tqdm(train_data + val_data + test_data, desc="Extracting graph metric data"):
                graph_metrics = calculate_graph_metrics(data.edge_index, data.x)
                if graph_metrics is not None:
                    data.graph_metrics = graph_metrics  # Add graph metrics to the Data object
                else:
                    # Assign default metrics (e.g., zeros)
                    num_nodes = data.x.size(0)
                    num_metrics = 5  # Number of metrics (degree, betweenness, eigenvector, pagerank, clustering)
                    data.graph_metrics = torch.zeros((num_nodes, num_metrics), dtype=torch.float)
                    logger.warning(f"Using default metrics for graph: {data}")


        # Apply dimensionality reduction to train, val, and test data
        #new_feat_dim = 128
        #if config['embed_reduction']:    
        #    embedding_reduction = nn.Linear(768, new_feat_dim).to(device)
        #    train_data = reduce_dimension_linear(train_data, embedding_reduction, device)
        #    val_data = reduce_dimension_linear(val_data, embedding_reduction, device)
        #    test_data = reduce_dimension_linear(test_data, embedding_reduction, device)


        # *** Save data
        all_data = [train_data, val_data, test_data]
        #word_features = [train_words_emb, val_words_emb, test_words_emb]
        data_obj = {
            #"word_features": word_features, # word_emb method 3
            "vocab": vocab,
            "all_data": all_data,
            "word_to_index": word_to_index,
            "time_to_build_graph": time.time() - start,
            "config": config,
        }
        utils.save_data(data_obj, file_name_data, path=output_dir, format_file='.pkl', compress=False)
    else:
        data_obj = utils.load_data(file_name_data, path=output_dir, format_file='.pkl', compress=False)
        train_data = data_obj['all_data'][0]
        val_data = data_obj['all_data'][1]
        test_data = data_obj['all_data'][2]
        word_to_index = data_obj['word_to_index']

    print(train_data[0])
    init_metrics = {'_epoch_stop': 0,'_train_loss': 0,'_val_loss': 0,'_test_loss': 0,'_val_acc': 0,'_test_acc': 0,'_val_f1_macro': 0,'_test_f1_macro': 0,'_exec_time': 0,}

    for index, row in experiment_data.iterrows(): # batch experiments
        metrics = init_metrics.copy() 
        try:
            print("******************************************* Running experiment with ID: ", row['id'])
            start = time.time()
            print('device: ', device)

            if row['_done'] == True or row['_done'] == 'True':
                print('Experiment already DONE')
                continue
            print("exp_config: ", row)
            
            if row['graph_direction'] == 'undirected':
                for data in train_data + val_data + test_data:
                    # Add reverse edges
                    data.edge_index = torch.cat([data.edge_index, data.edge_index.flip(0)], dim=1)
                    # Add reverse edge attributes
                    if row['gnn_edge_attr'] or row['gnn_edge_attr'] == 'True': 
                        data.edge_attr = torch.cat([data.edge_attr, data.edge_attr], dim=0)
                    #else:
                    #    del data.edge_attr  # Remove edge attributes if not needed

            print(train_data[0])
            # Create DataLoader for train, validation, and test partitions
            train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
            val_loader = DataLoader(val_data, batch_size=config['batch_size'], shuffle=False)
            test_loader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=False)


            # Initialize the model
            input_dim = train_data[0].x.shape[1]
            test_utils.set_random_seed(42)
            model = GNN(
                input_dim, 
                hidden_dim = int(row['gnn_nhid']), 
                dense_hidden_dim = int(row['gnn_dense_nhid']), 
                output_dim = output_dim, 
                dropout = row['gnn_dropout'], 
                num_layers = row['gnn_layers_convs'], 
                gnn_type = row['gnn_type'],
                heads = int(row['gnn_heads']), 
                task = 'graph'
            )
            model = model.to(device)
            print(model)

            # Training loop
            optimizer = torch.optim.Adam(model.parameters(), lr=row['gnn_learning_rate'], weight_decay=config['weight_decay'])
            criterion = torch.nn.CrossEntropyLoss()
            early_stopper = EarlyStopper(patience=row['gnn_patience'], min_delta=0)

            logger.info("Init GNN training!")
            best_test_acc = 0
            best_test_f1score = 0
            epoch_best_test_acc = 0
            for epoch in range(row['epoch_num']):
                loss_train = train_cooc(model, train_loader, device, optimizer, criterion)
                val_acc, val_f1_macro, val_loss, preds_val, labels_val = test_cooc(val_loader, model, device, criterion)
                test_acc, test_f1_macro, test_loss, preds_test, labels_test = test_cooc(test_loader, model, device, criterion)

                metrics['_train_loss'] = loss_train
                metrics['_val_loss'] = val_loss
                metrics['_val_acc'] = val_acc
                metrics['_val_f1_macro'] = val_f1_macro
                
                print(f'Epoch {epoch + 1}, Loss Train: {loss_train:.4f}, Loss Val: {val_loss:.4f}, Loss Test: {test_loss:.4f}, '
                    f'Val Acc: {val_acc:.4f}, Val F1Score: {val_f1_macro:.4f}, Test Acc: {test_acc:.4f}, Test F1Score: {test_f1_macro:.4f}')
                #print("preds_test: ", sorted(Counter(preds_test).items()))
                #print("label_test: ", sorted(Counter(labels_test).items()))
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    best_test_f1score = test_f1_macro
                    epoch_best_test_acc = epoch

                if early_stopper.early_stop(val_loss):
                    print('Early stopping fue to not improvement!')
                    metrics['_epoch_stop'] = epoch
                    break

                # Free unused memory
                del preds_val, labels_val, preds_test, labels_test
                torch.cuda.empty_cache()
                gc.collect()

            logger.info("Done GNN training!")

            # Final evaluation on the test set
            test_acc, test_f1_macro, test_loss, _, _ = test_cooc(test_loader, model, device, criterion)
            print(f'Test Accuracy: {test_acc:.4f}')
            print(f'Test F1Score: {test_f1_macro:.4f}')
            metrics['_test_loss'] = test_loss
            metrics['_test_acc'] = test_acc
            metrics['_test_f1_macro'] = test_f1_macro
            metrics['_best_test_acc'] = best_test_acc
            metrics['_best_test_f1score'] = best_test_f1score
            metrics['_epoch_best_test_acc'] = epoch_best_test_acc
            metrics['_done'] = True

            # Free GPU memory
            del model, train_loader, val_loader, test_loader
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as err:
            metrics['_done'] = 'Error'
            metrics['_desc'] = str(err)
            print("traceback: ", str(traceback.format_exc()))
            print(f"An error ocurred running experiment {row['id']}: ", err)
        finally:
            metrics['_exec_time'] = time.time() - start
            experiments_data = utils.read_csv(f'{experiments_path_file}')
            for key, value in metrics.items():
                experiments_data.loc[index, key] = value
            utils.save_csv(experiments_data, file_path=f'{experiments_path_file}')
            
            time.sleep(10)


if __name__ == '__main__':
    print('*** INIT autext23')
    dataset_name='coling24' # autext23, semeval24, coling24, autext23_s2, semeval24_s2
    build_graph = False
    cut_off_dataset = '10-10-10'
    cuda_num = 1
    output_dim = 2
    # test_experiments
    # experiments_autext23_cooc_20250211
    # experiments_coling25_cooc_20250211
    # experiments_semeval24_cooc_20250211
    file_name = 'experiments_coling25_cooc_20250211'
    experiments_path_file = f'{utils.OUTPUT_DIR_PATH}test_graph/batch_files/final_acl/{file_name}.csv'
    experiment_data = utils.read_csv(f'{experiments_path_file}')
    print(experiment_data.info())
    
    main(build_graph, dataset_name, cut_off_dataset, experiment_data, experiments_path_file, cuda_num, output_dim)

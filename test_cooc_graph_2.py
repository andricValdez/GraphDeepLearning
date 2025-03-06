import pprint
import torch
import numpy as np
import seaborn as sns
from torch_geometric.data import Data, DataLoader
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
from sklearn.decomposition import PCA
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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers import logging as transform_loggin
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
        self.norm1 = nn.LayerNorm(hidden_dim * heads) # LayerNorm, BatchNorm1d
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

        x = global_mean_pool(x, batch)

        logits = self.post_mp(x)
        probs  = F.log_softmax(logits, dim=1)
        # self.sigmoid(x)
        return emb, logits, probs

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

def get_influential_words(model, data_loader, device, vocab, top_k=20):
    model.eval()
    word_scores = {0: defaultdict(list), 1: defaultdict(list)}  # Separate storage for human (0) & machine (1)
    class_counts = {0: 0, 1: 0}  # Track class occurrences

    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            embeddings, logits, probs = model(
                data.x, data.edge_index, 
                data.edge_attr if hasattr(data, 'edge_attr') else None, 
                data.batch
            )

            # Get predicted class (human=0, machine=1)
            graph_preds = probs.argmax(dim=1).cpu().numpy()

            for graph_idx, graph_class in enumerate(graph_preds):
                class_counts[graph_class] += 1  # Track occurrences per class

                # Get words belonging to this graph
                mask = (data.batch.cpu().numpy() == graph_idx)
                graph_words = [vocab[i] for i, included in enumerate(mask) if included]

                # Compute word influence based on L2 norm
                word_importance = torch.norm(embeddings[mask], dim=1).cpu().numpy()

                # Store per-class influence scores
                for word, score in zip(graph_words, word_importance):
                    word_scores[graph_class][word].append(score)

    # Average importance scores PER CLASS
    avg_word_scores = {
        class_label: {word: np.mean(scores) for word, scores in word_scores[class_label].items()}
        for class_label in class_counts.keys()
    }

    # Get Top-K Words for Each Class
    influential_words = {
        class_label: sorted(avg_word_scores[class_label].items(), key=lambda x: x[1], reverse=True)[:top_k]
        for class_label in class_counts.keys()
    }

    return influential_words



def main():    
    config = {
        'build_graph': False,
        'dataset_name': 'semeval24', # autext23, semeval24, coling24, autext23_s2, semeval24_s2
        'cut_off_dataset': '5-10-10', # train-val-test
        "nfi": 'llm', # llm, w2v, random
        'cuda_num': 1,

        'window_size': 10,
        'graph_direction': 'undirected', # undirected | directed 
        'special_chars': False,
        'stop_words': False,
        'min_df': 2, # 1->autext | 5->semeval | 5-coling
        'max_df': 0.9,
        'max_features': None, # None -> all | 5000, 10000
        'not_found_tokens': 'avg', # avg, remove, zeros, ones
        'add_edge_attr': True,
        'add_graph_metric': False,
        'embed_reduction': True, # speacially for llm to reduce emb_size from 768 -> 128 

        "gnn_type": 'TransformerConv', # GCNConv, GINConv, GATConv, TransformerConv
        "dropout": 0.5,
        "patience": 10, # 5-autext23 | 10-semeval | 10-coling
        "learnin_rate": 0.000005, # autext23_s2 -> llm: 0.0002 | autext23 -> llm: 0.00001 | semeval -> llm: 0.000005  | coling -> llm: 0.0001 
        "batch_size": 32 * 1,
        "hidden_dim": 100, # 300 autext_s2, 100 others
        "dense_hidden_dim": 32, # 64-autext23 | 32-semeval | 64-coling
        "num_layers": 1,
        "heads": 1,
        "output_dim": 2, # 2-bin | 6-multi 
        "weight_decay": 0.0001, 
        'input_dim': 128, 
        'epochs': 100,
        "llm_name": 'microsoft/deberta-v3-base',
    }
    ## google-bert/bert-base-uncased
    ## FacebookAI/roberta-base
    ## microsoft/deberta-v3-base
    
    file_name_data = f"cooc_data_{config['dataset_name']}_{config['cut_off_dataset']}perc"
    #output_dir = f'{utils.OUTPUT_DIR_PATH}test_graph/{config["llm_name"].split("/")[1]}/'
    #output_dir = f'{utils.OUTPUT_DIR_PATH}test_graph/'
    output_dir = f'{test_utils.EXTERNAL_DISK_PATH}cooc_graph'

    nfi_dir = config["llm_name"].split("/")[1] # nfi -> llm
    if config['nfi'] == 'w2v':
        nfi_dir = 'w2v'
    if config['nfi'] == 'random':
        nfi_dir = 'random'

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
        if config['nfi'] == 'llm':
            embedding_reduction = nn.Linear(768, 128).to(device)
            train_words_emb = get_word_embeddings2(train_texts_norm, train_texts_tokenized, tokenizer, language_model, vocab, device, embedding_reduction, set_corpus='train', not_found_tokens=config['not_found_tokens'])
            val_words_emb = get_word_embeddings2(val_texts_norm, val_texts_tokenized, tokenizer, language_model, vocab, device, embedding_reduction, set_corpus='val', not_found_tokens=config['not_found_tokens'])
            test_words_emb = get_word_embeddings2(test_texts_norm, test_texts_tokenized, tokenizer, language_model, vocab, device, embedding_reduction, set_corpus='test', not_found_tokens=config['not_found_tokens'])
            
        # Train WORD2VEC model on the corpus (or load a pre-trained model)
        if config['nfi'] == 'w2v':
            w2v_model = Word2Vec(sentences=train_texts_tokenized + val_texts_tokenized + test_texts_tokenized, vector_size=config['input_dim'], window=5, min_count=1, workers=4)
            word_features = {word: torch.tensor(w2v_model.wv[word], dtype=torch.float) for word in w2v_model.wv.index_to_key}
            train_words_emb = {idx: {'tokens': {word: word_features[word] for word in text if word in word_features}} for idx, text in enumerate(train_texts_tokenized)}
            val_words_emb = {idx: {'tokens': {word: word_features[word] for word in text if word in word_features}} for idx, text in enumerate(val_texts_tokenized)}
            test_words_emb = {idx: {'tokens': {word: word_features[word] for word in text if word in word_features}} for idx, text in enumerate(test_texts_tokenized)}
        
        
        # Use RANDOM embeddings for train, val, and test data
        if config['nfi'] == 'random':
            word_features = {word: generate_random_embedding(config['input_dim']) for word in vocab}  # Random embeddings for all words
            train_words_emb = {idx: {'tokens': {word: word_features[word] for word in text if word in word_features}} for idx, text in enumerate(train_texts_tokenized)}
            val_words_emb = {idx: {'tokens': {word: word_features[word] for word in text if word in word_features}} for idx, text in enumerate(val_texts_tokenized)}
            test_words_emb = {idx: {'tokens': {word: word_features[word] for word in text if word in word_features}} for idx, text in enumerate(test_texts_tokenized)}
            

        # extract doc edges
        train_data, val_data, test_data = [], [], []
        for idx, (text_tokenized, label) in enumerate(zip(tqdm(train_texts_tokenized, desc="Extracting doc train edges"), train_labels)):
            doc_edges = extract_doc_edges(text_tokenized, label, train_words_emb[idx]['tokens'], vocab, config['window_size'])
            if doc_edges:
                if config['dataset_name'] == 'autext23':
                    doc_edges.metadata = {"label": train_set.iloc[idx]["label"],"domain": train_set.iloc[idx]["domain"],"model": train_set.iloc[idx]["model"]}
                if config['dataset_name'] in ['semeval24', 'coling24']: 
                    doc_edges.metadata = {"label": train_set.iloc[idx]["label"],"domain": train_set.iloc[idx]["source"],"model": train_set.iloc[idx]["model"]}
                train_data.append(doc_edges)
        for idx, (text_tokenized, label) in enumerate(zip(tqdm(val_texts_tokenized, desc="Extracting doc val edges"), val_labels)):
            doc_edges = extract_doc_edges(text_tokenized, label, val_words_emb[idx]['tokens'], vocab, config['window_size'])
            if doc_edges:
                if config['dataset_name'] == 'autext23':
                    doc_edges.metadata = {"label": val_set.iloc[idx]["label"],"domain": val_set.iloc[idx]["domain"],"model": val_set.iloc[idx]["model"]}
                if config['dataset_name'] in ['semeval24', 'coling24']: 
                    doc_edges.metadata = {"label": val_set.iloc[idx]["label"],"domain": val_set.iloc[idx]["source"],"model": val_set.iloc[idx]["model"]}
                val_data.append(doc_edges)
        for idx, (text_tokenized, label) in enumerate(zip(tqdm(test_texts_tokenized, desc="Extracting doc test edges"), test_labels)):
            doc_edges = extract_doc_edges(text_tokenized, label, test_words_emb[idx]['tokens'], vocab, config['window_size'])
            if doc_edges:
                if config['dataset_name'] == 'autext23':
                    doc_edges.metadata = {"label": test_set.iloc[idx]["label"],"domain": test_set.iloc[idx]["domain"],"model": test_set.iloc[idx]["model"]}
                if config['dataset_name'] in ['coling24']: # semeval24 dont have this info in test_Set
                    doc_edges.metadata = {"label": test_set.iloc[idx]["label"],"domain": test_set.iloc[idx]["source"],"model": test_set.iloc[idx]["model"]}
                if config['dataset_name'] in ['semeval24']: # semeval24 dont have this info in test_Set
                    doc_edges.metadata = {"label": test_set.iloc[idx]["label"]}
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
        utils.save_data(data_obj, file_name_data, path=f'{output_dir}/{nfi_dir}/', format_file='.pkl', compress=False)
    else:
        data_obj = utils.load_data(file_name_data, path=f'{output_dir}/{nfi_dir}/', format_file='.pkl', compress=False)
        train_data = data_obj['all_data'][0]
        val_data = data_obj['all_data'][1]
        test_data = data_obj['all_data'][2]
        vocab = data_obj['vocab']
        word_to_index = data_obj['word_to_index']

    # ********** cutoff saved data - TMP
    '''
    cutoff = 50
    print("len_train_val_test: ", len(train_data), len(val_data), len(test_data))
    print("train_len: ", Counter([data.y for data in train_data]))
    print("val_len: ", Counter([data.y for data in val_data]))
    print("test_len: ", Counter([data.y for data in test_data]))
    train_data = train_data[:int(len(train_data) * (cutoff / 100))]
    val_data = val_data[:int(len(val_data) * (cutoff / 100))]
    test_data = test_data[:int(len(test_data) * (cutoff / 100))]
    print("len_train_val_test: ", len(train_data), len(val_data), len(test_data))
    print("train_len: ", Counter([data.y for data in train_data]))
    print("val_len: ", Counter([data.y for data in val_data]))
    print("test_len: ", Counter([data.y for data in test_data]))
    '''
    # ********** cutoff saved data - TMP

    # ********** Add metadata - TMP
    '''train_text_set, val_text_set, test_text_set = test_utils.read_dataset(config['dataset_name'])
    # Cut off datasets
    cut_off_train = int(config['cut_off_dataset'].split('-')[0])
    cut_off_val = int(config['cut_off_dataset'].split('-')[1])
    cut_off_test = int(config['cut_off_dataset'].split('-')[2])

    train_set = train_text_set[:int(len(train_text_set) * (cut_off_train / 100))][:]
    val_set = val_text_set[:int(len(val_text_set) * (cut_off_val / 100))][:]
    test_set = test_text_set[:int(len(test_text_set) * (cut_off_test / 100))][:]
    
    for idx, data in enumerate(train_data):
        if config['dataset_name'] == 'autext23':
            data.metadata = {"label": train_set.iloc[idx]["label"],"domain": train_set.iloc[idx]["domain"],"model": train_set.iloc[idx]["model"]}
    for idx, data in enumerate(val_data):
        if config['dataset_name'] == 'autext23':
            data.metadata = {"label": val_set.iloc[idx]["label"],"domain": val_set.iloc[idx]["domain"],"model": val_set.iloc[idx]["model"]}
    for idx, data in enumerate(test_data):
        if config['dataset_name'] == 'autext23':
            data.metadata = {"label": test_set.iloc[idx]["label"],"domain": test_set.iloc[idx]["domain"],"model": test_set.iloc[idx]["model"]}
    
    print(train_data[0])
    data_obj['all_data'] = [train_data, val_data, test_data]
    utils.save_data(data_obj, file_name_data, path=f'{output_dir}/{nfi_dir}/', format_file='.pkl', compress=False)
    return'''
    # ********** Add metadata - TMP

    # Add reverse edges edge attributes and graph_metrics
    for data in train_data + val_data + test_data:
        if config['add_graph_metric']:
            data.x = torch.cat([data.x, data.graph_metrics], dim=-1)
        else:
            del data.graph_metrics
            
    if config['graph_direction'] == 'undirected':
        for data in train_data + val_data + test_data:
            # Add reverse edges
            data.edge_index = torch.cat([data.edge_index, data.edge_index.flip(0)], dim=1)
            # Add reverse edge attributes
            if config['add_edge_attr']:
                data.edge_attr = torch.cat([data.edge_attr, data.edge_attr], dim=0)
            else:
                del data.edge_attr  # Remove edge attributes if not needed

    # Debug: Check shapes of edge_index and edge_attr
    #print("Validation data examples:")
    #for i, data in enumerate(val_data[:5]):  # Print first 10 validation examples
    #    print(f"Data {i}: x={data.x.shape}, edge_index={data.edge_index.shape}, edge_attr={data.edge_attr.shape if hasattr(data, 'edge_attr') else 'None'}, y={data.y}, unique_words={len(data.unique_words)}")

    # Create DataLoader for train, validation, and test partitions
    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=False)

    print(train_data[0])
    for batch in train_loader:
        print(batch)
        break

    # Initialize the model
    input_dim = train_data[0].x.shape[1]
    test_utils.set_random_seed(42)
    model = GNN(input_dim, 
                config['hidden_dim'], 
                config['dense_hidden_dim'], 
                config['output_dim'], 
                config['dropout'], 
                config['num_layers'], 
                config['add_edge_attr'], 
                gnn_type=config['gnn_type'], 
                heads=config['heads'], 
                task='graph')
    model = model.to(device)
    print(model)

    # Training loop (example)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learnin_rate'], weight_decay=config['weight_decay'])
    criterion = torch.nn.CrossEntropyLoss()
    early_stopper = EarlyStopper(patience=config['patience'], min_delta=0)

    logger.info("Init GNN training!")
    best_test_acc = 0
    best_test_f1score = 0
    epoch_test_acc = 0
    
    for epoch in range(1, config['epochs']):
        loss_train = train_cooc(model, train_loader, device, optimizer, criterion)
        val_acc, val_f1_macro, val_loss, preds_val, labels_val = test_cooc(val_loader, model, device, criterion)

        if epoch % 1 == 0:
            test_acc, test_f1_macro, test_loss, preds_test, labels_test = test_cooc(test_loader, model, device, criterion)
            print(f'Ep {epoch + 1}, Loss Val: {val_loss:.4f}, Loss Test: {test_loss:.4f}, '
            f'Val Acc: {val_acc:.4f}, Val F1s: {val_f1_macro:.4f}, Test Acc: {test_acc:.4f}, Test F1s: {test_f1_macro:.4f}')
        else:
            print(f'Ep {epoch + 1}, Loss Val: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1s: {val_f1_macro:.4f}')

        #print("preds_test: ", sorted(Counter(preds_test).items()))
        #print("label_test: ", sorted(Counter(labels_test).items()))
        
        if early_stopper.early_stop(val_loss):
            print('Early stopping fue to not improvement!')
            break
    logger.info("Done GNN training!")

    # Final evaluation on the test set
    test_acc, test_f1_macro, test_loss, preds_test, _ = test_cooc(test_loader, model, device, criterion)
    print(f'Test Accuracy: {test_acc:.4f}')
    print(f'Test F1Score: {test_f1_macro:.4f}')
    print(f'Test Loss: {test_loss:.4f}')
    print("preds_test: ", sorted(Counter(preds_test).items()))
    
    cm = confusion_matrix(preds_test, labels_test)
    print(cm)

    model_save_path = f"{output_dir}/models/gnn_model_{config['nfi']}_{file_name_data}.pt"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved at: {model_save_path}")

    # Compute influential words
    influential_words = get_influential_words(model, test_loader, device, vocab, top_k=10)
    # Print influential words per class
    print("Most Influential Words per Class:")
    for class_label, words in influential_words.items():
        print(f"\nClass {class_label} ({'Human' if class_label==0 else 'Machine'}):")
        for word, score in words:
            print(f"{word}: {score:.4f}")









def plot_confusion_matrix(y_true, y_pred, class_names, dataset_name, normalize=False, cmap="Blues"):
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap=cmap, 
                xticklabels=class_names, yticklabels=class_names, linewidths=0.5, square=True,
                annot_kws={"size": 18})  # Increase font size of the annotations)
    # Increase the font size of class labels
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("Predicted Labels", fontsize=18)
    plt.ylabel("True Labels", fontsize=18)
    #plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    plt.title("Confusion Matrix - " + dataset_name, fontsize=18)
    plt.show()
    plt.savefig(f'{test_utils.OUTPUT_DIR_PATH}test_graph/cm_{dataset_name}_cooc_graph.png', dpi=300, bbox_inches="tight")

def evaluate_with_metadata(model, dataloader, device):
    model.eval()
    
    all_ids = []
    all_labels = []
    all_domains = []
    all_models = []
    all_preds = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            _, _, out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            preds = out.argmax(dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(batch.y.cpu().numpy())
            
            # Collect metadata (ensure these fields exist in the batch)
            #all_ids.extend(batch.metadata['id'])
            all_domains.extend(batch.metadata['domain'])
            all_models.extend(batch.metadata['model'])
    
    # Create a DataFrame for error analysis
    df_results = pd.DataFrame({
        #"id": all_ids,
        "label": all_labels,
        "domain": all_domains,
        "model": all_models,
        "predicted_label": all_preds
    })
    
    # Compute accuracy and F1-score
    accuracy = accuracy_score(df_results["label"], df_results["predicted_label"])
    f1 = f1_score(df_results["label"], df_results["predicted_label"], average="macro")
    cm = confusion_matrix(df_results["predicted_label"], df_results["label"])

    return df_results, accuracy, f1, cm

def load_model_for_inference(model_class, model_path, config, device):
    """ Load the trained model for inference only. """
    model = model_class(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        dense_hidden_dim=config['dense_hidden_dim'],
        output_dim=config['output_dim'],
        dropout=config['dropout'],
        num_layers=config['num_layers'],
        edge_attr=config['add_edge_attr'],
        gnn_type=config['gnn_type'],
        heads=config['heads'],
        task='graph'
    ).to(device)

    # Load the model state dictionary
    state_dict = torch.load(model_path, map_location=device)
    
    # Load the state dict with strict=False to allow mismatches
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    # Debugging: Print missing or unexpected keys
    if missing_keys:
        print(f" Warning: Missing keys in state_dict -> {missing_keys}")
    if unexpected_keys:
        print(f" Warning: Unexpected keys in state_dict -> {unexpected_keys}")
    

    model.eval()  # Set model to evaluation mode
    print(f"Model loaded from: {model_path}")
    return model

def model_inference():
    cuda_num = 1
    nfi = 'llm' # llm, w2v, random
    cut_off_dataset = '10-10-10' # semeval: 5-10-10 | autext: 10-10-10 | coling: 1-1-1
    dataset_name = 'autext23'  # autext23, semeval24, coling24, autext23_s2, semeval24_s2
    file_name_data = f"cooc_data_{dataset_name}_{cut_off_dataset}perc"
    device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")
    llm_name = 'microsoft/deberta-v3-base'

    output_dir = f'{test_utils.EXTERNAL_DISK_PATH}cooc_graph'
    model_path = f"{output_dir}/models/gnn_model_{nfi}_{file_name_data}.pt"

    data_obj = utils.load_data(file_name_data, path=f'{output_dir}/{llm_name.split("/")[1]}/', format_file='.pkl', compress=False)
    val_data = data_obj['all_data'][1]
    test_data = data_obj['all_data'][2]
    vocab = data_obj['vocab']
    config = data_obj['config']
    print("config: ", config)

    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    config = {
        "input_dim": test_data[0].x.shape[1],
        "hidden_dim": 100,
        "dense_hidden_dim": 64,
        "output_dim": 2,
        "dropout": 0.8,
        "num_layers": 1,
        "add_edge_attr": True,
        "gnn_type": 'TransformerConv',
        "heads": 1  
    }

    # Load trained model
    model = load_model_for_inference(GNN, model_path, config, device)
    print(model)

    # Final evaluation on the test set
    criterion = torch.nn.CrossEntropyLoss()
    #test_acc, test_f1_macro, test_loss, preds_test, _ = test_cooc(test_loader, model, device, criterion)
    df_results, test_acc, test_f1_macro, cm = evaluate_with_metadata(model, test_loader, device)

    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1 Score: {test_f1_macro:.4f}")
    print(cm)

    class_names = ["Human", "Machine"]
    plot_confusion_matrix(df_results["label"], df_results["predicted_label"], class_names, dataset_name, normalize=True)
    
    # Analyze error distribution by domain
    df_errors = df_results[df_results["label"] != df_results["predicted_label"]]

    print(df_errors["label"].value_counts(normalize=False))
    domain_error_dist = df_errors["domain"].value_counts(normalize=False)
    model_error_dist = df_errors["model"].value_counts(normalize=True)
    print(df_errors.head())
    
    print("\nError Distribution by Domain:")
    print(domain_error_dist)
    
    print("\nError Distribution by AI Model:")
    print(model_error_dist)

    # Compute influential words
    influential_words = get_influential_words(model, test_loader, device, vocab, top_k=10)
    # Print influential words per class
    print("Most Influential Words per Class:")
    for class_label, words in influential_words.items():
        print(f"\nClass {class_label} ({'Human' if class_label==0 else 'Machine'}):")
        for word, score in words:
            print(f"{word}: {score:.4f}")


#        true
#      0     1
#  0  626   77
#  1  459 1021


if __name__ == '__main__':
    #main()
    model_inference()


# ***** AUTEXT CONFIG
'''
    'dataset_name': 'autext23', # autext23, semeval24, coling24, autext23_s2, semeval24_s2
    'cut_off_dataset': '100-100-100', # train-val-test
    'add_edge_attr': True,
    "learnin_rate": 0.0001
    "dropout": 0.8,
    "dense_hidden_dim": 64, # 64-autext23 | 32-semeval | 64-coling
    "num_layers": 1,
    "heads": 1,

'''

# ***** SEMEVAL CONFIG
'''
    'dataset_name': 'semeval24', # autext23, semeval24, coling24, autext23_s2, semeval24_s2
    'cut_off_dataset': '5-10-10', # train-val-test
    'graph_direction': 'undirected', # undirected | directed 
    'add_edge_attr': True,
    "dropout": 0.5,
    "learnin_rate": 0.000002, 
    "dense_hidden_dim": 64, # 64-autext23 | 32-semeval | 64-coling
    "num_layers": 1,
    "heads": 1,
'''

# ***** COLING CONFIG
'''
    'dataset_name': 'coling24', # autext23, semeval24, coling24, autext23_s2, semeval24_s2
    'cut_off_dataset': '1-1-1', # train-val-test
    'graph_direction': 'undirected', # undirected | directed 
    'add_edge_attr': True,
    "dropout": 0.5,
    "learnin_rate": 0.0001, 
    "dense_hidden_dim": 64, # 64-autext23 | 32-semeval | 64-coling
    "num_layers": 1,
    "heads": 1,
'''
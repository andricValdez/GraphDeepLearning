import pprint
import torch
import numpy as np
from sklearn.decomposition import PCA
from torch_geometric.data import Data, DataLoader
from gensim.models import Word2Vec
from torch.nn import Linear
from torch_geometric.loader import NeighborSampler, NeighborLoader
from torch_geometric.utils import dropout_edge
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from itertools import combinations
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import os
import scipy as sp
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
import copy
from tqdm import tqdm
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
from sklearn.feature_extraction.text import TfidfTransformer
import torch
import gc

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import contractions
nltk.download('stopwords')
nltk.download('punkt_tab')

import test_utils
import utils

#************************************* CONFIGS
warnings.filterwarnings("ignore")

#************************************* CONFIGS
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s; - %(levelname)s; - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

global cnt1
cnt1 = 0
global cnt2
cnt2 = 0
global cnt3
cnt3 = 0

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
    def __init__(self, input_dim, hidden_dim, dense_hidden_dim, output_dim, dropout, num_layers, edge_attr=False, gnn_type='GCNConv', heads=1, task='node', directed=False):
        super(GNN, self).__init__()
        self.task = task
        self.heads = heads
        self.gnn_type = gnn_type
        self.edge_attr = edge_attr
        self.embeddings = []  # List to store embeddings after each layer

        self.directed = directed  # Add directed parameter
        self.conv1 = self.build_conv_model(input_dim, hidden_dim, self.heads)
        self.norm1 = nn.LayerNorm(hidden_dim * heads) # BatchNorm1d, LayerNorm
        self.convs = nn.ModuleList()
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
        )

        self.dropout = dropout
        self.num_layers = num_layers

    def build_conv_model(self, input_dim, hidden_dim, heads):
        if self.gnn_type == 'GCNConv':
            return GCNConv(input_dim, hidden_dim)
        elif self.gnn_type == 'GATConv':
            return GATConv(input_dim, hidden_dim, heads=heads)
        elif self.gnn_type == 'TransformerConv':
            if self.edge_attr:
                return TransformerConv(input_dim, hidden_dim, heads=heads, edge_dim=1)
            else:    
                return TransformerConv(input_dim, hidden_dim, heads=heads)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # Handle undirected graphs by adding reverse edges
        #if not self.directed:
        #    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # Add reverse edges
        #    if self.edge_attr:
        #        edge_attr = torch.cat([edge_attr, edge_attr], dim=0)  # Duplicate edge_attr for reverse edges

        # Pass through the GNN layers
        #print(edge_index.shape, edge_attr.shape)
        self.embeddings.append(x.detach().cpu().numpy())

        if self.edge_attr:
            x = self.conv1(x, edge_index, edge_attr)
        else:
            x = self.conv1(x, edge_index)

        emb = x
        x = F.relu(x)
        self.embeddings.append(x.detach().cpu().numpy())
        x = self.norm1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)


        for i in range(self.num_layers):
            if self.edge_attr:
                x = self.convs[i](x, edge_index, edge_attr)
            else:
                x = self.convs[i](x, edge_index)

            emb = x
            x = F.relu(x)
            self.embeddings.append(x.detach().cpu().numpy())
            x = self.lns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.task == 'graph':
            x = global_mean_pool(x, batch)

        x = self.post_mp(x)
        #return emb, None, F.log_softmax(x, dim=1)
        return emb, self.embeddings, F.log_softmax(x, dim=1)


def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        train_mask = batch.train_mask
        
        #if model.directed:
        #    edge_index = batch.directed_edge_index
        #    edge_attr = batch.edge_attr
        #else:
        #    edge_index = batch.undirected_edge_index
        #    edge_attr = torch.cat([batch.edge_attr, batch.edge_attr], dim=0)  # Duplicate edge_attr for reverse edges

        #edge_index, _ = dropout_edge(batch.edge_index, p=0.25)  # Drop 50% of edges
        emb, _, out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = criterion(out[train_mask], batch.y[train_mask])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, criterion, mask_type, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_f1 = 0
    total_nodes = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if mask_type == 'train':
                mask = batch.train_mask
            elif mask_type == 'val':
                mask = batch.val_mask
            elif mask_type == 'test':
                mask = batch.test_mask
            else:
                raise ValueError("Invalid mask_type. Use 'train', 'val', or 'test'.")

            #if model.directed:
            #    edge_index = batch.directed_edge_index
            #    edge_attr = batch.edge_attr
            #else:
            #    edge_index = batch.undirected_edge_index
            #    edge_attr = torch.cat([batch.edge_attr, batch.edge_attr], dim=0)  # Duplicate edge_attr for reverse edges

            emb, _, out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(out[mask], batch.y[mask])
            pred = out.argmax(dim=1)
            correct = (pred[mask] == batch.y[mask]).sum()
            f1 = f1_score(batch.y[mask].cpu(), pred[mask].cpu(), average='macro')

            total_loss += loss.item()
            total_correct += int(correct)
            total_f1 += f1
            total_nodes += mask.sum().item()

            all_preds.extend(pred[mask].cpu().numpy())
            all_labels.extend(batch.y[mask].cpu().numpy())

    acc = total_correct / total_nodes
    f1_macro = total_f1 / len(loader)
    loss = total_loss / len(loader)

    return acc, f1_macro, loss, all_preds, all_labels

def get_document_embeddings(text, tokenizer, language_model, device):
    # Tokenize the document and convert to tensor
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)
    # Get LLM embeddings
    with torch.no_grad():
        outputs = language_model(**inputs)
    # Use the embeddings from the last hidden state (CLS token or average pooling)
    doc_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return doc_embedding

def get_word_embeddings(word, tokenizer, language_model, device):
    # Tokenize the word and convert to tensor
    inputs = tokenizer(word, return_tensors='pt', truncation=True, padding=True).to(device)
    # Get BERT embeddings
    with torch.no_grad():
        outputs = language_model(**inputs)
    # Use the embeddings from the last hidden state (CLS token or average pooling)
    word_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return word_embedding

def get_word_embeddings2(corpus, tokenizer, language_model, vocab, device, not_found_tokens='avg'):
    embeddings_word_dict = {}
    token_freq = defaultdict(int)

    for idx, text in enumerate(tqdm(corpus, desc="Extracting word embeddings")): # hetero
        #text = utils.text_normalize_v2(text['doc'])
        
        encoded_text = tokenizer.encode_plus(text, return_tensors="pt", padding=True, truncation=True)
        encoded_text.to(device)
        with torch.no_grad():
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

    #for token, freq in token_freq.items():
    #    embeddings_word_dict[token] = np.divide(embeddings_word_dict[token], freq).tolist()

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

def calculate_pmi(co_occurrence_matrix, word_freq, total_pairs):
    pmi_matrix = co_occurrence_matrix.copy()
    for (word1, word2), count in co_occurrence_matrix.items():
        pmi = log((count * total_pairs) / (word_freq[word1] * word_freq[word2]))
        pmi_matrix[(word1, word2)] = pmi
    return pmi_matrix

def min_max_normalize(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    if max_val == min_val:  # Avoid division by zero
        return torch.zeros_like(tensor)
    return (tensor - min_val) / (max_val - min_val)

def calculate_graph_metrics(edges, node_features):
    num_nodes = node_features.size(0)  # Ensure all metrics have this length
    print("Min node ID in edges:", edges.min().item())
    print("Max node ID in edges:", edges.max().item())

    # Create a NetworkX graph from edge_index
    G = nx.Graph()
    G.add_edges_from(edges.t().tolist())

    # Compute node-level metrics
    degree_centrality = torch.tensor(list(nx.degree_centrality(G).values()), dtype=torch.float).view(-1, 1)
    logger.info("degree_centrality: %s", str(degree_centrality.shape))
    betweenness_centrality = test_utils.approximate_betweenness_centrality(edges, num_nodes=num_nodes, k=100)
    logger.info("betweenness_centrality: %s", str(betweenness_centrality.shape))
    eigenvector_centrality = test_utils.approximate_eigenvector_centrality(edges, num_nodes=num_nodes, max_iter=50)
    logger.info("eigenvector_centrality: %s", str(eigenvector_centrality.shape))
    pagerank = test_utils.approximate_pagerank(edges, num_nodes=num_nodes, max_iter=50)
    logger.info("pagerank: %s", str(pagerank.shape))
    clustering_coefficient = test_utils.approximate_clustering_coefficient(edges, num_nodes=num_nodes, k=100)
    logger.info("clustering_coefficient: %s", str(clustering_coefficient.shape))

    # Ensure all tensors have the same number of rows
    def fix_size(tensor, num_nodes):
        if tensor.shape[0] < num_nodes:
            padding = torch.zeros(num_nodes - tensor.shape[0], tensor.shape[1], dtype=tensor.dtype)
            return torch.cat([tensor, padding], dim=0)
        elif tensor.shape[0] > num_nodes:
            return tensor[:num_nodes]
        return tensor

    degree_centrality = fix_size(degree_centrality, num_nodes)
    betweenness_centrality = fix_size(betweenness_centrality, num_nodes)
    eigenvector_centrality = fix_size(eigenvector_centrality, num_nodes)
    pagerank = fix_size(pagerank, num_nodes)
    clustering_coefficient = fix_size(clustering_coefficient, num_nodes)

    # Normalize metrics
    degree_centrality = min_max_normalize(degree_centrality)
    betweenness_centrality = min_max_normalize(betweenness_centrality)
    eigenvector_centrality = min_max_normalize(eigenvector_centrality)
    pagerank = min_max_normalize(pagerank)
    clustering_coefficient = min_max_normalize(clustering_coefficient)

    # Combine all metrics into a single tensor
    graph_metrics = torch.cat([
        degree_centrality,
        betweenness_centrality,
        eigenvector_centrality,
        pagerank,
        clustering_coefficient,
    ], dim=-1)

    return graph_metrics

def reduce_dimension_linear(data_x, reduction_layer, device, batch_size=128):
    # Process x in batches
    reduced_x_list = []
    for i in range(0, data_x.shape[0], batch_size):
        batch = data_x[i:i + batch_size]  # Extract batch
        batch = batch.to(device)
        reduced_batch = reduction_layer(batch)  # Apply reduction layer
        reduced_x_list.append(reduced_batch.cpu())  # Move back to CPU and store

    # Concatenate all reduced batches
    reduced_x = torch.cat(reduced_x_list, dim=0)
    return reduced_x

def get_class_distribution(data_list):
    """Compute class distribution from a list of Data objects."""
    labels = []
    for data in data_list:
        labels.extend(data.y.tolist())  # Convert tensor to list and add to labels
    
    return Counter(labels)  # Count occurrences of each class label


def extract_gnn_embeddings(model, data, data_loader):
    model.eval()  # Set model to evaluation mode
    test_embeddings = [[] for _ in range(model.num_layers + 2)]  # Store embeddings at each step
    test_node_indices = []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(next(model.parameters()).device)  # Move batch to device
            x, edge_index = batch.x, batch.edge_index

            emb, embeddings, _ = model(x, edge_index)  # Forward pass to get embeddings
            test_node_indices.append(batch.n_id.cpu().numpy())  # Store node indices

            for i, emb_layer in enumerate(embeddings):
                test_embeddings[i].append(emb_layer)

    return test_embeddings, test_node_indices


'''
training on the full dataset was not possible in some scenarios due to its large size and high computational cost. 
For this reason, we added the param config "cut_off_dataset" enabling cut-off of the dataset/partition in the form train_val_set
Also, for each partition, we ensured metadata balancing within each class: for machine-generated texts, we maintained a balanced distribution across different text generation models (e.g., LLaMA, ChatGPT, etc.), and for human-written texts, we preserved balance across source domains (e.g., papers, articles, essays).
E.g. let's say we want to train our model with 25 % train, 50 % val, and 100 % test; in this case, we set -> 'cut_off_dataset': '25-50-100'
This could be helpful for a very large dataset, e.g. the COLING dataset (having 606,189 training docs) or to do rapid testing for some configs or proof of concepts
'''
def main():    
    # autext23,     100-100-100
    # semeval24,    10-25-25
    # coling24,     5-10-10
    # autext23_s2, 
    # semeval24_s2
    config = {
        'build_graph': False,
        'dataset_name': 'autext23', # autext23, semeval24, coling24, autext23_s2, semeval24_s2
        'cut_off_dataset': '1-1-1', # train-val-test
        "nfi": 'llm', # llm, w2v, random
        'cuda_num': 1,

        'window_size': 10,
        'graph_direction': 'undirected', # undirected | directed (for now all are undirecte, pending to handle and review to support both)
        'special_chars': False,
        'stop_words': False,
        'min_df': 5, # 1->autext | 5->semeval | 5-coling
        'max_df': 0.9,
        'max_features': None, # None -> all | 5000
        'not_found_tokens': 'avg', # avg, remove, zeros, ones
        'add_edge_attr': False,
        'add_graph_metric': False,
        'embed_reduction': False,

        "gnn_type": 'TransformerConv', # GCNConv, GATConv, TransformerConv
        "dropout": 0.6,
        "patience": 5, # 5-autext23 | 10-semeval | 10-coling
        "learnin_rate": 0.000002, # autext23 -> llm: 0.00002 | semeval -> llm: 0.000005  | coling -> llm: 0.0001 
        "batch_size": 32 * 1,
        "hidden_dim": 100, # 300 autext_s2, 100 others
        "dense_hidden_dim": 32, # 64-autext23 | 32-semeval | 64-coling
        "num_layers": 1,
        "heads": 1,
        "output_dim": 2, # 2-bin | 6-multi 
        "weight_decay": 0.001, 
        "num_neighbors": [50, 40],  # Adjust sampling depth
        'input_dim': 128,
        'epochs': 100,
        "llm_name": 'microsoft/deberta-v3-base',

    }

    ## google-bert/bert-base-uncased
    ## FacebookAI/roberta-base
    ## microsoft/deberta-v3-base
    #llm_name = 'microsoft/deberta-v3-base'
    
    file_name_data = f"hetero_data_{config['dataset_name']}_{config['cut_off_dataset']}perc" # perc_128
    #output_dir = f'{utils.OUTPUT_DIR_PATH}test_graph/{llm_name.split("/")[1]}/'
    #output_dir = f'{utils.OUTPUT_DIR_PATH}test_graph/'
    output_dir = f'{test_utils.EXTERNAL_DISK_PATH}hetero_graph'

    nfi_dir = config["llm_name"].split("/")[1] # nfi -> llm
    if config['nfi'] == 'w2v':
        nfi_dir = 'w2v'
    if config['nfi'] == 'random':
        nfi_dir = 'random'

    device = torch.device(f"cuda:{config['cuda_num']}" if torch.cuda.is_available() else "cpu")
    pprint.pprint(config)
    
    if config['build_graph']:
        # Load and preprocess dataset
        train_text_set, val_text_set, test_text_set = test_utils.read_dataset(config['dataset_name'])

        # Cut off datasets
        cut_off_train = int(config['cut_off_dataset'].split('-')[0])
        cut_off_val = int(config['cut_off_dataset'].split('-')[1])
        cut_off_test = int(config['cut_off_dataset'].split('-')[2])

        train_set = train_text_set[:int(len(train_text_set) * (cut_off_train / 100))][:]
        val_set = val_text_set[:int(len(val_text_set) * (cut_off_val / 100))][:]
        test_set = test_text_set[:int(len(test_text_set) * (cut_off_test / 100))][:]

        print("cutoff_distro_train_val_test: ", len(train_set), len(val_set), len(test_set))
        print("label_distro_train_val_test: ", train_set.value_counts('label'), val_set.value_counts('label'), test_set.value_counts('label'))
        
        # Combine texts and labels
        all_texts = list(train_set['text']) + list(val_set['text']) + list(test_set['text'])
        all_labels = list(train_set['label']) + list(val_set['label']) + list(test_set['label'])

        # Tokenize and normalize texts
        tokenized_corpus = []
        all_texts_norm = []
        for text in tqdm(all_texts, desc="Normalizing corpus"):
            text_norm = test_utils.text_normalize(text, special_chars=config['special_chars'], stop_words=config['stop_words'])
            all_texts_norm.append(text_norm)
            tokenized_corpus.append(re.findall("[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+", text_norm))

        # Create vocabulary and TF-IDF features
        vectorizer = CountVectorizer(min_df=config['min_df'], max_df=config['max_df'], max_features=config['max_features'])
        X = vectorizer.fit_transform(all_texts_norm)
        vocab = vectorizer.get_feature_names_out()
        word_to_index = {word: idx for idx, word in enumerate(vocab)}
        print('vocab: ', len(vocab))

        tfidf_transformer = TfidfTransformer()
        X_tfidf = tfidf_transformer.fit_transform(X)

        # *** Generate embeddings method 2  - LLM 
        if config['nfi'] == 'llm':
            # Generate word and document embeddings
            tokenizer = AutoTokenizer.from_pretrained(config['llm_name'], model_max_length=512)
            language_model = AutoModel.from_pretrained(config['llm_name'], output_hidden_states=True).to(device)

            # Generate word embeddings
            word_features = get_word_embeddings2(all_texts_norm, tokenizer, language_model, vocab, device, not_found_tokens=config['not_found_tokens'])
            word_features = torch.tensor(word_features, dtype=torch.float)

            # Generate document embeddings
            doc_features = []
            for text in tqdm(all_texts_norm, desc="Extracting doc embeddings"):
                doc_embedding = get_document_embeddings(text, tokenizer, language_model, device)
                doc_features.append(doc_embedding)
            doc_features = torch.tensor(doc_features, dtype=torch.float)

        if config['nfi'] == 'w2v':
            # Generate word embeddings
            w2v_model = Word2Vec(sentences=tokenized_corpus, vector_size=config['input_dim'], window=5, min_count=1, workers=4)
            word_features, updated_vocab = [], []
            for word in vocab:
                if word in w2v_model.wv:
                    word_features.append(w2v_model.wv[word])
                    updated_vocab.append(word)
                else:
                    word_features.append(np.zeros(w2v_model.vector_size))  # Assign zero vector for missing words
            word_features = torch.tensor(word_features, dtype=torch.float)
            #vocab = updated_vocab
            #word_to_index = {word: idx for idx, word in enumerate(vocab)}

            # Generate document embeddings using Word2Vec-based averaging
            doc_features = []
            for text in tqdm(tokenized_corpus, desc="Extracting document embeddings"):
                word_embeddings = [w2v_model.wv[word] for word in text if word in w2v_model.wv]
                if len(word_embeddings) > 0:
                    doc_embedding = np.mean(word_embeddings, axis=0)  # Average word embeddings
                else:
                    doc_embedding = np.zeros(w2v_model.vector_size)  # Default to zero vector if no words are found 
                doc_features.append(torch.tensor(doc_embedding, dtype=torch.float))
            doc_features = torch.stack(doc_features)  # Convert list of tensors to a single tensor
                    
        if config['nfi'] == 'random':
             # Generate word embeddings
            word_features = np.random.uniform(low=-1, high=1, size=(len(vocab), config['input_dim']))
            word_features = torch.tensor(word_features, dtype=torch.float)
            # Generate random document embeddings
            doc_features = []
            for _ in tqdm(all_texts_norm, desc="Generating random document embeddings"):
                doc_embedding = np.random.uniform(low=-1, high=1, size=(config['input_dim'],))
                doc_features.append(torch.tensor(doc_embedding, dtype=torch.float))
            doc_features = torch.stack(doc_features)  # Convert list of tensors to a single tensor


        # **** Extracting document-word edges
        doc_word_edges = set()
        doc_word_attr = {}
        for doc_id, doc in enumerate(tqdm(X, desc="Extracting document-word edges")):
            for word_idx in doc.nonzero()[1]:
                word_node_id = len(all_texts) + word_idx
                # Add edge: word -> document
                edge = (word_node_id, doc_id)
                if edge not in doc_word_edges:
                    doc_word_edges.add(edge)
                    doc_word_attr[edge] = X_tfidf[doc_id, word_idx]
                # Add edge: document -> word
                #reverse_edge = (doc_id, word_node_id)
                #if reverse_edge not in doc_word_edges:
                #    doc_word_edges.add(reverse_edge)
                #    doc_word_attr[reverse_edge] = X_tfidf[doc_id, word_idx]


        # Convert set to a sorted list (ensuring consistent order)
        doc_word_edges = sorted(doc_word_edges)  # List of (source, target) tuples
        doc_word_attr = [doc_word_attr[edge] for edge in doc_word_edges]  # Corresponding TF-IDF weights as a list

        # *** Create word-word edges and edge attributes
        word_word_edges = set()
        word_word_attr = {}
        word_freq = Counter()
        co_occurrence_matrix = defaultdict(int)
        total_pairs = 0

        for doc in tqdm(tokenized_corpus, desc="Extracting word-word edges"):
            for i in range(len(doc)):
                window = doc[i:i + config['window_size'] + 1]
                for word1, word2 in combinations(window, 2):
                    if word1 in word_to_index and word2 in word_to_index:
                        word1_id = len(all_texts) + word_to_index[word1]
                        word2_id = len(all_texts) + word_to_index[word2]
                        word_word_edges.add((word1_id, word2_id))
                        co_occurrence_matrix[(word1, word2)] += 1
                        #word_word_edges.add((word2_id, word1_id))
                        #co_occurrence_matrix[(word2, word1)] += 1

                        #if config['graph_direction'] == 'undirected':
                        #    word_word_edges.add((word2_id, word1_id))
                        #    co_occurrence_matrix[(word2, word1)] += 1
                        word_freq[word1] += 1
                        word_freq[word2] += 1
                        total_pairs += 1

        # Calculate PMI values
        pmi_matrix = calculate_pmi(co_occurrence_matrix, word_freq, total_pairs)

        # Convert word-word edges and attributes to lists
        word_word_edges = list(word_word_edges)
        word_word_attr = [pmi_matrix[(vocab[word1_id - len(all_texts)], vocab[word2_id - len(all_texts)])] for (word1_id, word2_id) in word_word_edges]

        # Combine document and word features
        node_features = torch.cat([doc_features, word_features], dim=0)
        edge_attr = torch.tensor(doc_word_attr + word_word_attr, dtype=torch.float).view(-1, 1)

        # Normalize edge attributes
        edge_attr = min_max_normalize(edge_attr)

        # Create node labels
        node_labels = torch.tensor(all_labels + [-1] * len(vocab), dtype=torch.long)

        # Combine all edges
        edges = torch.tensor(doc_word_edges + word_word_edges, dtype=torch.long).t().contiguous()

        # Calculate graph metrics
        #graph_metrics = calculate_graph_metrics(edges, node_features)

        # Create directed and undirected edge indices
        directed_edge_index = edges
        undirected_edge_index = torch.cat([edges, edges.flip(0)], dim=1)  # Add reverse edges
        
        # Removes duplicate edges
        directed_edge_index = torch.unique(directed_edge_index, dim=1)  
        undirected_edge_index = torch.unique(undirected_edge_index, dim=1)  

        # Handle edge_attr for undirected edges
        #if config['graph_direction'] == 'undirected':
        #    edge_attr = torch.cat([edge_attr, edge_attr], dim=0)  # Duplicate edge_attr for reverse edges

        # Create masks for train, validation, and test sets
        num_nodes = len(node_features)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[:len(train_set)] = True
        val_mask[len(train_set):len(train_set) + len(val_set)] = True
        test_mask[len(train_set) + len(val_set):len(train_set) + len(val_set) + len(test_set)] = True

        # Create the PyG Data object
        data = Data(
            x = node_features,
            #edge_index = edges,
            directed_edge_index=directed_edge_index,
            undirected_edge_index=undirected_edge_index,
            edge_attr = edge_attr,
            train_mask = train_mask,
            val_mask = val_mask,
            test_mask = test_mask,
            #graph_metrics = graph_metrics,
            y = node_labels
        )

        # Apply dimensionality reduction to train, val, and test data
        new_feat_dim = 128
        if config['embed_reduction']:    
            embedding_reduction = nn.Linear(768, new_feat_dim).to(device)
            data.x = reduce_dimension_linear(data.x, embedding_reduction, device, batch_size=128)

        # Save the data object
        #utils.save_data(data, file_name_data, path=output_dir, format_file='.pkl', compress=False)
        utils.save_data(data, file_name_data, path=f'{output_dir}/{nfi_dir}/', format_file='.pkl', compress=False)

    else:
        # Load the data object
        #data = utils.load_data(file_name_data, path=output_dir, format_file='.pkl', compress=False)
        data = utils.load_data(file_name_data, path=f'{output_dir}/{nfi_dir}/', format_file='.pkl', compress=False)

    #embedding_reduction = nn.Linear(768, 128).to(device)
    #data.x = reduce_dimension_linear(data.x, embedding_reduction, device, , batch_size=128)
    #utils.save_data(data, file_name_data, path=output_dir, format_file='.pkl', compress=False)
    #return

    # add/remove attr from Data object (based on the above config, this reduce the traning time)
    if config['graph_direction'] == 'undirected':
        data.edge_index = data.undirected_edge_index
        if config['add_edge_attr']:
            data.edge_attr = torch.cat([data.edge_attr, data.edge_attr], dim=0)  # Duplicate edge_attr for reverse edges
    #        data.edge_attr = torch.unique(data.edge_attr, dim=1)  
    else:
        data.edge_index = data.directed_edge_index
    del data.directed_edge_index
    del data.undirected_edge_index

    if config['add_graph_metric']:
        data.x = torch.cat([data.x, data.graph_metrics], dim=-1)
    else:
        del data.graph_metrics

    if not config['add_edge_attr']:
        del data.edge_attr

    # Find isolated nodes
    #nodes_with_edges = torch.unique(data.edge_index.flatten())
    #all_nodes = torch.arange(data.num_nodes)
    #mask = ~torch.isin(all_nodes, nodes_with_edges)  # Find nodes not in nodes_with_edges
    #isolated_nodes = all_nodes[mask]  # Select isolated nodes
    #print("isolated_nodes: ", isolated_nodes)

    # move data object to devine
    data = data.to(device)
    print(data)

    # Create NeighborLoader instances
    train_nodes = torch.nonzero(data.train_mask, as_tuple=True)[0]  # Indices of train nodes
    val_nodes = torch.nonzero(data.val_mask, as_tuple=True)[0]  # Indices of train nodes
    test_nodes = torch.nonzero(data.test_mask, as_tuple=True)[0]  # Indices of train nodes
    #print(train_nodes)
    #print(val_nodes)
    #print(test_nodes)

    train_loader = NeighborLoader(
        data,
        num_neighbors=config['num_neighbors'], # Number of neighbors to sample at each layer
        batch_size=config['batch_size'],       # Batch size
        input_nodes=train_nodes, # also is valid to pass directly: data.train_mask
        shuffle=True,
        replace=False,
        num_workers=4,
        persistent_workers=True  # Caches samples for repeated training 
    )

    val_loader = NeighborLoader(
        data,
        num_neighbors=config['num_neighbors'], 
        batch_size=config['batch_size'],       
        input_nodes=val_nodes,
        shuffle=True,
        replace=False,
        num_workers=4,
        persistent_workers=True  # Caches samples for repeated training         
    )

    test_loader = NeighborLoader(
        data,
        num_neighbors=config['num_neighbors'], 
        batch_size=config['batch_size'],       
        input_nodes=test_nodes,
        shuffle=True,
        replace=False,
        num_workers=4,
        persistent_workers=True  # Caches samples for repeated training 
    )

    for batch in train_loader:
        print(f"Batch nodes: {batch.n_id.size(0)}, Batch data: {batch}")
        break
    
    # Initialize the model
    input_dim = data.x.shape[1]
    test_utils.set_random_seed(42)

    model = GNN(
        input_dim,
        hidden_dim=config['hidden_dim'],
        dense_hidden_dim=config['dense_hidden_dim'],
        output_dim=config['output_dim'],
        dropout=config['dropout'],
        num_layers=config['num_layers'],
        edge_attr=config['add_edge_attr'],
        gnn_type=config['gnn_type'],
        heads=config['heads'],
        task='node',
        directed=(config['graph_direction'] == 'directed'),
    )
    
    model = model.to(device)
    print(model)

    # Training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learnin_rate'], weight_decay=config['weight_decay'])
    # Add learning rate scheduler
    #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    criterion = torch.nn.CrossEntropyLoss()
    early_stopper = EarlyStopper(patience=config['patience'], min_delta=0)

    logger.info("Init GNN training!")
    best_test_acc, best_test_f1score, best_epoch = 0, 0, 0
    for epoch in range(config['epochs']):
        #with torch.cuda.amp.autocast():  # Use FP16 for faster computation
        loss_train = train(model, train_loader, optimizer, criterion, device)
        val_acc, val_f1_macro, val_loss, _, _, = evaluate(model, val_loader, criterion, 'val', device)
        
        if epoch % 1 == 0:
            test_acc, test_f1_macro, test_loss, preds_test, labels_test  = evaluate(model, test_loader, criterion, 'test', device)
            print(f'Ep {epoch + 1}, Loss Val: {val_loss:.4f}, Loss Test: {test_loss:.4f}, '
            f'Val Acc: {val_acc:.4f}, Val F1s: {val_f1_macro:.4f}, Test Acc: {test_acc:.4f}, Test F1s: {test_f1_macro:.4f}')
        else:
            print(f'Ep {epoch + 1}, Loss Val: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1s: {val_f1_macro:.4f}')
            
        # Get the current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        #print("preds_test: ", sorted(Counter(preds_test).items()))
        #print("label_test: ", sorted(Counter(labels_test).items()))
        
        #scheduler.step(val_loss)  # Update learning rate
        if early_stopper.early_stop(val_loss):
            print('Early stopping due to no improvement!')
            break
    logger.info("Done GNN training!")

    # Evaluate on test set
    test_acc, test_f1_macro, test_loss, _, _ = evaluate(model, test_loader, criterion, 'test', device)
    print()
    print(f'Test Accuracy: {test_acc:.4f}')
    print(f'Test F1Score: {test_f1_macro:.4f}')
    print(f'Test Loss: {test_loss:.4f}')

    model_save_path = f"{output_dir}/models/gnn_model_{config['nfi']}_{file_name_data}.pt"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved at: {model_save_path}")

    
    
    # Extract embeddings for the test set nodes
    test_embeddings, test_node_indices = extract_gnn_embeddings(model, data, test_loader)

    # Plot the embeddings at each step (input features, after 1st layer, and after 2nd layer)
    steps = ['Input Features', 'After 1st Layer', 'After 2nd Layer']
    for i, embeddings in enumerate(test_embeddings):
        # Combine embeddings from all batches to create a single set
        combined_embeddings = np.vstack(embeddings)
        test_nodes_combined = np.concatenate(test_node_indices)

        # Use PCA to reduce the dimensions for visualization
        reduced_embeddings = PCA(n_components=2).fit_transform(combined_embeddings)
        
        # Plot the reduced embeddings
        plt.figure(figsize=(8, 6))
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], 
                    c=data.y[test_nodes_combined].cpu().numpy(), cmap='viridis')
        plt.colorbar()
        plt.title(f'{steps[i]} Embeddings of Test Set')
        plt.show()
        plt.savefig(f'test_emb_{i}.png')

     


if __name__ == '__main__':
    main()

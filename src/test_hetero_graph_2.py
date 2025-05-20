import pprint
import torch
import numpy as np
import pickle as pkl
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
from torch_geometric.utils import from_scipy_sparse_matrix
from datasets import load_dataset
import joblib
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import seaborn as sns
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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
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
from torch_geometric.nn import MLP
from collections import OrderedDict
import warnings
from transformers import logging as transform_loggin
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, TransformerConv, TopKPooling, GraphConv, SAGPooling, GENConv, GINConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch.nn import Linear, BatchNorm1d, ModuleList, LayerNorm
from IPython.core.display import display, HTML
from sklearn.feature_extraction.text import TfidfTransformer
import torch
import gc
import spacy
import mlflow
from mlflow import MlflowClient

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import contractions
nltk.download('stopwords')
nltk.download('punkt_tab')

import test_utils
import utils

# --- Load spaCy tokenizer ---
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

#************************************* CONFIGS
warnings.filterwarnings("ignore")
mlflow.set_tracking_uri(uri="http://localhost:8081")
mlflow.set_tracking_uri("/home/avaldez/projects/GraphDeepLearning/mlruns")
client = MlflowClient()
experiment_id = "0"
run = client.create_run(experiment_id)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#************************************* CONFIGS
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s; - %(levelname)s; - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
    def __init__(
        self,
        input_dim,
        hidden_dim,
        dense_hidden_dim,
        output_dim,
        dropout,
        num_layers,
        edge_attr=False,
        gnn_type='GCNConv',
        heads=1,
        directed = False,
        norm_type='batchnorm',   # 'batchnorm', 'layernorm', or None
        post_mp_layers=3,        # number of layers after message passing
    ):
        super(GNN, self).__init__()
        self.heads = heads
        self.gnn_type = gnn_type
        self.edge_attr = edge_attr
        self.dropout = dropout
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.embeddings = []
        self.directed = directed  # Add directed parameter

        # First conv
        self.conv1 = self.build_conv_model(input_dim, hidden_dim, heads)
        self.norm1 = self.build_norm_layer(hidden_dim * heads)

        # Additional conv layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(self.build_conv_model(hidden_dim * heads, hidden_dim, heads))
            self.norms.append(self.build_norm_layer(hidden_dim * heads))

        # Post-message-passing MLP
        dims = [hidden_dim * heads] + [dense_hidden_dim // (2 ** i) for i in range(post_mp_layers - 1)] + [output_dim]
        post_mp = []
        for i in range(len(dims) - 1):
            post_mp.append(nn.Linear(dims[i], dims[i + 1]))
            #if i < len(dims) - 2:
            #    post_mp.append(nn.ReLU())
        self.post_mp = nn.Sequential(*post_mp)


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
    
    def build_norm_layer(self, dim):
        if self.norm_type == 'batchnorm':
            return nn.BatchNorm1d(dim)
        elif self.norm_type == 'layernrom':
            return nn.LayerNorm(dim)
        else:
            return nn.Identity()
        
    def forward(self, x, edge_index, edge_attr=None, batch=None, return_attention=False):
        # Handle undirected graphs by adding reverse edges
        #if not self.directed:
        #    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # Add reverse edges
        #    if self.edge_attr:
        #        edge_attr = torch.cat([edge_attr, edge_attr], dim=0)  # Duplicate edge_attr for reverse edges

        # Pass through the GNN layers
        #print(edge_index.shape, edge_attr.shape)
        #edge_attentions = []  # Store edge attention scores
        #self.embeddings.append(x.detach().cpu().numpy())

        if self.edge_attr:
            #x, attn = self.conv1(x, edge_index, edge_attr, return_attention_weights=True)
            x = self.conv1(x, edge_index, edge_attr) # return_attention_weights=True
        else:
            x = self.conv1(x, edge_index)

        #edge_attentions.append(attn)  # Store first-layer attention

        emb = x
        x = F.relu(x)
        #self.embeddings.append(x.detach().cpu().numpy())
        x = self.norm1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        for i in range(self.num_layers):
            if self.edge_attr:
                #x, attn = self.convs[i](x, edge_index, edge_attr, return_attention_weights=True)
                x = self.convs[i](x, edge_index, edge_attr)
            else:
                x = self.convs[i](x, edge_index)

            #edge_attentions.append(attn)  # Store attention for each layer
            emb = x
            x = F.relu(x)
            #self.embeddings.append(x.detach().cpu().numpy())
            x = self.norms[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        logits = self.post_mp(x)
        #if return_attention:
        #    return edge_attentions  # Return attention coefficients
        
        #return emb, self.embeddings, F.log_softmax(x, dim=1)
        return emb, self.embeddings, logits

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

def get_document_embeddings_batch(texts, tokenizer, model, device, batch_size=128):
    model.to(device)
    model.eval()
    all_cls_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting doc embeddings"):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            #cls_embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
            all_cls_embeddings.extend(cls_embeddings.cpu().numpy())
    return all_cls_embeddings

def get_word_embeddings(word, tokenizer, language_model, device):
    # Tokenize the word and convert to tensor
    inputs = tokenizer(word, return_tensors='pt', truncation=True, padding=True).to(device)
    # Get BERT embeddings
    with torch.no_grad():
        outputs = language_model(**inputs)
    # Use the embeddings from the last hidden state (CLS token or average pooling)
    word_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return word_embedding

def get_word_doc_embeddings(corpus, tokenizer, language_model, vocab, device, not_found_tokens='avg', batch_size=32):
    embeddings_word_dict = {}
    token_freq = defaultdict(int)
    doc_embeddings_list = []
    language_model.to(device)
    language_model.eval()

    for batch_start in tqdm(range(0, len(corpus), batch_size), desc="Extracting doc-word embeddings in batches"):
        batch_texts = corpus[batch_start:batch_start + batch_size]

        encoded_batch = tokenizer.batch_encode_plus(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True
        ).to(device)

        with torch.no_grad():
            outputs = language_model(**encoded_batch, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]  # shape: [batch_size, seq_len, hidden_dim]

        input_ids = encoded_batch["input_ids"]

        # Document embeddings from CLS
        cls_embeddings = hidden_states[:, 0, :].cpu().numpy()  # [batch_size, hidden_dim]
        doc_embeddings_list.extend(cls_embeddings)

        for i in range(hidden_states.size(0)):  # loop through each document in the batch
            doc_embeddings = hidden_states[i]  # shape: [seq_len, hidden_dim]
            doc_token_ids = input_ids[i]

            raw_tokens = [tokenizer.decode([token_id]) for token_id in doc_token_ids]

            for token, embedding in zip(raw_tokens, doc_embeddings):
                token = token.strip()
                token_freq[token] += 1
                current_emb = embedding.cpu().numpy()

                if token not in embeddings_word_dict:
                    embeddings_word_dict[token] = current_emb
                else:
                    #embeddings_word_dict[token] += current_emb
                    embeddings_word_dict[token] = np.add.reduce([embeddings_word_dict[token], current_emb])

    # Average embeddings by frequency
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
    return doc_embeddings_list, word_embeddings

def get_word_embeddings_batch(vocab, tokenizer, model, device, batch_size=128):
    model.to(device)
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(vocab), batch_size), desc="Extracting word embeddings"):
            words = vocab[i:i + batch_size]
            inputs = tokenizer(words, return_tensors='pt', padding=True, truncation=True).to(device)
            outputs = model(**inputs)
            cls = outputs.last_hidden_state[:, 0, :]
            embeddings.extend(cls.cpu().numpy())
    return embeddings

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
    reduced_x_list = []

    with torch.no_grad():  # <--- disables autograd tracking
        for i in range(0, data_x.shape[0], batch_size):
            batch = data_x[i:i + batch_size].to(device)
            reduced_batch = reduction_layer(batch)
            reduced_x_list.append(reduced_batch.cpu())

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

def log_conf_matrix(y_pred, y_true):
    # Log confusion matrix as image
    cm = confusion_matrix(y_pred, y_true)
    classes = ["0", "1"]
    df_cfm = pd.DataFrame(cm, index = classes, columns = classes)
    plt.figure(figsize = (10,7))
    cfm_plot = sns.heatmap(df_cfm, annot=True, cmap='Blues', fmt='g')
    cfm_plot.figure.savefig(f'{utils.OUTPUT_DIR_PATH}/images/cm.png')
    mlflow.log_artifact(f"{utils.OUTPUT_DIR_PATH}/images/cm.png")

def train_mlp():
    cut_off_dataset = '1-1-1'
    dataset = "coling24" # autext23, semeval24, coling24
    model_name = "microsoft/deberta-v3-base"

    cuda_num = 0
    leaening_rate = 0.0001
    epochs = 200 
    hidden_dim = 100
    num_layers = 3
    dropout = 0.5
    patience = 5

    device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")
    llm_model = AutoModel.from_pretrained(model_name)
    #llm_model_size = 256
    llm_model_size = llm_model.config.hidden_size
    
    nfi_dir = model_name.split("/")[1] # nfi -> llm
    output_dir = f'{test_utils.EXTERNAL_DISK_PATH}hetero_graph'
    file_name_data = f"hetero_data_{dataset}_{cut_off_dataset}perc"
    data = utils.load_data(file_name_data, path=f'{output_dir}/{nfi_dir}/', format_file='.pkl', compress=False)
    print(data)

    node_features = data.x.to(device)
    node_labels = data.y.to(device)
    train_mask = data.train_mask.to(device)
    val_mask = data.val_mask.to(device)
    test_mask = data.test_mask.to(device)
    output_dim = len(set(node_labels.cpu().numpy())) - 1

    mlp_model = MLP(
        in_channels = llm_model_size,     # size of your node features (e.g., 768 for BERT)
        hidden_channels = hidden_dim,     # size of hidden layer
        out_channels = output_dim,        # number of classes (e.g., 2 for GenAI detection)
        num_layers = num_layers,          # total layers (input + hidden + output)
        dropout = dropout,                    # regularization
        norm = 'batch_norm'               # optional (can be batch_norm, layer_norm,  None)
    ).to(device) 

    print(mlp_model)
    early_stopper = EarlyStopper(patience=patience, min_delta=0)
    optimizer = torch.optim.Adam(mlp_model.parameters(), lr=leaening_rate)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        mlp_model.train()
        out = mlp_model(node_features)
        loss = criterion(out[train_mask], node_labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mlp_model.eval()
        with torch.no_grad():
            preds = mlp_model(node_features).argmax(dim=1)

            val_loss = criterion(out[val_mask], node_labels[val_mask])
            val_acc = accuracy_score(node_labels[val_mask].cpu(), preds[val_mask].cpu())
            val_f1 = f1_score(node_labels[val_mask].cpu(), preds[val_mask].cpu(), average='macro')

            test_loss = criterion(out[test_mask], node_labels[test_mask])
            test_acc = accuracy_score(node_labels[test_mask].cpu(), preds[test_mask].cpu())
            test_f1 = f1_score(node_labels[test_mask].cpu(), preds[test_mask].cpu(), average='macro')

        if epoch % 1 == 0:
            print(f"Epoch {epoch:03d}  | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f} ")

        if early_stopper.early_stop(val_loss):
            print('Early stopping due to no improvement!')
            break

def nlp_pipeline(docs: list):
    doc_lst = []
    for nlp_doc in tqdm(nlp.pipe(docs, batch_size=64, n_process=4), total=len(docs), desc="nlp_spacy_docs"):
        doc_lst.append(nlp_doc)
    return doc_lst

def regex_tokenizer(text):
    return re.findall(r'\w+|[^\w\s]', text, re.UNICODE)

def create_vocab_v1(all_texts_norm, stop_words=False, special_chars=False, min_df=1, max_df=1.0, max_features=5000, tokenize_pattern=regex_tokenizer):
    tokenized_corpus = []
    '''for text_norm in tqdm(all_texts_norm, desc="Tokenazing corpus"):
            doc = nlp(text_norm)
            tokens = []
            for token in doc:
                if stop_words and token.is_stop:
                    continue
                if special_chars and token.is_punct:
                    continue
                tokens.append(token.text) # text, lemma_, pos_
            tokenized_corpus.append(tokens)'''

    # Create vocabulary and TF-IDF features
    vectorizer = CountVectorizer(min_df=min_df, max_df=max_df, max_features=max_features, token_pattern=None, tokenizer=regex_tokenizer)
    X = vectorizer.fit_transform(all_texts_norm)
    tfidf_transformer = TfidfTransformer()
    X_tfidf = tfidf_transformer.fit_transform(X)
    vocab = vectorizer.get_feature_names_out()
    word_to_index = {word: idx for idx, word in enumerate(vocab)}
    return vocab, word_to_index, tokenized_corpus, X_tfidf, X

def balance_df(df):
    if 'source' not in df.columns or 'label' not in df.columns:
        raise ValueError("DataFrame must contain 'source' and 'label' columns")

    balanced_parts = []
    for domain, domain_df in df.groupby("source"):
        label_counts = domain_df['label'].value_counts()
        if len(label_counts) < 2:
            balanced_parts.append(domain_df)
            continue

        min_count = min(label_counts[0], label_counts[1])
        df_0 = domain_df[domain_df['label'] == 0].sample(min_count, random_state=42)
        df_1 = domain_df[domain_df['label'] == 1].sample(min_count, random_state=42)
        balanced_parts.append(pd.concat([df_0, df_1]))

    return pd.concat(balanced_parts).sample(frac=1, random_state=42).reset_index(drop=True)

def create_vocab_v2(all_texts_norm, stop_words=False, special_chars=False, min_df=1, max_features=5000):
    docs_nlp_spacy = nlp_pipeline(all_texts_norm)
    vocab = set()
    word_freq = Counter()
    tokenized_corpus = []
    for doc in docs_nlp_spacy:
        tokens = []
        for token in doc:
            if stop_words and token.is_stop:
                continue
            if special_chars and token.is_punct:
                continue
            vocab.add(token.text)
            word_freq[token.text] += 1
            tokens.append(token.text) # text, lemma_, pos_
        tokenized_corpus.append(tokens)
    vocab = list(vocab)
    # Filter words by min frequency
    filtered_vocab = [word for word in vocab if word_freq[word] >= min_df]
    # Sort remaining words by frequency (descending) and take top_k
    filtered_vocab = sorted(filtered_vocab, key=lambda w: word_freq[w], reverse=True)[:max_features]
    # Rebuild word_id_map and vocab_size
    vocab = filtered_vocab
    word_to_index = {w: i for i, w in enumerate(vocab)}
    
    vectorizer = CountVectorizer(vocabulary=word_to_index)
    X = vectorizer.fit_transform(all_texts_norm)
    tfidf_transformer = TfidfTransformer()
    X_tfidf = tfidf_transformer.fit_transform(X)
 
    return vocab, word_to_index, tokenized_corpus, X_tfidf, X

def build_projector(input_dim=768, hidden_dim=512, output_dim=256, dropout=0.1, num_layers=2, use_norm=True, activation='relu'):
    layers = []
    dim_in = input_dim

    for i in range(num_layers):
        dim_out = output_dim if i == num_layers - 1 else hidden_dim
        layers.append(nn.Linear(dim_in, dim_out))

        if i < num_layers - 1:
            if use_norm:
                layers.append(nn.BatchNorm1d(dim_out))  # or nn.BatchNorm1d
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(negative_slope=0.1))
            layers.append(nn.Dropout(dropout))

        dim_in = dim_out
    return nn.Sequential(*layers)

# ************************************************************ TMP DELETE
def load_corpus(output_path, dataset_str):
    names = ['vocab', 'y', 'valy', 'ty', 'node_labels', 'adj']
    objects = []
    for i in range(len(names)):
        with open(output_path + "ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    vocab, y, valy, ty, node_labels, adj = tuple(objects)
    train_size = y.shape[0]
    val_size = valy.shape[0]
    test_size = ty.shape[0]
    num_nodes = len(node_labels)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[ : train_size] = True
    val_mask[train_size : train_size + val_size] = True
    test_mask[train_size + val_size : train_size + val_size + test_size] = True
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    return vocab, adj, node_labels, train_mask, val_mask, test_mask, train_size, test_size, val_size

# ************************************************************ TMP DELETE
def normalize_adj(adj):
    row_sum = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(row_sum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

# ************************************************************ TMP DELETE
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



def main():    
    # autext23,     100-100-100
    # semeval24,    10-25-25
    # coling24,     5-10-10
    # autext23_s2, 
    # semeval24_s2
    config = {
        'build_graph': False,
        'dataset_name': 'autext24', # autext23, autext24, semeval24, coling24, autext23_s2, semeval24_s2
        'cut_off_dataset': '100-100-100', # train-val-test
        "nfi": 'llm', # llm, w2v, random
        'cuda_num': 0,

        'window_size': 10,
        'graph_direction': 'undirected', # undirected | directed (for now all are undirecte, pending to handle and review to support both)
        'special_chars': False,
        'stop_words': False,
        'min_df': 5, # 1->autext | 5->semeval | 5-coling
        'max_df': 1.0,
        'max_features': 15000, # None -> all | 5000
        'not_found_tokens': 'avg', # avg, remove, zeros, ones
        'add_edge_attr': True,
        'add_graph_metric': False,
        'embed_reduction': False, # False -> 768
        'reduce_dim_to': 256, # 128, 256 

        "gnn_type": 'TransformerConv', # GCNConv, GATConv, TransformerConv
        "dropout": 0.5,
        "patience":10, # 5-autext23 | 10-semeval | 10-coling
        "learnin_rate": 0.0002, # autext23 -> llm: 0.00002 | semeval -> llm: 0.000005  | coling -> llm: 0.0001 
        "batch_size": 512 * 1,
        "hidden_dim": 100, # 300 autext_s2, 100 others
        "dense_hidden_dim": 64, # 64-autext23 | 32-semeval | 64-coling
        "num_layers": 1,
        "heads": 1,
        "norm_type": None,   # 'batchnorm', 'layernorm', or None
        "post_mp_layers": 2,        # number of layers after message passing

        "weight_decay": 5e-4, 
        "num_neighbors": [100, 75],  # Autext -> [100, 75]
        "output_dim": 2, # 2-bin | 6-multi 
        'input_dim': 768, # setting below
        'epochs': 200,

        ## intfloat/multilingual-e5-large
        ## google-bert/bert-base-multilingual-uncased
        ## FacebookAI/xlm-roberta-base
        ## microsoft/deberta-v3-base
        "llm_name": 'FacebookAI/roberta-base',
        'leave_out_sources': True, # True, False

    }

    mlflow.set_experiment(f"GNN - HETERO")
    run_description = f"""Run experiment for GNN Classification Task using dataset {config['dataset_name']} with {config['cut_off_dataset']} % cutoff."""
    run_tags = {
        'mlflow.note.content': run_description,
        'mlflow.source.type': "LOCAL"
    }

    with mlflow.start_run(tags=run_tags):
        mlflow.set_tag("mlflow.runName", f"run_{config['dataset_name']}_{config['cut_off_dataset']}perc")

        if config['leave_out_sources']:
            if config['dataset_name'] == 'autext23':
                # Autext: ["wiki", "tweets", "legal"]
                config['leave_out_sources'] = ["tweets"] 
            elif config['dataset_name'] == 'autext24':
                # Autext: ["literary", "news", "reviews", "tweets", "wikipedia"]
                config['leave_out_sources'] = ["news", "literary"] 
            elif config['dataset_name'] == 'semeval24':
                # Semeval ["arxiv", "peerread", "reddit", "wikihow", "wikipedia"]
                config['leave_out_sources'] = ["wikihow", "wikipedia"]
            elif config['dataset_name'] == 'coling24':
                # Coling: ["hc3", "m4gt", "mage"]
                config['leave_out_sources'] = ["mage"]

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
            start_time = time.time()
            # Load and preprocess dataset
            train_text_set, val_text_set, test_text_set = test_utils.read_dataset(config['dataset_name'])

            if config['dataset_name'] == 'autext24':
                train_text_set = train_text_set[train_text_set['language'] == 'en'].reset_index(drop=True)
                val_text_set = val_text_set[val_text_set['language'] == 'en'].reset_index(drop=True)
                test_text_set = test_text_set[test_text_set['language'] == 'en']

            if config['leave_out_sources']:
                print(f"[INFO] Leave-One-Source-Out setting: removing '{config['leave_out_sources']}' from train.")
                #train_text_set = train_text_set[train_text_set['source'] != leave_out_source]
                #val_text_set = val_text_set[val_text_set['source'] == leave_out_source]
                
                # Split train set: keep everything NOT in leave_out_sources, swap the rest
                train_keep = train_text_set[~train_text_set['source'].isin(config['leave_out_sources'])]
                train_swap = train_text_set[train_text_set['source'].isin(config['leave_out_sources'])]

                # Split val set: keep only leave_out_sources, swap the rest
                val_keep = val_text_set[val_text_set['source'].isin(config['leave_out_sources'])]
                val_swap = val_text_set[~val_text_set['source'].isin(config['leave_out_sources'])]

                # Combine to form new sets
                #train_text_set = train_keep
                train_text_set = pd.concat([train_keep, val_swap], ignore_index=True)
                #val_text_set = val_keep
                val_text_set = pd.concat([val_keep, train_swap], ignore_index=True)
                
                print("Train set distro:\n", train_text_set.groupby("source")["label"].value_counts())
                print("Val set distro:\n", val_text_set.groupby("source")["label"].value_counts())

            # Cut off datasets
            cut_off_train = int(config['cut_off_dataset'].split('-')[0])
            cut_off_val = int(config['cut_off_dataset'].split('-')[1])
            cut_off_test = int(config['cut_off_dataset'].split('-')[2])

            train_set = train_text_set[:int(len(train_text_set) * (cut_off_train / 100))][:]
            val_set = val_text_set[:int(len(val_text_set) * (cut_off_val / 100))][:]
            test_set = test_text_set[:int(len(test_text_set) * (cut_off_test / 100))][:]

            #train_set = balance_df(train_set)
            #val_set = balance_df(val_set)
            print("distro_train_val_test: ", len(train_set), len(val_set), len(test_set))
            print("label_distro_train_val_test: ", train_set.value_counts('label'), val_set.value_counts('label'), test_set.value_counts('label'))
            print("Label distribution per source in Train set:\n", train_set.groupby("source")["label"].value_counts())
            print("Label distribution per source in Validation set:\n", val_set.groupby("source")["label"].value_counts())
            print("Label distribution per source in Test set:\n", test_set.groupby("source")["label"].value_counts())
            
            # Combine texts and labels
            all_texts = list(train_set['text']) + list(val_set['text']) + list(test_set['text'])
            all_labels = list(train_set['label']) + list(val_set['label']) + list(test_set['label'])

            # Normalize texts and Tokenize texts
            tokenized_corpus = []
            all_texts_norm = []
            for text in tqdm(all_texts, desc="Normalizing corpus"):
                text_norm = test_utils.text_normalize(text, special_chars=config['special_chars'], stop_words=config['stop_words'])
                all_texts_norm.append(text_norm)
                #tokenized_corpus.append(re.findall(r'\w+|[^\w\s]', text, re.UNICODE))

            # create vocab V1
            #vocab, word_to_index, _, X_tfidf, X = create_vocab_v1(all_texts_norm, stop_words=config['stop_words'], special_chars=config['special_chars'], min_df=config['min_df'], max_features=config['max_features'], tokenize_pattern=regex_tokenizer)

            # create vocab V2
            vocab, word_to_index, tokenized_corpus, X_tfidf, X = create_vocab_v2(all_texts_norm, stop_words=config['stop_words'], special_chars=config['special_chars'], min_df=config['min_df'], max_features=config['max_features'])

            print('vocab: ', len(vocab))
            with open(f"{output_dir}/{config['dataset_name']}_{config['cut_off_dataset']}.vocab.txt", 'w') as file:
                for item in vocab:
                    file.write(str(item) + '\n')

            # *** Generate embeddings method 2  - LLM 
            if config['nfi'] == 'llm':
                # Generate word and document embeddings
                tokenizer = AutoTokenizer.from_pretrained(config['llm_name'], model_max_length=512)
                language_model = AutoModel.from_pretrained(config['llm_name'], output_hidden_states=True).to(device)
                config['input_dim'] = language_model.config.hidden_size

                # ******************** Generate word embeddings
                # ********** batches
                doc_features, word_features = get_word_doc_embeddings(all_texts_norm, tokenizer, language_model, vocab, device, not_found_tokens=config['not_found_tokens'], batch_size=32)
                word_features = torch.tensor(word_features, dtype=torch.float)
                doc_features = torch.tensor(doc_features, dtype=torch.float)

                # ********** zeros init
                #word_features = torch.zeros((len(vocab), config['input_dim']))
                
                # ********** batches - simple
                #word_features = get_word_embeddings_batch(list(vocab), tokenizer, language_model, device, batch_size=128)
                #word_features = torch.tensor(word_features, dtype=torch.float)

                # ******************** Generate document embeddings
                # ********** one by one
                #doc_features = []
                #for text in tqdm(all_texts_norm, desc="Extracting doc embeddings"):
                #    doc_embedding = get_document_embeddings(text, tokenizer, language_model, device)
                #    doc_features.append(doc_embedding)
                #doc_features = torch.tensor(doc_features, dtype=torch.float)
                
                # ********** batches
                #doc_features = get_document_embeddings_batch(all_texts_norm, tokenizer, language_model, device, batch_size=32)
                #doc_features = torch.tensor(doc_features, dtype=torch.float)

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


            # **** Create document-word edges
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

            # Apply dimensionality reduction to train, val, and test data
            if config['embed_reduction']:    
                #embedding_reduction = nn.Linear(768, config['reduce_dim_to']).to(device)
                embedding_reduction = build_projector(input_dim=768, hidden_dim=512, output_dim=config['reduce_dim_to'], dropout=0.1, num_layers=1, use_norm=False, activation='gelu').to(device)

                node_features = reduce_dimension_linear(node_features, embedding_reduction, device, batch_size=128)

            # Create directed and undirected edge indices
            directed_edge_index = edges
            undirected_edge_index = torch.cat([edges, edges.flip(0)], dim=1)  # Add reverse edges

            # Removes duplicate edges (CHECK)
            #directed_edge_index = torch.unique(directed_edge_index, dim=1)  
            #undirected_edge_index = torch.unique(undirected_edge_index, dim=1)  

            #print(len(doc_word_attr), len(word_word_attr), len(edge_attr))
            #print(len(doc_word_edges), len(word_word_edges), edges.shape)
            #print(undirected_edge_index.shape, directed_edge_index.shape)

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

            num_docs = len(train_set) + len(val_set) + len(test_set)

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
                y = node_labels,
                num_docs = num_docs,
                #vocab = vocab,
                #word_to_index = word_to_index,
                #index_to_word = index_to_word,
                #all_texts_norm = all_texts_norm
            )

            # add/remove attr from Data object (based on the above config, this reduce the traning time)
            if config['graph_direction'] == 'undirected':
                data.edge_index = data.undirected_edge_index
                if config['add_edge_attr']:
                    data.edge_attr = torch.cat([data.edge_attr, data.edge_attr], dim=0)  # Duplicate edge_attr for reverse edges
            else:
                data.edge_index = data.directed_edge_index
            del data.directed_edge_index
            del data.undirected_edge_index

            # Save the data object
            #utils.save_data(data, file_name_data, path=output_dir, format_file='.pkl', compress=False)
            utils.save_data(data, file_name_data, path=f'{output_dir}/{nfi_dir}/', format_file='.pkl', compress=False)
            print("--- %s Graph Built Time: ---" % (time.time() - start_time))
        else:
            # Load the data object
            #data = utils.load_data(file_name_data, path=output_dir, format_file='.pkl', compress=False)
            data = utils.load_data(file_name_data, path=f'{output_dir}/{nfi_dir}/', format_file='.pkl', compress=False)

            '''
            # **************** TMP DELETE
            out_p = f'/home/avaldez/projects/TrainingGNN/outputs/'
            vocab, adj, node_labels, train_mask, val_mask, test_mask, train_size, test_size, val_size = load_corpus(out_p, config['dataset_name'])
        
            node_features = utils.load_data(
                f'node_features_{config["dataset_name"]}',
                path=out_p, 
                format_file='.pkl', compress=False)
            adj = normalize_adj(adj + sp.eye(adj.shape[0]))
            edge_index, edge_weight = from_scipy_sparse_matrix(adj)
            edge_weight = edge_weight.to(torch.float32).to(device)
            edge_index = edge_index.to(torch.long).to(device)
            data.train_mask = train_mask 
            data.val_mask = val_mask
            data.test_mask = test_mask
            data.x = node_features
            data.edge_index = edge_index
            data.edge_attr = edge_weight
            data.y = node_labels
            '''
            #print(data)
        

        #embedding_reduction = nn.Linear(768, 128).to(device)
        #data.x = reduce_dimension_linear(data.x, embedding_reduction, device, , batch_size=128)
        #utils.save_data(data, file_name_data, path=output_dir, format_file='.pkl', compress=False)
        #return

        for k, v in config.items():
            mlflow.log_param(k, v)
            
        if config['add_graph_metric']:
            data.x = torch.cat([data.x, data.graph_metrics], dim=-1)
        else:
            del data.graph_metrics

        if not config['add_edge_attr']:
            del data.edge_attr

        del data.vocab

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
            num_workers=2,
            persistent_workers=False  # Caches samples for repeated training 
        )

        val_loader = NeighborLoader(
            data,
            num_neighbors=config['num_neighbors'], 
            batch_size=config['batch_size'],       
            input_nodes=val_nodes,
            shuffle=True,
            replace=False,
            num_workers=2,
            persistent_workers=False  # Caches samples for repeated training         
        )

        test_loader = NeighborLoader(
            data,
            num_neighbors=config['num_neighbors'], 
            batch_size=config['batch_size'],       
            input_nodes=test_nodes,
            shuffle=True,
            replace=False,
            num_workers=2,
            persistent_workers=False  # Caches samples for repeated training 
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
            directed=(config['graph_direction'] == 'directed'),
            norm_type=config['norm_type'], 
            post_mp_layers=config['post_mp_layers'],   
        )

        mlflow.log_param('model_params', str(model))
        model = model.to(device)
        print(model)

        # Training loop
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learnin_rate'], weight_decay=config['weight_decay'])
        # Add learning rate scheduler
        #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

        criterion = torch.nn.CrossEntropyLoss()
        early_stopper = EarlyStopper(patience=config['patience'], min_delta=0)

        logger.info("Init GNN training!")
        start_time = time.time()

        best_test_acc_score = 0
        best_test_f1_score = 0
        stop_epoch = 0
        for epoch in range(config['epochs']):
            #with torch.cuda.amp.autocast():  # Use FP16 for faster computation
            loss_train = train(model, train_loader, optimizer, criterion, device)
            val_acc, val_f1_macro, val_loss, _, _ = evaluate(model, val_loader, criterion, 'val', device)
            
            if epoch % 1 == 0:
                test_acc, test_f1_macro, test_loss, _, _  = evaluate(model, test_loader, criterion, 'test', device)
                print(f'Ep {epoch + 1}, Loss Val: {val_loss:.4f}, Loss Test: {test_loss:.4f}, '
                f'Val Acc: {val_acc:.4f}, Val F1s: {val_f1_macro:.4f}, Test Acc: {test_acc:.4f}, Test F1s: {test_f1_macro:.4f}')
            else:
                print(f'Ep {epoch + 1}, Loss Val: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1s: {val_f1_macro:.4f}')
                
            # Get the current learning rate
            current_lr = optimizer.param_groups[0]['lr']

            #print("preds_test: ", sorted(Counter(preds_test).items()))
            #print("label_test: ", sorted(Counter(labels_test).items()))
            
            mlflow.log_metric(key=f"F1Score-val", value=float(val_f1_macro), step=epoch)
            mlflow.log_metric(key=f"Accuracy-val", value=float(val_acc), step=epoch)
            mlflow.log_metric(key=f"Loss-val", value=float(val_loss), step=epoch)
            mlflow.log_metric(key=f"F1Score-test", value=float(test_f1_macro), step=epoch)
            mlflow.log_metric(key=f"Accuracy-test", value=float(test_acc), step=epoch)
            mlflow.log_metric(key=f"Loss-test", value=float(test_loss), step=epoch)

            if test_acc > best_test_acc_score:
                best_test_acc_score = test_acc
            if test_f1_macro > best_test_f1_score:
                best_test_f1_score = test_f1_macro

            stop_epoch = epoch
            #scheduler.step(val_loss)  # Update learning rate
            if early_stopper.early_stop(val_loss):
                print('Early stopping due to no improvement!')
                break

            torch.cuda.empty_cache()
        
        logger.info("Done GNN training!")
        print("--- %s Graph Training Time ---" % (time.time() - start_time))

        # Evaluate on test set
        test_acc, test_f1_macro, test_loss, preds_test, labels_test = evaluate(model, test_loader, criterion, 'test', device)
        print()
        print(f'Test Accuracy: {test_acc:.4f}')
        print(f'Test F1Score: {test_f1_macro:.4f}')
        print(f'Test Loss: {test_loss:.4f}')

        print("preds_test:  ", sorted(Counter(preds_test).items()))
        print("labels_test: ", sorted(Counter(labels_test).items()))
        cm = confusion_matrix(preds_test, labels_test)
        print(cm)

        log_conf_matrix(preds_test, labels_test)
        mlflow.log_metric(key=f"num_epochs", value=int(config['epochs']))
        mlflow.log_metric(key=f"stop_epoch", value=int(stop_epoch))
        mlflow.log_metric(key=f"Final-F1Macro-val", value=float(val_f1_macro))
        mlflow.log_metric(key=f"Final-Accuracy-test", value=float(val_acc))
        mlflow.log_metric(key=f"Final-Loss-test", value=float(val_loss))
        mlflow.log_metric(key=f"Final-F1Macro-test", value=float(test_f1_macro))
        mlflow.log_metric(key=f"Final-Accuracy-test", value=float(test_acc))
        mlflow.log_metric(key=f"Final-Loss-test", value=float(test_loss))
        mlflow.log_metric(key=f"Best-Accuracy-test", value=float(best_test_acc_score))
        mlflow.log_metric(key=f"Best-F1Macro-test", value=float(best_test_f1_score))

        model_save_path = f"{output_dir}/models/gnn_model_{nfi_dir}_{file_name_data}.pth"
        
        # save model state dict (weights, layers, etc)
        model_checkpoint = {
            'model_state_dict': model.state_dict(),
            'input_dim': config['input_dim'],
            'hidden_dim': config['hidden_dim'],
            'dense_hidden_dim': config['dense_hidden_dim'],
            'output_dim': config['output_dim'],
            'dropout': config['dropout'],
            'num_layers': config['num_layers'],
            'edge_attr': config['add_edge_attr'],
            'gnn_type': config['gnn_type'],
            'heads': config['heads'],
            'task': 'node',
            'graph_direction': config['graph_direction'],
        }
        torch.save(model_checkpoint, model_save_path)

        # save entire model (this save hyperparam, architecture and model params)
        #torch.save(model, model_save_path)
        
        print(f"Model saved at: {model_save_path}")

        return

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



# **************** GNN Explicability
def create_node_masks(data, num_docs):
    
    num_nodes = data.x.shape[0]  # Total number of nodes
    num_words = num_nodes - num_docs  # Compute the number of word nodes dynamically

    # Create masks
    doc_mask = torch.zeros(num_nodes, dtype=torch.bool)
    word_mask = torch.zeros(num_nodes, dtype=torch.bool)

    # Assign document nodes (first num_docs)
    doc_mask[:num_docs] = True

    # Assign word nodes (remaining nodes after documents)
    word_mask[num_docs:] = True
    print(doc_mask, word_mask)
    return doc_mask, word_mask


def print_top_words_by_class(class_word_score_list, num_top_words=10):
    """
    Prints the most frequently attended words with average attention score.
    """
    print("\n Top Influential Words per Class (with average attention):")
    for label, name in zip([0, 1], ["Human", "Machine"]):
        print(f"\n Class: {name} (label={label})")

        # Aggregate count and attention scores
        word_counter = Counter()
        word_scores = defaultdict(list)
        for word, score in class_word_score_list[label]:
            word_counter[word] += 1
            word_scores[word].append(score)

        for word, count in word_counter.most_common(num_top_words):
            avg_score = sum(word_scores[word]) / len(word_scores[word])
            print(f"   {word:<20} → in {count:3d} docs | avg attention: {avg_score:.4f}")


def get_doc_word_attention_undirected(model, data, num_docs, layer=0):
    """
    From undirected edge attention, extract attention scores for document-word pairs.
    """
    model.eval()
    with torch.no_grad():
        attn_layers = model(data.x, data.edge_index, data.edge_attr, return_attention=True)

    attn_data = {
        'edge_index': attn_layers[layer][0].cpu().numpy(),
        'attention': attn_layers[layer][1].cpu().numpy().flatten()
    }

    doc_word_attention = defaultdict(list)
    sources, targets = attn_data['edge_index']
    attentions = attn_data['attention']

    for src, tgt, score in zip(sources, targets, attentions):
        if (src < num_docs and tgt >= num_docs):  # doc -- word
            doc_id, word_id = src, tgt
        elif (tgt < num_docs and src >= num_docs):  # word -- doc
            doc_id, word_id = tgt, src
        else:
            continue  # skip word-word or doc-doc edges
        doc_word_attention[doc_id].append((word_id, score))

    return doc_word_attention


def aggregate_top_words_by_class(doc_word_attention, doc_labels, vocab, num_docs, top_k=5):
    """
    Collect top-k attended words (with scores) per class label.
    Returns: class_word_counter[label] = list of (word, attention_score)
    """
    word_id_to_token = {idx + num_docs: token for idx, token in enumerate(vocab)}
    class_word_score_list = {0: [], 1: []}

    for doc_id, word_scores in doc_word_attention.items():
        label = doc_labels[doc_id]
        top_words = sorted(word_scores, key=lambda x: x[1], reverse=True)[:top_k]
        for word_id, score in top_words:
            token = word_id_to_token.get(word_id, f"<W{word_id}>")
            class_word_score_list[label].append((token, float(score)))

    return class_word_score_list


def analyze_attention_top_words(doc_word_attention, data, top_k=10, num_top_words=10):
    doc_labels = data.y[:data.num_docs].cpu().numpy()
    class_word_counter = aggregate_top_words_by_class(doc_word_attention, doc_labels, data.vocab, data.num_docs, top_k)
    print_top_words_by_class(class_word_counter, num_top_words)
    return class_word_counter


def get_top_k_attended_words_per_class(doc_word_attention, labels, vocab, num_docs, top_k_per_doc=5, top_k_final=10):
    word_id_to_token = {i + num_docs: token for i, token in enumerate(vocab)}
    class_top_attended = {0: [], 1: []}  # list of (token, attn_score)

    for doc_id, word_attn in doc_word_attention.items():
        label = labels[doc_id]
        top_words = sorted(word_attn, key=lambda x: x[1], reverse=True)[:top_k_per_doc]
        for word_id, attn_score in top_words:
            token = word_id_to_token.get(word_id, f"<W{word_id}>")
            class_top_attended[label].append((token, attn_score))

    # Now select top-k by attention score (not averaging!)
    final_top_k = {}
    for label in [0, 1]:
        seen = set()
        sorted_by_score = sorted(class_top_attended[label], key=lambda x: x[1], reverse=True)
        top_unique = []
        for word, score in sorted_by_score:
            if word not in seen:
                top_unique.append((word, score))
                seen.add(word)
            if len(top_unique) == top_k_final:
                break
        final_top_k[label] = top_unique 

    return final_top_k


def sample_docs_by_class(data, labels, class_id=0, n=10):
    doc_indices = torch.nonzero(data.y[:data.num_docs] == class_id, as_tuple=True)[0]
    sampled = doc_indices[torch.randperm(len(doc_indices))[:n]]
    return sampled.cpu().numpy()


def extract_doc_word_attention(attn_layer, num_docs, doc_ids):
    edge_index = attn_layer[0].cpu().numpy()
    attention = attn_layer[1].cpu().numpy().flatten()
    
    doc_word_attn = defaultdict(list)
    src, tgt = edge_index

    for s, t, a in zip(src, tgt, attention):
        if s in doc_ids and t >= num_docs:
            doc_word_attn[s].append((t, a))
        elif t in doc_ids and s >= num_docs:
            doc_word_attn[t].append((s, a))
    return doc_word_attn


def plot_word_saliency(doc_id, original_text, word_attention, word_id_offset, vocab, top_k=5, class_label=None):
    word_scores = [(wid - word_id_offset, attn) for wid, attn in word_attention]
    word_scores = [(vocab[i], attn) for i, attn in word_scores if 0 <= i < len(vocab)]
    word_scores = sorted(word_scores, key=lambda x: x[1], reverse=True)[:top_k]

    words, attns = zip(*word_scores) if word_scores else ([], [])

    fig, ax = plt.subplots(figsize=(min(12, 2 + 0.5 * len(words)), 1.5))
    bars = ax.barh(range(len(words)), attns, color='crimson', alpha=0.8)
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words)
    ax.invert_yaxis()
    ax.set_title(f"Doc #{doc_id} — Class: {class_label}", fontsize=12)
    ax.set_xlabel("Attention Coefficient")
    plt.tight_layout()
    plt.show()

    # Save figure
    save_path = os.path.join(test_utils.OUTPUT_DIR_PATH, f"doc_{doc_id}_class_{class_label}.png")
    plt.savefig(save_path)
    plt.close()


def render_html_saliency(doc_id, text, word_attention, word_id_offset, vocab, top_k=5, class_label=None, save_dir="saliency_html"):
    #os.makedirs(save_dir, exist_ok=True)

    word_scores = [(wid - word_id_offset, attn) for wid, attn in word_attention]
    word_scores = [(vocab[i], attn) for i, attn in word_scores if 0 <= i < len(vocab)]
    word_scores = sorted(word_scores, key=lambda x: x[1], reverse=True)[:top_k]

    if not word_scores:
        return

    word_attn_dict = {word.lower(): attn for word, attn in word_scores}
    max_attn = max(word_attn_dict.values())

    def colorize(word):
        attn = word_attn_dict.get(word.lower(), 0)
        opacity = min(1.0, attn / max_attn) if max_attn > 0 else 0
        return f"<span style='background-color: rgba(255, 0, 0, {opacity:.2f}); padding:1px'>{word}</span>"

    tokens = re.findall(r'\w+|\W+', text)
    highlighted = ''.join([colorize(w) if w.lower() in word_attn_dict else w for w in tokens])

    html = f"""
    <h3>📄 Document #{doc_id} — Class: {class_label}</h3>
    <p><strong>Top-{top_k} attended words:</strong></p>
    <ul>
        {''.join(f"<li>{w}: {s:.4f}</li>" for w, s in word_scores)}
    </ul>
    <p style="font-family:monospace; line-height:1.5;">{highlighted}</p>
    <hr/>
    """

    file_path = os.path.join(test_utils.OUTPUT_DIR_PATH, f"doc_{doc_id}_class_{class_label}.html")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"📄 Saved: {file_path}")


def gnn_explicability():
    dataset_name = 'semeval24' # autext23, semeval24, coling24, autext23_s2, semeval24_s2
    cut_off_dataset = '5-5-5' # train-val-test
    llm_name = 'microsoft/deberta-v3-base'
    file_name_data = f"hetero_data_{dataset_name}_{cut_off_dataset}perc" # perc_128
    output_dir = f'{test_utils.EXTERNAL_DISK_PATH}hetero_graph'
    nfi_dir = llm_name.split("/")[1] # nfi -> llm

    # Load the entire model
    model_save_path = f"{output_dir}/models/gnn_model_{nfi_dir}_{file_name_data}.pth"
    checkpoint = torch.load(model_save_path, map_location=torch.device('cpu'))
    model = GNN(
        checkpoint['input_dim'],
        checkpoint['hidden_dim'],
        checkpoint['dense_hidden_dim'],
        checkpoint['output_dim'],
        checkpoint['dropout'],
        checkpoint['num_layers'],
        edge_attr=checkpoint['edge_attr'],
        gnn_type=checkpoint['gnn_type'],
        heads=checkpoint['heads'],
        task=checkpoint['task'],
        directed=(checkpoint['graph_direction'] == 'directed')
    )

    # Load the model's state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    print(model)

    # load Data object
    data = utils.load_data(file_name_data, path=f'{output_dir}/{nfi_dir}/', format_file='.pkl', compress=False)
    data.doc_mask, data.word_mask = create_node_masks(data, data['num_docs'])
    #print(data)

    #doc_word_attention = get_doc_word_attention_undirected(model, data, num_docs=data.num_docs, layer=1)
    #utils.save_plain_text(doc_word_attention, test_utils.OUTPUT_DIR_PATH + 'data.txt')

    # After training and having `model`, `data`, `vocab`
    '''
    top_words = analyze_attention_top_words(
        doc_word_attention=doc_word_attention,
        data=data,
        top_k=10,
        num_top_words=10,
    )
    '''
    # get top k attended words
    '''
    final_top_k = get_top_k_attended_words_per_class(
        doc_word_attention, 
        data.y[:data.num_docs].cpu().numpy(), 
        data.vocab, 
        data.num_docs, 
        top_k_per_doc=10, 
        top_k_final=10
    )
    print("\n Top-K Highest Attention Words per Class:")
    for label, name in zip([0, 1], ["Human", "Machine"]):
        print(f"\n🔹 Class: {name} (label={label})")
        for word, score in final_top_k[label]:
            print(f"   {word:<20} → attention score: {score:.4f}")
    '''

    # plot top_k words per class (png image)
    '''
    doc_ids = sample_docs_by_class(data, data.y, class_id=0, n=10)
    for doc_id in doc_ids:
        text = data.all_texts_norm[doc_id]
        word_attn = doc_word_attention.get(doc_id, [])
        plot_word_saliency(doc_id, text, word_attn, word_id_offset=data.num_docs, vocab=data.vocab, top_k=10, class_label=0)
    doc_ids = sample_docs_by_class(data, data.y, class_id=1, n=10)
    for doc_id in doc_ids:
        text = data.all_texts_norm[doc_id]
        word_attn = doc_word_attention.get(doc_id, [])
        plot_word_saliency(doc_id, text, word_attn, word_id_offset=data.num_docs, vocab=data.vocab, top_k=10, class_label=1)
    '''

    #generate html
    '''
    doc_ids = sample_docs_by_class(data, data.y, class_id=0, n=10)    
    for doc_id in doc_ids:
        text = data.all_texts_norm[doc_id]
        word_attn = doc_word_attention.get(doc_id, [])
        render_html_saliency(doc_id, text, word_attn, word_id_offset=data.num_docs, vocab=data.vocab, top_k=10, class_label=0, save_dir='')
    '''

    # get top-k word-score per document
    cnt = 0
    doc_labels = data.y[:data.num_docs].cpu().numpy()
    word_id_to_token = {idx + data.num_docs: token for idx, token in enumerate(data.vocab)}
    for doc_id, word_scores in doc_word_attention.items():
        print(f"\nDocument {doc_id} ({doc_labels[doc_id]}):")
        top_words = sorted(word_scores, key=lambda x: x[1], reverse=True)[:5]
        for word_id, attn in top_words:            
            word = word_id_to_token.get(word_id, f"<W{word_id}>")
            print(f"  {word} ({attn:.4f})")
        
        if cnt == 10:
            break
        cnt += 1


    topk_per_doc = []
    top_k = 10
    for doc_id, word_scores in doc_word_attention.items():
        #print(f"\nDocument {doc_id} ({doc_labels[doc_id]}):")
        sorted_top = sorted(word_scores, key=lambda x: x[1], reverse=True)[:top_k]
        entry = {
            "doc_id": int(doc_id),
            "label": int(data.y[doc_id].item()),
            "top_words": [
                {"word": str(word_id_to_token.get(w)), "attention": float(a)}
                for w, a in sorted_top
            ]
        }
        topk_per_doc.append(entry)
        
    utils.save_json(topk_per_doc, test_utils.OUTPUT_DIR_PATH + dataset_name + '_data.json')


    
if __name__ == '__main__':
    main()
    #train_mlp()
    #gnn_explicability()


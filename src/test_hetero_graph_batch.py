import pprint
import copy
import torch
import numpy as np
from torch_geometric.data import Data, DataLoader
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
        return emb, None, F.log_softmax(x, dim=1)

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
        loss.backward(retain_graph=True)
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




def main(build_graph, dataset_name, cut_off_dataset, experiment_data, experiments_path_file, cuda_num, output_dim):    
    config = {
        'build_graph': build_graph,
        'dataset_name': dataset_name, # autext23, semeval24, coling24, autext23_s2, semeval24_s2
        'cut_off_dataset': cut_off_dataset, # train-val-test
        'cuda_num': cuda_num,

        'window_size': 10,
        'graph_direction': 'undirected', # undirected | directed (for now all are undirecte, pending to handle and review to support both)
        'special_chars': False,
        'stop_words': False,
        'min_df': 2, # 1->autext | 5->semeval | 5-coling
        'max_df': 0.9,
        'max_features': None, # None -> all | 5000, 10000
        'not_found_tokens': 'avg', # avg, remove, zeros, ones
        'add_edge_attr': False,
        'embed_reduction': True,

        "gnn_type": 'TransformerConv', # GCNConv, GATConv, TransformerConv
        "dropout": 0.8,
        "patience": 10, # 5-autext23 | 10-semeval | 10-coling
        "learnin_rate": 0.0001, # autext23 -> llm: 0.00002 | semeval -> llm: 0.000005  | coling -> llm: 0.0001 
        "batch_size": 32 * 1,
        "hidden_dim": 100, # 300 autext_s2, 100 others
        "dense_hidden_dim": 64, # 64-autext23 | 32-semeval | 64-coling
        "num_layers": 1,
        "heads": 1,
        "output_dim": output_dim, # 2-bin | 6-multi 
        "weight_decay": 0.00001, 
        "num_neighbors": [20, 10],  # Adjust sampling depth
        'input_dim': 768,
        'epochs': 100

    }

    ## google-bert/bert-base-uncased
    ## FacebookAI/roberta-base
    ## microsoft/deberta-v3-base
    llm_name = 'microsoft/deberta-v3-base'
    
    file_name_data = f"hetero_data_{config['dataset_name']}_{config['cut_off_dataset']}perc" # perc_128
    #output_dir = f'{utils.OUTPUT_DIR_PATH}test_graph/{llm_name.split("/")[1]}/'
    #output_dir = f'{utils.OUTPUT_DIR_PATH}test_graph/'
    output_dir = f'{test_utils.EXTERNAL_DISK_PATH}hetero_graph/{llm_name.split("/")[1]}/'

    device = torch.device(f"cuda:{config['cuda_num']}" if torch.cuda.is_available() else "cpu")
    #device = 'cpu'
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

        print("distro_train_val_test: ", len(train_set), len(val_set), len(test_set))
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

        # Generate word and document embeddings
        tokenizer = AutoTokenizer.from_pretrained(llm_name, model_max_length=512)
        language_model = AutoModel.from_pretrained(llm_name, output_hidden_states=True).to(device)

        embedding_reduction = nn.Linear(768, 128).to(device)
        word_features = get_word_embeddings2(all_texts_norm, tokenizer, language_model, vocab, device, not_found_tokens=config['not_found_tokens'])
        word_features = torch.tensor(word_features, dtype=torch.float)

        doc_features = []
        for text in tqdm(all_texts_norm, desc="Extracting doc embeddings"):
            doc_embedding = get_document_embeddings(text, tokenizer, language_model, device)
            doc_features.append(doc_embedding)
        doc_features = torch.tensor(doc_features, dtype=torch.float)

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

        # Convert set to a sorted list (ensuring consistent order)
        doc_word_edges = sorted(doc_word_edges)  # List of (source, target) tuples
        doc_word_attr = [doc_word_attr[edge] for edge in doc_word_edges]  # Corresponding TF-IDF weights as a list

        # Create word-word edges and edge attributes
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

        # Create directed and undirected edge indices
        directed_edge_index = edges
        undirected_edge_index = torch.cat([edges, edges.flip(0)], dim=1)  # Add reverse edges
        
        # Removes duplicate edges
        directed_edge_index = torch.unique(directed_edge_index, dim=1)  
        undirected_edge_index = torch.unique(undirected_edge_index, dim=1)  

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
            y = node_labels
        )

        # Apply dimensionality reduction to train, val, and test data
        new_feat_dim = 128
        if config['embed_reduction']:    
            embedding_reduction = nn.Linear(768, new_feat_dim).to(device)
            data.x = reduce_dimension_linear(data.x, embedding_reduction, device, batch_size=128)


        # Save the data object
        utils.save_data(data, file_name_data, path=output_dir, format_file='.pkl', compress=False)
    else:
        # Load the data object
        data = utils.load_data(file_name_data, path=output_dir, format_file='.pkl', compress=False)


    print(data)
    directed_edge_index = copy.deepcopy(data.directed_edge_index)
    undirected_edge_index = copy.deepcopy(data.undirected_edge_index)
    edge_attr = copy.deepcopy(data.edge_attr)
    del data.directed_edge_index
    del data.undirected_edge_index
    
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
            
            # add/remove attr from Data object (based on the above config, this reduce the traning time)
            if row['graph_direction'] == 'undirected':
                data.edge_index = undirected_edge_index
                if row['gnn_edge_attr']:
                    data.edge_attr = torch.cat([edge_attr, edge_attr], dim=0)  # Duplicate edge_attr for reverse edges
            if row['graph_direction'] == 'directed':
                data.edge_index = directed_edge_index
                if row['gnn_edge_attr']:
                    data.edge_attr = edge_attr

            if row['gnn_edge_attr'] == False or row['gnn_edge_attr'] == 'False': 
                del data.edge_attr

            # move data object to devine
            data = data.to(device)
            print(data)

            # Create NeighborLoader instances
            train_nodes = torch.nonzero(data.train_mask, as_tuple=True)[0]  # Indices of train nodes
            val_nodes = torch.nonzero(data.val_mask, as_tuple=True)[0]  # Indices of train nodes
            test_nodes = torch.nonzero(data.test_mask, as_tuple=True)[0]  # Indices of train nodes

            train_loader = NeighborLoader(data, num_neighbors=config['num_neighbors'], batch_size=config['batch_size'], input_nodes=train_nodes, shuffle=True, replace=False, num_workers=2, persistent_workers=True)
            val_loader = NeighborLoader(data, num_neighbors=config['num_neighbors'], batch_size=config['batch_size'], input_nodes=val_nodes, shuffle=True, replace=False, num_workers=2, persistent_workers=True)
            test_loader = NeighborLoader(data, num_neighbors=config['num_neighbors'], batch_size=config['batch_size'], input_nodes=test_nodes, shuffle=True, replace=False, num_workers=2, persistent_workers=True)
            
            for batch in train_loader:
                print(f"Batch nodes: {batch.n_id.size(0)}, Batch data: {batch}")
                break
            
            # Initialize the model
            input_dim = data.x.shape[1]
            test_utils.set_random_seed(42)
            model = GNN(
                input_dim, 
                hidden_dim = row['gnn_nhid'], 
                dense_hidden_dim = row['gnn_dense_nhid'], 
                output_dim = output_dim, 
                dropout = row['gnn_dropout'], 
                num_layers = row['gnn_layers_convs'], 
                gnn_type = row['gnn_type'],
                heads = row['gnn_heads'], 
                task = 'node'
            )
            model = model.to(device)
            print(model)

            # Training loop
            optimizer = torch.optim.Adam(model.parameters(), lr=row['gnn_learning_rate'], weight_decay=config['weight_decay'])
            # Add learning rate scheduler
            #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

            criterion = torch.nn.CrossEntropyLoss()
            early_stopper = EarlyStopper(patience=row['gnn_patience'], min_delta=0)

            logger.info("Init GNN training!")
            best_test_acc = 0
            best_test_f1score = 0
            epoch_best_test_acc = 0
            for epoch in range(row['epoch_num']):
                loss_train = train(model, train_loader, optimizer, criterion, device)
                val_acc, val_f1_macro, val_loss, _, _, = evaluate(model, val_loader, criterion, 'val', device)
                test_acc, test_f1_macro, test_loss, preds_test, labels_test  = evaluate(model, test_loader, criterion, 'test', device)
                
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
                    print('Early stopping due to no improvement!')
                    metrics['_epoch_stop'] = epoch
                    break

                # Free unused memory
                del preds_test, labels_test
                torch.cuda.empty_cache()
                gc.collect()
            logger.info("Done GNN training!")

            # Evaluate on test set
            test_acc, test_f1_macro, test_loss, _, _ = evaluate(model, test_loader, criterion, 'test', device)
            print()
            print(f'Test Accuracy: {test_acc:.4f}')
            print(f'Test F1Score: {test_f1_macro:.4f}')
            print(f'Test Loss: {test_loss:.4f}')
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

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            time.sleep(10)


if __name__ == '__main__':
    print('*** INIT EXPERIMENTS')
    dataset_name='semeval24' # autext23, semeval24, coling24, autext23_s2, semeval24_s2
    build_graph = False
    cut_off_dataset = '10-25-25'
    cuda_num = 0
    output_dim = 2
    # test_experiments
    # experiments_autext23_hetero_20250211
    # experiments_coling25_hetero_20250211
    # experiments_semeval24_hetero_20250211
    file_name = 'experiments_semeval24_hetero_20250211'
    experiments_path_file = f'{utils.OUTPUT_DIR_PATH}test_graph/batch_files/final_acl/{file_name}.csv'
    experiment_data = utils.read_csv(f'{experiments_path_file}')
    print(experiment_data.info())
    
    main(build_graph, dataset_name, cut_off_dataset, experiment_data, experiments_path_file, cuda_num, output_dim)


import torch
import numpy as np
from torch_geometric.data import Data, DataLoader
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.decomposition import PCA

from torch.nn import Linear
from torch_geometric.loader import NeighborSampler, NeighborLoader
from torch_geometric.nn import GCNConv, global_mean_pool
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
from torch_geometric.utils import degree
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
warnings.filterwarnings("ignore")

#************************************* CONFIGS
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s; - %(levelname)s; - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Visualize the important subgraph
def visualize_subgraph(edge_index, edge_mask, node_idx):
    important_edges = edge_index[:, edge_mask > 0.5]
    G = nx.Graph()
    G.add_edges_from(important_edges.t().tolist())
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500)
    plt.show()


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
    def __init__(self, input_dim, hidden_dim, dense_hidden_dim, output_dim, dropout, num_layers, gnn_type='GCNConv', heads=1, task='node'):
        super(GNN, self).__init__()
        self.task = task
        self.heads = heads
        self.gnn_type = gnn_type
        self.embeddings = []  # List to store embeddings after each layer
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
            nn.Linear(dense_hidden_dim, int(dense_hidden_dim // 2)),
            nn.Linear(int(dense_hidden_dim // 2), output_dim),
            #nn.Linear(int(dense_hidden_dim // 4), output_dim)
        )

        self.dropout = dropout
        self.num_layers = num_layers

    def build_conv_model(self, input_dim, hidden_dim, heads):
        if self.gnn_type == 'GCNConv':
            return GCNConv(input_dim, hidden_dim)
        elif self.gnn_type == 'GATConv':
            return GATConv(input_dim, hidden_dim, heads=heads)
        elif self.gnn_type == 'TransformerConv':
            return TransformerConv(input_dim, hidden_dim, heads=heads)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        self.embeddings.append(x.detach().cpu().numpy())
        x = self.conv1(x, edge_index)
        emb = x
        x = F.relu(x)
        self.embeddings.append(x.detach().cpu().numpy())
        x = self.norm1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            emb = x
            x = F.relu(x)
            self.embeddings.append(x.detach().cpu().numpy())
            x = self.lns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.task == 'graph':
            x = global_mean_pool(x, batch)

        x = self.post_mp(x)
        # F.log_softmax(x, dim=1)
        # self.sigmoid(x)
        return emb, self.embeddings, F.log_softmax(x, dim=1)

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        # Ensure the mask is applied to the batch
        train_mask = batch.train_mask
        emb, _, out = model(batch.x, batch.edge_index, None, batch.batch)
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
    misclassified_nodes = []  # List to store misclassified node indices
    misclassified_true_labels = []  # List to store ground truth labels of misclassified nodes
    misclassified_pred_labels = []  # List to store predicted labels of misclassified nodes
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            # Select the appropriate mask for evaluation
            if mask_type == 'train':
                mask = batch.train_mask
            elif mask_type == 'val':
                mask = batch.val_mask
            elif mask_type == 'test':
                mask = batch.test_mask
            else:
                raise ValueError("Invalid mask_type. Use 'train', 'val', or 'test'.")

            emb, _, out = model(batch.x, batch.edge_index, None, batch.batch)
            loss = criterion(out[mask], batch.y[mask])
            pred = out.argmax(dim=1)
            correct = (pred[mask] == batch.y[mask])
            total_correct += int(correct.sum())
            total_f1 += f1_score(batch.y[mask].cpu(), pred[mask].cpu(), average='macro')
            total_loss += loss.item()
            total_nodes += mask.sum().item()

            all_preds.extend(pred[mask].cpu().numpy())
            all_labels.extend(batch.y[mask].cpu().numpy())

            # Collect misclassified node indices, ground truth labels, and predicted labels
            misclassified_mask = ~correct
            misclassified_nodes.extend((misclassified_mask).nonzero(as_tuple=True)[0].cpu().tolist())
            misclassified_true_labels.extend(batch.y[mask][misclassified_mask].cpu().tolist())
            misclassified_pred_labels.extend(pred[mask][misclassified_mask].cpu().tolist())

    acc = total_correct / total_nodes
    f1_macro = total_f1 / len(loader)
    loss = total_loss / len(loader)

    return acc, f1_macro, loss, misclassified_nodes, misclassified_true_labels, misclassified_pred_labels, all_preds, all_labels


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
            not_found_tokens_dict[word] = []
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
                ... # remove

    if not_found_tokens == 'remove':
        vocab = list(set(vocab) - set(not_found_tokens_dict.keys()))

    print("emb_words cnt_found: ", cnt_found)
    print("emb_words cnt_not_found: ", cnt_not_found)
    print("not_found_tokens_dict: ", len(not_found_tokens_dict))
    return word_embeddings, vocab

def visualize_misclassified_subgraph(data, misclassified_nodes, misclassified_true_labels, misclassified_pred_labels, edge_index):
    # Create a subgraph for misclassified nodes
    G = nx.Graph()
    for node in misclassified_nodes:
        # Add edges connected to the misclassified node
        edges = edge_index[:, (edge_index[0] == node) | (edge_index[1] == node)]
        G.add_edges_from(edges.t().tolist())
    
    # Draw the subgraph
    pos = nx.spring_layout(G)
    labels = {node: f"True: {true}, Pred: {pred}" for node, true, pred in zip(misclassified_nodes, misclassified_true_labels, misclassified_pred_labels)}
    nx.draw(G, pos, with_labels=True, labels=labels, node_color='lightblue', edge_color='gray', node_size=500)
    plt.title("Subgraph of Misclassified Nodes")
    plt.show()

def extract_embeddings(model, loader, device):
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _, embeddings, _ = model(batch.x, batch.edge_index, None, batch.batch)
            all_embeddings.append(embeddings[-1])  # Use the final layer embeddings
            all_labels.append(batch.y.cpu().numpy())

    # Concatenate embeddings and labels
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return all_embeddings, all_labels


def main():    
    build_graph=False
    
    # autext23, semeval24, coling24, autext23_s2, semeval24_s2
    dataset_name = 'semeval24_s2'
    cut_off_dataset = 100

    cuda_num = 1
    window_size = 10
    special_chars = False
    stop_words = False
    min_df = 1 # 1->autext | 5->semeval | 5-coling
    max_df = 0.9
    max_features = None # None -> all | 5000
    not_found_tokens = 'avg' # avg, remove, zeros, ones

    ## google-bert/bert-base-uncased
    ## FacebookAI/roberta-base
    ## microsoft/deberta-v3-base
    llm_name = 'microsoft/deberta-v3-base'
    
    file_name_data = f'hetero_data_{dataset_name}_{cut_off_dataset}perc'
    output_dir = f'{utils.OUTPUT_DIR_PATH}test_graph/{llm_name.split("/")[1]}/'
    #output_dir = f'{utils.OUTPUT_DIR_PATH}test_graph/'
    #output_dir = f'{test_utils.EXTERNAL_DISK_PATH}hetero_graph/'

    # TransformerConv, GATConv, GCNonv
    gnn_type = 'TransformerConv'
    input_dim = 768 # shared_feature_dim
    hidden_dim = 100 # 300 autext_s2, 100 others
    dense_hidden_dim = 32 # 64-autext23 | 32-semeval | 64-coling
    num_layers = 1
    heads = 1
    dropout = 0.5
    output_dim = 6 # 2-bin | 6-multi 
    epochs = 100
    patience = 100 # 5-autext23 | 10-semeval | 10-coling
    learnin_rate = 0.0001 # autext23 -> llm: 0.00001 | semeval -> llm: 0.000001  | coling -> llm: 0.0001 
    weight_decay = 1e-5 
    batch_size = 32 * 1
    num_neighbors = [50, 40]  # Adjust sampling depth
    num_workers = 0

    device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")

    
    if build_graph == True:
        # ****************************** PROCESS AUTEXT DATASET && CUTOF
        train_text_set, val_text_set, test_text_set  = test_utils.read_dataset(dataset_name)

        # *** TRAIN
        cut_dataset_train = len(train_text_set) * (int(cut_off_dataset) / 100)
        train_set = train_text_set[:int(cut_dataset_train)]
        # *** VAL
        cut_dataset_val = len(val_text_set) * (int(cut_off_dataset) / 100)
        val_set = val_text_set[:int(cut_dataset_val)]
        # *** TEST
        cut_dataset_test = len(test_text_set) * (int(cut_off_dataset) / 100)
        test_set = test_text_set[:int(cut_dataset_test)]

        print("distro_train_val_test: ", len(train_set), len(val_set), len(test_set))
        print("label_distro_train_val_test: ", train_set.value_counts('label'), val_set.value_counts('label'), test_set.value_counts('label'))
        
        # Example text data (split into train, validation, and test sets)
        train_texts = list(train_set['text'])[:5]
        val_texts = list(val_set['text'])[:5]
        test_texts = list(test_set['text'])[:5]

        # Labels (binary classification: 0 or 1)
        train_labels = list(train_set['label'])[:]
        val_labels = list(val_set['label'])[:]
        test_labels = list(test_set['label'])[:]

        # Combine all texts for vocabulary and word embedding training
        all_texts = train_texts + val_texts + test_texts
        all_labels = train_labels + val_labels + test_labels

        # Tokenize the corpus for Word2Vec training
        tokenized_corpus = []
        all_texts_norm = []
        tokenize_pattern = "[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+"
        for text in tqdm(all_texts, desc="normalizing corpus"):
            text_norm = test_utils.text_normalize(text, special_chars=special_chars, stop_words=stop_words)  # Assuming text_normalize is defined
            all_texts_norm.append(text_norm)
            text_doc_tokens = re.findall(tokenize_pattern, text_norm)
            tokenized_corpus.append(text_doc_tokens)

        # Tokenize the text and create a vocabulary
        #vectorizer = CountVectorizer()
        vectorizer = CountVectorizer(min_df=min_df, max_df=max_df, max_features=max_features)
        X = vectorizer.fit_transform(all_texts_norm)
        vocab = vectorizer.get_feature_names_out()
        print('len_vocab: ', len(vocab))

        #**** PRe-Trained Langue Model as NFI    
        # Extract embedding from language model
        tokenizer = AutoTokenizer.from_pretrained(llm_name, model_max_length=512)
        language_model = AutoModel.from_pretrained(llm_name, output_hidden_states=True).to(device)

        # Generate embeddings for all words in the vocabulary
        #word_features = []
        #for word in tqdm(vocab[:]):
        #    word_embedding = get_word_embeddings(word, tokenizer, language_model, device)
        #    word_features.append(word_embedding)
        #word_features = torch.tensor(word_features, dtype=torch.float)
        
        word_features, vocab = get_word_embeddings2(all_texts_norm, tokenizer, language_model, vocab, device, not_found_tokens)
        word_features = torch.tensor(word_features, dtype=torch.float)
        print("word_features", len(word_features))
        print('len_vocab2: ', len(vocab))

         # Create a word-to-index dictionary for fast lookups
        word_to_index = {word: idx for idx, word in enumerate(vocab)}
        
        # Generate LLM embeddings for all documents
        doc_features = []
        for text in tqdm(all_texts_norm, desc="Extracting doc embeddings"):
            doc_embedding = get_document_embeddings(text, tokenizer, language_model, device)
            doc_features.append(doc_embedding)
        doc_features = torch.tensor(doc_features, dtype=torch.float)
        
        # Document-word edges (if a word exists in a document)
        doc_word_edges = []
        for doc_id, doc in enumerate(tqdm(X, desc="Extracting document-word edges")):
            for word_idx in doc.nonzero()[1]:
                doc_word_edges.append([doc_id, len(all_texts) + word_idx])  # Connect doc to word

        # Word-word edges (co-occurrence within a window size)
        word_word_edges = set()  # Use a set to avoid duplicate edges
        for doc in tqdm(tokenized_corpus[:], desc="Extracting word-word edges"):
            for i in range(len(doc)):
                # Get the window of words
                window = doc[i:i + window_size + 1]
                # Generate all unique pairs of words in the window
                for word1, word2 in combinations(window, 2):
                    if word1 in word_to_index and word2 in word_to_index:
                        word1_id = len(all_texts) + word_to_index[word1]
                        word2_id = len(all_texts) + word_to_index[word2]
                        # Add edges in both directions (undirected graph)
                        word_word_edges.add((word1_id, word2_id))
                        #word_word_edges.add((word2_id, word1_id))
        
        # Combine document and word features
        node_features = torch.cat([doc_features, word_features], dim=0)

        # Create node labels (only for document nodes, word nodes are unlabeled)
        node_labels = torch.tensor(all_labels + [-1] * len(vocab), dtype=torch.long)  # -1 for word nodes

        # Convert the set to a list of edges
        word_word_edges = list(word_word_edges)

        # Combine all edges
        print("doc_word_edges: ", len(doc_word_edges))
        print("word_word_edges: ", len(word_word_edges))
        edges = torch.tensor(doc_word_edges + word_word_edges, dtype=torch.long).t().contiguous()

        # Create the PyG Data object
        data =  Data(x=node_features, edge_index=edges, y=node_labels, vocab_len=len(vocab))

        # Create masks for train, validation, and test sets
        num_nodes = data.x.shape[0]

        # Initialize masks
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        # Set masks for document nodes
        train_mask[:len(train_texts)] = True
        val_mask[len(train_texts):len(train_texts) + len(val_texts)] = True
        test_mask[len(train_texts) + len(val_texts):len(train_texts) + len(val_texts) + len(test_texts)] = True

        # Add masks to the Data object
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        # Create the PyG Data object
        utils.save_data(data, file_name_data, path=output_dir, format_file='.pkl', compress=False)
        # pending to save doc embeddings, labels, mask, etc (train MLP or ML models to compare with GNN)
        #utils.save_data(doc_features, file_name_doc_emb, path=output_dir, format_file='.pkl', compress=False)
    else:
        data = utils.load_data(file_name_data, path=output_dir, format_file='.pkl', compress=False)


    print(data.edge_index.shape)
    # Identify word-to-word edges
    #word_mask = (data.edge_index[0] < data.vocab_len) & (data.edge_index[1] < data.vocab_len)  # Both source and target are word nodes
    #word_to_word_edges = data.edge_index[:, word_mask]  # Extract word-to-word edges
    ## Add reverse edges for word-to-word connections
    #reversed_edges = word_to_word_edges.flip(0)  # Reverse the edges
    ## Concatenate original edges with the reversed word-to-word edges
    #data.edge_index = torch.cat([data.edge_index, reversed_edges], dim=1)
    #print(data.edge_index.shape)
    #data.edge_index = torch.unique(data.edge_index, dim=1)  # Removes duplicate columns
    #print(data.edge_index.shape)

    #print(data.edge_index.shape)
    #data.edge_index = torch.cat([data.edge_index, data.edge_index.flip(0)], dim=1)  # Add reverse edges
    #print(data.edge_index.shape)
    #data.edge_index = torch.unique(data.edge_index, dim=1)  # Removes duplicate columns
    #print(data.edge_index.shape)

    data = data.to(device)
    print(data)

    # Create NeighborLoader instances
    train_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors, # Number of neighbors to sample at each layer
        batch_size=batch_size,       # Batch size
        input_nodes=data.train_mask,
        shuffle=True,
        num_workers=num_workers      # Number of workers for data loading
    )

    val_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors, 
        batch_size=batch_size,       
        input_nodes=data.val_mask,
        shuffle=True,
        num_workers=num_workers              
    )

    test_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors, 
        batch_size=batch_size,       
        input_nodes=data.test_mask,
        shuffle=True,
        num_workers=num_workers  
    )

    for batch in train_loader:
        print(f"Batch nodes: {batch.n_id.size(0)}, Batch data: {batch}")
        break
    

    # Initialize the model
    input_dim = data.x.shape[1]
    test_utils.set_random_seed(42)
    model = GNN(input_dim, hidden_dim, dense_hidden_dim, output_dim, dropout, num_layers, gnn_type=gnn_type, heads=heads, task='node')
    model = model.to(device)
    print(model)

    # Training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=learnin_rate, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    early_stopper = EarlyStopper(patience=patience, min_delta=0)

    logger.info("Init GNN training!")
    for epoch in range(epochs):
        loss_train = train(model, train_loader, optimizer, criterion, device)
        val_acc, val_f1_macro, val_loss, _, _, _, _, _ = evaluate(model, val_loader, criterion, 'val', device)
        test_acc, test_f1_macro, test_loss, misclassified_nodes, misclassified_true_labels, misclassified_pred_labels, preds_test, labels_test  = evaluate(model, test_loader, criterion, 'test', device)
        
        print(f'Epoch {epoch + 1}, Loss Train: {loss_train:.4f}, Loss Val: {val_loss:.4f}, Loss Test: {test_loss:.4f}, '
          f'Val Acc: {val_acc:.4f}, Val F1Score: {val_f1_macro:.4f}, Test Acc: {test_acc:.4f}, Test F1Score: {test_f1_macro:.4f}')
        print("preds_test: ", sorted(Counter(preds_test).items()))
        print("label_test: ", sorted(Counter(labels_test).items()))

        if early_stopper.early_stop(val_loss):
            print('Early stopping due to no improvement!')
            break
    logger.info("Done GNN training!")

    # Evaluate on test set
    test_acc, test_f1_macro, test_loss, misclassified_nodes, misclassified_true_labels, misclassified_pred_labels, preds_test, labels_test  = evaluate(model, test_loader, criterion, 'test', device)
    print(f'Test Accuracy: {test_acc:.4f}')
    print(f'Test F1Score: {test_f1_macro:.4f}')
    print(f'Test Loss: {test_loss:.4f}')
    
    print(f'Number of Misclassified Nodes: {len(misclassified_nodes)}')
    #print(f'Misclassified Node Indices: {misclassified_nodes}')
    print(f'Ground Truth Labels of Misclassified Nodes: {Counter(misclassified_true_labels)}')
    print(f'Predicted Labels of Misclassified Nodes: {Counter(misclassified_pred_labels)}')

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

    return 
    # Create confusion matrix
    conf_matrix = confusion_matrix(misclassified_true_labels, misclassified_pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(output_dim), yticklabels=range(output_dim))
    plt.xlabel('Predicted Labels')
    plt.ylabel('Ground Truth Labels')
    plt.title('Confusion Matrix for Misclassified Nodes')
    plt.show()
    plt.savefig('my_plot.png')

    # Visualize the subgraph of misclassified nodes
    #visualize_misclassified_subgraph(data, misclassified_nodes, misclassified_true_labels, misclassified_pred_labels, data.edge_index)

    return
    # Define the model configuration
    model_config = ModelConfig(
        mode="binary_classification",  # Change to "binary_classification" if applicable
        task_level="node",
        return_type="raw"  # Change to "probs" if your model returns probabilities
    )

    # Initialize the Explainer
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        model_config=model_config,
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
    )

    # Select a node to explain
    node_idx = 42  # Example node from the test set

    # Generate explanation
    explanation = explainer(data.x, data.edge_index, index=node_idx)

    # Get edge importance mask
    edge_mask = explanation.edge_mask
    print(edge_mask)

    # Visualize the explanation
    explanation.visualize_graph(node_idx, data.edge_index, explanation.edge_mask)

if __name__ == '__main__':
    main()
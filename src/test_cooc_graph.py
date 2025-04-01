
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
    def __init__(self, input_dim, hidden_dim, dense_hidden_dim, output_dim, dropout, num_layers, gnn_type='GCNConv', heads=1, task='node'):
        super(GNN, self).__init__()
        self.task = task
        self.heads = heads
        self.gnn_type = gnn_type
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

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x, edge_index)
        emb = x
        x = F.relu(x)
        x = self.norm1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        for i in range(self.num_layers):
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
            emb, _, out = model(data.x, data.edge_index, None, data.batch)
            loss = criterion(out, data.y)
            loss.backward()
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
            emb, _, out = model(data.x, data.edge_index, None, data.batch)
        pred = out.argmax(dim=1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(data.y.cpu().numpy())
        correct += int((pred == data.y).sum())
        loss = criterion(out, data.y)
        test_loss += loss.item()

    f1_macro = f1_score(all_labels, all_preds, average='macro')
    accuracy = correct / len(loader.dataset)
    return accuracy, f1_macro, test_loss / len(loader)

def extract_doc_edges(text_tokenized, label, word_features, vocab, window_size):
    #def extract_doc_edges(text_tokenized, label, word_features, word_to_index, vocab, window_size):
    try:
        # Get unique words in the document
        unique_words = list(set(text_tokenized))
        unique_words = [word for word in unique_words if (word in vocab and word in word_features.keys())]  # Filter out OOV words
        #unique_words = [word for word in unique_words if (word in vocab)]  # Filter out OOV words
        #print("\n unique_words: ", len(unique_words), unique_words)

        # Create node features for this graph (only words in the document)
        #node_features = torch.stack([word_features[word_to_index[word]] for word in unique_words])
        node_features = torch.stack([word_features[word] for word in unique_words])

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
        data = Data(x=node_features, edge_index=edges, y=label, unique_words=unique_words)
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

def get_word_embeddings2(texts_set, texts_tokenized, tokenizer, language_model, vocab, device, set_corpus='train', not_found_tokens='avg'):
    doc_words_embeddings_dict = {}

    for idx, text in enumerate(tqdm(texts_set, desc=f"Extracting {set_corpus} word embeddings")): # hetero
        #for idx, text_tokens in enumerate(tqdm(texts_tokenized, desc=f"Extracting {set_corpus} word embeddings")): # hetero
        #text = ' '.join(list(set(vocab) & set(text_tokens)))

        doc_tokesn_freq = defaultdict(int)
        doc_words_embeddings_dict[idx] = {'tokens': {}, 'not_found_tokens': []}
        try:
            encoded_text = tokenizer.encode_plus(text, return_tensors="pt", padding=True, truncation=True)
            encoded_text.to(device)
            with torch.no_grad():
                outputs_model = language_model(**encoded_text, output_hidden_states=True)
            last_hidden_state = outputs_model.hidden_states[-1]

            for i in range(0, len(last_hidden_state)):
                raw_tokens = [tokenizer.decode([token_id]) for token_id in encoded_text['input_ids'][i]]
                for token, embedding in zip(raw_tokens, last_hidden_state[i]):
                    token = str(token).strip()
                    if token not in vocab:
                        continue
                    doc_tokesn_freq[token] += 1
                    current_emb = embedding.cpu().detach().numpy().tolist()
                    if token not in doc_words_embeddings_dict[idx]['tokens'].keys():
                        doc_words_embeddings_dict[idx]['tokens'][token] = current_emb
                    else:
                        doc_words_embeddings_dict[idx]['tokens'][token] = np.add.reduce([doc_words_embeddings_dict[idx]['tokens'][token], current_emb])
            for token, freq in doc_tokesn_freq.items():
                doc_words_embeddings_dict[idx]['tokens'][token] = torch.tensor(np.divide(doc_words_embeddings_dict[idx]['tokens'][token], freq).tolist(), dtype=torch.float)
        except Exception as e:
            print('Error: %s', str(e))

    for idx, doc_tokens in enumerate(tqdm(texts_tokenized)): 
        try:
            # get tokens that were not found in doc_words_embeddings_dict according to doc_tokens AND only exist in vocab
            merge_tokens = list(set(vocab) & (set(doc_tokens) - set(doc_words_embeddings_dict[idx]['tokens'].keys())))
            for token in merge_tokens:
            #for token in doc_tokens: 
                not_found_tokens_dict = {}
                #if (token in vocab) and (token not in doc_words_embeddings_dict[idx]['tokens']):
                if not_found_tokens == 'ones':
                    doc_words_embeddings_dict[idx]['tokens'][token] = torch.tensor(np.ones(language_model.config.hidden_size), dtype=torch.float)
                elif not_found_tokens == 'zeros':
                    doc_words_embeddings_dict[idx]['tokens'][token] = torch.tensor(np.zeros(language_model.config.hidden_size), dtype=torch.float)
                else: # avg
                    node_tokens = tokenizer.encode_plus(token, return_tensors="pt", padding=True, truncation=True)
                    raw_tokens = [tokenizer.decode([token_id]) for token_id in node_tokens['input_ids'][0]]
                    not_found_tokens_dict[token] = raw_tokens[1:-1]
                    avg_emb, emb_list = [], []
                    for token, subtokens in not_found_tokens_dict.items():
                        if token in doc_words_embeddings_dict[idx]['tokens']:
                            continue
                        for subtoken in subtokens:
                            if subtoken in doc_words_embeddings_dict[idx]['tokens']:
                                emb_list.append(doc_words_embeddings_dict[idx]['tokens'][subtoken])
                    if len(emb_list) == 0:
                        avg_emb = np.zeros(language_model.config.hidden_size)
                    else:
                        avg_emb = np.mean(emb_list, axis=0).flatten().tolist()
                    doc_words_embeddings_dict[idx]['tokens'][token] = torch.tensor(avg_emb, dtype=torch.float)
        except Exception as e:
            print('Error: %s', str(e))

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


def main():    
    build_graph=False

    # autext23, semeval24, coling24, autext23_s2, semeval24_s2
    dataset_name = 'autext23'
    cut_off_dataset = 100
    cuda_num = 1
    window_size = 10 # 10 -> auetxt23 | 20 -> semeval/coling
    special_chars = False
    stop_words = False
    min_df = 1 # 2 -> auetxt23 | 5 -> semeval/coling
    max_df = 0.9
    max_features = None # None -> all | 5000
    batch_size = 64
    not_found_tokens = 'avg' # avg, ones, zeros

    ## google-bert/bert-base-uncased
    ## FacebookAI/roberta-base
    ## microsoft/deberta-v3-base
    llm_name = 'microsoft/deberta-v3-base'

    file_name = f'cooc_data_{dataset_name}_{cut_off_dataset}perc'
    output_dir = f'{utils.OUTPUT_DIR_PATH}test_graph/{llm_name.split("/")[1]}/'
    #output_dir = f'{utils.OUTPUT_DIR_PATH}test_graph/'
    #output_dir = f'{test_utils.EXTERNAL_DISK_PATH}cooc_graph/'
    
    # TransformerConv, GATConv
    gnn_type='TransformerConv'
    input_dim = 768 # shared_feature_dim
    hidden_dim = 100
    dense_hidden_dim = 64
    num_layers = 3
    heads = 2
    dropout = 0.5
    output_dim = 2
    epochs = 100
    patience = 10
    learnin_rate = 0.00002 # Autext -> llm: 0.00001 | semeval -> llm: 0.000001  | coling -> llm: 0.0001 
    weight_decay = 1e-5
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
        train_texts = list(train_set['text'])[:]
        val_texts = list(val_set['text'])[:]
        test_texts = list(test_set['text'])[:]

        # Labels (binary classification: 0 or 1)
        train_labels = list(train_set['label'])[:]
        val_labels = list(val_set['label'])[:]
        test_labels = list(test_set['label'])[:]

        # Normalize and Tokenize the corpus 
        tokenize_pattern = "[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+"
        train_texts_norm, train_texts_tokenized = normalize_text(train_texts, tokenize_pattern, special_chars, stop_words, set='train')
        val_texts_norm, val_texts_tokenized = normalize_text(val_texts, tokenize_pattern, special_chars, stop_words, set='val')
        test_texts_norm, test_texts_tokenized = normalize_text(test_texts, tokenize_pattern, special_chars, stop_words, set='test')

        # create a vocabulary
        all_texts_norm = train_texts_norm + val_texts_norm + test_texts_norm
        vocab, word_to_index = create_vocab(all_texts_norm, min_df=min_df, max_df=max_df, max_features=max_features)
        print("not_found_tokens approach: ", not_found_tokens)
        # LLM        
        # Extract embedding from language model
        tokenizer = AutoTokenizer.from_pretrained(llm_name, model_max_length=512)
        language_model = AutoModel.from_pretrained(llm_name, output_hidden_states=True).to(device)

        # *** Generate embeddings method 1
        '''
        word_features = []
        for word in tqdm(vocab[:], desc="Extracting word embeddings"):
            word_embedding = get_word_embeddings(word, tokenizer, language_model, device)
            word_features.append(word_embedding)
        word_features = torch.tensor(word_features, dtype=torch.float)
        '''

        # *** Generate embeddings method 2     
        train_words_emb = get_word_embeddings2(train_texts_norm, train_texts_tokenized, tokenizer, language_model, vocab, device, set_corpus='train', not_found_tokens=not_found_tokens)
        val_words_emb = get_word_embeddings2(val_texts_norm, val_texts_tokenized, tokenizer, language_model, vocab, device, set_corpus='val', not_found_tokens=not_found_tokens)
        test_words_emb = get_word_embeddings2(test_texts_norm, test_texts_tokenized, tokenizer, language_model, vocab, device, set_corpus='test', not_found_tokens=not_found_tokens)
        #print(len(train_words_emb[0]['tokens'].keys()), train_words_emb[0]['tokens'].keys())

        train_data, val_data, test_data = [], [], []
        for idx, (text_tokenized, label) in enumerate(zip(tqdm(train_texts_tokenized, desc="Extracting doc train edges"), train_labels)):
            doc_edges = extract_doc_edges(text_tokenized, label, train_words_emb[idx]['tokens'], vocab, window_size)
            if doc_edges:
                train_data.append(doc_edges)
        for idx, (text_tokenized, label) in enumerate(zip(tqdm(val_texts_tokenized, desc="Extracting doc val edges"), val_labels)):
            doc_edges = extract_doc_edges(text_tokenized, label, val_words_emb[idx]['tokens'], vocab, window_size)
            if doc_edges:
                val_data.append(doc_edges)
        for idx, (text_tokenized, label) in enumerate(zip(tqdm(test_texts_tokenized, desc="Extracting doc test edges"), test_labels)):
            doc_edges = extract_doc_edges(text_tokenized, label, test_words_emb[idx]['tokens'], vocab, window_size)
            if doc_edges:
                test_data.append(doc_edges)


        # *** Generate embeddings method 3
        '''
        word_features = get_word_embeddings3(all_texts_norm, tokenizer, language_model, vocab, device, not_found_tokens=not_found_tokens)
        word_features = torch.tensor(word_features, dtype=torch.float)
        print("word_features", len(word_features))
        train_data = [extract_doc_edges2(text_tokenized, label, word_features, word_to_index, vocab, window_size) for idx, (text_tokenized, label) in enumerate(zip(tqdm(train_texts_tokenized, desc="Extracting doc train edges"), train_labels))]
        val_data = [extract_doc_edges2(text_tokenized, label, word_features, word_to_index, vocab, window_size) for idx, (text_tokenized, label) in enumerate(zip(tqdm(val_texts_tokenized, desc="Extracting doc val edges"), train_labels))]
        test_data = [extract_doc_edges2(text_tokenized, label, word_features, word_to_index, vocab, window_size) for idx, (text_tokenized, label) in enumerate(zip(tqdm(test_texts_tokenized, desc="Extracting doc test edges"), train_labels))]
        '''

        # *** Save data
        all_data = [train_data, val_data, test_data]
        #word_features = [train_words_emb, val_words_emb, test_words_emb]
        data_obj = {
            #"word_features": word_features, # word_emb method 3
            "vocab": vocab,
            "all_data": all_data,
            "word_to_index": word_to_index,
        }
        utils.save_data(data_obj, file_name, path=output_dir, format_file='.pkl', compress=False)
    else:
        data_obj = utils.load_data(file_name, path=output_dir, format_file='.pkl', compress=False)
        train_data = data_obj['all_data'][0]
        val_data = data_obj['all_data'][1]
        test_data = data_obj['all_data'][2]
        word_to_index = data_obj['word_to_index']
        #word_features = data_obj['word_features'] # word_emb method 3
        #train_words_emb = word_features[0]
        #val_words_emb = word_features[1]
        #test_words_emb = word_features[2]

    # *** for embeddings method 2
    #train_data = [extract_node_features(data, train_words_emb[idx]) for idx, data in enumerate(tqdm(train_data, desc="Extracting train node feat"))]
    #val_data = [extract_node_features(data, val_words_emb[idx]) for idx, data in enumerate(tqdm(val_data, desc="Extracting val node feat"))]
    #test_data = [extract_node_features(data, test_words_emb[idx]) for idx, data in enumerate(tqdm(test_data, desc="Extracting test node feat"))]
    
    # *** for embeddings method 3
    #train_data = [extract_node_features(data, word_features, word_to_index) for idx, data in enumerate(tqdm(train_data, desc="Extracting train node feat"))]
    #val_data = [extract_node_features(data, word_features, word_to_index) for idx, data in enumerate(tqdm(val_data, desc="Extracting val node feat"))]
    #test_data = [extract_node_features(data, word_features, word_to_index) for idx, data in enumerate(tqdm(test_data, desc="Extracting test node feat"))]

    print(train_data[0])

    # Create DataLoader for train, validation, and test partitions
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    for batch in train_loader:
        print(batch)
        break

    # Initialize the model
    test_utils.set_random_seed(42)
    model = GNN(input_dim, hidden_dim, dense_hidden_dim, output_dim, dropout, num_layers, gnn_type=gnn_type, heads=heads, task='graph')
    model = model.to(device)
    print(model)

    # Training loop (example)
    optimizer = torch.optim.Adam(model.parameters(), lr=learnin_rate, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    early_stopper = EarlyStopper(patience=patience, min_delta=0)

    logger.info("Init GNN training!")
    for epoch in range(1, epochs):
        loss_train = train_cooc(model, train_loader, device, optimizer, criterion)
        val_acc, val_f1_macro, val_loss = test_cooc(val_loader, model, device, criterion)
        #train_acc, train_f1_macro, train_loss = test_cooc(train_loader)

        if early_stopper.early_stop(val_loss):
            print('Early stopping fue to not improvement!')
            break
        print(f'Epoch {epoch}, Loss Train: {loss_train:.4f}, Loss Val: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1Score: {val_f1_macro:.4f}')
    logger.info("Done GNN training!")

    # Final evaluation on the test set
    test_acc, test_f1_macro, test_loss = test_cooc(test_loader, model, device, criterion)
    print(f'Test Accuracy: {test_acc:.4f}')
    print(f'Test F1Score: {test_f1_macro:.4f}')


if __name__ == '__main__':
    main()

# + colab={"base_uri": "https://localhost:8080/"} id="wHhjCNAR0Kh0" outputId="515d9593-e6c4-45ae-c094-22aef02f73b2"
from sklearn.datasets import fetch_20newsgroups

from text2graphapi.src.Cooccurrence  import Cooccurrence
from text2graphapi.src.Heterogeneous  import Heterogeneous
from text2graphapi.src.IntegratedSyntacticGraph  import ISG
import time
import numpy as np
import pandas as pd
import glob

import joblib
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
from sklearn.feature_extraction.text import TfidfVectorizer
import itertools
from math import log

#from datasets import load_dataset, Dataset, DatasetDict
from transformers import logging
from transformers import AutoTokenizer, AutoModel, Trainer, AutoModelForSequenceClassification, TrainingArguments
from transformers import TrainingArguments, Trainer
from transformers import get_scheduler

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

from scipy.sparse import coo_array
import gc
import glob
import torch.nn.functional as F
from torch_geometric.data import DataLoader, Data
from collections import OrderedDict
import warnings
from transformers import logging as transform_loggin

import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, TransformerConv, TopKPooling, GraphConv, SAGPooling
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch.nn import Linear, BatchNorm1d, ModuleList, LayerNorm
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

import torch

from nltk.corpus import stopwords
nltk.download('stopwords')

# %matplotlib inline
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

warnings.filterwarnings(action='ignore')

LANGUAGE = 'en' #es, en, fr

ROOT_DRIVE_PATH = '/content/drive/MyDrive/Docto/datasets/'
AUTEXT23_PATH = ROOT_DRIVE_PATH + 'Autextification23/'
COLING24_PATH = ROOT_DRIVE_PATH + 'COLING_2024/'

#ROOT_DIR = os.path.dirname(os.path.dirname(__file__)) + '/GraphDeepLearning'
DATASET_DIR = '/home/avaldez/projects/GraphDeepLearning' + '/datasets/'


# *************** TEXT TO GRAPHS
def custom_tokenizer(nlp):
    #infix_re = re.compile(r'''[.\,\?\:\;\...\‘\’\`\“\”\"\'~]''')
    infix_re = re.compile(r'(?:[\\(){}[\]=&|^+<>/*%;.\'"?!~-]|(?:\w+|\d+))')
    prefix_re = compile_prefix_regex(nlp.Defaults.prefixes)
    suffix_re = compile_suffix_regex(nlp.Defaults.suffixes)
    return Tokenizer(nlp.vocab, prefix_search=prefix_re.search, suffix_search=suffix_re.search, infix_finditer=infix_re.finditer, token_match=None)

class Text2CoocGraph():
    def __init__(self, graph_type, apply_prep=True,parallel_exec=False, window_size=1,language='en', steps_preprocessing={}):
        self.apply_prep = apply_prep
        self.window_size = window_size
        self.graph_type = graph_type
        self.parallel_exec = parallel_exec
        self.language = language
        self.steps_prep = steps_preprocessing
        self.stop_words = set(stopwords.words('english'))

        exclude_modules = ["ner", "textcat"]
        self.nlp = spacy.load('en_core_web_sm', exclude=exclude_modules)
        self.nlp.tokenizer = custom_tokenizer(self.nlp)

    def _get_entities(self, doc_instance) -> list:
        nodes = []
        for token in doc_instance:
            if token.text in ['[CLS]', '[SEP]', '[UNK]']:
                continue
            node = (f'{str(token.lemma_)}', {'lemma_': token.lemma_, 'pos_tag': token.pos_}) # (word, {'node_attr': value}) | {'pos_tag': token.pos_} | token.lemma_ | token.text
            nodes.append(node)
        #print(nodes)
        return nodes

    def _get_relations(self, doc) -> list:
        d_cocc = defaultdict(int)
        text_doc_tokens, edges = [], []
        for token in doc:
            if token.text in ['[CLS]', '[SEP]', '[UNK]']:
                continue
            text_doc_tokens.append(f'{str(token.lemma_)}')
        for i in range(len(text_doc_tokens)):
            word = text_doc_tokens[i]
            next_word = text_doc_tokens[i+1 : i+1 + self.window_size]
            for t in next_word:
                key = (word, t)
                d_cocc[key] += 1

        unigram_freq = nltk.FreqDist(text_doc_tokens)
        bigram_freq = nltk.FreqDist(d_cocc)
        for words, value in d_cocc.items():
            pmi_val = self._pmi(words, unigram_freq, bigram_freq)
            edge = (words[0], words[1], {'freq': value, 'weight': round(pmi_val,4)})  # freq, pmi | (word_i, word_j, {'edge_attr': value})
            edges.append(edge)
        return edges

    def _pmi(self, words, unigram_freq, bigram_freq):
        prob_word1 = unigram_freq[words[0]] / float(sum(unigram_freq.values()))
        prob_word2 = unigram_freq[words[1]] / float(sum(unigram_freq.values()))
        prob_word1_word2 = bigram_freq[words] / float(sum(bigram_freq.values()))
        return math.log(prob_word1_word2/float(prob_word1*prob_word2),2)

    def _handle_stop_words(self, text) -> str:
        tokens = nltk.word_tokenize(text)
        without_stopwords = [word for word in tokens if not word.lower().strip() in self.stop_words]
        return " ".join(without_stopwords)

    def _text_normalize(self, text: str) -> list:
        if self.apply_prep:
            if self.steps_prep['to_lowercase']:
                text = text.lower() # text to lower case
            if self.steps_prep['handle_blank_spaces']:
                text = re.sub(r'\s+', ' ', text).strip() # remove blank spaces
            if self.steps_prep['handle_html_tags']:
                text = re.compile('<.*?>').sub(r'', text) # remove html tags
            if self.steps_prep['handle_special_chars']:
                text = re.sub('[^A-Za-z0-9]+ ', '', text) # remove special chars
                text = re.sub('\W+ ','', text)
                text = text.replace('"',"")
            if self.steps_prep['handle_stop_words']:
                text = self._handle_stop_words(text) # remove stop words
        return text

    def _nlp_pipeline(self, docs: list, params = {'get_multilevel_lang_features': False}):
        doc_tuples = []
        Doc.set_extension("multilevel_lang_info", default=[], force=True)
        for doc, context in list(self.nlp.pipe(docs, as_tuples=True, n_process=4, batch_size=1000)):
            if params['get_multilevel_lang_features'] == True:
                doc._.multilevel_lang_info = self.get_multilevel_lang_features(doc)
            doc_tuples.append((doc, context))
        return doc_tuples

    def _build_graph(self, nodes: list, edges: list) -> networkx:
        if self.graph_type == 'DiGraph':
            graph = nx.DiGraph()
        else:
            graph = nx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        return graph

    def _transform_cooc_pipeline(self, doc_instance: tuple) -> list:
        output_dict = {
            'doc_id': doc_instance['id'],
            'context': doc_instance['context'],
            'graph': None,
            'number_of_edges': 0,
            'number_of_nodes': 0,
            'status': 'success'
        }
        try:
            # get_entities
            nodes = self._get_entities(doc_instance['doc'])
            # get_relations
            edges = self._get_relations(doc_instance['doc'])
            # build graph
            graph = self._build_graph(nodes, edges)
            output_dict['number_of_edges'] += graph.number_of_edges()
            output_dict['number_of_nodes'] += graph.number_of_nodes()
            output_dict['graph'] = graph
        except Exception as e:
            print('Error: %s', str(e))
            output_dict['status'] = 'fail'
        finally:
            return output_dict

    def transform(self, corpus_texts) -> list:
        print("Init transformations: Text to Co-Ocurrence Graph")
        print("Transforming %s text documents...", len(corpus_texts))
        prep_docs, corpus_output_graph, delayed_func = [], [], []

        for doc_data in corpus_texts:
            if self.apply_prep == True:
                doc_data['doc'] = self._text_normalize(doc_data['doc'])
            prep_docs.append(
                (doc_data['doc'], {'id': doc_data['id'], "context": doc_data['context']})
            )

        docs = self._nlp_pipeline(prep_docs)

        for doc, context in list(docs):
            corpus_output_graph.append(
                self._transform_cooc_pipeline(
                    {
                        'id': context['id'],
                        'doc': doc,
                        'context': context['context']
                    }
                )
            )

        print("Done transformations")

        return corpus_output_graph

class Text2HeteroGraph():
    def __init__(self, graph_type, apply_prep=True, parallel_exec=False, window_size=1, language='en', steps_preprocessing={}):
        self.apply_prep = apply_prep
        self.window_size = window_size
        self.graph_type = graph_type
        self.parallel_exec = parallel_exec
        self.language = language
        self.steps_prep = steps_preprocessing
        self.stop_words = set(stopwords.words('english'))

        exclude_modules = ["ner", "textcat"]
        self.nlp = spacy.load('en_core_web_sm', exclude=exclude_modules)
        self.nlp.tokenizer = custom_tokenizer(self.nlp)

    def __get_windows(self, doc_words_list, window_size):
        word_window_freq = defaultdict(int)
        word_pair_count = defaultdict(int)
        len_doc_words_list = len(doc_words_list)
        len_windows = 0

        for i, doc in enumerate(doc_words_list):
            windows = []
            doc_words = doc['words']
            length = len(doc_words)

            if length <= window_size:
                windows.append(doc_words)
            else:
                for j in range(length - window_size + 1):
                    window = doc_words[j: j + window_size]
                    windows.append(list(set(window)))
            for window in windows:
                for word in window:
                    word_window_freq[word] += 1
                for word_pair in itertools.combinations(window, 2):
                    word_pair_count[word_pair] += 1
            len_windows += len(windows)

        return word_window_freq, word_pair_count, len_windows

    def __get_pmi(self, doc_words_list, window_size):
        word_window_freq, word_pair_count, len_windows = self.__get_windows(doc_words_list, window_size)
        word_to_word_pmi = []
        for word_pair, count in word_pair_count.items():
            word_freq_i = word_window_freq[word_pair[0]]
            word_freq_j = word_window_freq[word_pair[1]]
            pmi = log((1.0 * count / len_windows) / (1.0 * word_freq_i * word_freq_j/(len_windows * len_windows)))
            if pmi <= 0:
                continue
            word_to_word_pmi.append((word_pair[0], word_pair[1], {'weight': round(pmi, 2)}))
        return word_to_word_pmi

    def __get_tfidf(self, corpus_docs_list, vocab):
        vectorizer = TfidfVectorizer(vocabulary=vocab, norm=None, use_idf=True, smooth_idf=False, sublinear_tf=False, lowercase=False, tokenizer=None)
        tfidf = vectorizer.fit_transform(corpus_docs_list)
        words_docs_tfids = []
        len_tfidf = tfidf.shape[0]

        for ind, row in enumerate(tfidf):
            for col_ind, value in zip(row.indices, row.data):
                edge = ('D-' + str(ind+1), vocab[col_ind], {'weight': round(value, 2)})
                words_docs_tfids.append(edge)
        return words_docs_tfids

    def _handle_stop_words(self, text) -> str:
        tokens = nltk.word_tokenize(text)
        without_stopwords = [word for word in tokens if not word.lower().strip() in self.stop_words]
        return " ".join(without_stopwords)

    def _nlp_pipeline(self, docs: list, params = {'get_multilevel_lang_features': False}):
        doc_tuples = []
        Doc.set_extension("multilevel_lang_info", default=[], force=True)
        for doc, context in list(self.nlp.pipe(docs, as_tuples=True, n_process=4, batch_size=1000)):
            if params['get_multilevel_lang_features'] == True:
                doc._.multilevel_lang_info = self.get_multilevel_lang_features(doc)
            doc_tuples.append((doc, context))
        return doc_tuples

    def _text_normalize(self, text: str) -> list:
        if self.apply_prep:
            if self.steps_prep['to_lowercase']:
                text = text.lower() # text to lower case
            if self.steps_prep['handle_blank_spaces']:
                text = re.sub(r'\s+', ' ', text).strip() # remove blank spaces
            if self.steps_prep['handle_html_tags']:
                text = re.compile('<.*?>').sub(r'', text) # remove html tags
            if self.steps_prep['handle_special_chars']:
                text = re.sub('[^A-Za-z0-9]+ ', '', text) # remove special chars
                text = re.sub('\W+ ','', text)
                text = text.replace('"',"")
            if self.steps_prep['handle_stop_words']:
                text = self._handle_stop_words(text) # remove stop words
        return text

    def __get_entities(self, doc_words_list: list) -> list:
        nodes = []
        for d in doc_words_list:
            node_doc =  ('D-' + str(d['doc']), {})
            nodes.append(node_doc)
            for word in d['words']:
                node_word = (str(word), {})
                nodes.append(node_word)
        return nodes

    def __get_relations(self, corpus_docs_list, doc_words_list, vocab) -> list:
        edges = []
        #tfidf
        word_to_doc_tfidf = self.__get_tfidf(corpus_docs_list, vocab)
        edges.extend(word_to_doc_tfidf)
        #pmi
        word_to_word_pmi = self.__get_pmi(doc_words_list, self.window_size)
        edges.extend(word_to_word_pmi)
        return edges

    def __build_graph(self, nodes: list, edges: list) -> networkx:
        if self.graph_type == 'DiGraph':
            graph = nx.DiGraph()
        else:
            graph = nx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        return graph

    def __transform_pipeline(self, corpus_docs: list) -> list:
        output_dict = {
            'doc_id': 1,
            'graph': None,
            'number_of_edges': 0,
            'number_of_nodes': 0,
            'status': 'success'
        }
        try:
            #1. text preprocessing
            corpus_docs_list = []
            doc_words_list = []
            len_corpus_docs = len(corpus_docs)
            vocab = set()
            delayed_func = []
            prep_docs = []

            for doc_data in corpus_docs:
                if self.apply_prep == True:
                    doc_data['doc'] = self._text_normalize(doc_data['doc'])
                prep_docs.append((doc_data['doc'], {'id': doc_data['id']}))

            docs = self._nlp_pipeline(prep_docs)

            for doc, context in docs:
                doc_tokens = [str(token.text) for token in doc] # text,  lemma_
                corpus_docs_list.append(str(" ".join(doc_tokens)))
                doc_words_list.append({'doc': context['id'], 'words': doc_tokens})
                vocab.update(set(doc_tokens))

            #2. get node/entities
            nodes = self.__get_entities(doc_words_list)
            #3. get edges/relations
            edges = self.__get_relations(corpus_docs_list, doc_words_list, list(vocab))
            #4. build graph
            graph = self.__build_graph(nodes, edges)
            output_dict['number_of_edges'] = graph.number_of_edges()
            output_dict['number_of_nodes'] = graph.number_of_nodes()
            output_dict['graph'] = graph
        except Exception as e:
            print('Error: %s', str(e))
            output_dict['status'] = 'fail'
        finally:
            return output_dict

    def transform(self, corpus_docs: list) -> list:
        print("Init transformations: Text to Heterogeneous Graph")
        print("Transforming %s text documents...", len(corpus_docs))
        corpus_output_graph = [self.__transform_pipeline(corpus_docs)]
        print("Done transformations")
        return corpus_output_graph

def cooccur_graph_instance(steps_preprocessing, lang='en', window_size=5):
    # create co_occur object
    #co_occur = Cooccurrence(
    co_occur = Text2CoocGraph(
            graph_type = 'DiGraph',
            window_size = window_size,
            apply_prep = True,
            steps_preprocessing = steps_preprocessing,
            language = lang, #es, en, fr
        )
    return co_occur

def hetero_graph_instance(steps_preprocessing, lang='en', window_size=5):
    # create co_occur object
    #hetero_graph = Heterogeneous(
    hetero_graph = Text2HeteroGraph(
        graph_type = 'DiGraph',
        window_size = window_size,
        apply_prep = True,
        steps_preprocessing = steps_preprocessing,
        language = lang, #sp, en, fr
    )
    return hetero_graph

def isg_graph_instance(lang='en'):
    # create isg object
    isg = ISG(
        graph_type = 'DiGraph',
        apply_prep = True,
        steps_preprocessing = {
            "handle_blank_spaces": True,
            "handle_emoticons": True,
            "to_lowercase": True,
            "handle_html_tags": True,
            "handle_contractions": True,
            "handle_stop_words": False,
            "handle_special_chars": False,
        },
        parallel_exec = False,
        language = lang, #spanish (sp), english (en), french (fr)
        output_format = 'networkx'
    )
    return isg

def transform(corpus_text_docs, steps_preprocessing, type='cooc', window_size=5):
    print("Init transform text to graph: ")
    if type == 'cooc':
      t2graph = cooccur_graph_instance(steps_preprocessing=steps_preprocessing, window_size=window_size)
    elif type == 'hetero':
      t2graph = hetero_graph_instance(steps_preprocessing=steps_preprocessing, window_size=window_size)
    elif type == 'isg':
      t2graph = isg_graph_instance()
    else:
      t2graph = cooccur_graph_instance()

    # Apply t2g transformation
    start_time = time.time() # time init
    graph_output = t2graph.transform(corpus_text_docs)
    for corpus_text_doc in corpus_text_docs:
        for g in graph_output:
            if g['doc_id'] == corpus_text_doc['id']:
                g['context'] = corpus_text_doc['context']
                break
    end_time = (time.time() - start_time)
    print("\t * TOTAL TIME:  %s seconds" % end_time)
    return graph_output


# *************** DATASETS
def read_custom_dataset():
  return [
      {'id': 1, 'doc': "Artificial Intelligence is the ability of a digital computer or computer-controlled robot to perform tasks commonly associated with intelligent beings !!!!!!", "context": {"target": 1}},
      {'id': 2, 'doc': "Natural      language processing refers to the branch of computer science that focus on the ability of computers to understand text and spoken words in much the same way human beings can", "context": {"target": 1}},
      {'id': 3, 'doc': "Authorship verification @ is the task of deciding whether two texts have been written by the same author based on comparing the texts' writing styles", "context": {"target": 0}},
      {'id': 4, 'doc': "Feature     extraction !!!! refers to the process of transforming raw data into numerical features ", "context": {"target": 1}},
      {'id': 5, 'doc': "A graph (test) neural network is a class of artificial neural networks for processing data that can be represented as graphs.", "context": {"target": 0}},
      {'id': 6, 'doc': "The concept  ¿? of artificial intelligence (AI) is to create machines that can learn, reason, and act in ways that would normally require human intelligence. AI is a broad field that draws from many disciplines, including computer science, data analytics, and neuroscience.", "context": {"target": 0}},
  ]

def read_20_newsgroups_dataset(subset='train'):
    newsgroups_dataset = fetch_20newsgroups(subset=subset) #subset='train', fetch from sci-kit learn
    id = 1
    corpus_text_docs = []
    for index in range(len(newsgroups_dataset.data)):
        doc = {"id": id, "doc": newsgroups_dataset.data[index], "context": {"target": newsgroups_dataset.target[index]}}
        corpus_text_docs.append(doc)
        id += 1
    return corpus_text_docs

def read_semeval_dataset():
  ...

def read_coling25_dataset():
  #dataset = load_dataset("Jinyan1/COLING_2025_MGT_en")
  #return dataset
  train_set = read_jsonl(dir_path=f'{DATASET_DIR}coling2024/en_train.jsonl')
  val_set = read_jsonl(dir_path=f'{DATASET_DIR}coling2024/en_dev.jsonl')
  test_set = read_jsonl(dir_path=f'{DATASET_DIR}coling2024/test_set_en_with_label.jsonl')
  return train_set, val_set, test_set

def read_autext23_dataset():
  train_set = read_csv(file_path=f'{DATASET_DIR}autext2023/subtask1/train_set.csv') 
  val_set = read_csv(file_path=f'{DATASET_DIR}autext2023/subtask1/val_set.csv') 
  test_set = read_csv(file_path=f'{DATASET_DIR}autext2023/subtask1/test_set.csv') 
  return train_set, val_set, test_set

def read_pan24_dataset(dataset_dir, subset='train'):
  text_data_lst = []
  if subset=='train':
    corpus_text_docs = read_json(file_path=dataset_dir + 'pan24/Partition_A_train_set.jsonl')
    corpus_text_docs = shuffle(corpus_text_docs)
    corpus_text_docs_dict = corpus_text_docs.to_dict('records')
    for instance in corpus_text_docs_dict:
      if len(instance['text'].split()) < 10:
          continue
      doc = {
          "id": instance['id'],
          "doc": instance['text'][:],
          "context": {"id": instance['id'], "topic": instance['topic'], "target": instance['class'],  "model": instance['model'],  "art-id": instance['art-id'] }
      }
      text_data_lst.append(doc)
    return text_data_lst

  else:
    corpus_text_docs = read_json(file_path=dataset_dir + 'pan24/Partition_A_test_set.jsonl')
    corpus_text_docs = shuffle(corpus_text_docs)
    corpus_text_docs_dict = corpus_text_docs.to_dict('records')
    for instance in corpus_text_docs_dict:
      if len(instance['text1'].split()) < 10:
          continue
      if len(instance['text2'].split()) < 10:
          continue
      doc1 = {
          "id": 'txt1' + str(instance['id']),
          "doc": instance['text1'][:],
          "context": {"problem_id": instance['id'], "model": instance['model1'], "target": instance['label1']}
      }
      doc2 = {
          "id": 'txt2' + str(instance['id']),
          "doc": instance['text2'][:],
          "context": {"problem_id": instance['id'], "model": instance['model2'], "target": instance['label2']}
      }
      text_data_lst.append(doc1)
      text_data_lst.append(doc2)
    return text_data_lst


# *************** UTILS
def read_csv(file_path):
  df = pd.read_csv(file_path)
  return df

def read_json(file_path):
  df = pd.read_json(file_path, lines=True)
  df = df.sort_values('id', ascending=True)
  return df

def read_jsonl(dir_path):
    return pd.read_json(path_or_buf=dir_path, lines=True)

def save_data(data, file_name, path='/', format_file='.pkl', compress=False):
    path_file = os.path.join(path, file_name + format_file)
    joblib.dump(data, path_file, compress=compress)

def delete_dir_files(dir_path):
  files = glob.glob(dir_path + '/*')
  for f in files:
      os.remove(f)

def create_dir(dir_path):
  if not os.path.exists(dir_path):
    os.makedirs(dir_path)

def text_metrics(dataframe):
    text_lens, max_text_len, min_text_len = 0, 0, 1000000000
    for index, row in dataframe.iterrows():
        text_i_len = len(row['text'].split())
        text_lens += text_i_len
        if text_i_len > max_text_len:
            max_text_len = text_i_len
        if text_i_len < min_text_len:
            min_text_len = text_i_len
    print("Text AVG Tokens: ", text_lens/len(dataframe))
    print("Text Max Tokens: ", max_text_len)
    print("Text Min Tokens: ", min_text_len)

def process_dataset(corpus_text_docs):
    text_data_lst = []
    corpus_text_docs_dict = corpus_text_docs.to_dict('records')
    for idx, instance in enumerate(corpus_text_docs_dict):
        doc = {
            "id": idx+1,
            "doc": instance['text'][:],
            "label": instance['label'],
            #"model": instance['model'],
            "context": {
                #"id": instance['id'], 
                "id": idx+1, 
                "target": instance['label'],
            }
        }
        text_data_lst.append(doc)
    return text_data_lst

def cut_dataset(corpus_text_docs, cut_percentage_dataset):
  cut_dataset = len(corpus_text_docs) * (int(cut_percentage_dataset) / 100)
  return corpus_text_docs[:int(cut_dataset)]

def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())
    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])
    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()

def get_masks(graph, corpus_all_text_docs, set_idxs):
  # show corpus_graph_docs
  train_mask, val_mask, test_mask, y_mask = [], [], [], []
  t1,t2,t3 = [],[],[]

  # set idxs
  train_idx = set_idxs['train']
  val_idx = set_idxs['train'] + set_idxs['val']
  test_idx = set_idxs['train'] + set_idxs['val'] + set_idxs['test']

  # get train_mask
  doc_cnt = 0
  for idx, node in enumerate(graph['graph'].nodes(data=True)):
    if node[0].startswith("D-"):
      doc_cnt += 1
      if doc_cnt <= train_idx:
        t1.append(node[0])
        train_mask.append(True)
        continue
    train_mask.append(False)

  # get val_mask
  doc_cnt = 0
  for idx, node in enumerate(graph['graph'].nodes(data=True)):
    if node[0].startswith("D-"):
      doc_cnt += 1
      if doc_cnt > train_idx and doc_cnt <= val_idx:
        t2.append(node[0])
        val_mask.append(True)
        continue
    val_mask.append(False)

  # get test_mask
  doc_cnt = 0
  for idx, node in enumerate(graph['graph'].nodes(data=True)):
    if node[0].startswith("D-"):
      doc_cnt += 1
      if doc_cnt > train_idx and doc_cnt > val_idx and doc_cnt <= test_idx:
        t3.append(node[0])
        test_mask.append(True)
        continue
    test_mask.append(False)

  # obtain y ground truth
  for idx, node in enumerate(graph['graph'].nodes(data=True)):
    #print((node[0], list(graph['graph'].neighbors(node[0])))) # nodes per node
    if node[0].startswith("D-"):
      doc_id = node[0].split('-')[1]
      doc = corpus_all_text_docs[int(doc_id)-1]
      y_mask.append(doc["context"]['target'])
      #print((node[0], list(graph['graph'].neighbors(node[0])))) # nodes per doc
    else:
      y_mask.append(False)

  print("total_docs: ", doc_cnt)
  #print("train_docs: ", t1)
  #print("val_docs:   ", t2)
  #print("test_docs:  ", t3)
  return train_mask, val_mask, test_mask, y_mask

def w2v_train(graph_data, num_features):
    sent_tokens = []
    for g in graph_data:
        sent_tokens.append(list(g['graph'].nodes))
    model_w2v = gensim.models.Word2Vec(sent_tokens, min_count=1,vector_size=num_features, window=3)
    return model_w2v

def d2v_train(data, num_features):
    # preproces the documents, and create TaggedDocuments
    tagged_data = [TaggedDocument(words=word_tokenize(d['doc'].lower()), tags=[str(i)]) for i, d in enumerate(data)]
    # train the Doc2vec model
    model = Doc2Vec(vector_size=num_features, min_count=2, epochs=50)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    return model

def w2v_train_v2(graph, num_features, set_idxs, graph_type='hetero'):
  X_train, X_val, X_test = [], [], []

  if graph_type == 'hetero':
    cnt_train = 0
    for idx, node in enumerate(graph['graph'].nodes):
        if node.startswith("D-"):
          doc_id = node.split('-')[1]
          #print(doc_id)
          if int(doc_id) <= set_idxs['train']:
            X_train.append(list(graph['graph'].neighbors(node)))
          elif int(doc_id) <= set_idxs['train'] + set_idxs['val']:
            X_val.append(list(graph['graph'].neighbors(node)))
          elif int(doc_id) <= set_idxs['train'] + set_idxs['val'] + set_idxs['test']:
            X_test.append(list(graph['graph'].neighbors(node)))
          else:
            X_train.append(node)

  if graph_type == 'cooc':
    graph_train = graph[ : set_idxs['train']]
    graph_val = graph[set_idxs['train'] : set_idxs['train']+set_idxs['val']]
    graph_test = graph[set_idxs['train']+set_idxs['val'] : ]
    for g in graph_train:
      X_train.append(list(g['graph'].nodes))
    for g in graph_val:
      X_val.append(list(g['graph'].nodes))
    for g in graph_test:
      X_test.append(list(g['graph'].nodes))

  model_w2v = gensim.models.Word2Vec(X_train, min_count=1,vector_size=num_features, window=3)
  words = set(model_w2v.wv.index_to_key)
  print("X_train_val_test: ", len(X_train), len(X_val), len(X_test))
  print("words: ", len(words), model_w2v)

  X_train_vect = [[model_w2v.wv[i] for i in ls if i in words] for ls in X_train]
  X_val_vect = [[model_w2v.wv[i] for i in ls if i in words] for ls in X_val]
  X_test_vect = [[model_w2v.wv[i] for i in ls if i in words] for ls in X_test]

  X_train_vect_avg = []
  for vect in X_train_vect:
      vect = np.array(vect)
      if vect.size:
          X_train_vect_avg.append(vect.mean(axis=0))
      else:
          X_train_vect_avg.append(np.zeros(num_features, dtype=float))

  X_val_vect_avg = []
  for vect in X_val_vect:
      vect = np.array(vect)
      if vect.size:
          X_val_vect_avg.append(vect.mean(axis=0))
      else:
          X_val_vect_avg.append(np.zeros(num_features, dtype=float))

  X_test_vect_avg = []
  for vect in X_test_vect:
      vect = np.array(vect)
      if vect.size:
          X_test_vect_avg.append(vect.mean(axis=0))
      else:
          X_test_vect_avg.append(np.zeros(num_features, dtype=float))

  X_all_vect_avg = []
  X_all_vect_avg.extend(X_train_vect_avg)
  X_all_vect_avg.extend(X_val_vect_avg)
  X_all_vect_avg.extend(X_test_vect_avg)

  return model_w2v, X_all_vect_avg

def llm_get_embbedings(dataset, subset, emb_type='llm_cls', graph_type='hetero', device='cpu', output_path='', save_emb=False, llm_finetuned_name='', num_labels=2, print_info=False):
  print('inside llm_get_embbedings')
  tokenizer = AutoTokenizer.from_pretrained(llm_finetuned_name)
  model = AutoModel.from_pretrained(llm_finetuned_name)
  model = model.to(device)
  model.eval()

  if emb_type == 'llm_cls':
      with torch.no_grad():
          embeddings_lst = []
          for row in tqdm(dataset):
              inputs = tokenizer(row["text"], return_tensors='pt', padding=True, truncation=True, max_length=512)
              inputs.to(device)
              outputs_model = model(**inputs, output_hidden_states=True)
              last_hidden_state = outputs_model.hidden_states[-1]
              embedding = last_hidden_state[0,0,:].cpu().detach().numpy().tolist()
              embeddings_lst.append(embedding)
          return embeddings_lst

  if emb_type == 'llm_word':
      with torch.no_grad():
          embeddings_word_dict = {}
          cnt = 1
          word_cnt = 0
          found_tokens_total = 0
          set_tokens = set()
          set_raw_tokens = set()
          lst_raw_tokens = []

          for index, row  in enumerate(tqdm(dataset)):
            #print("row: ", row)
            inputs = tokenizer(row["text"], return_tensors='pt', padding=True, truncation=True, max_length=512)
            inputs.to(device)
            outputs_model = model(**inputs, output_hidden_states=True)
            last_hidden_state = outputs_model.hidden_states[-1]
            word_cnt += last_hidden_state.shape[1]
            #print("len_input_ids: ", inputs['input_ids'][0].shape)
            found_tokens = 0
            for token in row['text'].split():
              set_tokens.add(token)
            
            if graph_type == 'cooc':
              embeddings_word_dict[str(row['id'])] = {"doc_id": row['id'], 'label': row['label'], 'embedding': {}} 

            for i in range(0, len(last_hidden_state)):
                raw_tokens = [tokenizer.decode([token_id]) for token_id in inputs['input_ids'][i]]
                lst_raw_tokens.extend(raw_tokens)
                #print(len(raw_tokens), len(last_hidden_state[i]))
                #print(raw_tokens)
                #print(row['text'].split())
                for token, embedding in zip(raw_tokens, last_hidden_state[i]):
                    #print(str(token).strip())
                    set_raw_tokens.add(str(token).strip())
                    if str(token).strip() in row['text'].split():
                      found_tokens += 1
                    if graph_type == 'cooc':
                      embeddings_word_dict[str(row['id'])]['embedding'][str(token).strip()] = embedding.cpu().detach().numpy().tolist()
                    if graph_type == 'hetero':
                      embeddings_word_dict[str(token).strip()] = embedding.cpu().detach().numpy().tolist()

            found_tokens_total += found_tokens
            if print_info:
              print("row: ", last_hidden_state.shape, inputs['input_ids'][0].shape, found_tokens, len(row['text'].split())) #row['id']
            #if cnt == 100:
            #  break
            cnt += 1

          if print_info:
            print("*** word_cnt: ", word_cnt)
            print("*** raw_tokens Counter: ", Counter(lst_raw_tokens))
            print("*** found_tokens_total: ", found_tokens_total)
            print("*** embeddings_word_dict: ", len(embeddings_word_dict.keys()))
            print("*** text set_tokens: ", len(set_tokens))
            print("*** set_raw_tokens: ", len(set_raw_tokens))


      return embeddings_word_dict


# *************** GNNs
class BuildDataset():
    def __init__(self, graphs_data, corpus_texts, set_idxs, subset, device, nfi='w2v', num_labels=2, num_features=256, graph_type='cooc', oov_feat_type='random', doc_embs=None, model_w2v=None, llm_finetuned_name=''):
        self.graphs_data = graphs_data
        self.nfi_model = {}
        self.subset = subset
        self.nfi = nfi
        self.num_labels = num_labels
        self.num_features = num_features
        self.graph_type = graph_type
        self.nfi_model_w2v = model_w2v
        self.doc_embs = doc_embs
        self.device = device
        self.corpus_texts = corpus_texts
        self.set_idxs = set_idxs
        self.oov_feat_type = oov_feat_type # random, remove
        self.llm_finetuned_name = llm_finetuned_name

    def process_dataset(self):
      data_list = []

      if self.graph_type == 'hetero':
        if self.nfi == 'llm':
          dataset, lst_nodes = [], []
          cnt = 1
          for idx, node in enumerate(list(self.graphs_data['graph'].nodes)): # for hetero
            if str(node).startswith("D-"):
              continue
            lst_nodes.append(str(node))
            if cnt == 100:
              #emb_dict = llm_get_embbedings(dataset=[{'text': " ".join(lst_nodes)}], subset=self.subset, emb_type='llm_word', device=self.device, save_emb=False, llm_finetuned_name=self.llm_finetuned_name, num_labels=self.num_labels)
              #for k, v in emb_dict.items():
              #    tmp_nfi_model.setdefault(k, []).append(v)
              dataset.append({'text': " ".join(lst_nodes)})
              lst_nodes = []
              cnt = 0
            cnt += 1

          self.nfi_model = llm_get_embbedings(dataset, subset=self.subset, emb_type='llm_word', graph_type=self.graph_type, device=self.device, save_emb=False, llm_finetuned_name=self.llm_finetuned_name, num_labels=self.num_labels, print_info=False)
          print("nfi_model.keys: ", len(self.nfi_model.keys()))
        
        else: 
          self.nfi_model = self.nfi_model_w2v
        
        try:
          # Get node features
          node_feats, oov_cnt = self.get_node_features(self.graphs_data, nfi_type=self.nfi)
          train_mask, val_mask, test_mask, y_mask = get_masks(self.graphs_data, self.corpus_texts, self.set_idxs)

          # Get adjacency info
          edge_index = self.get_adjacency_info(self.graphs_data['graph'])
          data = Data(
              x = node_feats,
              edge_index = edge_index,
              y = torch.tensor(np.asarray(y_mask), dtype=torch.int64),
              train_mask = torch.tensor(np.asarray(train_mask), dtype=torch.bool),
              val_mask = torch.tensor(np.asarray(val_mask), dtype=torch.bool),
              test_mask = torch.tensor(np.asarray(test_mask), dtype=torch.bool),
          )
          data_list.append(data)
        except Exception as e:
          ...
          print(e)
        else:
          print("oov_cnt_total: ", oov_cnt)

      if self.graph_type == 'cooc':
        if self.nfi == 'llm':
          dataset = [{'id': str(d['context']['id']), 'label': d['context']['target'], 'text': " ".join(list(d['graph'].nodes))} for d in self.graphs_data]
          self.nfi_model = llm_get_embbedings(dataset, subset=self.subset, emb_type='llm_word', graph_type=self.graph_type, device=self.device, save_emb=False, llm_finetuned_name=self.llm_finetuned_name, num_labels=self.num_labels, print_info=False)
          print("nfi_model.keys: ", len(self.nfi_model.keys()))
        else: 
          self.nfi_model = self.nfi_model_w2v

        oov_cnt_total = 0
        for index_in_batch, g in enumerate(tqdm(self.graphs_data)):
          #print(g['graph'])
          try:
              # Get node features
              node_feats, oov_cnt = self.get_node_features(g, nfi_type=self.nfi)
              # Get adjacency info
              edge_index = self.get_adjacency_info(g['graph'])
              # Get labels info
              label = self.get_labels(g["context"]["target"])

              #print(node_feats.shape, edge_index.shape, label.shape)
              data = Data(
                  x = node_feats,
                  edge_index = edge_index,
                  y = label,
                  pred = '',
                  context = g["context"]
              )
              data_list.append(data)
              oov_cnt_total += oov_cnt
          except Exception as e:
              ...
              print(e)
              #print(g)
          else:
              ...
              #print(g['graph'], " | oov_cnt: ", oov_cnt)
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
          words = set(self.nfi_model_w2v.wv.index_to_key)
          doc_idx = 0
          for node in list(g['graph'].nodes):
              try:
                if str(node).startswith("D-"):
                  d_emb = self.doc_embs[doc_idx]
                  #d_emb = self.get_random_emb(emb_dim=self.num_features)
                  #d_emb = torch.ones(self.num_features)
                  #d_emb = torch.zeros(self.num_features)
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
                #g['graph'].remove_node(node)
                #graph_node_feat.append(self.get_random_emb(emb_dim=self.num_features))
                #oov_cnt += 1
          #print("doc_idx: ", doc_idx)
          #print("oov_cnt: ", oov_cnt)

        elif nfi_type == 'llm':
          doc_idx = 0
          for node in list(g['graph'].nodes):
            try:
              if str(node).startswith("D-"):
                d_emb = self.doc_embs[doc_idx]
                #d_emb = self.get_random_emb(emb_dim=self.num_features)
                graph_node_feat.append(d_emb)
                doc_idx += 1
              else:
                if self.graph_type == 'hetero':
                  if str(node) in self.nfi_model.keys():
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
                elif self.graph_type == 'cooc':
                  if str(node) in self.nfi_model[str(g['id'])].keys():
                    graph_node_feat.append(self.nfi_model[str(node)])
                  else:
                    graph_node_feat.append(self.get_random_emb(emb_dim=self.num_features))
                    #g['graph'].remove_node(node)
                    #graph_node_feat.append(self.nfi_model_w2v.wv(str(node)))
                    oov_cnt += 1

            except Exception as e:
              #print('Error: %s', str(e))
              graph_node_feat.append(self.get_random_emb(emb_dim=self.num_features))
              #oov_cnt += 1
          #print("doc_idx: ", doc_idx)
          #print("oov_cnt: ", oov_cnt)

        elif nfi_type == 'ones': # generate a vect-emb of 1s
            word_nodes = list(g['graph'].nodes)
            graph_node_feat = [torch.ones(self.num_features) for indx, word_node in enumerate(word_nodes)]

        elif nfi_type == 'identity': # generate a vect-emb of 0s
            #word_vectors_emb = sp.identity(len(list(g['graph'].nodes)))
            #graph_node_feat = word_vectors_emb.toarray()

            graph_node_feat = np.eye(len(list(g['graph'].nodes)))
            #graph_node_feat = sp.csr_matrix(I).toarray()

        else: # random init
            # initialise an Embedding layer from Torch
            word_nodes = list(g['graph'].nodes)
            encoded_word_nodes = [indx for indx, word_node in enumerate(word_nodes)]
            emb = nn.Embedding(len(word_nodes), self.num_features)
            word_vectors_emb = emb(torch.tensor(encoded_word_nodes))
            graph_node_feat = word_vectors_emb.detach().numpy()

        #return torch.from_numpy(graph_node_feat).to_sparse(), oov_cnt
        graph_node_feat = np.asarray(graph_node_feat)
        return torch.tensor(graph_node_feat, dtype=torch.float), oov_cnt


    def get_adjacency_info(self, g):
        adj_tmp = nx.to_scipy_sparse_array(g,  weight='weight', dtype=np.cfloat)
        adj_coo = sp.coo_array(adj_tmp)
        #print("adj_coo: ", adj_coo)
        edge_indices = []
        for index in range(len(g.edges)):
            edge_indices += [[adj_coo.row[index], adj_coo.col[index]]]

        edge_indices = torch.tensor(edge_indices)
        t = edge_indices.t().to(torch.long).view(2, -1)
        #print("edge_index:", t)
        return edge_indices.t().to(torch.long).view(2, -1)


    def get_labels(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)

    def get_random_emb(self, emb_dim):
        emb = nn.Embedding(1, emb_dim)
        word_emb = emb(torch.tensor([0]))
        return word_emb.detach().numpy()[0]

class GNN(torch.nn.Module):
    def __init__(self,
        gnn_type='TransformerConv',
        num_features=768,
        hidden_channels=64,
        out_emb_size=256,
        num_classes=2,
        heads=1,
        dropout=0.5,
        pooling='gmeanp',
        batch_norm='BatchNorm1d',
        layers_convs=3,
        dense_nhid=64,
        edge_dim=None,
        task='graph'
    ):
        super(GNN, self).__init__()
        torch.manual_seed(1234567)

        # setting vars
        self.n_layers = layers_convs
        self.dropout_rate = dropout
        self.dense_neurons = dense_nhid
        self.batch_norm = batch_norm
        self.pooling = pooling
        self.out_emb_size = out_emb_size
        self.top_k_every_n = 2
        self.top_k_ratio = 0.5
        self.edge_dim = edge_dim
        self.task = task

        # setting ModuleList
        self.conv_layers = ModuleList([])
        self.transf_layers = ModuleList([])
        self.pooling_layers = ModuleList([])
        self.bn_layers = ModuleList([])

        # select convolution layer
        GNN_LAYER_BY_NAME = {
            "GCNConv": GCNConv,
            "GATConv": GATConv,
            "GraphConv": GraphConv,
            "TransformerConv": TransformerConv,
        }
        conv_layer = GNN_LAYER_BY_NAME[gnn_type]
        if gnn_type in ['GATConv', 'TransformerConv']:
            self.support_edge_attr = True
        else:
            self.support_edge_attr = False

        # Transformation layer
        if self.support_edge_attr:
            self.conv1 = conv_layer(num_features, hidden_channels, heads, edge_dim=self.edge_dim)
        else:
            self.conv1 = conv_layer(num_features, hidden_channels, heads)

        self.transf1 = Linear(hidden_channels*heads, hidden_channels)

        if batch_norm != None:
            #self.bn1 = BatchNorm1d(hidden_channels*heads)
            self.bn1 = BatchNorm1d(hidden_channels)
            #self.bn1 = LayerNorm(hidden_channels)

        # Other layers
        for i in range(self.n_layers):
            if self.support_edge_attr:
                self.conv_layers.append(conv_layer(hidden_channels, hidden_channels, heads, edge_dim=self.edge_dim))
            else:
                self.conv_layers.append(conv_layer(hidden_channels, hidden_channels, heads))

            self.transf_layers.append(Linear(hidden_channels*heads, hidden_channels))

            if batch_norm != None:
                #self.bn_layers.append(BatchNorm1d(hidden_channels*heads))
                self.bn_layers.append(BatchNorm1d(hidden_channels))
                #self.bn_layers.append(LayerNorm(hidden_channels))
            if pooling == 'topkp':
                if i % self.top_k_every_n == 0:
                    #self.pooling_layers.append(TopKPooling(hidden_channels*heads, ratio=self.top_k_ratio))
                    self.pooling_layers.append(TopKPooling(hidden_channels, ratio=self.top_k_ratio))
            if pooling == 'sagp':
                if i % self.top_k_every_n == 0:
                    self.pooling_layers.append(SAGPooling(hidden_channels, ratio=0.5))

        # Linear layers
        len_lin1_vect = 1
        if self.pooling in ['gmeanp_gaddp']:
            len_lin1_vect = 2

        self.linear1 = Linear(hidden_channels*len_lin1_vect, self.dense_neurons)
        #self.linear2 = Linear(int(self.dense_neurons), num_classes)
        self.linear2 = Linear(self.dense_neurons, int(self.dense_neurons)//2)
        self.linear3 = Linear(int(self.dense_neurons)//2, num_classes)

    def forward(self, x, edge_index, edge_attr, batch):
        # Initial transformation
        if self.support_edge_attr:
            x = self.conv1(x, edge_index, edge_attr)
        else:
            x = self.conv1(x, edge_index)
        x = torch.relu(self.transf1(x))
        #x = x.relu()
        if self.batch_norm != None:
            x = self.bn1(x)

        # Holds the intermediate graph representations only for TopKPooling
        global_representation = []

        # iter trought n_layers, apply convs
        for i in range(self.n_layers):
            if self.support_edge_attr:
                x = self.conv_layers[i](x, edge_index, edge_attr)
            else:
                x = self.conv_layers[i](x, edge_index)
            #x = x.relu()
            x = torch.relu(self.transf_layers[i](x))
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

            if self.batch_norm != None:
                x = self.bn_layers[i](x)

            if self.pooling in ['topkp', 'sagp']:
                if i % self.top_k_every_n == 0 or i == self.n_layers:
                    if self.support_edge_attr:
                        x, edge_index, edge_attr, batch, _, _  = self.pooling_layers[int(i/self.top_k_every_n)](x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
                    else:
                        x, edge_index, _, batch, _, _  = self.pooling_layers[int(i/self.top_k_every_n)](x=x, edge_index=edge_index, batch=batch)
                    global_representation.append(global_mean_pool(x, batch))

        emb = x

        if self.task == 'graph':
          # Aplpy graph pooling
          if self.pooling in ['topkp', 'sagp']:
              x = sum(global_representation)
          elif self.pooling == 'gmeanp':
              x = global_mean_pool(x, batch)
          elif self.pooling == 'gaddp':
              x = global_add_pool(x, batch)
          elif self.pooling == 'gmaxp':
              x = global_max_pool(x, batch)
          elif self.pooling == 'gmeanp_gaddp':
              x = torch.cat([global_mean_pool(x, batch), global_add_pool(x, batch)], dim=1)
          else:
              x = global_mean_pool(x, batch)

        out = torch.relu(self.linear1(x))
        out = F.dropout(out, p=self.dropout_rate, training=self.training)
        out = torch.relu(self.linear2(out))
        out = F.dropout(out, p=self.dropout_rate, training=self.training)
        #out = self.linear3(out)

        out = F.softmax(self.linear3(out), dim=1)

        return x, out
        #return x

class GNN_2(torch.nn.Module):
    def __init__(self, gnn_type, hidden_channels, num_features=256, heads=1, num_classes=2, task='graph'):
        super().__init__()
        torch.manual_seed(1234567)
        self.task = task
        GNN_LAYER_BY_NAME = {
            "GCNConv": GCNConv,
            "GATConv": GATConv,
            "GraphConv": GraphConv,
            "TransformerConv": TransformerConv,
        }
        conv_layer = GNN_LAYER_BY_NAME[gnn_type]

        self.conv1 = conv_layer(num_features, hidden_channels, heads)
        self.conv2 = conv_layer(hidden_channels*heads, hidden_channels, heads)
        self.conv3 = conv_layer(hidden_channels*heads, hidden_channels, heads)
        #self.conv4 = conv_layer((int(hidden_channels)//4)*heads, int(hidden_channels)//4, heads)
        #self.conv5 = conv_layer((int(hidden_channels)//8)*heads, int(hidden_channels)//16, heads)
        self.bn1 = BatchNorm1d(hidden_channels*heads)
        self.linear1 = nn.Linear(hidden_channels*heads, num_classes)
        #self.linear2 = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, _, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = x.relu()
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        x = x.relu()
        #x = F.dropout(x, p=0.5, training=self.training)
        #x = self.conv4(x, edge_index)
        #x = x.relu()
        #x = F.dropout(x, p=0.5, training=self.training)
        #x = self.conv5(x, edge_index)
        #x = x.relu()

        if self.task == 'graph':
          x = global_mean_pool(x, batch)

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.bn1(x)
        out = torch.relu(self.linear1(x))

        #x = F.dropout(x, p=0.6, training=self.training)
        #x = self.conv1(x, edge_index)
        #x = F.relu(x)
        #x = F.dropout(x, p=0.6, training=self.training)
        #x = self.conv2(x, edge_index)

        #out = torch.relu(self.linear1(x))
        #out = F.dropout(out, p=0.6, training=self.training)
        #out = torch.relu(self.linear2(out))
        return x, out

class GCN(torch.nn.Module):
    def __init__(self, gnn_type, hidden_channels, num_features=256, heads=1, num_classes=2, task='graph'):

        super().__init__()
        torch.manual_seed(1234567)

        GNN_LAYER_BY_NAME = {
            "GCNConv": GCNConv,
            "GATConv": GATConv,
            "GraphConv": GraphConv,
            "TransformerConv": TransformerConv,
        }
        conv_layer = GNN_LAYER_BY_NAME[gnn_type]

        self.task = task
        self.conv1 = conv_layer(num_features, hidden_channels, heads=heads)
        self.conv2 = conv_layer(hidden_channels*heads, hidden_channels, heads=heads)
        self.conv3 = conv_layer(hidden_channels*heads, hidden_channels, heads=heads)
        #self.conv4 = conv_layer(hidden_channels*heads, hidden_channels, heads=heads)
        #self.conv5 = conv_layer(hidden_channels*heads, hidden_channels, heads=heads)
        self.bn1 = BatchNorm1d(hidden_channels)
        self.linear1 = nn.Linear(hidden_channels, int(hidden_channels)//2)
        self.linear2 = nn.Linear(int(hidden_channels)//2, num_classes)

        #self.post_mp = nn.Sequential(
        #    nn.Linear(hidden_channels, hidden_channels//2),
        #    nn.Dropout(0.25),
        #    nn.Linear(hidden_channels//2, num_classes)
        #)

    def forward(self, x, edge_index, _, batch):
        #print("init_x: ", x.shape, x[0])
        x = self.conv1(x, edge_index)
        #print("conv1: ", x.shape, x[0])
        x = x.relu()
        #x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        #print("conv2: ", x.shape, x[0])
        x = x.relu()
        #x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv3(x, edge_index)
        ##print("conv3: ", x.shape, x)
        x = x.relu()
        #x = F.dropout(x, p=0.5, training=self.training)
        #x = self.conv4(x, edge_index)
        ##print("conv4: ", x.shape, x)
        #x = x.relu()
        #x = F.dropout(x, p=0.5, training=self.training)
        #x = self.conv5(x, edge_index)
        #print("conv5: ", x.shape, x)
        if self.task == 'graph':
          x = global_mean_pool(x, batch)

        #x = self.post_mp(x)
        x = self.bn1(x)
        out = torch.relu(self.linear1(x))
        out = F.dropout(out, p=0.6, training=self.training)
        out = torch.relu(self.linear2(out))

        #return x, F.log_softmax(x, dim=1)
        return x, out

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss <= self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def train_graph(loader, model, device, criterion, optimizer):
    model.train()
    train_loss = 0.0
    steps = 0
    embeddings_data = []
    all_preds, all_labels = [], []
    for step, data in enumerate(loader):  # Iterate in batches over the training dataset.
        data.to(device)
        emb, out = model(data.x, data.edge_index, None, data.batch)
        embeddings_data.append({'batch': step, 'doc_id': data.context['id'], 'labels': data.y, 'embedding': emb})
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()
        steps += 1
        all_preds.append(out.argmax(dim=1).cpu().detach().numpy())
        all_labels.append(data.y.cpu().detach().numpy())
    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()

    return model, accuracy_score(all_labels, all_preds), train_loss / steps, embeddings_data, loader

def test_graph(loader, model, device, criterion):
    model.eval()
    correct = 0
    test_loss = 0.0
    steps = 0
    pred_loader = []
    embeddings_data = []
    all_preds, all_labels = [], []

    with torch.no_grad():
        for step, data in enumerate(loader):  # Iterate in batches over the training/test dataset.
            data.to(device)
            #print('testing batch...', step)
            emb, out = model(data.x, data.edge_index, None, data.batch)
            embeddings_data.append({'batch': step, 'doc_id': data.context['id'], 'labels': data.y, 'embedding': emb})
            loss = criterion(out, data.y)
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            data.pred = pred
            correct += int((pred == data.y).sum())  # Check against ground-truth labels.
            test_loss += loss.item()
            steps += 1
            pred_loader.append(data)

            all_preds.append(out.argmax(dim=1).cpu().detach().numpy())
            all_labels.append(data.y.cpu().detach().numpy())
        all_preds = np.concatenate(all_preds).ravel()
        all_labels = np.concatenate(all_labels).ravel()

        acc_score = accuracy_score(all_labels, all_preds)
        f1_score_macro = f1_score(all_labels, all_preds, average='macro')
        f1_score_micro = f1_score(all_labels, all_preds, average='micro')

        return test_loss / steps, acc_score, f1_score_macro, f1_score_micro, embeddings_data, pred_loader

def train_node(data, model, criterion, optimizer):
    model.train()
    optimizer.zero_grad()
    emb, out = model(data.x, data.edge_index, None, None)

    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    loss_val, val_acc, val_f1_score_macro, val_f1_score_micro, loss_test, test_acc, test_f1_score_macro, test_f1_score_micro = test_node(data, model, criterion)
    return loss, emb, loss_val, val_acc, val_f1_score_macro, val_f1_score_micro, loss_test, test_acc, test_f1_score_macro, test_f1_score_micro

def test_node(data, model, criterion):
    model.eval()
    with torch.no_grad():
        emb, out = model(data.x, data.edge_index, None, None)
        pred = out.argmax(dim=1)
        loss_val = criterion(out[data.val_mask], data.y[data.val_mask])
        truth_val = data.y[data.val_mask].cpu().detach().numpy().tolist()
        pred_val = pred[data.val_mask].cpu().detach().numpy().tolist()
        val_acc = accuracy_score(truth_val, pred_val)
        val_f1_score_macro = f1_score(truth_val, pred_val, average='macro')
        val_f1_score_micro = f1_score(truth_val, pred_val, average='micro')

        loss_test = criterion(out[data.test_mask], data.y[data.test_mask])
        truth_test = data.y[data.test_mask].cpu().detach().numpy().tolist()
        pred_test = pred[data.test_mask].cpu().detach().numpy().tolist()
        test_acc = accuracy_score(truth_test, pred_test)
        test_f1_score_macro = f1_score(truth_test, pred_test, average='macro')
        test_f1_score_micro = f1_score(truth_test, pred_test, average='micro')

        return loss_val, val_acc, val_f1_score_macro, val_f1_score_micro, loss_test, test_acc, test_f1_score_macro, test_f1_score_micro

def eval_d2v_gnn(doc_embs, graph_type, data, train_set, val_set, test_set, hetero_gnn_emb, cooc_gnn_train_emb, cooc_gnn_val_emb, cooc_gnn_test_emb):
    # doc embeddings (w2v, llm, etc)
    X_train_d2v = np.asarray(doc_embs[:len(train_set)])
    X_val_d2v = np.asarray(doc_embs[len(train_set):len(train_set)+len(val_set)])
    X_test_d2v = np.asarray(doc_embs[len(train_set)+len(val_set):])
    print(X_train_d2v.shape, X_val_d2v.shape, X_test_d2v.shape)

    # ground truth
    y_train = np.asarray(train_set['label'].to_list()[:], dtype=np.float32).reshape(-1, 1)
    y_val = np.asarray(val_set['label'].to_list()[:], dtype=np.float32).reshape(-1, 1)
    y_test = np.asarray(test_set['label'].to_list()[:], dtype=np.float32).reshape(-1, 1)
    print(y_train.shape, y_val.shape, y_test.shape)

    clf_d2v = LinearSVC()
    #clf_d2v = xgb.XGBClassifier(n_jobs=-1)
    clf_model_d2v = clf_d2v.fit(X_train_d2v, y_train)

    #*** PREDICT val-test
    y_val_pred_d2v = clf_model_d2v.predict(X_val_d2v)
    y_test_pred_d2v = clf_model_d2v.predict(X_test_d2v)

    #*** GET metrics

    print("****** clf_d2v")
    print('Val accuracy %s' % accuracy_score(y_val, y_val_pred_d2v))
    print('Val F1 score Macro: ', f1_score(y_val, y_val_pred_d2v, average='macro'))
    print('Val F1 score Micro: ', f1_score(y_val, y_val_pred_d2v, average='micro'))

    print()
    print('Test accuracy %s' % accuracy_score(y_test, y_test_pred_d2v))
    print('Test F1 score Macro: ', f1_score(y_test, y_test_pred_d2v, average='macro'))
    print('Test F1 score Micro: ', f1_score(y_test, y_test_pred_d2v, average='micro'))

    print("\n")
    # best_embs GNN

    if graph_type == 'cooc':
        X_train_gnn, X_val_gnn, X_test_gnn = [], [], []
        y_train_gnn, y_val_gnn, y_test_gnn = [], [], []
        for t_emb in cooc_gnn_train_emb:
            for label, emb in zip(t_emb['labels'], t_emb['embedding']):
                X_train_gnn.append(emb.cpu().detach().numpy())
                y_train_gnn.append(label.cpu().detach().numpy())
        for t_emb in cooc_gnn_val_emb:
            for label, emb in zip(t_emb['labels'], t_emb['embedding']):
                X_val_gnn.append(emb.cpu().detach().numpy())
                y_val_gnn.append(label.cpu().detach().numpy())
        for t_emb in cooc_gnn_test_emb:
            for label, emb in zip(t_emb['labels'], t_emb['embedding']):
                X_test_gnn.append(emb.cpu().detach().numpy())
                y_test_gnn.append(label.cpu().detach().numpy())
        # doc embeddings (w2v, llm, etc)
        X_train_gnn = np.asarray(X_train_gnn)
        X_val_gnn = np.asarray(X_val_gnn)
        X_test_gnn = np.asarray(X_test_gnn)
        y_train = np.asarray(y_train_gnn, dtype=np.float32).reshape(-1, 1)
        y_val = np.asarray(y_val_gnn, dtype=np.float32).reshape(-1, 1)
        y_test = np.asarray(y_test_gnn, dtype=np.float32).reshape(-1, 1)


    if graph_type == 'hetero':
        X_train_gnn = hetero_gnn_emb[data.train_mask].cpu().detach().numpy()
        X_val_gnn = hetero_gnn_emb[data.val_mask].cpu().detach().numpy()
        X_test_gnn = hetero_gnn_emb[data.test_mask].cpu().detach().numpy()

    print(X_train_gnn.shape, X_val_gnn.shape, X_test_gnn.shape)

    clf_gnn = LinearSVC()
    #clf_gnn = xgb.XGBClassifier(n_jobs=-1)
    clf_model_gnn = clf_gnn.fit(X_train_gnn, y_train)

    #*** PREDICT val-test
    y_val_pred_gnn = clf_model_gnn.predict(X_val_gnn)
    y_test_pred_gnn = clf_model_gnn.predict(X_test_gnn)

    #*** GET metrics
    print("****** clf_gnn")
    print('Val accuracy %s' % accuracy_score(y_val, y_val_pred_gnn))
    print('Val F1 score Macro: ', f1_score(y_val, y_val_pred_gnn, average='macro'))
    print('Val F1 score Micro: ', f1_score(y_val, y_val_pred_gnn, average='micro'))
    print()
    print('Test accuracy %s' % accuracy_score(y_test, y_test_pred_gnn))
    print('Test F1 score Macro: ', f1_score(y_test, y_test_pred_gnn, average='macro'))
    print('Test F1 score Micro: ', f1_score(y_test, y_test_pred_gnn, average='micro'))
    print()






# *************** MAIN
def main():
    #*** Read Dataset
    # Autext 2023
    #train_set, val_set, test_set = read_autext23_dataset() 
    # coling 2024
    train_set, val_set, test_set = read_coling25_dataset()

    cutoff_train, cutoff_val, cutoff_test = 5, 1, 5

    train_set['word_len'] = train_set['text'].str.split().str.len()
    print("min_token_len: ", train_set['word_len'].min())
    print("max_token_len: ", train_set['word_len'].max())
    print("avg_token_len: ", train_set['word_len'].mean())
    #train_set = train_set[train_set['word_len'] <= 1000]
    #train_set = train_set[train_set['word_len'] >= 10]
    print("label_distro_train_val_test: ", train_set.value_counts('label'), val_set.value_counts('label'), test_set.value_counts('label'))
    print(train_set.nlargest(5, ['word_len']) )
    

    #*** Setting Params General
    cuda_num = 0
    num_features = 768 # llm: 768 | w2v: 128, 256, 512
    nfi = 'llm' # w2v, llm
    oov_feat_type = 'remove' # remove, random, zeros, ones
    graph_type = 'hetero' # cooc, hetero
    num_classes = 2
    batch_size_gnn = 128 # only for cooc
    
    llm_finetuned_name = 'andricValdez/roberta-base-finetuned-autext23' # andricValdez/roberta-base-finetuned-autext23 | andricValdez/bert-base-uncased-finetuned-autext23

    #*** Settings GNN Params
    gnn_type = 'GATConv'
    heads_gnn = 1
    hidden_channels = 128
    dense_nhid = 64
    learning_rate = 0.0001
    epochs = 500
    best_val_loss = 100000
    best_val_acc = 0
    best_val_macro_f1score = 0
    best_val_micro_f1score = 0
    best_epoch = 1
    early_stopper = EarlyStopper(patience=15, min_delta=0)
    device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")

    #*** Settings T2G Params
    window_size = 5
    steps_preprocessing = {
        "handle_blank_spaces": True,
        "handle_emoticons": True,
        "to_lowercase": True,
        "handle_html_tags": True,
        "handle_contractions": True,
        "handle_stop_words": False,
        "handle_special_chars": False,
    }


    #*** cutoff dataset
    train_set = cut_dataset(train_set, cutoff_train)
    val_set = cut_dataset(val_set, cutoff_val)
    test_set = cut_dataset(test_set, cutoff_test)
    all_sets = pd.concat([train_set, val_set, test_set], axis=0)
    corpus_all_text_docs = process_dataset(all_sets)
    print(all_sets.info())
    print("train-val-test: ", train_set.shape, val_set.shape, test_set.shape)
    print("label_distro_train_val_test: ", train_set.value_counts('label'), val_set.value_counts('label'), test_set.value_counts('label'))
    
    #*** Text to Graph
    graphs_output = transform(corpus_all_text_docs,steps_preprocessing=steps_preprocessing, type=graph_type, window_size=window_size) # cooc, hetero, isg

    if graph_type == 'hetero':
        graph = graphs_output[0]
    else:
        graph = graphs_output

    #*** set_idxs, graph len per set
    set_idxs = {'train': len(train_set), 'val': len(val_set), 'test': len(test_set)}

    #*** print graph info
    if graph_type == 'hetero':
        print("nodes: ", len(graph['graph'].nodes(data=True)))
        print("edges: ", len(graph['graph'].edges(data=True)))
        train_mask, val_mask, test_mask, y_mask = get_masks(graph, corpus_all_text_docs, set_idxs)
        print(len(y_mask))
        print(len(train_mask), sum(train_mask))
        print(len(val_mask), sum(val_mask))
        print(len(test_mask), sum(test_mask))
        print(Counter([str(l) for l in y_mask]))
        print("isolates_nodes: ", list(nx.isolates(graph['graph'])))
    else:
        graph_train = graph[ : set_idxs['train']]
        graph_val = graph[set_idxs['train'] : set_idxs['train'] + set_idxs['val']]
        graph_test = graph[set_idxs['train'] + set_idxs['val'] : ]
        print("num_graphs: ", len(graph))
        print("graph_train: ", len(graph_train))
        print("graph_val: ", len(graph_val))
        print("graph_test: ", len(graph_test))
        print("graph_train[0]: ", graph_train[0])
        print("isolates_nodes: ", [list(nx.isolates(g['graph'])) for g in graph if len(list(nx.isolates(g['graph'])))])
        print("num_nodes: ", sum([len(list(g['graph'].nodes)) for g in graph]))
        cooc_nodes = []
        for g in graph:
            cooc_nodes.extend([n for n in list(g['graph'].nodes)])
        print(len(set(cooc_nodes)), len(cooc_nodes))


    #*** Create a deepcopy graph
    graph_cp = copy.deepcopy(graph)

    #*** Get Doc Embeddings
    # W2vect
    model_w2v, doc_embs = w2v_train_v2(graph=graph_cp, num_features=num_features, set_idxs=set_idxs, graph_type=graph_type)

    # Transformer - LLM for Docs
    if nfi == 'llm':
        text_data_lst = [{'id': d['context']['id'], 'label': d['context']['target'], 'text': d['doc']} for d in corpus_all_text_docs]
        doc_embs = llm_get_embbedings(text_data_lst, subset='train', emb_type='llm_cls', device=device, save_emb=False, llm_finetuned_name=llm_finetuned_name, num_labels='2')

    # *** Build Dataset and get Dataloader
    build_dataset = BuildDataset(graph_cp, corpus_texts=corpus_all_text_docs, set_idxs=set_idxs, subset='train', device=device, 
                                 nfi=nfi, num_labels=num_classes, num_features=num_features, oov_feat_type=oov_feat_type, 
                                 graph_type=graph_type, doc_embs=doc_embs, model_w2v=model_w2v, llm_finetuned_name=llm_finetuned_name)
    proc_dataset = build_dataset.process_dataset()

    if graph_type == 'hetero':
        train_loader = DataLoader(proc_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        data = proc_dataset[0]
        data.to(device)
        print(data)
        print(data.y.shape)
        print(data.train_mask.shape)
        print(data.val_mask.shape)
        print("nodes: ", len(graph_cp['graph'].nodes(data=True)))
        print("edges: ", len(graph_cp['graph'].edges(data=True)))
        print("isolates_nodes: ", list(nx.isolates(graph_cp['graph'])))

    if graph_type == 'cooc':
        train_loader = DataLoader(proc_dataset[ : set_idxs['train']], batch_size=batch_size_gnn, shuffle=True)
        val_loader = DataLoader(proc_dataset[set_idxs['train'] : set_idxs['train'] + set_idxs['val']], batch_size=batch_size_gnn, shuffle=True)
        test_loader = DataLoader(proc_dataset[set_idxs['train'] + set_idxs['val'] : ], batch_size=batch_size_gnn, shuffle=True)
        print("isolates_nodes: ", [list(nx.isolates(g['graph'])) for g in graph_cp if len(list(nx.isolates(g['graph'])))])
        print("num_nodes: ", sum([len(list(g['graph'].nodes)) for g in graph_cp]))


    # *** Set GNN

    if graph_type == 'hetero':
        task = 'node'
    if graph_type == 'cooc':
        task = 'graph'

    model = GNN_2(gnn_type=gnn_type, hidden_channels=hidden_channels, num_features=num_features, heads=heads_gnn, num_classes=num_classes, task=task)

    #model = GCN(gnn_type='GATConv', hidden_channels=hidden_channels, num_features=num_features, heads=1, num_classes=num_classes, task='node')
    '''
    model = GNN(
            gnn_type = 'GATConv', # GCNConv, GATConv, TransformerConv, GraphConv
            num_features = num_features,
            hidden_channels = hidden_channels,
            dense_nhid = dense_nhid,
            num_classes = num_classes,
            pooling = 'gmeanp',
            batch_norm = 'BatchNorm1d', # BatchNorm1d
            layers_convs = 4,
            heads = 3,
            dropout = 0.5,
            edge_dim = None,
            task = task
    )
    '''
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    hetero_gnn_emb = []
    cooc_gnn_train_emb = []
    cooc_gnn_val_emb = []
    cooc_gnn_test_emb = []
    
    if graph_type == 'hetero':
        for epoch in range(1, epochs+1):
            loss, emb, loss_val, val_acc, val_f1_score_macro, val_f1_score_micro, loss_test, test_acc, test_f1_score_macro, test_f1_score_micro = train_node(data, model, criterion, optimizer)
            if epoch % 10 == 0:
                print(f'Epoch: {epoch:03d}, Train_Loss: {loss:.4f}, Val_Acc: {val_acc:.4f}, Val_F1_Score_Micro: {val_f1_score_micro:.4f}, Val_F1_Score_Macro: {val_f1_score_macro:.4f}, Val_Loss {loss_val:.4f}')

            if val_f1_score_macro > best_val_macro_f1score:
                best_epoch = epoch
                hetero_gnn_emb = emb
                best_val_loss = loss_val
                best_val_acc = val_acc
                best_val_macro_f1score = val_f1_score_macro
                best_val_micro_f1score = val_f1_score_micro

            if early_stopper.early_stop(loss_val):
                print('Early stopping fue to not improvement!')
                break

        print(f'Best_Epoch: {best_epoch:03d}, Best_Loss: {best_val_loss:.4f}, Best_Val_Acc: {best_val_acc:.4f}, Val_F1_Score_Micro: {best_val_micro_f1score:.4f}, Val_F1_Score_Macro: {best_val_macro_f1score:.4f}')


    if graph_type == 'cooc':
        for epoch in range(1, epochs+1):
            model, train_acc, train_loss, train_emb, _ = train_graph(train_loader, model, device, criterion, optimizer)
            loss_val, val_acc, val_f1_score_macro, val_f1_score_micro, val_emb, pred_loader = test_graph(val_loader)
            if epoch % 10 == 0:
                print(f'Epoch: {epoch:03d} | Train Loss {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val_Acc: {val_acc:.4f}, Val_F1_Score_Micro: {val_f1_score_micro:.4f}, Val_F1_Score_Macro: {val_f1_score_macro:.4f}, Val_Loss {loss_val:.4f}')

            if val_f1_score_macro > best_val_macro_f1score:
                best_epoch = epoch
                best_val_loss = loss_val
                best_val_acc = val_acc
                best_val_macro_f1score = val_f1_score_macro
                best_val_micro_f1score = val_f1_score_micro
                cooc_gnn_train_emb = train_emb
                cooc_gnn_val_emb = val_emb

            if early_stopper.early_stop(loss_val):
                print('Early stopping fue to not improvement!')
                break
        print(f'Best_Epoch: {best_epoch:03d}, Best_Loss: {best_val_loss:.4f}, Best_Val_Acc: {best_val_acc:.4f}, Val_F1_Score_Micro: {best_val_micro_f1score:.4f}, Val_F1_Score_Macro: {best_val_macro_f1score:.4f}')


    if graph_type == 'hetero':
        loss_val, val_acc, val_f1_score_macro, val_f1_score_micro, loss_test, test_acc, test_f1_score_macro, test_f1_score_micro = test_node(data, model, criterion)
        print()
        print(f'Val Accuracy: {val_acc:.4f}')
        print(f'Val F1_score_Macro: {val_f1_score_macro:.4f}')
        print(f'Val F1_score_Micro: {val_f1_score_micro:.4f}')
        print(f'Val Loss: {loss_val:.4f}')
        print()
        print(f'Test Accuracy: {test_acc:.4f}')
        print(f'Test F1_score_Macro: {test_f1_score_macro:.4f}')
        print(f'Test F1_score_Micro: {test_f1_score_micro:.4f}')
        print(f'Test Loss: {loss_test:.4f}')
        print()

    if graph_type == 'cooc':
        loss_test, test_acc, test_f1_score_macro, test_f1_score_micro, cooc_gnn_test_emb, pred_loader = test_graph(test_loader, model, device, criterion)
        print()
        print(f'Test Accuracy: {test_acc:.4f}')
        print(f'Test F1_score_Macro: {test_f1_score_macro:.4f}')
        print(f'Test F1_score_Micro: {test_f1_score_micro:.4f}')
        print(f'Test Loss: {loss_test:.4f}')
        print()


    eval_d2v_gnn(doc_embs, graph_type, data, train_set, val_set, test_set, hetero_gnn_emb, cooc_gnn_train_emb, cooc_gnn_val_emb, cooc_gnn_test_emb)



if __name__ == '__main__':
   main()
import networkx as nx
import networkx
from collections import defaultdict
import logging
import sys
import traceback
import pandas as pd
import time
import warnings
import nltk 
from itertools import chain
import re
from spacy.tokens import Doc
import spacy
from tqdm import tqdm 
from torch_geometric.utils.convert import from_networkx
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from collections import Counter, defaultdict
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GATConv, TransformerConv
from torch_geometric.nn import (
    GCNConv, GATConv, TransformerConv,
    global_mean_pool, global_max_pool, global_add_pool,
    GlobalAttention, Set2Set
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
from torch.nn import Linear, BatchNorm1d, ModuleList, LayerNorm
from torch_geometric.nn import MLP
from functools import lru_cache
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

import gc
import mlflow
from mlflow import MlflowClient
import matplotlib.pyplot as plt 
import seaborn as sns
import argparse
import json
import sys
import os
import ast  

import utils
import test_utils

#from vanilla_gtn import VanillaGTN  # wherever you saved it
from vanilla_gtn import ISG_VanillaGTN  
from vanilla_gtn import VanillaGTNWithEdgeConcat 
from vanilla_gtn import HybridISG_GTN 

try:
    from xgboost import XGBClassifier
    xgb_installed = True
except ImportError:
    xgb_installed = False

mlflow.set_tracking_uri(uri="http://localhost:8081")
mlflow.set_tracking_uri("/home/avaldez/projects/GraphDeepLearning/mlruns")
client = MlflowClient()
experiment_id = "0"
run = client.create_run(experiment_id)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configs
warnings.filterwarnings("ignore")
log_file_path = "training.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s; - %(levelname)s; - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Load spaCy tokenizer ---
nlp = spacy.load("en_core_web_sm", disable=[])

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('wordnet')
    nltk.data.find('omw-1.4')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')
finally:
    from nltk.corpus import wordnet


# ****************** POSITIONAL ENCODINGS

from torch_geometric.utils import to_dense_adj, degree
from torch_geometric.transforms import AddLaplacianEigenvectorPE

def add_lap_pe(data, k=8, undirected=True, attr_name="lap_pe"):
    # In-place: adds data.lap_pe [N, k]
    T = AddLaplacianEigenvectorPE(k=k, attr_name=attr_name, is_undirected=undirected)
    return T(data)

@torch.no_grad()
def random_walk_landing_probs(edge_index, num_nodes, steps=(1,2,4,8), device="cpu"):
    """
    Returns [N, len(steps)] with landing probabilities at the same node after t steps.
    Dense per-doc is OK since your graphs are per-document (usually small).
    """
    A = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0).to(device)  # [N,N]
    # Row-normalize (avoid divide-by-zero)
    deg = A.sum(dim=1, keepdim=True).clamp(min=1)
    P = A / deg
    N = num_nodes
    eye = torch.eye(N, device=device)

    feats = []
    Pt = P.clone()
    for t in range(1, max(steps) + 1):
        if t > 1:
            Pt = Pt @ P
        if t in steps:
            # diag(P^t)
            feats.append(torch.diagonal(Pt).unsqueeze(1))
    return torch.cat(feats, dim=1) if feats else torch.zeros(N, 0, device=device)



# ****************** POS_TAGS
POS_TAGS = [
    "NOUN", "VERB", "ADJ", "ADV", "PRON", "PROPN", "AUX", "CCONJ", "SCONJ", 
    "ADP", "DET", "NUM", "PART", "INTJ", "PUNCT", "SYM", "X", "ROOT_0"
]
POS2IDX = {pos: idx for idx, pos in enumerate(POS_TAGS)}
NUM_POS_TAGS = len(POS_TAGS)

# ****************** DEPENDENCY_TAGS
DEPENDENCY_TAGS = [
    'ROOT', 'nsubj', 'obj', 'iobj', 'obl', 'nmod', 'amod', 'advmod', 'aux',
    'cop', 'mark', 'cc', 'conj', 'det', 'case', 'compound', 'xcomp', 'ccomp',
    'acl', 'appos', 'nummod', 'punct', 'dep', 'parataxis'
]
DEP2IDX = {tag: idx for idx, tag in enumerate(DEPENDENCY_TAGS)}
NUM_DEP_TAGS = len(DEPENDENCY_TAGS)

DOMAIN2ID = {
    'hc3': 0,
    'm4gt': 1,
    'mage': 2,
    # add test domains if you plan to train on them
    'unknown': 3
}

def log_conf_matrix(y_pred, y_true):
    # Log confusion matrix as image
    cm = confusion_matrix(y_pred, y_true)
    classes = ["0", "1"]
    df_cfm = pd.DataFrame(cm, index = classes, columns = classes)
    plt.figure(figsize = (10,7))
    cfm_plot = sns.heatmap(df_cfm, annot=True, cmap='Blues', fmt='g')
    cfm_plot.figure.savefig(f'{utils.OUTPUT_DIR_PATH}/images/cm.png')
    mlflow.log_artifact(f"{utils.OUTPUT_DIR_PATH}/images/cm.png")

@lru_cache(maxsize=10000)
def get_synonyms(word):
    synsets = wordnet.synsets(word)
    return list(set(chain.from_iterable([w.lemma_names() for w in synsets])))[:5]

class EmbeddingProjector(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, output_dim=256, dropout=0.2):
        super(EmbeddingProjector, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.proj(x)

def get_pos_encoding_ohe(graph):
    """
    graph_nodes: list of strings like 'roof_NOUN', 'flat_ADJ', 'ROOT_0'
    Returns tensor of one-hot encoded POS tags
    """
    graph_nodes = list(graph.nodes())
    pos_encodings = []
    for node in graph_nodes:
        if "_" not in node:
            pos = "X"
        else:
            pos = node.split("_")[-1]
        one_hot = torch.zeros(NUM_POS_TAGS)
        if pos in POS2IDX:
            one_hot[POS2IDX[pos]] = 1.0
        pos_encodings.append(one_hot)
    return torch.stack(pos_encodings)

def get_pos_encoding(graph):
    """
    Return POS IDs (not one-hot) for input to nn.Embedding
    """
    graph_nodes = list(graph.nodes())
    pos_ids = []
    for node in graph_nodes:
        if "_" not in node:
            pos = "X"
        else:
            pos = node.split("_")[-1]
        pos_id = POS2IDX.get(pos, POS2IDX["X"])
        pos_ids.append(pos_id)
    return torch.tensor(pos_ids, dtype=torch.long)  # [num_nodes]

def add_graph_attr(graph):
    for u, v, attrs in graph.edges(data=True):
        try:
            tag = attrs.get('gramm_relation', 'dep')
            tag_id = DEP2IDX.get(tag, DEP2IDX['dep'])
            graph[u][v]['dep_id'] = tag_id
            graph[u][v]['token_distance'] = float(attrs.get('token_distance', 1))
        except Exception as e:
            # Log error and assign default values
            logger.warning(f"Missing edge attributes for ({u}, {v}) — using defaults. Error: {e}")
            graph[u][v]['dep_id'] = DEP2IDX['dep']
            graph[u][v]['token_distance'] = 1.0
    return graph

def get_multilevel_lang_features(doc) -> list:
    """Extract multilevel features from a spaCy Doc or Span object (e.g., sentence)"""
    doc_tokens = [] 
    for token in doc:
        #print(f"'{token.text}'", end=' | ')
        #synonyms_token = get_synonyms(token.text)
        #synonyms_token_head = get_synonyms(token.head.text)

        token_info = {
            'token': token,
            'token_text': token.text,
            'token_lemma': token.lemma_,
            'token_pos': token.pos_,
            'token_dependency': token.dep_,
            'token_head': token.head,
            'token_head_text': token.head.text,
            'token_head_lemma': token.head.lemma_,
            'token_head_pos': token.head.pos_,
            #'token_synonyms': synonyms_token[:5],
            #'token_head_synonyms': synonyms_token_head[:5],
            'is_root_token': token.dep_ == 'ROOT'
        }
        doc_tokens.append(token_info)

    return doc_tokens

def nlp_pipeline(docs: list, params = {'get_multilevel_lang_features': False}):
    doc_lst = []
    Doc.set_extension("multilevel_lang_info", default=[], force=True)

    #for doc in list(nlp.pipe(docs, as_tuples=False, n_process=4, batch_size=128)):
    for nlp_doc in tqdm(nlp.pipe(docs, batch_size=64, n_process=4), total=len(docs), desc="nlp_spacy_docs"):
        if params['get_multilevel_lang_features'] == True:
            nlp_doc._.multilevel_lang_info = get_multilevel_lang_features(nlp_doc)
        doc_lst.append(nlp_doc)
    return doc_lst
    
    #for doc in tqdm(docs, desc="nlp_spacy_docs: "):
    #    nlp_doc = nlp(doc)
    #    if params['get_multilevel_lang_features'] == True:
    #        nlp_doc._.multilevel_lang_info = get_multilevel_lang_features(nlp_doc)
    #    doc_lst.append(nlp_doc)
    #return doc_lst


# Add language code mapping and spaCy model loading
LANGUAGE_MODELS = {
    'en': spacy.load("en_core_web_sm"),
    #'es': spacy.load("es_core_news_sm"),
    #'pt': spacy.load("pt_core_news_sm"),
    #'ca': spacy.load("es_core_news_sm"),  # fallback
    #'gl': spacy.load("es_core_news_sm"),  # fallback
    #'eu': spacy.load("es_core_news_sm")   # fallback
}

LANGUAGE_NAME2CODE = {
    'English': 'en', 'Spanish': 'es', 'Portuguese': 'pt',
    'Catalan': 'ca', 'Gallego': 'gl', 'Euskera': 'eu',
    'en': 'en', 'es': 'es', 'pt': 'pt', 'ca': 'ca', 'gl': 'gl', 'eu': 'eu'
}

def nlp_pipeline_multilang(texts: list, langs: list):
    """Process documents with appropriate spaCy model based on language list."""
    doc_lst = []
    for text, lang in tqdm(zip(texts, langs), total=len(texts), desc="nlp_spacy_docs"):
        lang_code = LANGUAGE_NAME2CODE.get(lang, 'es')  # fallback to Spanish
        nlp = LANGUAGE_MODELS.get(lang_code, LANGUAGE_MODELS['es'])
        nlp_doc = nlp(text)
        doc_lst.append(nlp_doc)
    return doc_lst
 
def normalize_text(texts, special_chars=False, stop_words=False, set='train'):    
    all_texts_norm = []
    for text in tqdm(texts, desc=f"Normalizing {set} corpus"):
        text_norm = test_utils.text_normalize(text, special_chars, stop_words) 
        all_texts_norm.append(text_norm)
    return all_texts_norm

class ISG():
    def __init__(self, 
                graph_type,
                vocab,
                output_format='', 
                apply_prep=True, 
                parallel_exec=False, 
                language='en', 
                steps_preprocessing={},
                use_synonyms=True):
        self.apply_prep = apply_prep
        self.parallel_exec = parallel_exec
        self.graph_type = graph_type
        self.output_format = output_format
        self.use_synonyms = use_synonyms
        self.vocab = vocab

    def _set_synonyms(doc: dict, word_synonnyms: list):
        if len(word_synonnyms):
            return [] 
        else:
            for syn in doc['token_synonyms']:
                if str(doc['token_lemma']) == str(syn):
                    continue
                else:
                    return [(f"{syn}_{doc['token_pos']}",{'pos_tag': doc['token_pos']})]
            return []

    def _get_entities(self, text_doc: list) -> list:  
        nodes = [('ROOT_0', {'pos_tag': 'ROOT_0'})]
        vocab = self.vocab
        use_synonyms = self.use_synonyms

        for d in text_doc:
            token_text = d['token_text'].lower()
            token_pos = d['token_pos']
            key = f"{token_text}_{token_pos}"
            if token_text not in vocab:
                continue

            nodes.append((key, {'pos_tag': token_pos}))

            if use_synonyms and d['token_synonyms']:
                syn = d['token_synonyms'][0].lower()
                if syn in vocab:
                    nodes.append((f"{syn}_{token_pos}", {'pos_tag': token_pos}))
        return nodes


    def _get_relations(self, text_doc: list) -> list:
        edges = []
        vocab = self.vocab
        use_synonyms = self.use_synonyms

        for d in text_doc:
            word = d['token_text'].lower()
            head_word = d['token_head_text'].lower()
            pos = d['token_pos']
            head_pos = d['token_head_pos']
            dep = d['token_dependency']
            distance = abs(d['token'].i - d['token'].head.i)
            norm_dist = 1.0 / distance if distance > 0 else 1.0

            edge_attr = {"gramm_relation": dep, "token_distance": norm_dist}

            if d['is_root_token']:
                if word not in vocab:
                    continue
                edges.append(('ROOT_0', f"{word}_{pos}", edge_attr))
                if use_synonyms and d['token_synonyms']:
                    syn = d['token_synonyms'][0].lower()
                    if syn in vocab:
                        edges.append(('ROOT_0', f"{syn}_{pos}", edge_attr))
            else:
                if word not in vocab or head_word not in vocab:
                    continue
                edges.append((f"{head_word}_{head_pos}", f"{word}_{pos}", edge_attr))
                if use_synonyms and d['token_head_synonyms']:
                    syn = d['token_head_synonyms'][0].lower()
                    if syn in vocab:
                        edges.append((f"{syn}_{head_pos}", f"{word}_{pos}", edge_attr))
        return edges

    def _build_graph(self, nodes: list, edges: list) -> networkx:
        if self.graph_type == 'undirected':
            graph = nx.Graph()
        else:
            graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        return graph

    def _get_frequency_weight(self, graph: nx.DiGraph):
        freq_dict = defaultdict(int)    
        for edge in graph.edges(data=True):
            freq_dict[edge[2]['gramm_relation']] += 1

        for edge in graph.edges(data=True):
            edge[2]['gramm_relation'] = f'{edge[2]["gramm_relation"]}_{freq_dict[edge[2]["gramm_relation"]]+graph.number_of_edges(edge[0], edge[1])}'

    def _build_ISG_graph(self, graphs: list) -> networkx:
        if self.graph_type == 'undirected':
            int_synt_graph = nx.Graph()
        else:
            int_synt_graph = nx.DiGraph()

        for graph in graphs:
            int_synt_graph = nx.compose(int_synt_graph, graph)
        return int_synt_graph

    def _build_ISG_graph_v2(self, graphs: list) -> nx.DiGraph:
        G = nx.DiGraph() if self.graph_type == 'directed' else nx.Graph()
        for g in graphs:
            G.add_nodes_from(g.nodes(data=True))
            G.add_edges_from(g.edges(data=True))
        return G

    def _transform_pipeline(self, docs: list) -> list:
        doc_graphs = []
        for doc in tqdm(docs, desc="Building ISG per doc"):
            try:
                sentence_graphs = []
                #print("\n ************************************")
                #print(doc.text)
                for sent in doc.sents:
                    #print("\n", sent)
                    sent_info = get_multilevel_lang_features(sent)
                    #print("\n", sent_info)
                    nodes = self._get_entities(sent_info)
                    edges = self._get_relations(sent_info)
                    graph = self._build_graph(nodes, edges)
                    #print(graph.nodes())
                    #print(graph.edges(data=True))
                    sentence_graphs.append(graph)
                doc_graph = self._build_ISG_graph(sentence_graphs)
                #print(doc_graph.nodes())
                #print(doc_graph.edges())
                doc_graphs.append(doc_graph)
            except Exception as e:
                logger.error('Error processing doc: %s', str(e))
                logger.error('Traceback: %s', traceback.format_exc())
                doc_graphs.append(nx.DiGraph())
        return doc_graphs

    def _build_dummy_graph(self) -> nx.Graph:
        G = nx.DiGraph() if self.graph_type == 'directed' else nx.Graph()
        # Add minimal dummy structure
        G.add_node("DUMMY_NOUN", pos_tag="NOUN")
        G.add_node("FILLER_VERB", pos_tag="VERB")
        G.add_edge("DUMMY_NOUN", "FILLER_VERB", gramm_relation="dep", token_distance=1.0)
        return G


    def transform(self, nlp_docs) -> list:
        logger.info("Init transformations: Text to Integrated Syntactic Graphs")
        logger.info("Transforming %s text documents...", len(nlp_docs))
        logger.debug("Spacy nlp_pipeline")
        logger.debug("Transform_pipeline")

        doc_graphs = self._transform_pipeline(nlp_docs)
        logger.info("Done transformations")

        output_list = []
        avg_nodes = 0
        avg_edges = 0

        for i, graph in enumerate(doc_graphs):
            # Fallback for malformed or short graphs
            if graph.number_of_nodes() < 2 or graph.number_of_edges() < 1:
                logger.warning(f"[ISG] Using fallback graph for doc_id={i} (nodes={graph.number_of_nodes()}, edges={graph.number_of_edges()})")
                graph = self._build_dummy_graph()

            output_list.append({
                'doc_id': i,
                'graph': graph,
                'number_of_edges': graph.number_of_edges(),
                'number_of_nodes': graph.number_of_nodes(),
                'status': 'success'
            })

            avg_nodes += graph.number_of_nodes()
            avg_edges += graph.number_of_edges()

        return output_list, avg_nodes / len(output_list), avg_edges / len(output_list)

def extract_token_distances_aligned(graph, pyg_data):
    """
    Extract token distances aligned with pyg_data.edge_index
    Handles both directed and undirected graphs
    """
    edge_index = pyg_data.edge_index  # [2, E]
    num_edges = edge_index.size(1)
    
    # Get node list for indexing
    node_list = list(graph.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}
    
    token_distances = []
    
    for i in range(num_edges):
        src_idx = edge_index[0, i].item()
        dst_idx = edge_index[1, i].item()
        
        src_node = node_list[src_idx]
        dst_node = node_list[dst_idx]
        
        # Check if edge exists in graph
        if graph.has_edge(src_node, dst_node):
            dist = graph[src_node][dst_node].get('token_distance', 1.0)
        elif graph.has_edge(dst_node, src_node):  # Undirected case
            dist = graph[dst_node][src_node].get('token_distance', 1.0)
        else:
            dist = 1.0  # Default
        
        token_distances.append(dist)
    
    return torch.tensor(token_distances, dtype=torch.float)

def compute_isg_positional_encodings(graph, num_nodes):
    """
    Compute ISG-aware positional encodings
    """
    pe_dict = {}
    
    # 1) ROOT distance (syntactic depth)
    root_distances = {}
    for node in graph.nodes():
        if node == 'ROOT_0':
            root_distances[node] = 0
        else:
            try:
                root_distances[node] = nx.shortest_path_length(graph, 'ROOT_0', node)
            except:
                root_distances[node] = num_nodes  # Unreachable from root
    
    root_depth = torch.tensor([root_distances[n] for n in graph.nodes()], dtype=torch.float)
    pe_dict['root_depth'] = (root_depth / root_depth.max()).unsqueeze(1) if root_depth.max() > 0 else torch.zeros(num_nodes, 1)
    
    # 2) Betweenness centrality (information flow hubs)
    try:
        betweenness = nx.betweenness_centrality(graph)
        pe_dict['betweenness'] = torch.tensor([betweenness[n] for n in graph.nodes()], dtype=torch.float).unsqueeze(1)
    except:
        pe_dict['betweenness'] = torch.zeros(num_nodes, 1)
    
    # 3) Clustering coefficient (local connectivity)
    try:
        clustering = nx.clustering(graph.to_undirected())
        pe_dict['clustering'] = torch.tensor([clustering[n] for n in graph.nodes()], dtype=torch.float).unsqueeze(1)
    except:
        pe_dict['clustering'] = torch.zeros(num_nodes, 1)
    
    # 4) Is synonym node indicator
    is_synonym = torch.tensor([1.0 if graph.degree(n) == 1 and n != 'ROOT_0' else 0.0 
                               for n in graph.nodes()], dtype=torch.float).unsqueeze(1)
    pe_dict['is_synonym'] = is_synonym
    
    return pe_dict

def compute_dep_diversity_features(graph):
    """
    Compute local dependency relation diversity per node
    LLMs tend to use fewer diverse dependency types locally
    """
    from collections import defaultdict
    
    node_dep_types = defaultdict(set)
    
    for u, v, attr in graph.edges(data=True):
        dep_rel = attr.get('gramm_relation', 'unk')
        node_dep_types[u].add(dep_rel)
        node_dep_types[v].add(dep_rel)
    
    diversity = []
    for node in graph.nodes():
        if node in node_dep_types:
            diversity.append(len(node_dep_types[node]))
        else:
            diversity.append(0)
    
    diversity_tensor = torch.tensor(diversity, dtype=torch.float).unsqueeze(1)
    
    # Optional: normalize to [0, 1]
    max_div = diversity_tensor.max()
    if max_div > 0:
        diversity_tensor = diversity_tensor / max_div
    
    return diversity_tensor

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
        task='node',
        norm_type='batchnorm',   # 'batchnorm', 'layernorm', or None
        post_mp_layers=3,        # number of layers after message passing
        pooling_type='mean'      # 'mean', 'max', 'sum', 'attention', 'set2set'
    ):
        super(GNN, self).__init__()
        self.task = task
        self.heads = heads
        self.gnn_type = gnn_type
        self.edge_attr = edge_attr
        self.dropout = dropout
        self.num_layers = num_layers
        self.norm_type = norm_type

        # First conv
        self.conv1 = self.build_conv_model(input_dim, hidden_dim, heads)
        self.norm1 = self.build_norm_layer(hidden_dim * heads)

        # Additional conv layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(self.build_conv_model(hidden_dim * heads, hidden_dim, heads))
            self.norms.append(self.build_norm_layer(hidden_dim * heads))

        # Global Pooling
        self.global_pool = self.get_pooling_layer(pooling_type, hidden_dim * heads)

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
        elif self.gnn_type == 'GINConv':
            return GCNConv(input_dim, hidden_dim)
        elif self.gnn_type == 'GATConv':
            return GATConv(input_dim, hidden_dim, heads=heads)
        elif self.gnn_type == 'TransformerConv':
            if self.edge_attr:
                return TransformerConv(input_dim, hidden_dim, heads=heads, edge_dim=32)
            else:
                return TransformerConv(input_dim, hidden_dim, heads=heads)
        else:
            raise ValueError(f"Unsupported GNN type: {self.gnn_type}")

    def build_norm_layer(self, dim):
        if self.norm_type == 'batchnorm':
            return nn.BatchNorm1d(dim)
        elif self.norm_type == 'layernrom':
            return nn.LayerNorm(dim)
        else:
            return nn.Identity()

    def get_pooling_layer(self, pooling_type, hidden_dim):
        if pooling_type == 'mean':
            return global_mean_pool
        elif pooling_type == 'max':
            return global_max_pool
        elif pooling_type == 'sum':
            return global_add_pool
        elif pooling_type == 'attention':
            gate_nn = nn.Sequential(nn.Linear(hidden_dim, 1))
            return GlobalAttention(gate_nn)
        elif pooling_type == 'set2set':
            return Set2Set(hidden_dim, processing_steps=3)
        else:
            raise ValueError(f"Unsupported pooling type: {pooling_type}")


    def get_graph_embedding(self, x, edge_index, edge_attr=None, batch=None):
        if self.edge_attr:
            x = self.conv1(x, edge_index, edge_attr)
        else:
            x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.norm1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        for i in range(self.num_layers):
            if self.edge_attr:
                x = self.convs[i](x, edge_index, edge_attr)
            else:
                x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = self.norms[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.global_pool(x, batch)
        return x

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        x = self.get_graph_embedding(x, edge_index, edge_attr, batch)
        logits = self.post_mp(x)
        return logits
        
class GCNClassifier(torch.nn.Module):
    def __init__(self, in_dim=768, hidden_dim=128, out_dim=2):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch  # batch maps nodes to graphs
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)  # → (batch_size, hidden_dim)
        return F.log_softmax(x, dim=1)

def get_node_features_from_doc(graph, text, tokenizer, model, device, max_length=512):
    # 1. Tokenize entire document
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    graph_nodes = list(graph.nodes())

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = outputs.last_hidden_state.squeeze(0)  # [seq_len, hidden_dim]

    input_tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze(0))
    token_map = {
        token.lstrip("Ġ▁").lower(): i
        for i, token in enumerate(input_tokens)
    }
    node_features = []

    match_cnt = 0
    no_match_cnt = 0
    cls_embedding = hidden_states[0].detach().cpu()
    for node in graph_nodes:
        word = node.split("_")[0].lower()
        
        if node == "ROOT_0":
            match_cnt += 1
            node_features.append(cls_embedding)
            continue

        idx = token_map.get(word)
        if idx is not None:
            match_cnt += 1
            node_features.append(hidden_states[idx].detach().cpu())
        else:
            no_match_cnt += 1
            node_features.append(torch.zeros(model.config.hidden_size).detach().cpu())

    #print(graph.number_of_nodes(), graph.number_of_edges(), match_cnt, no_match_cnt)
    return torch.stack(node_features)  # shape: [num_nodes, hidden_dim]

def extract_embeddings(model, loader, device):
    model.eval()
    X, y = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            emb = model.get_graph_embedding(data.x, data.edge_index, data.edge_attr, data.batch)
            X.append(emb.cpu())
            y.append(data.y.cpu())
    return torch.cat(X).numpy(), torch.cat(y).numpy()

def train_sklearn_classifier(X, y, classifier_type='logistic'):
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    
    if classifier_type == 'logistic':
        clf = LogisticRegression(max_iter=1000).fit(X_scaled, y)
    elif classifier_type == 'svm':
        clf = SVC(kernel='linear', probability=True).fit(X_scaled, y)
    elif classifier_type == 'random_forest':
        clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_scaled, y)
    elif classifier_type == 'mlp':
        clf = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300).fit(X_scaled, y)
    elif classifier_type == 'xgboost':
        if not xgb_installed:
            raise ImportError("XGBoost is not installed.")
        clf = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='mlogloss').fit(X_scaled, y)
    else:
        raise ValueError("Unsupported classifier")
    return clf, scaler

def evaluate_sklearn_classifier(clf, scaler, X, y):
    X_scaled = scaler.transform(X)
    preds = clf.predict(X_scaled)
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, average='macro')
    return acc, f1, preds

def save_model(model, path_weights="gnn_model.pth", path_config="gnn_config.json"):
    # Save weights
    torch.save(model.state_dict(), path_weights)

    # Save model config
    config = {
        "input_dim": model.conv1.in_channels,
        "hidden_dim": model.conv1.out_channels,
        "dense_hidden_dim": model.post_mp[0].in_features,  # first layer in post_mp
        "output_dim": model.post_mp[-1].out_features,      # last layer output
        "dropout": model.dropout,
        "num_layers": model.num_layers,
        "edge_attr": model.edge_attr,
        "gnn_type": model.gnn_type,
        "heads": model.heads,
        "task": model.task,
        "norm_type": model.norm_type,
        "post_mp_layers": len(model.post_mp) // 1,  # assuming one Linear per layer
        "pooling_type": model.global_pool.__class__.__name__.lower().replace("global", "").replace("pool", "")
    }

    with open(path_config, "w") as f:
        json.dump(config, f)

def load_model(path_weights="gnn_model.pth", path_config="gnn_config.json", device="cpu"):
    with open(path_config, "r") as f:
        config = json.load(f)

    model = GNN(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        dense_hidden_dim=config["dense_hidden_dim"],
        output_dim=config["output_dim"],
        dropout=config["dropout"],
        num_layers=config["num_layers"],
        edge_attr=config["edge_attr"],
        gnn_type=config["gnn_type"],
        heads=config["heads"],
        task=config["task"],
        norm_type=config["norm_type"],
        post_mp_layers=config["post_mp_layers"],
        pooling_type=config["pooling_type"]
    ).to(device)

    model.load_state_dict(torch.load(path_weights, map_location=device))
    model.eval()
    return model


def extract_feat(graph_data, texts, labels, domains, device, tokenizer, model, projector,
                 pos_embedding, domain_embedding, domain2id, dep_embedding,
                 add_domain_feat=False, reduce_dim_emb=False, add_pos_feat=True,
                 project_after_concat=False, 
                 # --- NEW PE knobs ---
                 use_lap_pe=True, lap_k=8, undirected_for_lap=True,
                 use_degree_pe=True,
                 use_rw_pe=True, rw_steps=(1,2,4,8),
                 use_isg_structural_pe=True,  # NEW
                 set="train"):
    
    data_list = []
    for g_instance in tqdm(graph_data, desc=f"graph-feat {set}: "):
        try:
            graph = g_instance['graph']
            doc_id = g_instance['doc_id']
            num_nodes = graph.number_of_nodes()

            # Domain label
            if set == "test":
                source_id = domain2id['unknown']
            else:
                source = domains[doc_id]
                source_id = domain2id.get(source, domain2id['unknown'])

            # 1) LLM node features (your existing)
            node_feats = get_node_features_from_doc(graph, texts[doc_id], tokenizer, model, device)  # [N, d_llm]
            
            # 2) Graph-level attrs (existing)
            graph = add_graph_attr(graph)

            # 3) Optional POS embeddings (existing)
            if add_pos_feat:
                with torch.no_grad():
                    pos_ids = get_pos_encoding(graph)                        # [N]
                    pos_feats = pos_embedding(pos_ids).cpu()                 # [N, d_pos]
            else:
                pos_feats = None

            # 4) Optional domain embedding per node (existing)
            if add_domain_feat:
                with torch.no_grad():
                    domain_feat = domain_embedding(torch.tensor(source_id).to(device))  # [d_dom]
                    domain_feats = domain_feat.unsqueeze(0).repeat(graph.number_of_nodes(), 1).cpu()
            else:
                domain_feats = None

            # ---------- Build a PyG Data to compute PEs ----------
            pyg_data = from_networkx(graph)  # contains .edge_index, .dep_id if you added it in add_graph_attr
            num_nodes = graph.number_of_nodes()

            # Edge attributes 
            with torch.no_grad():
                dep_embed_tensor = dep_embedding(pyg_data.dep_id.to(device)).cpu()  # [E, dep_emb_dim]
            pyg_data.edge_attr = dep_embed_tensor

            # Continuous token distance
            pyg_data.token_distance = extract_token_distances_aligned(graph, pyg_data)

            #pyg_data.token_distance = torch.tensor(
            #    [graph[u][v]['token_distance'] for u, v in graph.edges()],
            #    dtype=torch.float, device=device
            #)

            # ---------- Positional / Structural Encodings ----------
            pe_list = []

            # (a) Degree PE (1-d)
            if use_degree_pe:
                deg = degree(pyg_data.edge_index[0], num_nodes=num_nodes).unsqueeze(1)  # [N,1]
                pe_list.append(deg.cpu())

            # (b) Laplacian eigenvectors (k)
            if use_lap_pe and num_nodes >= 2:
                pyg_data = add_lap_pe(pyg_data, k=lap_k, undirected=undirected_for_lap, attr_name="lap_pe")
                # guard NaNs (isolated nodes)
                lap = torch.nan_to_num(pyg_data.lap_pe, nan=0.0, posinf=0.0, neginf=0.0).cpu()  # [N, k]
                pe_list.append(lap)

            # (c) Random-walk landing probs at steps
            if use_rw_pe and num_nodes >= 2:
                rw = random_walk_landing_probs(
                    pyg_data.edge_index.to(device),
                    num_nodes=num_nodes,
                    steps=rw_steps,
                    device=device
                ).cpu()  # [N, len(steps)]
                pe_list.append(rw)

            # ========== NEW ISG STRUCTURAL PEs ==========
            if use_isg_structural_pe:
                isg_pe = compute_isg_positional_encodings(graph, num_nodes)
                dep_diversity = compute_dep_diversity_features(graph)  # NEW
                pe_list.extend([
                    isg_pe['root_depth'],
                    isg_pe['betweenness'],
                    isg_pe['clustering'],
                    #isg_pe['is_synonym']
                    dep_diversity  
                ])

            pe_feats = torch.cat(pe_list, dim=1) if len(pe_list) > 0 else None  # [N, d_pe]

            # ---------- Combine node features + POS/Domain + PEs ----------
            # Your original projection logic preserved
            if reduce_dim_emb:
                if project_after_concat:
                    feats = [node_feats]
                    if pos_feats is not None:    feats.append(pos_feats)
                    if domain_feats is not None: feats.append(domain_feats)
                    if pe_feats is not None:     feats.append(pe_feats)
                    full_input = torch.cat(feats, dim=1)
                    with torch.no_grad():
                        node_feats = projector(full_input.to(device)).cpu()
                else:
                    with torch.no_grad():
                        node_feats = projector(node_feats.to(device)).cpu()
                    feats = [node_feats]
                    if pos_feats is not None:    feats.append(pos_feats)
                    if domain_feats is not None: feats.append(domain_feats)
                    if pe_feats is not None:     feats.append(pe_feats)
                    node_feats = torch.cat(feats, dim=1)
            else:
                feats = [node_feats]
                if pos_feats is not None:    feats.append(pos_feats)
                if domain_feats is not None: feats.append(domain_feats)
                if pe_feats is not None:     feats.append(pe_feats)
                node_feats = torch.cat(feats, dim=1)

            # ---------- Finalize PyG Data ----------
            pyg_data.domain_id = source_id
            pyg_data.x = node_feats
            pyg_data.y = torch.tensor([labels[doc_id]])
            data_list.append(pyg_data)

        except Exception as e:
            logger.error(f"[extract_feat] Error in doc_id={doc_id}: {str(e)}")

    return data_list


def custom_tokenizer(text):
    return re.findall(r'\w+|[^\w\s]', text)

def create_vocab(texts, set='all', min_df=1, max_df=0.9, max_features=5000):
    # Create a vocabulary
    vectorizer = CountVectorizer(tokenizer=custom_tokenizer, token_pattern=None, min_df=min_df, max_df=max_df, max_features=max_features)
    vectorizer.fit_transform(texts)
    vocab = vectorizer.get_feature_names_out()
    # Create a word-to-index dictionary for fast lookups
    word_to_index = {word: idx for idx, word in enumerate(vocab)}
    print(f'vocab {set} set: ', len(vocab))
    return vocab, word_to_index

def create_vocab_v2(texts_nlp, stop_words=False, special_chars=False, min_df=1, max_features=5000):
    vocab = set()
    word_freq = Counter()
    for doc in texts_nlp:
        for token in doc:
            if stop_words and token.is_stop:
                continue
            if special_chars and token.is_punct:
                continue
            vocab.add(token.text)
            word_freq[token.text] += 1
    vocab = list(vocab)
    # Filter words by min frequency
    filtered_vocab = [word for word in vocab if word_freq[word] >= min_df]
    # Sort remaining words by frequency (descending) and take top_k
    filtered_vocab = sorted(filtered_vocab, key=lambda w: word_freq[w], reverse=True)[:max_features]
    # Rebuild word_id_map and vocab_size
    vocab = filtered_vocab
    word_to_index = {w: i for i, w in enumerate(vocab)}
    return vocab, word_to_index

def train_mlp():
    # Hyperparameters
    cuda_num = 0
    learning_rate = 0.0001
    epochs = 500 
    hidden_dim = 200
    num_layers = 3
    dropout = 0.5
    patience = 15
    output_dim = 2

    # Dataset settings
    cut_off_dataset = '10-10-10'
    dataset = "autext23"
    model_name = "microsoft/deberta-v3-base"

    # Device and model
    device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")
    llm_model = AutoModel.from_pretrained(model_name)
    llm_model_size = llm_model.config.hidden_size

    # Load node-level ISG data
    file_name_data = f"isg_data_{dataset_name}_{cut_off_dataset}perc" 
    output_dir = f'{test_utils.EXTERNAL_DISK_PATH}isg_graph'
    data = utils.load_data(file_name_data, path=f'{output_dir}/', format_file='.pkl', compress=False)

    # Create PyG loaders
    train_loader = DataLoader(data["data_train_list"], batch_size=256, shuffle=True)
    val_loader = DataLoader(data["data_val_list"], batch_size=256, shuffle=False)
    test_loader = DataLoader(data["data_test_list"], batch_size=256, shuffle=False)

    # MLP Model
    mlp_model = MLP(
        in_channels=llm_model_size,
        hidden_channels=hidden_dim,
        out_channels=output_dim,
        num_layers=num_layers,
        dropout=dropout,
        norm='batch_norm'
    ).to(device)

    print(mlp_model)
    optimizer = torch.optim.Adam(mlp_model.parameters(), lr=learning_rate)

    criterion = torch.nn.CrossEntropyLoss()
    early_stopper = EarlyStopper(patience=patience, min_delta=0)

    # Training loop
    for epoch in range(epochs):
        mlp_model.train()
        total_loss = 0

        for batch in train_loader:
            batch = batch.to(device)
            x = global_mean_pool(batch.x, batch.batch)  
            out = mlp_model(x)                            
            loss = criterion(out, batch.y)               
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Evaluation
        mlp_model.eval()
        def eval(loader):
            all_preds, all_labels = [], []
            total_val_loss = 0
            for batch in loader:
                batch = batch.to(device)
                with torch.no_grad():
                    x = global_mean_pool(batch.x, batch.batch)  
                    out = mlp_model(x)
                    loss = criterion(out, batch.y)
                    preds = out.argmax(dim=1)
                    all_preds.extend(preds.cpu().tolist())
                    all_labels.extend(batch.y.cpu().tolist())
                    total_val_loss += loss.item()
            acc = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average='macro')
            return total_val_loss / len(loader), acc, f1

        val_loss, val_acc, val_f1 = eval(val_loader)
        test_loss, test_acc, test_f1 = eval(test_loader)

        print(f"Epoch {epoch:03d} | Train Loss: {total_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}")

        # Early stopping logic (optional)
        if early_stopper.early_stop(val_loss):
            print('Early stopping triggered!')
            break

def balance_df(df, group_by="source"):
    if group_by not in df.columns or 'label' not in df.columns:
        raise ValueError(f"DataFrame must contain '{group_by}' and 'label' columns")

    balanced_parts = []
    for group_val, group_df in df.groupby(group_by):
        label_counts = group_df['label'].value_counts()
        if len(label_counts) < 2:
            balanced_parts.append(group_df)
            continue

        min_count = min(label_counts[0], label_counts[1])
        df_0 = group_df[group_df['label'] == 0].sample(min_count, random_state=42)
        df_1 = group_df[group_df['label'] == 1].sample(min_count, random_state=42)
        balanced_parts.append(pd.concat([df_0, df_1]))

    return pd.concat(balanced_parts).sample(frac=1, random_state=42).reset_index(drop=True)


def build_domain2id(train_set, val_set, test_set):
    # Extract unique domain values from the source column
    all_sources = pd.concat([train_set, val_set])['source'].unique().tolist()
    domain2id = {domain: idx for idx, domain in enumerate(sorted(all_sources))}

    # Add 'unknown' to handle unseen test domains
    domain2id['unknown'] = len(domain2id)
    return domain2id
       
def test_gnn(model, device, loader, criterion):
    model.eval()
    all_loss = 0.0
    correct = 0
    all_preds = []
    all_labels = []
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            #out = model(data)
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        pred = out.argmax(dim=1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(data.y.cpu().numpy())
        correct += int((pred == data.y).sum())
        loss = criterion(out, data.y)
        all_loss += loss.item()

    f1_macro = f1_score(all_labels, all_preds, average='macro')
    accuracy = correct / len(loader.dataset)
    return f1_macro, accuracy, all_loss / len(loader), all_preds, all_labels

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


def main(dataset_name, cut_off_dataset, cuda_num=0, 
         graph_type='directed', edge_attr = False, 
         build_graph = True, max_features=None,
         reduce_dim_emb = True,  reduced_dim = 256,
         add_pos_feat = True, lr = 0.0001, min_df = 2,
         max_df = 0.9, patience = 10, hidden_gnn_dim = 100,
         dense_hidden_gnn_dim = 64, num_gnn_layers = 1, 
         heads_gnn = 1, gnn_type = 'GATConv', add_domain_feat=False,
         dropout = 0.5, lang_model_name = 'microsoft/deberta-v3-base',
         project_after_concat=False, stop_words=False, special_chars=False,
         leave_out_sources=None, data=None, file_name_data='', output_dir='',
         norm_type='batchnorm', post_mp_layers=2, pooling_type='mean', 
         balance_dataset = True,
         use_lap_pe=False, lap_k=8, undirected_for_lap=True,
         use_degree_pe=False,
         use_rw_pe=False, rw_steps=(1,2,4,8),
         use_isg_structural_pe=True,  # NEW
         use_token_distance_attn=True,  # NEW
         use_global_pooling=False,  # NEW
    ):

    mlflow.log_param("dataset", dataset_name)
    mlflow.log_param("cut_off_dataset", cut_off_dataset)

    device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")

    pos_emb_dim = 32
    dep_emb_dim = 32
    domain_emb_dim = 32
    batch_size = 512
    num_epochs = 300

    tokenizer = AutoTokenizer.from_pretrained(lang_model_name)
    lang_model = AutoModel.from_pretrained(lang_model_name).to(device)

    input_gnn_dim = lang_model.config.hidden_size 
    input_proj_dim = lang_model.config.hidden_size 

    if build_graph: 

        train_text_set, val_text_set, test_text_set = test_utils.read_dataset(dataset_name)
        if dataset_name == 'autext24':
            train_text_set = train_text_set[train_text_set['language'] == 'en'].reset_index(drop=True)
            val_text_set = val_text_set[val_text_set['language'] == 'en'].reset_index(drop=True)
            test_text_set = test_text_set[test_text_set['language'] == 'en']
        
        if leave_out_sources:
            print(f"[INFO] Leave-One-Source-Out setting: removing '{leave_out_sources}' from train.")
            #train_text_set = train_text_set[train_text_set['source'] != leave_out_source]
            #val_text_set = val_text_set[val_text_set['source'] == leave_out_source]
            
            # Split train set: keep everything NOT in leave_out_sources, swap the rest
            train_keep = train_text_set[~train_text_set['source'].isin(leave_out_sources)]
            train_swap = train_text_set[train_text_set['source'].isin(leave_out_sources)]

            # Split val set: keep only leave_out_sources, swap the rest
            val_keep = val_text_set[val_text_set['source'].isin(leave_out_sources)]
            val_swap = val_text_set[~val_text_set['source'].isin(leave_out_sources)]

            # Combine to form new sets
            #train_text_set = train_keep
            train_text_set = pd.concat([train_keep, val_swap], ignore_index=True)
            #val_text_set = val_keep
            val_text_set = pd.concat([val_keep, train_swap], ignore_index=True)
            
            print("Train set distro:\n", train_text_set.groupby("source")["label"].value_counts())
            print("Val set distro:\n", val_text_set.groupby("source")["label"].value_counts())

        # Cut off datasets
        cut_off_train = int(cut_off_dataset.split('_')[0])
        cut_off_val = int(cut_off_dataset.split('_')[1])
        cut_off_test = int(cut_off_dataset.split('_')[2])

        train_set = train_text_set[:int(len(train_text_set) * (cut_off_train / 100))][:]
        val_set = val_text_set[:int(len(val_text_set) * (cut_off_val / 100))][:]
        test_set = test_text_set[:int(len(test_text_set) * (cut_off_test / 100))][:]

        group_by = "source"
        #if dataset_name == 'autext24':
        #    group_by = 'language'

        if balance_dataset:
            train_set = balance_df(train_set, group_by) # source, language
            val_set = balance_df(val_set, group_by)

        print("distro_train_val_test: ", len(train_set), len(val_set), len(test_set))
        print("label_distro_train_val_test: ", train_set.value_counts('label'), val_set.value_counts('label'), test_set.value_counts('label'))
        print("Label distribution per source in Train set:\n", train_set.groupby("source")["label"].value_counts())
        print("Label distribution per source in Validation set:\n", val_set.groupby("source")["label"].value_counts())
        print("Label distribution per source in Test set:\n", test_set.groupby("source")["label"].value_counts())

        if dataset_name == 'autext24':
            print("Language distribution per source in Train set:\n", train_set.groupby("language")["label"].value_counts())
            print("Language distribution per source in Val set:\n", val_set.groupby("language")["label"].value_counts())
            print("Language distribution per source in Test set:\n", test_set.groupby("language")["label"].value_counts())

        domain2id = build_domain2id(train_set, val_set, test_set)
        limit = None

        train_texts = list(train_set['text'])[:limit]
        val_texts = list(val_set['text'])[:limit]
        test_texts = list(test_set['text'])[:limit]

        #test_texts[0] = ""  # for testing short text

        #train_texts[0] = "My, my, I was forgetting all about the children and the mysterious fern seed. I wonder if it has changed them back into real little children again. Yes, here they come." 
        #train_texts[0] = "Neural networks can detect patterns in complex data. They are often used in image recognition tasks. Training deep models requires significant computational resources."
        #AI uses data to improve automation systems
        #AI transforms industries through automation 

        train_labels = list(train_set['label'])[:limit]
        val_labels = list(val_set['label'])[:limit]
        test_labels = list(test_set['label'])[:limit]

        train_domains = list(train_set['source'])[:limit]
        val_domains = list(val_set['source'])[:limit]
        test_domains = list(test_set['source'])[:limit]

        train_texts_norm = normalize_text(train_texts, set='train')
        val_texts_norm = normalize_text(val_texts, set='val')
        test_texts_norm = normalize_text(test_texts, set='test')

        #if dataset_name == 'autext24':
        #    train_nlp_docs = nlp_pipeline_multilang(train_texts_norm, langs=list(train_set['language']))
        #    val_nlp_docs = nlp_pipeline_multilang(val_texts_norm, langs=list(val_set['language']))
        #    test_nlp_docs = nlp_pipeline_multilang(test_texts_norm, langs=list(test_set['language']))
        #else:
        train_nlp_docs = nlp_pipeline(train_texts_norm)
        val_nlp_docs = nlp_pipeline(val_texts_norm)
        test_nlp_docs = nlp_pipeline(test_texts_norm)

        all_nlp_docs = train_nlp_docs + val_nlp_docs + test_nlp_docs
        vocab, word_to_index = create_vocab_v2(all_nlp_docs, stop_words=stop_words, special_chars=special_chars, min_df=min_df, max_features=max_features)
        print("vocab_size: ", len(vocab))
        with open(f"{test_utils.OUTPUT_DIR_PATH}/{dataset_name}_vocab.txt", 'w', encoding='utf-8') as file:
            for item in vocab:
                file.write(str(item) + '\n') 

        isg = ISG(graph_type=graph_type, vocab=vocab, output_format='networkx', apply_prep=True, parallel_exec=False, language='en', use_synonyms=False)

        graph_train, avg_train_nodes, avg_train_edges = isg.transform(train_nlp_docs)
        graph_val, avg_val_nodes, avg_val_edges = isg.transform(val_nlp_docs)
        graph_test, avg_test_nodes, avg_test_edges = isg.transform(test_nlp_docs)

        g = graph_train[0]
        print(train_texts_norm[0])
        print(g['graph'].nodes(data=True))
        print(g['graph'].edges(data=True))


        # PROJECTION
        if project_after_concat and add_pos_feat:
            input_proj_dim += pos_emb_dim
        if project_after_concat and add_domain_feat:
            input_proj_dim += domain_emb_dim

        #llm_projector = nn.Linear(input_proj_dim, reduced_dim).to(device)
        llm_projector = build_projector(input_dim=input_proj_dim, hidden_dim=512, output_dim=reduced_dim, dropout=0.1, num_layers=1, use_norm=False, activation='gelu').to(device)

        pos_embedding = nn.Embedding(NUM_POS_TAGS, pos_emb_dim)
        domain_embedding = nn.Embedding(len(domain2id), domain_emb_dim).to(device)
        dep_embedding = nn.Embedding(NUM_DEP_TAGS, dep_emb_dim).to(device)

        data_train_list = extract_feat(graph_train, train_texts, train_labels, train_domains, device, tokenizer, 
                                       lang_model, llm_projector, pos_embedding, domain_embedding, domain2id, dep_embedding, 
                                       add_domain_feat, reduce_dim_emb, add_pos_feat, project_after_concat, 
                                       use_lap_pe, lap_k, undirected_for_lap, use_degree_pe, use_rw_pe, rw_steps, use_isg_structural_pe, set="train")
        
        data_val_list = extract_feat(graph_val, val_texts, val_labels, val_domains, device, tokenizer, 
                                     lang_model, llm_projector, pos_embedding, domain_embedding, domain2id, dep_embedding, 
                                     add_domain_feat, reduce_dim_emb, add_pos_feat, project_after_concat, 
                                     use_lap_pe, lap_k, undirected_for_lap, use_degree_pe, use_rw_pe, rw_steps, use_isg_structural_pe, set="val")
        
        data_test_list = extract_feat(graph_test, test_texts, test_labels, test_domains,  device, tokenizer, 
                                      lang_model, llm_projector, pos_embedding, domain_embedding, domain2id, dep_embedding, 
                                      add_domain_feat, reduce_dim_emb, add_pos_feat, project_after_concat, 
                                      use_lap_pe, lap_k, undirected_for_lap, use_degree_pe, use_rw_pe, rw_steps, use_isg_structural_pe, set="test")
        data = {
            "graph_train": graph_train,
            "graph_val": graph_val,
            "graph_test": graph_test,
            "data_train_list": data_train_list,
            "data_val_list": data_val_list,
            "data_test_list": data_test_list,
        }

        utils.save_data(data, file_name_data, path=f'{output_dir}/', format_file='.pkl', compress=False)

    elif not build_graph and not data:
        data = utils.load_data(file_name_data, path=f'{output_dir}/', format_file='.pkl', compress=False)
        data_train_list = data["data_train_list"]
        data_val_list = data["data_val_list"]
        data_test_list = data["data_test_list"]
    elif not build_graph and data:
        data_train_list = data["data_train_list"]
        data_val_list = data["data_val_list"]
        data_test_list = data["data_test_list"]
    else:
        ...

    train_loader = DataLoader(data_train_list, batch_size=batch_size, shuffle=True, num_workers=0) 
    val_loader = DataLoader(data_val_list, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(data_test_list, batch_size=batch_size, shuffle=False, num_workers=0)

    if reduce_dim_emb and project_after_concat:
        input_gnn_dim = reduced_dim
    elif reduce_dim_emb and (not project_after_concat):
        input_gnn_dim = reduced_dim
        if add_pos_feat:
            input_gnn_dim += pos_emb_dim
        if add_domain_feat:
            input_gnn_dim += domain_emb_dim
    elif not reduce_dim_emb:
        if add_pos_feat:
            input_gnn_dim += pos_emb_dim
        if add_domain_feat:
            input_gnn_dim += domain_emb_dim
        # NEW:
        if use_degree_pe: input_gnn_dim += 1
        if use_lap_pe: input_gnn_dim += lap_k
        if use_rw_pe: input_gnn_dim += len(rw_steps)
        if use_isg_structural_pe:  # NEW
            input_gnn_dim += 3  # root_depth, betweenness, clustering, is_synonym
            input_gnn_dim += 1  # dep_diversity

    test_utils.set_random_seed(42)
    


    if gnn_type == 'Vanilla-GTN-NoEdge':
        # Baseline: No edge features
        model = VanillaGTNWithEdgeConcat(
            input_dim=input_gnn_dim,
            hidden_dim=hidden_gnn_dim,
            dense_hidden_dim=dense_hidden_gnn_dim,
            output_dim=2,
            dropout=dropout,
            num_layers=num_gnn_layers,
            use_edge_attr=False,
            heads=heads_gnn,
            norm_type=norm_type,
            pooling_type=pooling_type,
            post_mp_layers=post_mp_layers,
            edge_attr_dim=None,
            ffn_mult=4,
            attn_dropout=0.1,
        ).to(device)
        print("🔵 Vanilla GTN (No Edge)")
    
    elif gnn_type == 'Vanilla-GTN-EdgeConcat':
        # Standard Vanilla GTN with edge concat
        model = VanillaGTNWithEdgeConcat(
            input_dim=input_gnn_dim,
            hidden_dim=hidden_gnn_dim,
            dense_hidden_dim=dense_hidden_gnn_dim,
            output_dim=2,
            dropout=dropout,
            num_layers=num_gnn_layers,
            use_edge_attr=True,
            heads=heads_gnn,
            norm_type=norm_type,
            pooling_type=pooling_type,
            post_mp_layers=post_mp_layers,
            edge_attr_dim=(data_train_list[0].edge_attr.size(-1) if edge_attr else None),
            ffn_mult=4,
            attn_dropout=0.1,
        ).to(device)
        print("🟢 Vanilla GTN (Edge Concat)")
    
    elif gnn_type == 'ISG-VanillaGTN':
        # Your original ISG-GTN
        model = ISG_VanillaGTN(
            input_dim=input_gnn_dim,
            hidden_dim=hidden_gnn_dim,
            dense_hidden_dim=dense_hidden_gnn_dim,
            output_dim=2,
            dropout=dropout,
            num_layers=num_gnn_layers,
            use_edge_attr=edge_attr,
            use_token_distance=use_token_distance_attn,
            heads=heads_gnn,
            norm_type=norm_type,
            pooling_type=pooling_type,
            post_mp_layers=post_mp_layers,
            edge_attr_dim=(data_train_list[0].edge_attr.size(-1) if edge_attr else None),
            ffn_mult=4,
            attn_dropout=0.1,
            use_global_pooling=use_global_pooling,
        ).to(device)
        print("🟠 ISG-VanillaGTN (Original)")
    
    elif gnn_type == 'Hybrid-ISG-GTN':
        # NEW: Hybrid combining both approaches
        model = HybridISG_GTN(
            input_dim=input_gnn_dim,
            hidden_dim=hidden_gnn_dim,
            dense_hidden_dim=dense_hidden_gnn_dim,
            output_dim=2,
            dropout=dropout,
            num_layers=num_gnn_layers,
            use_edge_attr=edge_attr,
            use_token_distance=use_token_distance_attn,
            use_edge_bias=True,  # NEW: Additional bias on top of concat
            heads=heads_gnn,
            norm_type=norm_type,
            pooling_type=pooling_type,
            post_mp_layers=post_mp_layers,
            edge_attr_dim=(data_train_list[0].edge_attr.size(-1) if edge_attr else None),
            ffn_mult=4,
            attn_dropout=0.1,
            use_global_pooling=use_global_pooling,
        ).to(device)
        print("🔥 Hybrid-ISG-GTN (Edge Concat + ISG Enhancements)")
    
    else:
        raise ValueError(f"Unknown gnn_type: {gnn_type}")


    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    criterion = torch.nn.CrossEntropyLoss()
    early_stopper = EarlyStopper(patience=patience, min_delta=0)
    print(model)
    mlflow.log_param('model_params', str(model))

    logger.info("Init GNN training!")
    start_time = time.time()

    best_test_acc_score = 0
    best_test_f1_score = 0
    best_val_acc_score = 0
    best_val_f1_score = 0
    patience_counter = 0
    stop_epoch = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            # Pass token_distance if available
            token_dist = data.token_distance if hasattr(data, 'token_distance') else None
            out = model(data.x, data.edge_index, data.edge_attr, data.batch, token_dist)
            
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        '''
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        '''

        train_loss = train_loss / len(train_loader)

        val_f1_macro, val_accuracy, val_loss, _, _ = test_gnn(model, device, val_loader, criterion)
        test_f1_macro, test_accuracy, test_loss, _, _ = test_gnn(model, device, test_loader, criterion)

        print(f"Epoch {epoch:02d} | Train-Loss {train_loss:4f}  | Val-Loss {val_loss:4f} | Test-Loss {test_loss:4f} | Val-Acc: {val_accuracy:4f} | Test-Acc: {test_accuracy:4f} | Test-F1Macro: {test_f1_macro:4f}")
        

        mlflow.log_metric(key=f"F1Score-val", value=float(val_f1_macro), step=epoch)
        mlflow.log_metric(key=f"Accuracy-val", value=float(val_accuracy), step=epoch)
        mlflow.log_metric(key=f"Loss-val", value=float(val_loss), step=epoch)
        mlflow.log_metric(key=f"F1Score-test", value=float(test_f1_macro), step=epoch)
        mlflow.log_metric(key=f"Accuracy-test", value=float(test_accuracy), step=epoch)
        mlflow.log_metric(key=f"Loss-test", value=float(test_loss), step=epoch)

        # Step the scheduler based on val_loss
        #scheduler.step(val_loss)

        if test_accuracy > best_test_acc_score:
            best_test_acc_score = test_accuracy
        if test_f1_macro > best_test_f1_score:
            best_test_f1_score = test_f1_macro

        stop_epoch = epoch
        #scheduler.step(val_accuracy)

        # Early Stopping 1 (using val loss)
        
        if early_stopper.early_stop(val_loss):
            print('Early stopping triggered!')
            break
        

        # Early Stopping 2 (using acccuracy/f1score)
        '''
        if val_accuracy > best_val_acc_score:
            best_val_acc_score = val_accuracy
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        '''
        
    logger.info("Done GNN training!")
    print("--- %s Graph Training Time ---" % (time.time() - start_time))

    model_save_path = f"{output_dir}/models/gnn_model_{file_name_data}.pth"
    model_config_path = f"{output_dir}/models/config_{file_name_data}.json"
    #save_model(model, model_save_path, model_config_path)

    test_f1_macro, test_accuracy, test_loss, preds_test, labels_test = test_gnn(model, device, test_loader, criterion)
    print(f" ----> Test-Loss {test_loss:4f} | Test-Acc: {test_accuracy:4f} | Test-F1Macro: {test_f1_macro:4f}")

    print("preds_test:  ", sorted(Counter(preds_test).items()))
    print("labels_test: ", sorted(Counter(labels_test).items()))
    cm = confusion_matrix(preds_test, labels_test)
    print(cm)

    log_conf_matrix(preds_test, labels_test)
    mlflow.log_metric(key=f"num_epochs", value=int(num_epochs))
    mlflow.log_metric(key=f"stop_epoch", value=int(stop_epoch))
    mlflow.log_metric(key=f"Final-F1Macro-val", value=float(val_f1_macro))
    mlflow.log_metric(key=f"Final-Accuracy-test", value=float(val_accuracy))
    mlflow.log_metric(key=f"Final-Loss-test", value=float(val_loss))
    mlflow.log_metric(key=f"Final-F1Macro-test", value=float(test_f1_macro))
    mlflow.log_metric(key=f"Final-Accuracy-test", value=float(test_accuracy))
    mlflow.log_metric(key=f"Final-Loss-test", value=float(test_loss))
    mlflow.log_metric(key=f"Best-Accuracy-test", value=float(best_test_acc_score))
    mlflow.log_metric(key=f"Best-F1Macro-test", value=float(best_test_f1_score))
    mlflow.log_artifact(log_file_path)
    return

    # ********************************************************************************* analysis preds

    _, _, test_text_set = test_utils.read_dataset(dataset_name, print_info=False)

    if dataset_name == 'autext24':
        test_text_set = test_text_set[test_text_set['language'] == 'en']
    
    cut_off_test = int(cut_off_dataset.split('_')[2])
    test_set = test_text_set[:int(len(test_text_set) * (cut_off_test / 100))][:]

    test_set['preds_test'] = preds_test
    test_set['labels_test'] = labels_test
    test_set['id_test'] = test_set['id']
    test_set = test_set[['id', 'id_test', 'label', 'labels_test', 'preds_test', 'source', 'model']]
    test_set.to_csv(f"{utils.OUTPUT_DIR_PATH}preds_{dataset_name}.csv", index=False)
    mlflow.log_artifact(f"{utils.OUTPUT_DIR_PATH}preds_{dataset_name}.csv")

    source_accuracy = (
        test_set.groupby("source")
        .apply(lambda test_set: (test_set["preds_test"] == test_set["labels_test"]).mean())
        .reset_index(name="accuracy")
    )
    print(source_accuracy)

    source_label_accuracy = (
        test_set.groupby(["source", "label"])
        .apply(lambda test_set: (test_set["preds_test"] == test_set["labels_test"]).mean())
        .reset_index(name="accuracy")
    )
    print(source_label_accuracy)

    label_accuracy = (
        test_set.groupby("label")
        .apply(lambda test_set: (test_set["preds_test"] == test_set["labels_test"]).mean())
        .reset_index(name="accuracy")
    )
    print(label_accuracy)

    model_accuracy = (
        test_set.groupby("model")
        .apply(lambda test_set: (test_set["preds_test"] == test_set["labels_test"]).mean())
        .reset_index(name="accuracy")
    )
    print(model_accuracy.to_string(index=False))

    summary_path = "accuracy_breakdown.txt"
    with open(summary_path, "w") as f:
        f.write("=== Accuracy by Model ===\n")
        f.write(model_accuracy.to_string(index=False))
        f.write("\n\n=== Accuracy by Label ===\n")
        f.write(label_accuracy.to_string(index=False))
        f.write("\n\n=== Accuracy by Source ===\n")
        f.write(source_accuracy.to_string(index=False))
        f.write("\n\n=== Accuracy by Source-Label ===\n")
        f.write(source_label_accuracy.to_string(index=False))

    mlflow.log_artifact(summary_path)
    os.remove(summary_path)
    return


    #  Freeze GNN and train an external classifier 
    print("[INFO] Extracting graph embeddings from trained GNN...")
    X_train, y_train = extract_embeddings(model, train_loader, device)
    X_val, y_val = extract_embeddings(model, val_loader, device)
    X_test, y_test = extract_embeddings(model, test_loader, device)

    classifier_type = 'svm' # logistic, svm, random_forest, mlp, xgboost
    clf, scaler = train_sklearn_classifier(X_train, y_train, classifier_type=classifier_type) 
    acc, f1, preds = evaluate_sklearn_classifier(clf, scaler, X_test, y_test)
    print(f"[{classifier_type} classifier] Test Accuracy: {acc:.4f} | Test F1 Macro: {f1:.4f}")
    mlflow.log_param("ml_classifier", classifier_type)
    mlflow.log_metric(key=f"ml_classifier_accuracy", value=float(acc))
    mlflow.log_metric(key=f"ml_classifier_f1macro", value=float(f1))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default=None)
    args = parser.parse_args()

    if args.config_path:
        with open(args.config_path, "r") as f:
            config = json.load(f)
        if config['_done'] == True or config['_done'] == 'True':
            sys.exit("Experiment already DONE") 
        del config['_done']
    else:
        config = {
            'graph_type': 'undirected', # directed, undirected
            'dataset_name': 'autext23', # autext23, autext24, semeval24, coling24
            'cut_off_dataset': '100_100_100',
            # autext23:  10-10-10 | 50-50-50 | 100-100-100*
            # semeval24: 1-1-1 | 5-5-5 | 10-10-10* | 25-25-25 | 50-50-50*
            # coling24: 1-1-1 | 2-2-2 | 5-10-5 | 10-10-10*

            'cuda_num': 0,
            'build_graph': False,
            'edge_attr': True,
            'reduce_dim_emb': False, # False->768
            'add_pos_feat': True, 
            'add_domain_feat': False,
            'project_after_concat': True, # True -> concat all and project | False -> project_llm and then concat
            'reduced_dim': 256,
            
            'balance_dataset': True,
            'max_features': 20000, # None -> all | 5000, 10000, 15000, 20000
            'min_df': 3, # 2->autext | 5->semeval | 5-coling
            'max_df': 1.0,
            'stop_words': False,
            'special_chars': False,
 
            'patience': 10,
            'hidden_gnn_dim': 300,
            'dense_hidden_gnn_dim': 64,
            'num_gnn_layers': 1,
            'heads_gnn': 1,
            'dropout': 0.5,
            'lr': 0.00001, # 0001, 00001, 000001
            'gnn_type': 'Hybrid-ISG-GTN', # Vanilla-GTN-NoEdge', Vanilla-GTN-EdgeConcat, ISG-VanillaGTN, Hybrid-ISG-GTN
            'norm_type': 'batchnorm',        # 'batchnorm', 'layernorm', or None
            'post_mp_layers': 3,             # 1, 2, or 3 layers
            'pooling_type': 'mean',           # 'mean', 'attention', 'max', 'sum', 'set2set' 

            'use_lap_pe': True, 
            'lap_k': 8, 
            'use_degree_pe': False,
            'use_rw_pe': False, 
            'rw_steps': (1,2,4,8),

            "use_isg_structural_pe": False,  
            "use_token_distance_attn": True,  
            "use_global_pooling": False,  

            ## intfloat/multilingual-e5-large
            ## google-bert/bert-base-multilingual-uncased
            ## google-bert/bert-base-uncased
            ## FacebookAI/roberta-base
            ## microsoft/deberta-v3-base
            'lang_model_name': 'microsoft/deberta-v3-base',
            'leave_out_sources': True, # True, False, 'LODO'
        }

    dataset_name = config['dataset_name']
    lodo_domains = {
        'autext23': ["tweets", "legal", "wiki"],
        'autext24': ["literary", "news", "reviews", "tweets", "wikipedia"],
        'semeval24': ["arxiv", "peerread", "reddit", "wikihow", "wikipedia"],
        'coling24': ["hc3", "m4gt", "mage"]
    }

    # Perform LODO if configured
    if config.get('leave_out_sources') == "LODO":
        for domain in lodo_domains[dataset_name]:
            config['leave_out_sources'] = [domain]
            config['file_name_data'] = f"isg_data_{dataset_name}_{config['cut_off_dataset']}perc"
            config['output_dir'] = f'{test_utils.EXTERNAL_DISK_PATH}isg_graph'

            mlflow.set_experiment(f"GNN - ISG")
            run_description = f"""Run experiment for GNN Classification Task using dataset {dataset_name} with {config['cut_off_dataset']} % cutoff leaving out {domain} domains"""
            run_tags = {
                'mlflow.note.content': run_description,
                'mlflow.source.type': "LOCAL"
            }

            with mlflow.start_run(tags=run_tags):
                mlflow.set_tag("mlflow.runName", f"lodo_run_{dataset_name}_{config['cut_off_dataset']}perc")
                for k, v in config.items():
                    mlflow.log_param(k, v)
                main(**config)
    # Fallback: standard run
    else:

        if type(config["leave_out_sources"]) == list:
            ...
        elif config['leave_out_sources']:
            if config['dataset_name'] == 'autext23':
                # Autext: ["wiki", "tweets", "legal"]
                config['leave_out_sources'] = ["tweets"] 
            elif config['dataset_name'] == 'autext24':
                # Autext: ["literary", "news", "reviews", "tweets", "wikipedia"]
                # OK - "news", "literary"
                # OK - "news", "wikipedia"
                # OK - "literary", "wikipedia"
                # OK - "literary", "reviews"
                #  - "news", "reviews"
                #  - "news", "tweets"
                config['leave_out_sources'] = ["literary", "news"] 
            elif config['dataset_name'] == 'semeval24':
                # Semeval ["arxiv", "peerread", "reddit", "wikihow", "wikipedia"]
                config['leave_out_sources'] = ["wikihow", "wikipedia"]
            elif config['dataset_name'] == 'coling24':
                # Coling: ["hc3", "m4gt", "mage"]
                config['leave_out_sources'] = ["mage"]

        config['file_name_data'] = f"isg_data_{dataset_name}_{config['cut_off_dataset']}perc"
        config['output_dir'] = f'{test_utils.EXTERNAL_DISK_PATH}isg_graph'

        if not config['build_graph']:
            config['data'] = utils.load_data(config['file_name_data'], path=f'{config["output_dir"]}/', format_file='.pkl', compress=False)

        mlflow.set_experiment(f"GNN - ISG")
        run_description = f"""Run experiment for GNN Classification Task using dataset {dataset_name} with {config['cut_off_dataset']} % cutoff."""
        run_tags = {
            'mlflow.note.content': run_description,
            'mlflow.source.type': "LOCAL"
        }

        with mlflow.start_run(tags=run_tags):
            mlflow.set_tag("mlflow.runName", f"run_{dataset_name}_{config['cut_off_dataset']}perc")
            for k, v in config.items():
                mlflow.log_param(k, v)
            main(**config)

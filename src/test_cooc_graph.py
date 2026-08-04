import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
import sys
import os
import copy
import time
import logging
import warnings
import json
import argparse
import inspect
import math
from itertools import combinations
from collections import Counter, defaultdict
from tqdm import tqdm

import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers import AutoTokenizer, AutoModel

from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GATConv, TransformerConv
from torch_geometric.nn import global_mean_pool

from joblib import Parallel, delayed

import mlflow
import pandas as pd

import test_utils
import utils

# ─── Config ───────────────────────────────────────────────────────────────────
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

log_file_path = "training_cooc.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s; - %(levelname)s; - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

mlflow.set_tracking_uri("/home/avaldez/projects/GraphDeepLearning/mlruns")

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')


# ─── Helpers ──────────────────────────────────────────────────────────────────

def canonicalize_dataset_name(dataset_name):
    aliases = {'autext_s2': 'autext23_s2'}
    return aliases.get(dataset_name, dataset_name)


def get_main_kwargs(config: dict) -> dict:
    accepted = inspect.signature(main).parameters
    return {k: v for k, v in config.items() if k in accepted}


def configure_mlflow_run(config: dict, dataset_name: str):
    mlflow.set_experiment(config.get("mlflow_exp_name", "CoOc-Graph"))
    name = config.get("name", "manual")
    cutoff = config["cut_off_dataset"]
    run_name = f"{name}_{dataset_name}_{cutoff}perc"
    return {
        "mlflow.note.content": f"CoOc GNN | dataset={dataset_name} cutoff={cutoff}",
        "mlflow.source.type": "LOCAL",
        "mlflow.runName": run_name,
    }


def infer_num_classes(*label_lists):
    labels = sorted({int(l) for ll in label_lists for l in ll})
    return len(labels), labels


def compute_class_weights(data_list, num_classes, device):
    targets = torch.tensor([int(d.y.item()) for d in data_list], dtype=torch.long)
    counts = torch.bincount(targets, minlength=num_classes).float()
    weights = targets.numel() / (num_classes * counts.clamp(min=1.0))
    return weights.to(device), counts


# ─── Graph improvements: PMI and TF-IDF ───────────────────────────────────────

def compute_pmi(texts_tokenized: list, vocab: set, window_size: int,
                min_count: int = 5) -> dict:
    """
    Compute Normalized PMI (NPMI) for all word pairs that co-occur within
    a sliding window across the corpus. Returns {(w1,w2): npmi} for pairs
    with positive association (npmi > 0).

    NPMI is bounded in [-1, 1]: 1 means always co-occurring, 0 means
    independent, -1 means never co-occurring. Only pairs > 0 are returned.
    """
    word_count: Counter = Counter()
    pair_count: Counter = Counter()

    for tokens in tqdm(texts_tokenized, desc="Computing PMI", leave=False):
        valid = [t for t in tokens if t in vocab]
        for i, w1 in enumerate(valid):
            word_count[w1] += 1
            for w2 in valid[i + 1: i + window_size + 1]:
                if w2 != w1:
                    pair_count[tuple(sorted([w1, w2]))] += 1

    total_w = max(sum(word_count.values()), 1)
    total_p = max(sum(pair_count.values()), 1)

    pmi_scores: dict = {}
    for (w1, w2), cnt in pair_count.items():
        if cnt < min_count:
            continue
        p_w1   = word_count[w1] / total_w
        p_w2   = word_count[w2] / total_w
        p_pair = cnt / total_p
        denom  = p_w1 * p_w2
        if denom <= 0 or p_pair <= 0:
            continue
        pmi  = math.log2(p_pair / denom)
        npmi = pmi / (-math.log2(p_pair))   # normalize to [-1, 1]
        if npmi > 0:
            pmi_scores[(w1, w2)] = npmi
            pmi_scores[(w2, w1)] = npmi

    logger.info(f"PMI computed: {len(pmi_scores) // 2} positive pairs")
    return pmi_scores


def compute_tfidf_scores(texts_norm: list, vocab: set) -> list:
    """
    Fit a TF-IDF vectorizer (with sublinear_tf) over all texts using the
    graph vocabulary, and return a list of per-document dicts
    {word: tfidf_score}.
    """
    vec = TfidfVectorizer(vocabulary=sorted(vocab), sublinear_tf=True)
    mat = vec.fit_transform(texts_norm)
    feat_names = vec.get_feature_names_out()

    doc_tfidf = []
    for i in range(mat.shape[0]):
        row = mat.getrow(i)
        doc_tfidf.append({feat_names[j]: float(row[0, j]) for j in row.indices})
    return doc_tfidf


# ─── Model ────────────────────────────────────────────────────────────────────

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = float('-inf') if mode == 'max' else float('inf')

    def early_stop(self, score):
        improved = (score >= self.best_score + self.min_delta) if self.mode == 'max' \
                   else (score <= self.best_score - self.min_delta)
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, dense_hidden_dim, output_dim,
                 dropout, num_layers, gnn_type='GCNConv', heads=1,
                 use_edge_attr=False, edge_attr_dim=1):
        super().__init__()
        self.gnn_type = gnn_type
        self.heads = heads if gnn_type != 'GCNConv' else 1
        self.dropout = dropout
        self.num_layers = num_layers
        self.use_edge_attr = use_edge_attr
        self.edge_attr_dim = edge_attr_dim

        out1 = hidden_dim * self.heads
        self.conv1 = self._build_conv(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(out1)

        self.convs = nn.ModuleList([self._build_conv(out1, hidden_dim) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(out1) for _ in range(num_layers)])

        self.post_mp = nn.Sequential(
            nn.Linear(out1, dense_hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dense_hidden_dim, dense_hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dense_hidden_dim // 2, output_dim),
        )

    def _build_conv(self, in_dim, out_dim):
        if self.gnn_type == 'GCNConv':
            return GCNConv(in_dim, out_dim)
        elif self.gnn_type == 'GATConv':
            if self.use_edge_attr:
                return GATConv(in_dim, out_dim, heads=self.heads, edge_dim=self.edge_attr_dim)
            return GATConv(in_dim, out_dim, heads=self.heads)
        elif self.gnn_type == 'TransformerConv':
            if self.use_edge_attr:
                return TransformerConv(in_dim, out_dim, heads=self.heads, edge_dim=self.edge_attr_dim)
            return TransformerConv(in_dim, out_dim, heads=self.heads)
        raise ValueError(f"Unsupported gnn_type: {self.gnn_type}")

    def _apply_conv(self, conv, x, edge_index, edge_attr):
        if self.use_edge_attr and edge_attr is not None:
            if self.gnn_type == 'GCNConv':
                return conv(x, edge_index, edge_attr.squeeze(1))  # GCNConv takes 1D edge_weight
            return conv(x, edge_index, edge_attr)
        return conv(x, edge_index)

    def get_graph_embedding(self, x, edge_index, edge_attr=None, batch=None):
        x = F.relu(self.norm1(self._apply_conv(self.conv1, x, edge_index, edge_attr)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        for conv, norm in zip(self.convs, self.norms):
            x = F.relu(norm(self._apply_conv(conv, x, edge_index, edge_attr)))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return global_mean_pool(x, batch)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        return self.post_mp(self.get_graph_embedding(x, edge_index, edge_attr, batch))


# ─── Training / Evaluation ────────────────────────────────────────────────────

def train_epoch(model, loader, device, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') and data.edge_attr is not None else None
        out = model(data.x, data.edge_index, edge_attr, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            edge_attr = data.edge_attr if hasattr(data, 'edge_attr') and data.edge_attr is not None else None
            out = model(data.x, data.edge_index, edge_attr, data.batch)
            total_loss += criterion(out, data.y).item()
            all_preds.extend(out.argmax(dim=1).cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
    f1  = f1_score(all_labels, all_preds, average='macro')
    acc = accuracy_score(all_labels, all_preds)
    return f1, acc, total_loss / len(loader), all_preds, all_labels


# ─── Graph Construction ───────────────────────────────────────────────────────

def normalize_text_corpus(texts, special_chars=False, stop_words=False, set_name='train'):
    tokenize_pattern = r"[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+"
    texts_norm, tokenized = [], []
    for text in tqdm(texts, desc=f"Normalizing {set_name}"):
        norm = test_utils.text_normalize(text, special_chars, stop_words)
        texts_norm.append(norm)
        tokenized.append(re.findall(tokenize_pattern, norm))
    return texts_norm, tokenized


def create_vocab(texts, min_df=1, max_df=0.9, max_features=None):
    vectorizer = CountVectorizer(min_df=min_df, max_df=max_df, max_features=max_features)
    vectorizer.fit_transform(texts)
    vocab = set(vectorizer.get_feature_names_out())
    print(f"Vocab size: {len(vocab)}")
    return vocab


def get_word_embeddings_batched(texts_norm, texts_tokenized, tokenizer, language_model,
                                vocab, device, set_corpus='train',
                                not_found='avg', batch_size=32):
    """Batch LLM inference to extract per-document contextual word embeddings."""
    doc_word_embs = [{} for _ in texts_norm]
    hidden_size = language_model.config.hidden_size

    for batch_start in tqdm(range(0, len(texts_norm), batch_size),
                            desc=f"LLM embeddings [{set_corpus}]"):
        batch_texts   = texts_norm[batch_start:batch_start + batch_size]
        batch_indices = list(range(batch_start, min(batch_start + batch_size, len(texts_norm))))

        encoded = tokenizer(
            batch_texts, return_tensors="pt",
            padding=True, truncation=True, max_length=512
        ).to(device)

        with torch.no_grad():
            last_hidden = language_model(**encoded, output_hidden_states=True).hidden_states[-1]

        for i, doc_idx in enumerate(batch_indices):
            token_freq: dict = defaultdict(int)
            seq_len    = int(encoded['attention_mask'][i].sum().item())
            raw_tokens = [tokenizer.decode([tid]) for tid in encoded['input_ids'][i, :seq_len]]

            for tok, emb in zip(raw_tokens, last_hidden[i, :seq_len]):
                tok = tok.strip()
                if tok not in vocab:
                    continue
                token_freq[tok] += 1
                cpu_emb = emb.cpu().detach()
                if tok not in doc_word_embs[doc_idx]:
                    doc_word_embs[doc_idx][tok] = cpu_emb.clone()
                else:
                    doc_word_embs[doc_idx][tok] += cpu_emb

            for tok, freq in token_freq.items():
                if freq > 1:
                    doc_word_embs[doc_idx][tok] /= freq

    for doc_idx, tokens in enumerate(
            tqdm(texts_tokenized, desc=f"Filling missing tokens [{set_corpus}]")):
        missing = (set(vocab) & set(tokens)) - doc_word_embs[doc_idx].keys()
        for word in missing:
            if not_found == 'zeros':
                doc_word_embs[doc_idx][word] = torch.zeros(hidden_size)
            elif not_found == 'ones':
                doc_word_embs[doc_idx][word] = torch.ones(hidden_size)
            else:  # avg subtokens
                subtoks  = [t.strip() for t in tokenizer.tokenize(word)]
                emb_list = [doc_word_embs[doc_idx][t] for t in subtoks
                            if t in doc_word_embs[doc_idx]]
                doc_word_embs[doc_idx][word] = (
                    torch.stack(emb_list).mean(0) if emb_list else torch.zeros(hidden_size)
                )
    return doc_word_embs


def _build_one_graph(
    text_tokens:      list,
    label:            int,
    word_embs:        dict,
    vocab:            set,
    window_size:      int,
    pmi_scores:       dict | None = None,
    tfidf_scores:     dict | None = None,
    use_edge_weights: bool = True,
    use_pmi:          bool = False,
    use_doc_node:     bool = True,
    use_self_loops:   bool = True,
):
    """
    Build one co-occurrence graph for a single document.

    Improvements over the binary baseline:
      - Edge weights: log(freq) x sum(1/distance) captures how often and how
        close two words appear together within the window.
      - PMI filtering/boosting: edges with NPMI <= 0 are removed; positive
        pairs are boosted by (1 + NPMI). Requires pmi_scores precomputed.
      - TF-IDF node feature: appends a scalar TF-IDF score to each node's
        LLM embedding so the GNN knows which words are discriminative.
      - Virtual document node: one extra node (feat = mean of all words)
        connected to every word node, acting as a global information hub
        and reducing the effective graph diameter.
      - Self-loops: every node receives its own feature during aggregation,
        critical for GCN / GAT / TransformerConv stability.
    """
    try:
        # Preserve first-occurrence order (better than set() which is unordered)
        seen: dict = {}
        for w in text_tokens:
            if w not in seen and w in vocab and w in word_embs:
                seen[w] = len(seen)
        unique_words = list(seen.keys())
        if not unique_words:
            return None

        local_idx = seen  # {word: local_node_id}

        # ── Build edge weights: frequency x distance decay ─────────────────
        edge_w:    dict = defaultdict(float)   # (i,j) -> sum(1/dist)
        edge_freq: dict = defaultdict(int)     # (i,j) -> co-occurrence count

        for pos, w1 in enumerate(text_tokens):
            if w1 not in local_idx:
                continue
            end = min(pos + window_size + 1, len(text_tokens))
            for j in range(pos + 1, end):
                w2 = text_tokens[j]
                if w2 not in local_idx or w2 == w1:
                    continue
                dist = j - pos
                key  = (local_idx[w1], local_idx[w2])
                edge_w[key]    += 1.0 / dist
                edge_freq[key] += 1

        if not edge_w:
            return None

        # ── Assemble edge lists ────────────────────────────────────────────
        src, dst, attrs = [], [], []
        for (i1, i2), w_sum in edge_w.items():
            freq   = edge_freq[(i1, i2)]
            weight = math.log1p(freq) * w_sum if use_edge_weights else 1.0

            # PMI: filter non-informative pairs, boost informative ones
            if use_pmi and pmi_scores is not None:
                npmi = pmi_scores.get((unique_words[i1], unique_words[i2]), 0.0)
                if npmi <= 0:
                    continue                  # remove edges with no positive association
                weight *= (1.0 + npmi)        # boost by NPMI in (0, 1]

            src   += [i1, i2]
            dst   += [i2, i1]
            attrs += [weight, weight]         # undirected: both directions

        if not src:
            return None

        # ── Node features ──────────────────────────────────────────────────
        base_feats = torch.stack([word_embs[w] for w in unique_words])   # [N, H]

        # Append TF-IDF score as an extra node dimension (discriminative weight)
        if tfidf_scores is not None:
            tfidf_vec  = torch.tensor(
                [tfidf_scores.get(w, 0.0) for w in unique_words], dtype=torch.float
            ).unsqueeze(1)                                                # [N, 1]
            node_feats = torch.cat([base_feats, tfidf_vec], dim=1)       # [N, H+1]
        else:
            node_feats = base_feats

        n_words = len(unique_words)

        # ── Virtual document node ──────────────────────────────────────────
        # Feature = mean of all word embeddings (global summary of the doc)
        # Connected to every word node bidirectionally with neutral weight 1.0
        if use_doc_node:
            doc_feat   = node_feats.mean(0, keepdim=True)                # [1, H(+1)]
            node_feats = torch.cat([node_feats, doc_feat], dim=0)        # [N+1, H(+1)]
            doc_idx    = n_words
            for wi in range(n_words):
                src   += [wi, doc_idx]
                dst   += [doc_idx, wi]
                attrs += [1.0, 1.0]

        # ── Finalize edge tensors ──────────────────────────────────────────
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr  = torch.tensor(attrs, dtype=torch.float).unsqueeze(1)  # [E, 1]

        # ── Self-loops ─────────────────────────────────────────────────────
        # Added after doc_node so every node (including the virtual one) gets one.
        # Essential for GCN/GAT/TransformerConv to retain the node's own representation.
        if use_self_loops:
            n_nodes    = node_feats.shape[0]
            loop_idx   = torch.arange(n_nodes, dtype=torch.long)
            loop_ei    = torch.stack([loop_idx, loop_idx], dim=0)         # [2, N]
            loop_ea    = torch.ones(n_nodes, 1, dtype=torch.float)        # [N, 1]
            edge_index = torch.cat([edge_index, loop_ei], dim=1)
            edge_attr  = torch.cat([edge_attr,  loop_ea], dim=0)

        return Data(x=node_feats, edge_index=edge_index, edge_attr=edge_attr,
                    y=torch.tensor([label]))

    except Exception as e:
        logger.warning(f"Graph build failed: {e}")
        return None


def build_graph_data(texts_tokenized, labels, word_embs, vocab, window_size,
                     pmi_scores=None, doc_tfidf=None,
                     use_edge_weights=True, use_pmi=False,
                     use_doc_node=True, use_self_loops=True,
                     set_name='train', n_jobs=4):
    results = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(_build_one_graph)(
            tokens, label, word_embs[idx], vocab, window_size,
            pmi_scores=pmi_scores,
            tfidf_scores=doc_tfidf[idx] if doc_tfidf is not None else None,
            use_edge_weights=use_edge_weights,
            use_pmi=use_pmi,
            use_doc_node=use_doc_node,
            use_self_loops=use_self_loops,
        )
        for idx, (tokens, label) in enumerate(
            tqdm(zip(texts_tokenized, labels), total=len(labels),
                 desc=f"Building graphs [{set_name}]"))
    )
    return [r for r in results if r is not None]


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(
    dataset_name, cut_off_dataset, cuda_num=0,
    build_graph=True, window_size=10,
    special_chars=False, stop_words=False,
    min_df=1, max_df=0.9, max_features=None,
    batch_size=64, llm_batch_size=32, not_found_tokens='avg',
    llm_name='microsoft/deberta-v3-base',
    gnn_type='TransformerConv',
    hidden_dim=100, dense_hidden_dim=64,
    num_layers=2, heads=2, dropout=0.5,
    epochs=100, patience=10, lr=0.00002,
    weight_decay=1e-5, n_jobs=4,
    # ── Graph improvement flags ──────────────────────────────────────────
    use_edge_weights=True,   # log(freq) x sum(1/dist) edge weighting
    use_pmi=False,           # PMI-based edge filtering + boosting
    pmi_min_count=5,         # minimum co-occurrence count to compute PMI
    use_tfidf_feat=True,     # append TF-IDF score as extra node feature (+1 dim)
    use_doc_node=True,       # add virtual document node connected to all words
    use_self_loops=True,     # add self-loops for stable GNN aggregation
    use_edge_attr=True,      # pass edge_attr to the GNN convolutions
    # ────────────────────────────────────────────────────────────────────
    file_name_data='', output_dir='',
):
    dataset_name = canonicalize_dataset_name(dataset_name)
    mlflow.log_params({
        "dataset":          dataset_name,
        "cut_off_dataset":  cut_off_dataset,
        "llm_name":         llm_name,
        "gnn_type":         gnn_type,
        "window_size":      window_size,
        "use_edge_weights": use_edge_weights,
        "use_pmi":          use_pmi,
        "pmi_min_count":    pmi_min_count,
        "use_tfidf_feat":   use_tfidf_feat,
        "use_doc_node":     use_doc_node,
        "use_self_loops":   use_self_loops,
        "use_edge_attr":    use_edge_attr,
    })

    device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")

    # Support both '10_10_10' and single-int cut_off formats
    if isinstance(cut_off_dataset, str) and '_' in cut_off_dataset:
        cut_train, cut_val, cut_test = [int(x) for x in cut_off_dataset.split('_')]
    else:
        cut_train = cut_val = cut_test = int(cut_off_dataset)

    if build_graph:
        train_text_set, val_text_set, test_text_set = test_utils.read_dataset(
            dataset_name, print_info=False)

        train_set = train_text_set[:int(len(train_text_set) * cut_train / 100)]
        val_set   = val_text_set[:int(len(val_text_set)   * cut_val   / 100)]
        test_set  = test_text_set[:int(len(test_text_set) * cut_test  / 100)]

        print("distro_train_val_test:", len(train_set), len(val_set), len(test_set))
        print("label_distro:", train_set.value_counts('label').to_dict(),
              val_set.value_counts('label').to_dict(),
              test_set.value_counts('label').to_dict())

        train_texts = list(train_set['text'])
        val_texts   = list(val_set['text'])
        test_texts  = list(test_set['text'])

        train_labels = list(train_set['label'])
        val_labels   = list(val_set['label'])
        test_labels  = list(test_set['label'])

        num_classes, label_list = infer_num_classes(train_labels, val_labels, test_labels)
        mlflow.log_param("num_classes", num_classes)

        train_texts_norm, train_tokens = normalize_text_corpus(train_texts, special_chars, stop_words, 'train')
        val_texts_norm,   val_tokens   = normalize_text_corpus(val_texts,   special_chars, stop_words, 'val')
        test_texts_norm,  test_tokens  = normalize_text_corpus(test_texts,  special_chars, stop_words, 'test')

        all_texts_norm = train_texts_norm + val_texts_norm + test_texts_norm
        all_tokens     = train_tokens     + val_tokens     + test_tokens
        vocab = create_vocab(all_texts_norm, min_df=min_df, max_df=max_df, max_features=max_features)

        # ── Corpus-level PMI ─────────────────────────────────────────────
        pmi_scores = None
        if use_pmi:
            pmi_scores = compute_pmi(all_tokens, vocab, window_size, pmi_min_count)
            mlflow.log_param("pmi_pairs", len(pmi_scores) // 2)

        # ── Per-document TF-IDF scores ────────────────────────────────────
        train_tfidf = val_tfidf = test_tfidf = None
        if use_tfidf_feat:
            n_train     = len(train_texts_norm)
            n_val       = len(val_texts_norm)
            all_tfidf   = compute_tfidf_scores(all_texts_norm, vocab)
            train_tfidf = all_tfidf[:n_train]
            val_tfidf   = all_tfidf[n_train: n_train + n_val]
            test_tfidf  = all_tfidf[n_train + n_val:]

        # ── LLM contextual embeddings ─────────────────────────────────────
        tokenizer      = AutoTokenizer.from_pretrained(llm_name, model_max_length=512)
        language_model = AutoModel.from_pretrained(llm_name, output_hidden_states=True).to(device)
        language_model.eval()

        train_word_embs = get_word_embeddings_batched(
            train_texts_norm, train_tokens, tokenizer, language_model,
            vocab, device, 'train', not_found_tokens, llm_batch_size)
        val_word_embs = get_word_embeddings_batched(
            val_texts_norm, val_tokens, tokenizer, language_model,
            vocab, device, 'val', not_found_tokens, llm_batch_size)
        test_word_embs = get_word_embeddings_batched(
            test_texts_norm, test_tokens, tokenizer, language_model,
            vocab, device, 'test', not_found_tokens, llm_batch_size)

        del language_model
        torch.cuda.empty_cache()

        # ── Build graphs ──────────────────────────────────────────────────
        train_data = build_graph_data(
            train_tokens, train_labels, train_word_embs, vocab, window_size,
            pmi_scores=pmi_scores, doc_tfidf=train_tfidf,
            use_edge_weights=use_edge_weights, use_pmi=use_pmi,
            use_doc_node=use_doc_node, use_self_loops=use_self_loops,
            set_name='train', n_jobs=n_jobs)
        val_data = build_graph_data(
            val_tokens, val_labels, val_word_embs, vocab, window_size,
            pmi_scores=pmi_scores, doc_tfidf=val_tfidf,
            use_edge_weights=use_edge_weights, use_pmi=use_pmi,
            use_doc_node=use_doc_node, use_self_loops=use_self_loops,
            set_name='val', n_jobs=n_jobs)
        test_data = build_graph_data(
            test_tokens, test_labels, test_word_embs, vocab, window_size,
            pmi_scores=pmi_scores, doc_tfidf=test_tfidf,
            use_edge_weights=use_edge_weights, use_pmi=use_pmi,
            use_doc_node=use_doc_node, use_self_loops=use_self_loops,
            set_name='test', n_jobs=n_jobs)

        os.makedirs(output_dir, exist_ok=True)
        utils.save_data(
            {"vocab": vocab, "train_data": train_data,
             "val_data": val_data, "test_data": test_data},
            file_name_data, path=f'{output_dir}/', format_file='.pkl', compress=False)

    else:
        data_obj   = utils.load_data(file_name_data, path=f'{output_dir}/', format_file='.pkl', compress=False)
        train_data = data_obj['train_data']
        val_data   = data_obj['val_data']
        test_data  = data_obj['test_data']

    inferred_labels = sorted({int(d.y.item()) for d in train_data + val_data + test_data})
    num_classes = len(inferred_labels)
    label_list  = inferred_labels

    # input_dim inferred from the first graph (accounts for TF-IDF +1 dim automatically)
    input_dim     = train_data[0].x.shape[1]
    has_edge_attr = hasattr(train_data[0], 'edge_attr') and train_data[0].edge_attr is not None

    print(f"num_classes={num_classes} | input_dim={input_dim} | "
          f"edge_attr={has_edge_attr} | "
          f"train={len(train_data)} | val={len(val_data)} | test={len(test_data)}")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_data,  batch_size=batch_size, shuffle=False, num_workers=0)

    test_utils.set_random_seed(42)
    model = GNN(
        input_dim, hidden_dim, dense_hidden_dim, num_classes,
        dropout, num_layers, gnn_type=gnn_type, heads=heads,
        use_edge_attr=(use_edge_attr and has_edge_attr),
        edge_attr_dim=1,
    ).to(device)
    print(model)

    class_weights, class_counts = compute_class_weights(train_data, num_classes, device)
    optimizer     = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)
    early_stopper = EarlyStopper(patience=patience, min_delta=0, mode='max')

    print("class_counts:",  class_counts.tolist())
    print("class_weights:", [round(float(w), 4) for w in class_weights.cpu()])
    mlflow.log_params({
        "class_counts":  class_counts.tolist(),
        "class_weights": [round(float(w), 6) for w in class_weights.cpu()],
        "model_params":  str(model),
    })

    best_val_f1      = float('-inf')
    best_test_f1     = 0.0
    best_model_state = copy.deepcopy(model.state_dict())
    stop_epoch       = 0

    logger.info("Init GNN training!")
    start = time.time()

    for epoch in range(epochs):
        train_loss               = train_epoch(model, train_loader, device, optimizer, criterion)
        val_f1,  val_acc,  val_loss,  _, _ = evaluate(model, val_loader,  device, criterion)
        test_f1, test_acc, test_loss, _, _ = evaluate(model, test_loader, device, criterion)

        print(f"Epoch {epoch:02d} | "
              f"Train-Loss {train_loss:.4f} | Val-Loss {val_loss:.4f} | Test-Loss {test_loss:.4f} | "
              f"Val-Acc {val_acc:.4f} | Val-F1 {val_f1:.4f} | "
              f"Test-Acc {test_acc:.4f} | Test-F1 {test_f1:.4f}")

        mlflow.log_metrics({
            "F1Score-val": val_f1, "Accuracy-val": val_acc, "Loss-val": val_loss,
            "F1Score-test": test_f1, "Accuracy-test": test_acc, "Loss-test": test_loss,
        }, step=epoch)

        if val_f1 > best_val_f1:
            best_val_f1      = val_f1
            best_model_state = copy.deepcopy(model.state_dict())
        if test_f1 > best_test_f1:
            best_test_f1 = test_f1

        stop_epoch = epoch
        if early_stopper.early_stop(val_f1):
            print("Early stopping triggered!")
            break

    logger.info("Done GNN training!")
    print(f"--- {time.time() - start:.1f}s Graph Training Time ---")

    model.load_state_dict(best_model_state)
    test_f1, test_acc, test_loss, preds_test, labels_test = evaluate(
        model, test_loader, device, criterion)
    print(f"----> Test-Loss {test_loss:.4f} | Test-Acc {test_acc:.4f} | Test-F1 {test_f1:.4f}")
    print(confusion_matrix(labels_test, preds_test, labels=label_list))

    os.makedirs(f"{output_dir}/models", exist_ok=True)
    torch.save(model.state_dict(), f"{output_dir}/models/cooc_model_{file_name_data}.pth")

    mlflow.log_metrics({
        "Final-F1Macro-test":  test_f1,
        "Final-Accuracy-test": test_acc,
        "Final-Loss-test":     test_loss,
        "Best-F1Macro-test":   best_test_f1,
        "stop_epoch":          stop_epoch,
    })
    mlflow.log_artifact(log_file_path)

    # ── Prediction breakdown ───────────────────────────────────────────────
    _, _, test_text_set = test_utils.read_dataset(dataset_name, print_info=False)
    test_set_df = test_text_set[:int(len(test_text_set) * cut_test / 100)].copy()
    if 'model' not in test_set_df.columns:
        test_set_df['model'] = 'unknown'
    test_set_df['preds_test']  = preds_test
    test_set_df['labels_test'] = labels_test

    preds_path = f"{utils.OUTPUT_DIR_PATH}preds_cooc_{dataset_name}.csv"
    test_set_df[['id', 'label', 'labels_test', 'preds_test', 'source', 'model']].to_csv(
        preds_path, index=False)
    mlflow.log_artifact(preds_path)

    source_acc = (
        test_set_df.groupby("source")
        .apply(lambda df: (df["preds_test"] == df["labels_test"]).mean())
        .reset_index(name="accuracy")
    )
    print(source_acc)


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default=None)
    args = parser.parse_args()

    if args.config_path:
        with open(args.config_path, "r") as f:
            config = json.load(f)
        if config.get('_done') in (True, 'True'):
            sys.exit("Experiment already DONE")
        config.pop('_done', None)
    else:
        config = {
            'name':             'manual_run',
            'mlflow_exp_name':  'CoOc-Graph',

            # ── Dataset ──────────────────────────────────────────────────
            'dataset_name':    'autext23',    # autext23 | autext23_s2 | autext24 | semeval24 | coling24
            'cut_off_dataset': '10_10_10',    # '10_10_10' | '50_50_50' | '100_100_100'
            'cuda_num':        1,

            # ── Graph construction ────────────────────────────────────────
            'build_graph':  True,
            'window_size':  10,               # 10 -> autext | 20 -> semeval/coling

            # ── Vocabulary ───────────────────────────────────────────────
            'min_df':       3,                # 2 -> autext | 5 -> semeval/coling
            'max_df':       0.9,
            'max_features': None,             # None -> all | 5000, 10000, ...

            # ── Text normalization ────────────────────────────────────────
            'special_chars': False,
            'stop_words':    False,

            # ── LLM ──────────────────────────────────────────────────────
            ## google-bert/bert-base-uncased | FacebookAI/roberta-base
            ## microsoft/deberta-v3-base
            'llm_name':         'microsoft/deberta-v3-base',
            'not_found_tokens': 'avg',        # avg | ones | zeros
            'llm_batch_size':   32,           # texts per LLM forward pass

            # ── Graph improvements ────────────────────────────────────────
            # Each flag is independent — mix and match for ablation studies.
            #
            # use_edge_weights: weight each co-occurrence edge by log(freq) x sum(1/dist)
            #   closer words and more frequent pairs get stronger edges
            # use_pmi: filter edges with NPMI <= 0 and boost surviving edges by (1+NPMI)
            #   removes trivial pairs (the-of, is-a, etc.) that add noise
            #   NOTE: slower build; set pmi_min_count higher to skip rare pairs
            # use_tfidf_feat: appends a 1D TF-IDF score to each node's LLM embedding
            #   gives the GNN a signal for how discriminative each word is in its doc
            #   NOTE: increases input_dim by 1 (handled automatically)
            # use_doc_node: adds one virtual node (mean embedding) connected to all words
            #   shortens graph diameter; helps aggregate global document context
            # use_self_loops: every node (including doc node) gets a self-loop
            #   required for GCN/GAT/TransformerConv to retain own features during aggregation
            # use_edge_attr: whether the GNN convolutions consume the edge_attr tensor
            #   if False, edge_attr is still built but ignored by the model
            'use_edge_weights': True,
            'use_pmi':          False,
            'pmi_min_count':    5,
            'use_tfidf_feat':   True,
            'use_doc_node':     True,
            'use_self_loops':   True,
            'use_edge_attr':    True,

            # ── GNN architecture ──────────────────────────────────────────
            'gnn_type':         'TransformerConv',  # GCNConv | GATConv | TransformerConv
            'hidden_dim':       100,
            'dense_hidden_dim': 64,
            'num_layers':       2,
            'heads':            2,
            'dropout':          0.5,

            # ── Training ─────────────────────────────────────────────────
            'batch_size':    64,
            'epochs':        100,
            'patience':      10,
            'lr':            0.00002,         # autext: 2e-5 | semeval: 1e-6 | coling: 1e-4
            'weight_decay':  1e-5,
            'n_jobs':        4,               # parallel workers for graph construction
        }

    config['dataset_name'] = canonicalize_dataset_name(config['dataset_name'])
    dataset_name = config['dataset_name']

    # setdefault: runner-supplied values win; these are fallbacks for manual runs
    config.setdefault('file_name_data', f"cooc_data_{dataset_name}_{config['cut_off_dataset']}perc")
    config.setdefault('output_dir', f'{test_utils.EXTERNAL_DISK_PATH}cooc_graph')

    run_tags = configure_mlflow_run(config, dataset_name)
    with mlflow.start_run(tags=run_tags):
        for k, v in config.items():
            mlflow.log_param(k, v)
        main(**get_main_kwargs(config))

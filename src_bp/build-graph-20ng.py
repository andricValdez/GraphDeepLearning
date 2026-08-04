import os
import random
import numpy as np
import pickle as pkl
import scipy.sparse as sp
from math import log
from collections import defaultdict, Counter
from tqdm import tqdm
import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

import utils
import test_utils

# Utility Functions

def load_dataset(dataset_path):
    doc_name_list, doc_train_list, doc_test_list = [], [], []
    with open(dataset_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            doc_name_list.append(line.strip())
            temp = line.split("\t")
            if 'test' in temp[1]:
                doc_test_list.append(line.strip())
            elif 'train' in temp[1]:
                doc_train_list.append(line.strip())
    return doc_name_list, doc_train_list, doc_test_list

def load_corpus(corpus_path):
    with open(corpus_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def build_vocab(documents):
    vocab, word_freq = set(), {}
    
    for doc in tqdm(documents, desc="build_vocab"):
        for word in doc.split():
            vocab.add(word)
            word_freq[word] = word_freq.get(word, 0) + 1
    vocab = list(vocab)
    word_id_map = {word: idx for idx, word in enumerate(vocab)}
    return vocab, word_freq, word_id_map

def build_windows(documents, window_size):
    windows = []
    for doc in tqdm(documents, desc="build_windows"):
        words = doc.split()
        if len(words) <= window_size:
            windows.append(words)
        else:
            for j in range(len(words) - window_size + 1):
                windows.append(words[j:j + window_size])
    return windows

def compute_pmi(windows, vocab, word_id_map):
    word_window_freq = defaultdict(int)
    word_pair_count = defaultdict(int)
    for window in tqdm(windows, desc="compute_pmi"):
        appeared = set()
        for word in window:
            if word not in appeared:
                word_window_freq[word] += 1
                appeared.add(word)
        for i in range(len(window)):
            for j in range(i):
                if window[i] == window[j]:
                    continue
                word_pair_count[(window[i], window[j])] += 1
                word_pair_count[(window[j], window[i])] += 1

    num_window = len(windows)
    row, col, weight = [], [], []
    for (word_i, word_j), count in word_pair_count.items():
        p_ij = count / num_window
        p_i = word_window_freq[word_i] / num_window
        p_j = word_window_freq[word_j] / num_window
        pmi = log(p_ij / (p_i * p_j))
        if pmi > 0:
            row.append(word_id_map[word_i])
            col.append(word_id_map[word_j])
            weight.append(pmi)
    return row, col, weight

def build_doc_word_edges(documents, word_id_map):
    tokenized_docs = [doc.split() for doc in documents]
    
    # Precompute IDF
    doc_freq = defaultdict(int)
    for tokens in tokenized_docs:
        for word in set(tokens):
            doc_freq[word] += 1
    idf_map = {word: log(len(documents) / (df + 1e-10)) for word, df in doc_freq.items()}
    # Compute term frequency
    doc_word_freq = defaultdict(int)
    for doc_id, tokens in enumerate(tokenized_docs):
        for word in tokens:
            word_id = word_id_map[word]
            doc_word_freq[(doc_id, word_id)] += 1

    row, col, weight = [], [], []
    for doc_id, tokens in enumerate(tqdm(tokenized_docs, desc="build_doc_word_edges")):
        seen = set()
        for word in tokens:
            if word in seen: continue
            seen.add(word)
            word_id = word_id_map[word]
            tf = doc_word_freq[(doc_id, word_id)]
            idf = idf_map[word]
            row.append(doc_id)
            col.append(word_id)
            weight.append(tf * idf)
    return row, col, weight

def build_adjacency_matrix(num_docs, vocab_size, word_word_edges, doc_word_edges):
    row, col, weight = [], [], []
    word_row, word_col, word_weight = word_word_edges
    doc_row, doc_col, doc_weight = doc_word_edges

    # Shift word ids for unified index space
    row.extend([num_docs + r for r in word_row])
    col.extend([num_docs + c for c in word_col])
    weight.extend(word_weight)

    row.extend(doc_row)
    col.extend([num_docs + c for c in doc_col])
    weight.extend(doc_weight)

    node_size = num_docs + vocab_size
    adj = sp.csr_matrix((weight, (row, col)), shape=(node_size, node_size))
    return adj

def normalize_adj(adj):
    """Symmetric normalization:  D^{-1/2} A D^{-1/2}"""
    adj = adj + sp.eye(adj.shape[0])  # Add self-loops
    rowsum = np.array(adj.sum(1))[:, 0]
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    return D_inv_sqrt.dot(adj).dot(D_inv_sqrt)

def scipy_to_torch_sparse(adj_sp):
    adj_sp = adj_sp.tocoo()
    indices = torch.from_numpy(
        np.vstack((adj_sp.row, adj_sp.col)).astype(np.int64)
    )
    values = torch.from_numpy(adj_sp.data.astype(np.float32))
    shape = torch.Size(adj_sp.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def build_masks(num_nodes, train_ids, test_ids, doc_name_list, label_map):
    # Create full label array
    labels = []
    for idx in train_ids:
        label = doc_name_list[idx].split("\t")[2]
        labels.append(label_map[label])

    # Stratified split
    train_ids_strat, val_ids_strat = train_test_split(
        train_ids, test_size=0.1, stratify=labels, random_state=42
    )

    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)

    for idx in train_ids_strat:
        train_mask[idx] = True
    for idx in val_ids_strat:
        val_mask[idx] = True
    for idx in test_ids:
        test_mask[idx] = True

    return train_mask, val_mask, test_mask


def build_labels(doc_name_list, doc_ids, label_list, num_nodes, num_docs):
    label_map = {label: i for i, label in enumerate(label_list)}
    y = np.full(num_nodes, -1, dtype=int)
    for i in doc_ids:
        label = doc_name_list[i].split('\t')[2]
        y[i] = label_map[label]
    return y, label_map


def get_cls_embeddings_batch(texts, tokenizer, model, device, batch_size=32):
    model.to(device)
    model.eval() 
    all_cls_embeddings = []
    # Split into batches
    for i in  tqdm(range(0, len(texts), batch_size), desc="Extracting doc embeddings"):
        batch_texts = texts[i:i+batch_size]
        # Tokenize batch
        inputs = tokenizer(
            batch_texts,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=512
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            # Get CLS token: shape (batch_size, hidden_dim)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            all_cls_embeddings.extend(cls_embeddings.cpu().numpy())
    return all_cls_embeddings

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        x = self.linear(x)              # Step 1: Linear transform (W x)
        x = torch.sparse.mm(adj, x)    # Step 2: Aggregate neighbors (A x W x)
        return F.relu(x)               # Step 3: Non-linearity

class GCN2(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dense_hidden_dim=64, dropout=0.5):
        super().__init__()
        self.gcn1 = GCNLayer(in_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        self.dropout = dropout
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, dense_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_hidden_dim, out_dim)
        )


    def forward(self, x, adj):
        x = self.gcn1(x, adj)          # Layer 1
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gcn2(x, adj)          # Layer 2 (output logits for each class)
        x = self.post_mp(x)
        return x


def build_graph():
    dataset = "20ng"
    dts_path = "/home/avaldez/projects/GraphDeepLearning/datasets/20ng/"
    dataset_path = os.path.join(dts_path, f"{dataset}.txt")
    corpus_path = os.path.join(dts_path,  f"{dataset}.clean.txt")

    doc_name_list, doc_train_list, doc_test_list = load_dataset(dataset_path)
    doc_content_list = load_corpus(corpus_path)
    train_ids = [doc_name_list.index(name) for name in doc_train_list]
    test_ids = [doc_name_list.index(name) for name in doc_test_list]
    ids = train_ids + test_ids
    documents = [doc_content_list[i] for i in ids]
    print(train_ids[0], train_ids[-1])
    print(test_ids[0], test_ids[-1])
    print(len(ids))

    vocab, word_freq, word_id_map = build_vocab(documents)
    windows = build_windows(documents, window_size=20)
    word_word_edges = compute_pmi(windows, vocab, word_id_map)
    doc_word_edges = build_doc_word_edges(documents, word_id_map)
    adj = build_adjacency_matrix(len(documents), len(vocab), word_word_edges, doc_word_edges)
    label_set = list(set(name.split('\t')[2] for name in doc_name_list))
    

    #y = build_labels(doc_name_list, ids, label_set)
    #train_mask, val_mask, test_mask = build_masks(len(documents), train_ids, test_ids)
    num_docs = len(documents)
    vocab_size = len(vocab)
    num_nodes = num_docs + vocab_size
    y, label_map = build_labels(doc_name_list, ids, label_set, num_nodes, num_docs)
    train_mask, val_mask, test_mask = build_masks(num_nodes, train_ids, test_ids, doc_name_list, label_map)

    print("Adjacency matrix shape:", adj.shape)
    print("Labels shape:", y.shape)
    print("Train/Val/Test mask counts:", train_mask.sum(), val_mask.sum(), test_mask.sum())

    # Load model (BERT, etc)
    model_path = f'{utils.OUTPUT_DIR_PATH}baselines/bert-20ng-best'
    tokenizer = BertTokenizer.from_pretrained(model_path)
    llm_model = BertModel.from_pretrained(model_path) 
    hidden_dim = llm_model.config.hidden_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    doc_features = torch.tensor(get_cls_embeddings_batch(documents, tokenizer, llm_model, device, batch_size=32), dtype=torch.float)
    word_features = torch.empty((len(vocab), hidden_dim)).uniform_(-0.01, 0.01)

    # Node features and graph structure
    node_features = torch.cat([doc_features, word_features], dim=0)
    
    output_dir = f'{utils.OUTPUT_DIR_PATH}baselines/'
    filename = 'build-graph-20ng'
    data = {"adj": adj, 'node_features': node_features, 'node_labels': y, "train_mask": train_mask, 'val_mask': val_mask, 'test_mask': test_mask}
    utils.save_data(data, filename, path=f'{output_dir}', format_file='.pkl', compress=False)




def train_gnn():
    filename = 'build-graph-20ng'
    output_dir = f'{utils.OUTPUT_DIR_PATH}baselines/'
    data = utils.load_data(filename, path=output_dir, format_file='.pkl', compress=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adj = data['adj']
    node_features = data['node_features']
    node_labels = data['node_labels']
    train_mask = data['train_mask']
    val_mask = data['val_mask']
    test_mask = data['test_mask']
    print(Counter(train_mask))
    print(Counter(val_mask))
    print(Counter(test_mask))
    print(Counter(node_labels))

    #test_indices = torch.where(torch.tensor(test_mask, dtype=torch.bool))[0]
    #for i in test_indices[:5]:
    #    print(f"Node {i} degree: {adj[i].count_nonzero()}")

    # Convert to tensors
    node_features = torch.tensor(node_features, dtype=torch.float)
    node_labels = torch.tensor(node_labels, dtype=torch.long)
    train_mask = torch.tensor(train_mask, dtype=torch.bool)
    val_mask = torch.tensor(val_mask, dtype=torch.bool)
    test_mask = torch.tensor(test_mask, dtype=torch.bool)

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize_adj(adj)
    adj = scipy_to_torch_sparse(adj)
    print(adj.shape)
    print(node_features.shape)
    print(node_labels.shape)
    
    print("label classes:", set(node_labels[test_mask].tolist()))
    print("Any -1 in labels?", (node_labels[test_mask] == -1).any())
    print("class distribution:", Counter(node_labels[test_mask].tolist()))

    # Quick sanity check
    print("Train labels:", np.unique(node_labels[train_mask].cpu().numpy()))
    print("Val labels:", np.unique(node_labels[val_mask].cpu().numpy()))
    print("Test labels:", np.unique(node_labels[test_mask].cpu().numpy()))


    return

    model = GCN2(in_dim=node_features.shape[1], hidden_dim=200, out_dim=20)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    adj = adj.to(device)
    node_features = node_features.to(device)
    node_labels = node_labels.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)
    model = model.to(device)
    print(model)

    for epoch in range(300):
        model.train()
        logits = model(node_features, adj)

        # Only compute loss on train docs
        loss_train = criterion(logits[train_mask], node_labels[train_mask])

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        # Validation accuracy
        model.eval()
        with torch.no_grad():
            out = model(node_features, adj)  # raw logits
            probs = F.softmax(out, dim=1)  # convert to class probabilities
            pred = probs.argmax(dim=1)
            val_acc = (pred[val_mask] == node_labels[val_mask]).float().mean()
            loss_val = criterion(logits[val_mask], node_labels[val_mask])
            test_acc = (pred[test_mask] == node_labels[test_mask]).float().mean()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss_train: {loss_train:.4f} | Loss_val: {loss_val:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")

    model.eval()
    out = model(node_features, adj)  # raw logits
    probs = F.softmax(out, dim=1)  # convert to class probabilities
    pred = probs.argmax(dim=1)
    test_acc = (pred[test_mask] == node_labels[test_mask]).float().mean()
    print("test_acc: ", test_acc)

if __name__ == "__main__":
    #build_graph()
    train_gnn()

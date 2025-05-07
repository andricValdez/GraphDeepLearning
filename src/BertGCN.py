import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from transformers import BertTokenizer, BertModel
from collections import defaultdict
import math
from torch_geometric.data import Data
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from tqdm import tqdm
import re
from collections import Counter, defaultdict
import random
import torch.nn as nn
import random
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from sklearn.preprocessing import MinMaxScaler
import numpy as np

import pandas as pd
import utils

import test_utils

def tokenize_fn(example, tokenizer):
        return tokenizer(example['text'], truncation=True, padding='max_length', max_length=256)

def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        f1 = f1_score(labels, preds, average="weighted")
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1}

def fine_tune_model():
    # read dataset
    train_docs, test_docs, val_docs, train_labels, test_labels, val_labels = read_dataset()
    
    # Create DataFrames
    df_train = pd.DataFrame({'text': train_docs, 'label': train_labels})
    df_val = pd.DataFrame({'text': val_docs, 'label': val_labels})
    df_test = pd.DataFrame({'text': test_docs, 'label': test_labels})

    # Wrap into DatasetDict
    dataset = DatasetDict({
        'train': Dataset.from_pandas(df_train),
        'val': Dataset.from_pandas(df_val),
        'test': Dataset.from_pandas(df_test)
    })

    # tokenization
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenized_dataset = dataset.map(tokenize_fn, batched=True,  fn_kwargs={"tokenizer": tokenizer})
    tokenized_dataset = tokenized_dataset.remove_columns(['text'])
    tokenized_dataset.set_format(type='torch')

    # model and train setup
    num_labels = 20
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    training_args = TrainingArguments(
        output_dir=f'{utils.OUTPUT_DIR_PATH}baselines/bert-20ng',
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        num_train_epochs=4,
        weight_decay=0.01,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["val"],  # or a validation split
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(f'{utils.OUTPUT_DIR_PATH}baselines/bert-20ng-best')

    final_test_metrics = trainer.evaluate(tokenized_dataset['test'])
    print("Final Test Evaluation:", final_test_metrics)

def evaluate_fine_tune_model():
    # Replace with your actual model directory
    model_path = f'{utils.OUTPUT_DIR_PATH}baselines/bert-20ng-best'
    # read dataset
    train_docs, test_docs, val_docs, train_labels, test_labels, val_labels = read_dataset()
    
    # Create DataFrames
    df_test = pd.DataFrame({'text': test_docs, 'label': test_labels})

    # Wrap into DatasetDict
    test_set = DatasetDict({
        'test': Dataset.from_pandas(df_test)
    })

    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenized_test = test_set.map(tokenize_fn, batched=True, fn_kwargs={"tokenizer": tokenizer})
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)   
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        #data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Run evaluation
    metrics = trainer.evaluate(tokenized_test)
    print("Test set evaluation:", metrics) 


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
    
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, dense_hidden_dim=32, dropout=0.5):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.dropout = dropout
        #self.norm1 = nn.LayerNorm(hidden_channels) # BatchNorm1d, LayerNorm

        self.post_mp = nn.Sequential(
            nn.Linear(hidden_channels, dense_hidden_dim),
            nn.Linear(dense_hidden_dim, int(dense_hidden_dim // 2)),
            nn.Linear(int(dense_hidden_dim // 2), num_classes),
        )

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        #x = self.norm1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.post_mp(x)
        
        return x

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, dropout=0.5):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        return x

def build_adjacency(edge_index, edge_weight, num_nodes):
    row, col = edge_index
    indices = torch.stack([row, col], dim=0)
    values = edge_weight
    adj = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes))

    # Add self-loops
    self_loop = torch.arange(num_nodes)
    loop_index = torch.stack([self_loop, self_loop])
    loop_weight = torch.ones(num_nodes)
    loop_adj = torch.sparse_coo_tensor(loop_index, loop_weight, (num_nodes, num_nodes))

    return (adj + loop_adj).coalesce()

def normalize_adj(adj):
    row, col = adj.indices()
    deg = torch.zeros(adj.size(0)).scatter_add_(0, row, adj.values())
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.

    norm_vals = deg_inv_sqrt[row] * adj.values() * deg_inv_sqrt[col]
    return torch.sparse_coo_tensor(adj.indices(), norm_vals, adj.size())

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


# Train loop
def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    #out = model(data.x, data.edge_index, data.edge_attr)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# Test function
def test(model, criterion, data, mask):
    model.eval()
    out = model(data.x, data.edge_index)
    #out = model(data.x, data.edge_index, data.edge_attr)
    preds = out.argmax(dim=1)
    correct = preds[mask] == data.y[mask]
    acc = int(correct.sum()) / int(mask.sum())
    loss = criterion(out[mask], data.y[mask])
    return acc, loss


# BERT CLS Embeddings for Documents
def get_cls_embedding(text, tokenizer, model, device):
    model.to(device)
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    doc_embedding =  outputs.last_hidden_state[:, 0, :]
    return doc_embedding.cpu().detach().numpy().tolist()

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

def read_dataset():
    cut_off_dataset = '100-100-100'

    doc_content_list = []
    dataset_path = "/home/avaldez/projects/GraphDeepLearning/datasets/20ng/"
    f = open(dataset_path  + '20ng.clean.txt', 'r')
    lines = f.readlines()
    for line in lines:
        doc_content_list.append(line.strip())
    f.close()

    doc_name_list = []
    doc_train_list = []
    doc_test_list = []
    f = open(dataset_path + '20ng.txt', 'r')
    lines = f.readlines()
    for line in lines:
        doc_name_list.append(line.strip())
        temp = line.split("\t")
        if temp[1].find('test') != -1:
            doc_test_list.append(line.strip())
        elif temp[1].find('train') != -1:
            doc_train_list.append(line.strip())
    f.close()

    print(doc_content_list[0])
    print(doc_name_list[0])
    print(doc_train_list[0])
    print(doc_test_list[0])
    print(len(doc_content_list), len(doc_train_list), len(doc_test_list))

    labels_list = []
    labels_ohe = {}
    for doc in doc_name_list:
        labels_list.append(doc.split()[2])
    for label, idx in zip(set(labels_list), [i for i in range(20)]):
        labels_ohe[label] = idx 
    print(Counter(labels_list))
    print(labels_ohe)

    train_docs = doc_content_list[len(doc_test_list):]
    test_docs = doc_content_list[:len(doc_test_list)]

    train_labels = [labels_ohe[doc.split()[2]] for doc in doc_train_list]
    test_labels = [labels_ohe[doc.split()[2]] for doc in doc_test_list]
    
    train_docs, val_docs, train_labels, val_labels = train_test_split(
        train_docs,
        train_labels,
        test_size=0.1,
        random_state=42,
        stratify=train_labels
    )

    #test_shuffle = list(zip(test_docs, test_labels))
    #random.shuffle(test_shuffle)
    #test_docs, test_labels = zip(*test_shuffle)
    #test_docs, test_labels = list(test_docs), list(test_labels) 

    print("train: ", len(train_docs), len(train_labels))    
    print("val  : ", len(val_docs), len(val_labels))    
    print("test : ", len(test_docs), len(test_labels)) 
    print(Counter(train_labels)) 

    return train_docs, test_docs, val_docs, train_labels, test_labels, val_labels

# *************************************** 
#  PREPOCESSING DATA
# *************************************** 

def prep_data_2():
    # Parameters
    min_df = 1
    max_df = 1.0
    window_size = 5
    cut_off_dataset = '50-100-100'

    model_name = 'bert-base-uncased'
    output_dir = f'{utils.OUTPUT_DIR_PATH}baselines/'
    filename = 'BertGCN-20ng-testing'
    
    doc_content_list = []
    dataset_path = "/home/avaldez/projects/GraphDeepLearning/datasets/20ng/"
    f = open(dataset_path  + '20ng.clean.txt', 'r')
    lines = f.readlines()
    for line in lines:
        doc_content_list.append(line.strip())
    f.close()

    doc_name_list = []
    doc_train_list = []
    doc_test_list = []
    f = open(dataset_path + '20ng.txt', 'r')
    lines = f.readlines()
    for line in lines:
        doc_name_list.append(line.strip())
        temp = line.split("\t")
        if temp[1].find('test') != -1:
            doc_test_list.append(line.strip())
        elif temp[1].find('train') != -1:
            doc_train_list.append(line.strip())
    f.close()

    print(doc_content_list[0])
    print(doc_name_list[0])
    print(doc_train_list[0])
    print(doc_test_list[0])
    print(len(doc_content_list), len(doc_train_list), len(doc_test_list))

    labels_list = []
    labels_ohe = {}
    for doc in doc_name_list:
        labels_list.append(doc.split()[2])
    for label, idx in zip(set(labels_list), [i for i in range(20)]):
        labels_ohe[label] = idx 
    print(Counter(labels_list))
    print(labels_ohe)

    train_docs = doc_content_list[len(doc_test_list):]
    test_docs = doc_content_list[:len(doc_test_list)]

    train_labels = [labels_ohe[doc.split()[2]] for doc in doc_train_list]
    test_labels = [labels_ohe[doc.split()[2]] for doc in doc_test_list]
    
    train_docs, val_docs, train_labels, val_labels = train_test_split(
        train_docs,
        train_labels,
        test_size=0.1,
        random_state=42,
        stratify=train_labels
    )

    # Cut off datasets
    cut_off_train = int(cut_off_dataset.split('-')[0])
    cut_off_val = int(cut_off_dataset.split('-')[1])
    cut_off_test = int(cut_off_dataset.split('-')[2])
    train_docs = train_docs[:int(len(train_docs) * (cut_off_train / 100))][:]
    val_docs = val_docs[:int(len(val_docs) * (cut_off_val / 100))][:]
    test_docs = test_docs[:int(len(test_docs) * (cut_off_test / 100))][:]
    train_labels = train_labels[:int(len(train_labels) * (cut_off_train / 100))][:]
    val_labels = val_labels[:int(len(val_labels) * (cut_off_val / 100))][:]
    test_labels = test_labels[:int(len(test_labels) * (cut_off_test / 100))][:]

    train_shuffle = list(zip(train_docs, train_labels))
    random.shuffle(train_shuffle)
    train_docs, train_labels = zip(*train_shuffle)
    train_docs, train_labels = list(train_docs), list(train_labels) 

    val_shuffle = list(zip(val_docs, val_labels))
    random.shuffle(val_shuffle)
    val_docs, val_labels = zip(*val_shuffle)
    val_docs, val_labels = list(val_docs), list(val_labels) 

    test_shuffle = list(zip(test_docs, test_labels))
    random.shuffle(test_shuffle)
    test_docs, test_labels = zip(*test_shuffle)
    test_docs, test_labels = list(test_docs), list(test_labels) 

    print("train: ", len(train_docs), len(train_labels))    
    print("val  : ", len(val_docs), len(val_labels))    
    print("test : ", len(test_docs), len(test_labels)) 

    print(Counter(train_labels)) 

    # Combine data
    documents = train_docs + val_docs + test_docs
    labels = train_labels + val_labels + test_labels

    # tokenized docs: 
    vocab = []
    tokenized_docs = []
    for doc in documents:
        tokenized_docs.append(doc.split())
        vocab.extend(doc.split())
    vocab = list(set(vocab))
    print("vocab: ", len(vocab))
 
    #vectorizer = CountVectorizer(min_df=min_df, max_df=max_df)
    #vectorizer.fit_transform(documents)
    #vocab = vectorizer.get_feature_names_out()  # Filtered vocab

    word2id = {word: idx for idx, word in enumerate(vocab)}
    id2word = {idx: word for word, idx in word2id.items()}
    num_docs = len(documents)

    tfidf_vectorizer = TfidfVectorizer(vocabulary=word2id)
    tfidf = tfidf_vectorizer.fit_transform(documents)
    doc_word_edges = []
    doc_word_weights = []

    for doc_id in tqdm(range(num_docs), desc="Extracting doc_word_edges"):
        for word_id in tfidf[doc_id].nonzero()[1]:
            word_node = num_docs + word_id
            weight = tfidf[doc_id, word_id]
            # Add both directions
            doc_word_edges.append([doc_id, word_node])
            doc_word_edges.append([word_node, doc_id])
            doc_word_weights.extend([weight, weight])
            #doc_word_weights.extend([weight])

    # PMI Word-Word Edges
    word_window_count = defaultdict(int)
    word_pair_count = defaultdict(int)
    window_count = 0

    for tokens in tqdm(tokenized_docs, desc="Extracting word_windows"):
        for i in range(len(tokens)):
            window = tokens[i:i + window_size]
            window_ids = [word2id[token] for token in window]
            unique_ids = set(window_ids)
            for id1 in unique_ids:
                word_window_count[id1] += 1
            for id1 in unique_ids:
                for id2 in unique_ids:
                    if id1 != id2:
                        word_pair_count[(id1, id2)] += 1
            window_count += 1

    word_word_edges = []
    word_word_weights = []

    for (i, j), count in tqdm(word_pair_count.items(), desc="Extracting word_word_edges"):
        p_ij = count / window_count
        p_i = word_window_count[i] / window_count
        p_j = word_window_count[j] / window_count
        pmi = math.log(p_ij / (p_i * p_j) + 1e-8)
        if pmi > 0:
            node_i = num_docs + i
            node_j = num_docs + j
            # Add both directions
            word_word_edges.append([node_i, node_j])
            word_word_edges.append([node_j, node_i])
            word_word_weights.extend([pmi, pmi])
            #word_word_weights.extend([pmi])

    # Load model (BERT, etc)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    bert = BertModel.from_pretrained(model_name) 
    num_words = len(word2id)
    hidden_dim = bert.config.hidden_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    doc_features = torch.tensor(get_cls_embeddings_batch(documents, tokenizer, bert, device, batch_size=32), dtype=torch.float)
    #doc_features = torch.stack([get_cls_embedding(doc, tokenizer, bert, device) for doc in tqdm(documents, desc="Extracting doc embeddings")])
    #doc_features = torch.zeros((num_docs, hidden_dim))  # Zero vector
    word_features = torch.zeros((num_words, hidden_dim))  # Zero vector

    # Node features and graph structure
    node_features = torch.cat([doc_features, word_features], dim=0)
    edge_index = torch.tensor(doc_word_edges + word_word_edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(doc_word_weights + word_word_weights, dtype=torch.float)

    # Labels for document nodes (e.g., 0 or 1), shape: [num_docs]
    node_labels = torch.tensor(labels + [-1] * num_words, dtype=torch.long)

    print(num_docs, num_words, hidden_dim)

    # Original masks (only for document nodes)
    num_nodes = len(node_features)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[:len(train_docs)] = True
    val_mask[len(train_docs):len(train_docs) + len(val_docs)] = True
    test_mask[len(train_docs) + len(val_docs):len(train_docs) + len(val_docs) + len(test_docs)] = True

    # PyG Graph
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_weight, y=node_labels,
            train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    print(data)
    
    utils.save_data(data, filename, path=f'{output_dir}', format_file='.pkl', compress=False)

def prep_data():

    # Parameters
    min_df = 1
    max_df = 1.0
    window_size = 10
    cut_off_dataset = '10-10-10'

    model_name = 'bert-base-uncased'
    output_dir = f'{utils.OUTPUT_DIR_PATH}baselines/'
    filename = 'BertGCN-20ng-' + cut_off_dataset
    
    # Load subset of 20 Newsgroups
    categories = None
    #categories = ['comp.graphics', 'sci.space', 'rec.sport.baseball']
    # Load dataset
    # train, val
    newsgroups_train = fetch_20newsgroups(subset='train', categories=categories) # remove=('headers', 'footers', 'quotes')
    documents_train = newsgroups_train.data
    labels_train = newsgroups_train.target
    #print(documents_train[0], newsgroups_train.target[0])
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        documents_train,
        labels_train,
        test_size=0.1,
        random_state=42,
        stratify=labels_train
    )
    # test
    newsgroups_test = fetch_20newsgroups(subset='test', categories=categories) # remove=('headers', 'footers', 'quotes')
    test_texts = newsgroups_test.data
    test_labels = newsgroups_test.target
    print(len(train_texts), len(val_texts), len(test_texts))
    print(train_texts[0], train_labels[0])

    # Cut off datasets
    cut_off_train = int(cut_off_dataset.split('-')[0])
    cut_off_val = int(cut_off_dataset.split('-')[1])
    cut_off_test = int(cut_off_dataset.split('-')[2])
    train_texts = train_texts[:int(len(train_texts) * (cut_off_train / 100))][:]
    val_texts = val_texts[:int(len(val_texts) * (cut_off_val / 100))][:]
    test_texts = test_texts[:int(len(test_texts) * (cut_off_test / 100))][:]
    train_labels = train_labels[:int(len(train_labels) * (cut_off_train / 100))][:]
    val_labels = val_labels[:int(len(val_labels) * (cut_off_val / 100))][:]
    test_labels = test_labels[:int(len(test_labels) * (cut_off_test / 100))][:]

    # Combine data
    documents = train_texts + val_texts + test_texts
    labels = train_labels.tolist() + val_labels.tolist() + test_labels.tolist()

    print(len(train_texts), len(val_texts), len(test_texts))
    print(Counter(labels))

    # Load model (BERT, etc)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    bert = BertModel.from_pretrained(model_name)

    # Nornalize and Tokenize documents to build word vocab
    # Tokenize and normalize texts
    tokenized_docs = []
    all_texts_norm = []
    for text in tqdm(documents, desc="Normalizing corpus"):
        text_norm = test_utils.text_normalize(text, special_chars=False, stop_words=False)
        all_texts_norm.append(text_norm)
        # custom
        tokenized_docs.append(re.findall("[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+", text_norm))
        # LLM tokenizer
        #tokenized_docs = [tokenizer.tokenize(doc) for doc in documents]

    vectorizer = CountVectorizer(min_df=min_df, max_df=max_df)
    vectorizer.fit_transform(all_texts_norm)
    vocab = vectorizer.get_feature_names_out()  # Filtered vocab

    word2id = {word: idx for idx, word in enumerate(vocab)}
    id2word = {idx: word for word, idx in word2id.items()}
    num_docs = len(documents)

    filtered_tokenized_docs = []
    for tokens in tokenized_docs:
        filtered_tokens = [t for t in tokens if t in word2id]
        filtered_tokenized_docs.append(filtered_tokens)

    # TF-IDF Doc-Word Edges
    #corpus = [" ".join(tokens) for tokens in tokenized_docs]
    #tfidf_vectorizer = TfidfVectorizer(vocabulary=word2id)
    #tfidf = tfidf_vectorizer.fit_transform(X)
    corpus = [" ".join(tokens) for tokens in filtered_tokenized_docs]
    
    #print(all_texts_norm[0])
    #print(tokenized_docs[0])
    #print(corpus[0])
    #print(len(vocab))
    #return

    tfidf_vectorizer = TfidfVectorizer(vocabulary=word2id)
    tfidf = tfidf_vectorizer.fit_transform(corpus)

    doc_word_edges = []
    doc_word_weights = []
    for doc_id in  tqdm(range(num_docs), desc="Extracting doc_word_edges"):
        for word_id in tfidf[doc_id].nonzero()[1]:
            doc_word_edges.append([doc_id, num_docs + word_id])
            doc_word_weights.append(tfidf[doc_id, word_id])

    # PMI Word-Word Edges
    word_window_count = defaultdict(int)
    word_pair_count = defaultdict(int)
    window_count = 0
    for tokens in tqdm(filtered_tokenized_docs, desc="Extracting word_word_edges"):
        for i in range(len(tokens)):
            window = tokens[i:i + window_size]
            window_ids = [word2id[token] for token in window]
            unique_ids = set(window_ids)
            for id1 in unique_ids:
                word_window_count[id1] += 1
            for id1 in unique_ids:
                for id2 in unique_ids:
                    if id1 != id2:
                        word_pair_count[(id1, id2)] += 1
            window_count += 1

    word_word_edges = []
    word_word_weights = []
    for (i, j), count in tqdm(word_pair_count.items(), desc="Extracting PMI"):
        p_ij = count / window_count
        p_i = word_window_count[i] / window_count
        p_j = word_window_count[j] / window_count
        pmi = math.log(p_ij / (p_i * p_j) + 1e-8)
        if pmi > 0:
            word_word_edges.append([num_docs + i, num_docs + j])
            word_word_weights.append(pmi)

    num_words = len(word2id)
    hidden_dim = bert.config.hidden_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    doc_features = torch.tensor(get_cls_embeddings_batch(documents, tokenizer, bert, device, batch_size=32), dtype=torch.float)
    #doc_features = torch.stack([get_cls_embedding(doc, tokenizer, bert, device) for doc in tqdm(documents, desc="Extracting doc embeddings")])
    #doc_features = torch.zeros((num_docs, hidden_dim))  # Zero vector
    word_features = torch.zeros((num_words, hidden_dim))  # Zero vector

    # Node features and graph structure
    node_features = torch.cat([doc_features, word_features], dim=0)
    edge_index = torch.tensor(doc_word_edges + word_word_edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(doc_word_weights + word_word_weights, dtype=torch.float)

    # Labels for document nodes (e.g., 0 or 1), shape: [num_docs]
    node_labels = torch.tensor(labels + [-1] * num_words, dtype=torch.long)

    print(num_docs, num_words, hidden_dim)

    # Original masks (only for document nodes)
    num_nodes = len(node_features)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[:len(train_texts)] = True
    val_mask[len(train_texts):len(train_texts) + len(val_texts)] = True
    test_mask[len(train_texts) + len(val_texts):len(train_texts) + len(val_texts) + len(test_texts)] = True

    # PyG Graph
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_weight, y=node_labels,
            train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    print(data)
    
    utils.save_data(data, filename, path=f'{output_dir}', format_file='.pkl', compress=False)


# *************************************** 
#  TRAINING
# *************************************** 

def train_gcn():
    #cut_off_dataset = '10-10-10'
    cut_off_dataset = 'testing'
    filename = 'BertGCN-20ng-' + cut_off_dataset
    output_dir = f'{utils.OUTPUT_DIR_PATH}baselines/'
    data = utils.load_data(filename, path=output_dir, format_file='.pkl', compress=False)
    del data.edge_attr
    print(data)    
    print("num_classes: ", set(data.y.tolist()))    
    print(Counter(data.y.tolist()))

    # Load model and optimizer
    model = GCN(in_channels=data.num_features, hidden_channels=100, dense_hidden_dim=64, num_classes=20)
    #model = GAT(in_channels=data.num_features, hidden_channels=120, num_classes=len(set(data.y.tolist())))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    early_stopper = EarlyStopper(patience=10, min_delta=0)

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)
    model = model.to(device)
    print(model)

    # Run training
    for epoch in range(1, 201):
        train_loss = train(model, data, optimizer, criterion)
        val_acc, val_loss = test(model, criterion, data, data.val_mask)
        train_acc, _ = test(model, criterion, data, data.train_mask)
        test_acc, _ = test(model, criterion, data, data.test_mask)
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Train_Loss: {train_loss:.4f} | Val_Loss: {val_loss:.4f} |  Train_Acc: {train_acc:.4f} |  Val_Acc: {val_acc:.4f} |  Test_Acc: {test_acc:.4f}")
        if early_stopper.early_stop(val_loss):
            print('Early stopping due to no improvement!')
            break
        
    test_acc, test_loss = test(model, criterion, data, data.test_mask)
    print(f"Test Acc: {test_acc:.4f}")
    
def train_gcn_raw():
    # Parameters
    build_graph = False
    window_size = 10
    cut_off_dataset = '100-100-100'

    if build_graph:
        model_name = 'bert-base-uncased'
        output_dir = f'{utils.OUTPUT_DIR_PATH}baselines/'
        filename = 'BertGCN-20ng-raw'

        # Load model (BERT, etc)
        model_path = f'{utils.OUTPUT_DIR_PATH}baselines/bert-20ng-best'
        #llm_model = AutoModelForSequenceClassification.from_pretrained(model_path)
        #tokenizer = AutoTokenizer.from_pretrained(model_path)

        tokenizer = BertTokenizer.from_pretrained(model_path)
        llm_model = BertModel.from_pretrained(model_path) 
        
        train_docs, test_docs, val_docs, train_labels, test_labels, val_labels = read_dataset()

        # Combine data
        documents = train_docs + val_docs + test_docs
        labels = train_labels + val_labels + test_labels

        # tokenized docs: 
        vocab = []
        tokenized_docs = []
        for doc in documents:
            tokenized_docs.append(doc.split())
            vocab.extend(doc.split())
        vocab = list(set(vocab))
        print("vocab: ", len(vocab))

        word2id = {word: idx for idx, word in enumerate(vocab)}
        id2word = {idx: word for word, idx in word2id.items()}
        num_docs = len(documents)

        tfidf_vectorizer = TfidfVectorizer(vocabulary=word2id)
        tfidf = tfidf_vectorizer.fit_transform(documents)
        doc_word_edges = []
        doc_word_weights = []

        for doc_id in tqdm(range(num_docs), desc="Extracting doc_word_edges"):
            for word_id in tfidf[doc_id].nonzero()[1]:
                word_node = num_docs + word_id
                weight = tfidf[doc_id, word_id]
                # Add both directions
                doc_word_edges.append([doc_id, word_node])
                doc_word_edges.append([word_node, doc_id])
                doc_word_weights.extend([weight, weight])
                #doc_word_weights.extend([weight])

        # PMI Word-Word Edges
        word_window_count = defaultdict(int)
        word_pair_count = defaultdict(int)
        window_count = 0

        for tokens in tqdm(tokenized_docs, desc="Extracting word_windows"):
            for i in range(len(tokens)):
                window = tokens[i:i + window_size]
                window_ids = [word2id[token] for token in window]
                unique_ids = set(window_ids)
                for id1 in unique_ids:
                    word_window_count[id1] += 1
                for id1 in unique_ids:
                    for id2 in unique_ids:
                        if id1 != id2:
                            word_pair_count[(id1, id2)] += 1
                window_count += 1

        word_word_edges = []
        word_word_weights = []

        for (i, j), count in tqdm(word_pair_count.items(), desc="Extracting word_word_edges"):
            p_ij = count / window_count
            p_i = word_window_count[i] / window_count
            p_j = word_window_count[j] / window_count
            pmi = math.log(p_ij / (p_i * p_j) + 1e-8)
            if pmi > 0:
                node_i = num_docs + i
                node_j = num_docs + j
                # Add both directions
                word_word_edges.append([node_i, node_j])
                word_word_edges.append([node_j, node_i])
                word_word_weights.extend([pmi, pmi])
                #word_word_weights.extend([pmi])

        
        num_words = len(word2id)
        hidden_dim = llm_model.config.hidden_size
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        doc_features = torch.tensor(get_cls_embeddings_batch(documents, tokenizer, llm_model, device, batch_size=32), dtype=torch.float)
        #word_features = torch.zeros((num_words, hidden_dim))  # Zero vector
        word_features = torch.empty((num_words, hidden_dim)).uniform_(-0.01, 0.01)

        # Combine weights before creating edge_weight tensor
        all_weights = np.array(doc_word_weights + word_word_weights).reshape(-1, 1)
        scaler = MinMaxScaler()
        normalized_weights = scaler.fit_transform(all_weights).flatten()

        # Node features and graph structure
        node_features = torch.cat([doc_features, word_features], dim=0)
        edge_index = torch.tensor(doc_word_edges + word_word_edges, dtype=torch.long).t().contiguous()
        #edge_weight = torch.tensor(doc_word_weights + word_word_weights, dtype=torch.float)
        edge_weight = torch.tensor(normalized_weights, dtype=torch.float)

        # Labels for document nodes (e.g., 0 or 1), shape: [num_docs]
        node_labels = torch.tensor(labels + [-1] * num_words, dtype=torch.long)

        print(num_docs, num_words, hidden_dim)

        # Original masks (only for document nodes)
        num_nodes = len(node_features)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[:len(train_docs)] = True
        val_mask[len(train_docs):len(train_docs) + len(val_docs)] = True
        test_mask[len(train_docs) + len(val_docs):len(train_docs) + len(val_docs) + len(test_docs)] = True

        adj = build_adjacency(edge_index, edge_weight, num_nodes)
        adj = normalize_adj(adj)
        data = {"adj": adj, "node_features": node_features, 'node_labels': node_labels, "train_mask": train_mask, 'val_mask': val_mask, 'test_mask': test_mask}
        utils.save_data(data, filename, path=f'{output_dir}', format_file='.pkl', compress=False)


    filename = 'BertGCN-20ng-raw'
    output_dir = f'{utils.OUTPUT_DIR_PATH}baselines/'
    data = utils.load_data(filename, path=output_dir, format_file='.pkl', compress=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adj = data['adj']
    node_features = data['node_features']
    node_labels = data['node_labels']
    train_mask = data['train_mask']
    val_mask = data['val_mask']
    test_mask = data['test_mask']

    model = GCN2(in_dim=node_features.shape[1], hidden_dim=200, out_dim=20)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
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

        '''
        model.eval()
        pred = logits.argmax(dim=1)
        val_acc = (pred[val_mask] == node_labels[val_mask]).float().mean()
        loss_val = criterion(logits[val_mask], node_labels[val_mask])
        test_acc = (pred[test_mask] == node_labels[test_mask]).float().mean()
        '''

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


def main():
    #prep_data()
    #prep_data_2()
    #train_gcn()
    
    train_gcn_raw()
    
    #fine_tune_model()
    #evaluate_fine_tune_model()


if __name__ == '__main__':
    main()
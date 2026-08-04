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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import utils
import test_utils
from vanilla_gtn import VanillaGTN  # wherever you saved it

try:
    from xgboost import XGBClassifier
    xgb_installed = True
except ImportError:
    xgb_installed = False

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
       

def normalize_text(texts, special_chars=False, stop_words=False, set='train'):    
    all_texts_norm = []
    for text in tqdm(texts, desc=f"Normalizing {set} corpus"):
        text_norm = test_utils.text_normalize(text, special_chars, stop_words) 
        all_texts_norm.append(text_norm)
    return all_texts_norm


def extract_deberta_features(texts, lang_model, tokenizer, device, batch_size=32, set='train'):
    """Extrae características CLS de DeBERTa para una lista de textos"""
    lang_model.eval()
    all_features = []
    
    with torch.no_grad():
        #for i in range(0, len(texts), batch_size):
        for i in tqdm(range(0, len(texts), batch_size), desc=f"extract-feat {set}: "):

            batch_texts = texts[i:i+batch_size]
            
            # Tokenizar el batch
            inputs = tokenizer(
                batch_texts.tolist() if hasattr(texts, 'tolist') else batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(device)
            
            # Obtener embeddings
            outputs = lang_model(**inputs)
            
            # Extraer el token [CLS] (primera posición)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            
            all_features.append(cls_embeddings.cpu())
    
    return torch.cat(all_features, dim=0)


class SimpleNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], num_classes=2, dropout=0.3):
        super(SimpleNNClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Capas ocultas
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train_nn_classifier(model, train_loader, val_loader, test_loader, device, num_epochs=50, patience=7, lr=0.001):
    """Entrena el clasificador NN"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5, verbose=True)
    
    best_val_f1 = 0
    patience_counter = 0
    best_model_state = None
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_f1_scores = []
    test_losses = []
    test_accuracies = []
    test_f1_scores = []
    
    for epoch in range(num_epochs):
        # ***** Training
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # ***** Validation
        model.eval()
        val_preds = []
        val_true = []
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                _, predicted = torch.max(outputs.data, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_true.extend(batch_y.cpu().numpy())
                val_loss += loss.item()
        
        # ***** Test
        model.eval()
        test_preds = []
        test_true = []
        test_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                _, predicted = torch.max(outputs.data, 1)
                test_preds.extend(predicted.cpu().numpy())
                test_true.extend(batch_y.cpu().numpy())
                test_loss += loss.item()
        
        val_acc = accuracy_score(val_true, val_preds)
        val_f1 = f1_score(val_true, val_preds, average='macro')
        test_acc = accuracy_score(test_true, test_preds)
        test_f1 = f1_score(test_true, test_preds, average='macro')
        
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_acc)
        val_f1_scores.append(val_f1)
        test_losses.append(test_loss / len(test_loader))
        test_accuracies.append(test_acc)
        test_f1_scores.append(test_f1)
        
        print(f'Epoch [{epoch+1}/{num_epochs}] - Val-Loss: {val_loss/len(val_loader):.4f} - Test-Loss: {test_loss/len(test_loader):.4f} - Val Acc: {val_acc:.4f} - Val F1: {val_f1:.4f} - Test Acc: {test_acc:.4f} - Val F1: {test_f1:.4f}')
        
        # Early Stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            
        scheduler.step(val_f1)
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Cargar el mejor modelo
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, {
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'val_f1_scores': val_f1_scores,
        'best_val_f1': best_val_f1
    }

def evaluate_nn_model(model, test_loader, device):
    """Evalúa el modelo NN en el test set"""
    model.eval()
    test_preds = []
    test_true = []
    test_probs = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            test_preds.extend(predicted.cpu().numpy())
            test_true.extend(batch_y.cpu().numpy())
            test_probs.extend(probabilities.cpu().numpy())
    
    return test_true, test_preds, test_probs

def train_logistic_regression(deberta_features, labels, test_size=0.2, random_state=42):
    
    # Dividir en train/val
    X_train, X_val, y_train, y_val = train_test_split(
        deberta_features, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    # Entrenar Logistic Regression
    lr_model = LogisticRegression(
        random_state=random_state,
        max_iter=1000,
        class_weight='balanced',  # Para datasets desbalanceados
        C=1.0  # Parámetro de regularización
    )
    
    lr_model.fit(X_train, y_train)
    
    # Predecir y evaluar
    y_pred = lr_model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='macro')
    
    print(f"Logistic Regression - Accuracy: {accuracy:.4f}, F1-Macro: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))
    
    return lr_model, accuracy, f1


def main(extract_feat, algo, dataset_name, cut_off_dataset, cuda_num=0, 
         num_epochs=100, 
         lr=0.0001,
         patience=10,
         lang_model_name = 'microsoft/deberta-v3-base',
         leave_out_sources = None,
         balance_dataset = True,
    ):
    
    output_dir = f'{test_utils.EXTERNAL_DISK_PATH}pretrained_clf'
    file_name_data = f"{lang_model_name.split('/')[1]}_{dataset_name}_{config['cut_off_dataset']}perc"
    
    device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    tokenizer = AutoTokenizer.from_pretrained(lang_model_name)
    lang_model = AutoModel.from_pretrained(lang_model_name).to(device)
    
    if extract_feat:
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

        limit = None
        
        train_texts = list(train_set['text'])[:limit]
        val_texts = list(val_set['text'])[:limit]
        test_texts = list(test_set['text'])[:limit]

        train_labels = list(train_set['label'])[:limit]
        val_labels = list(val_set['label'])[:limit]
        test_labels = list(test_set['label'])[:limit]

        train_texts_norm = normalize_text(train_texts, set='train')
        val_texts_norm = normalize_text(val_texts, set='val')
        test_texts_norm = normalize_text(test_texts, set='test')


        print("\n" + "="*50)
        print("EXTRACTING DEBERTA FEATURES FOR LOGISTIC REGRESSION")
        print("="*50)
        
        # Extraer características para todos los conjuntos
        train_features = extract_deberta_features(train_texts_norm, lang_model, tokenizer, device, batch_size, set="train")
        val_features = extract_deberta_features(val_texts_norm, lang_model, tokenizer, device, batch_size, set="val")
        test_features = extract_deberta_features(test_texts_norm, lang_model, tokenizer, device, batch_size, set="test")

        # Preparar datos para PyTorch
        X_train = train_features.float()
        X_val = val_features.float()
        X_test = test_features.float()
        y_train = torch.tensor(train_set['label'].values, dtype=torch.long)
        y_val = torch.tensor(val_set['label'].values, dtype=torch.long)
        y_test = torch.tensor(test_set['label'].values, dtype=torch.long)

        data = {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
        }

        utils.save_data(data, file_name_data, path=f'{output_dir}/', format_file='.pkl', compress=False)

    else:
        data = utils.load_data(file_name_data, path=f'{output_dir}/', format_file='.pkl', compress=False)
        X_train = data["X_train"]
        X_val = data["X_val"]
        X_test = data["X_test"]
        y_train = data["y_train"]
        y_val = data["y_val"]
        y_test = data["y_test"]

    if algo == 'neu_net':
        # Crear DataLoaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Crear y entrenar el modelo NN
        input_dim = X_train.shape[1]  # Dimensión de las características DeBERTa (768 o 1024)
        num_classes = len(torch.unique(y_train))
        
        print(f"\nCreating Neural Network Classifier:")
        print(f"Input dimension: {input_dim}")
        print(f"Number of classes: {num_classes}")
        
        nn_model = SimpleNNClassifier(
            input_dim=input_dim,
            hidden_dims=[128, 128],  # Puedes ajustar estas dimensiones
            num_classes=num_classes,
            dropout=0.5
        ).to(device)
        
        print(f"Model architecture:\n{nn_model}")
        
        # Entrenar el modelo
        print("\nTraining Neural Network Classifier...")
        trained_nn_model, training_history = train_nn_classifier(
            nn_model, train_loader, val_loader, test_loader,
            device, 
            num_epochs=num_epochs, 
            lr=lr,
            patience=patience
        )
        
        # Evaluar en test set
        print("\nEvaluating on Test Set...")
        test_true, test_preds, test_probs = evaluate_nn_model(trained_nn_model, test_loader, device)
        
        test_accuracy = accuracy_score(test_true, test_preds)
        test_f1 = f1_score(test_true, test_preds, average='macro')
        
        print(f"\nNeural Network Test Results:")
        print(f"Accuracy: {test_accuracy:.4f}")
        print(f"F1-Macro: {test_f1:.4f}")
        print("\nTest Classification Report:")
        print(classification_report(test_true, test_preds))

    if algo == 'log_reg':
    
        # Convertir a numpy para scikit-learn
        X_train = X_train.numpy()
        X_val = X_val.numpy()
        X_test = X_test.numpy()
        
        # Entrenar Logistic Regression
        print("\nTraining Logistic Regression...")
        
        lr_model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced',
            C=1.0,
            solver='liblinear'  # Buen solver para datasets pequeños/medianos
        )
        
        lr_model.fit(X_train, y_train)
        
        # Evaluar en validation set
        y_val_pred = lr_model.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred, average='macro')
        
        print(f"\nLogistic Regression Validation Results:")
        print(f"Accuracy: {val_accuracy:.4f}")
        print(f"F1-Macro: {val_f1:.4f}")
        print("\nValidation Classification Report:")
        print(classification_report(y_val, y_val_pred))
        
        # Evaluar en test set
        y_test_pred = lr_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred, average='macro')
        
        print(f"\nLogistic Regression Test Results:")
        print(f"Accuracy: {test_accuracy:.4f}")
        print(f"F1-Macro: {test_f1:.4f}")
        print("\nTest Classification Report:")
        print(classification_report(y_test, y_test_pred))
        


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

            'extract_feat': True,
            'algo': 'log_reg', # neu_net, log_reg
            'dataset_name': 'autext24', # autext23, autext24, semeval24, coling24
            'cut_off_dataset': '100_100_100',
            # autext23:  10-10-10 | 50-50-50 | 100-100-100*
            # semeval24: 1-1-1 | 5-5-5 | 10-10-10* | 25-25-25 | 50-50-50*
            # coling24: 1-1-1 | 2-2-2 | 5-10-5 | 10-10-10*

            'num_epochs': 100, 
            'lr': 0.0001,
            'patience': 10,
            'cuda_num': 1,
            'balance_dataset': True,
 
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


    if config['leave_out_sources']:
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


    main(**config)

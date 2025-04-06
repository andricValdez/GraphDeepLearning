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
import networkx as nx 
import scipy as sp
import scipy.sparse as sp

import gensim
from scipy.sparse import coo_array 
from sklearn.datasets import fetch_20newsgroups
import gc
import glob
import torch.nn.functional as F
from torch_geometric.data import DataLoader, Data, ClusterData, GraphSAINTRandomWalkSampler

from torch.nn import Linear, BatchNorm1d, ModuleList, LayerNorm
from torch_geometric.nn import GCNConv, GATConv, TransformerConv, TopKPooling, GraphConv, SAGPooling
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch.nn.modules.module import Module
from torch_geometric.nn import DataParallel
from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from torch_geometric.loader import NeighborLoader, ClusterLoader
from sklearn.metrics import f1_score
from collections import OrderedDict
import warnings
from transformers import logging as transform_loggin
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
import mlflow
import matplotlib.pyplot as plt 
import seaborn as sns

import utils
import gnn_convs
import GraphDeepLearning.node_feat_init_test as node_feat_init_test
from stylometric import StyloCorpus
import text2graph
from dataset import BuildDataset

#************************************* CONFIGS
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s; - %(levelname)s; - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
warnings.filterwarnings("ignore")
transform_loggin.set_verbosity_error()

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class NeuralNetwork(torch.nn.Module):
    def __init__(self, in_channels, nhid, out_ch, layers_num):
        super(NeuralNetwork,self).__init__()
        self.flatten = torch.nn.Flatten()
        layers = [('linear1', torch.nn.Linear(in_channels, nhid)), ('relu1', torch.nn.ReLU())]

        for index in range(layers_num):
            layers.append((f'linear{index+2}', torch.nn.Linear(nhid, nhid)))
            layers.append((f'relu{index+2}', torch.nn.ReLU()))

        layers.append(('final_relu', torch.nn.Linear(nhid, out_ch)))
        layers.append(('sigmoid', torch.nn.Sigmoid()))
        #layers.append(('softmax', torch.nn.Softmax()))
        self.linear_relu_stack = torch.nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train_ml_clf_model(algo_clf, train_data, train_labels, val_data, val_labels):
    model = CalibratedClassifierCV(algo_clf()) #n_jobs=-3
    model.fit(train_data, train_labels)
    y_pred  = model.predict(val_data)
    y_true = val_labels#.view(1,-1)[0].numpy()
    print('\t Accuracy:', np.mean(y_true == y_pred))
    print('\t F1Score:', f1_score(y_true, y_pred , average='macro'))
    return model


def train_dense_rrnn_clf_model(dense_model, device, train_loader, val_data, val_labels):
    learning_rate = 0.00001
    early_stopper = EarlyStopper(patience=10, min_delta=0)
    optimizer = torch.optim.Adam(dense_model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss() # BCEWithLogitsLoss, CrossEntropyLoss, BCELoss
    sigmoid = torch.nn.Sigmoid()
    dense_model = dense_model.to(device) 
    print_preds_test = False
    train_loss = 0.0
    epochs = 100
    avg_acc = 0.0
    best_acc_epoch = 0
    best_acc = 0.0
    best_f1_scre_epoch = 0
    best_f1_scre = 0.0
    steps = 0
    for epoch in range(1, epochs):
        dense_model.train()
        for features, labels, in train_loader:
            features.to(device)           
            labels.to(device)           
            #print(batch.x.shape, batch.y)
            logits = dense_model(features)
            #logits = logits.argmax(dim=1)
            #print(logits)
            loss = criterion(logits, labels.float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()
            steps += 1
            
        acc, f1_scre, val_loss, preds_test = test_train_concat_emb(dense_model, criterion, val_data, val_labels, epoch)
        avg_acc += acc
        if acc > best_acc:
            best_acc = acc
            best_acc_epoch = epoch
        
        if f1_scre > best_f1_scre:
            best_f1_scre = f1_scre
            best_f1_scre_epoch = epoch
        
        print(f"Epoch: {epoch} | train_loss: {loss.item():4f} | val_loss: {val_loss:4f} | acc: {acc:4f} | avg_acc: {avg_acc/(epoch+1):4f} | best_acc (epoch {best_acc_epoch}): {best_acc:4f})")
        if early_stopper.early_stop(val_loss): 
            print('Early stopping fue to not improvement!') 
            print(f"Epoch: {epoch} | train_loss: {loss.item():4f} | val_loss: {val_loss:4f} | f1_scre: {f1_scre:4f}  | acc: {acc:4f} | avg_acc: {avg_acc/(epoch+1):4f} | best_acc (epoch {best_acc_epoch}): {best_acc:4f})")
            break

        if epoch == 10 and print_preds_test:
            print(preds_test['outs'])
    
    print(f"best_f1_scre (epoch {best_f1_scre_epoch}): {best_f1_scre:4f}) |  best_acc (epoch {best_acc_epoch}): {best_acc:4f})")
    return dense_model


def test_train_concat_emb(dense_model, criterion, val_data, val_labels, epoch):
    sigmoid = torch.nn.Sigmoid()
    dense_model.eval()
    targ = val_labels.float()
    preds_test = {}
    with torch.no_grad():
        logits = dense_model.forward(val_data)      
        loss = criterion(logits, targ)    

        acc = (logits.round() == targ).float().mean()
        acc = float(acc)

        y_pred = logits.round().cpu().numpy()
        f1_scre = f1_score(val_labels.float().cpu().numpy(), y_pred, average='macro')

        if epoch == 10:
            #pred_probab = sigmoid(logits)
            #y_pred = pred_probab.argmax(1)      
            #print(logits)
            #print(logits.round())
            #print(targ)
            preds_test = {'preds': logits.round().cpu().numpy().tolist(),'outs': logits.cpu().numpy().tolist()}
        
        return acc, f1_scre, loss, preds_test

 
def log_conf_matrix(y_pred, y_true, epoch):
    # Log confusion matrix as image
    cm = confusion_matrix(y_pred, y_true)
    classes = ["0", "1"]
    df_cfm = pd.DataFrame(cm, index = classes, columns = classes)
    plt.figure(figsize = (10,7))
    cfm_plot = sns.heatmap(df_cfm, annot=True, cmap='Blues', fmt='g')
    cfm_plot.figure.savefig(f'{utils.OUTPUT_DIR_PATH}/images/cm_{epoch}.png')
    mlflow.log_artifact(f"{utils.OUTPUT_DIR_PATH}/images/cm_{epoch}.png")


def calculate_metrics(y_pred, y_true, epoch, type):
    #print("*************** type:", type)
    cm = confusion_matrix(y_pred, y_true)
    f1 = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    mlflow.log_metric(key=f"F1Score-{type}", value=float(f1), step=epoch)
    mlflow.log_metric(key=f"Accuracy-{type}", value=float(acc), step=epoch)
    mlflow.log_metric(key=f"Precision-{type}", value=float(prec), step=epoch)
    mlflow.log_metric(key=f"Recall-{type}", value=float(rec), step=epoch)
    
    #print(f'Epoch: {epoch:03d} | F1Score-{type}: {f1:.4f} | Accuracy-{type}: {acc:.4f} | Precision-{type}: {prec:.4f} | Recall-{type}: {rec:.4f}')
    try:
        roc = roc_auc_score(y_true, y_pred)
        mlflow.log_metric(key=f"ROC-AUC-{type}", value=float(roc), step=epoch)
    except:
        mlflow.log_metric(key=f"ROC-AUC-{type}", value=float(0), step=epoch)
    else:
        return f1, acc, prec, rec


def gnn_model(
                text2graph_type, all_loader, train_loader, val_loader, 
                metrics, device, epoch_num, gnn_type, num_features, 
                hidden_channels, learning_rate, 
                gnn_dropout, gnn_pooling, gnn_batch_norm,
                gnn_layers_convs, gnn_heads, gnn_dense_nhid,
                num_classes, edge_dim, retrain_model_name='', 
                retrain_model=False, patience=10
            ):
    
    '''
    model = gnn_convs.GNN(gnn_type=gnn_type, num_features=num_features,  hidden_channels=hidden_channels, 
        num_classes=num_classes, heads=gnn_heads, dropout=gnn_dropout, 
        pooling=gnn_pooling, batch_norm=gnn_batch_norm, layers_convs=gnn_layers_convs, 
        dense_nhid=gnn_dense_nhid, edge_dim=edge_dim
    )
    '''
    '''
    model = gnn_convs.GNN_2(gnn_type=gnn_type, hidden_channels=hidden_channels, pooling=gnn_pooling, 
                  n_layers=gnn_layers_convs, dropout=gnn_dropout, num_features=num_features, 
                  edge_dim=edge_dim, dense_nhid=gnn_dense_nhid, heads=gnn_heads, 
                  num_classes=num_classes, task=text2graph_type)
    '''
    '''
    model = gnn_convs.GNN_3(gnn_type=gnn_type, hidden_channels=hidden_channels, pooling=gnn_pooling, 
                  n_layers=gnn_layers_convs, dropout=gnn_dropout, num_features=num_features, edge_dim=edge_dim,
                  dense_nhid=gnn_dense_nhid, heads=gnn_heads, num_classes=num_classes, task=text2graph_type)
    '''
    model = gnn_convs.GNNStanford(num_features, hidden_channels, gnn_dense_nhid, num_classes, 
                                  gnn_dropout, gnn_layers_convs, gnn_type=gnn_type, 
                                  heads=gnn_heads, task=text2graph_type)

    mlflow.log_param('model_params', str(model))
    #model= torch.nn.DataParallel(model)
    model = model.to(device)

    #model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss() # BCEWithLogitsLoss, CrossEntropyLoss, BCELoss
    early_stopper = EarlyStopper(patience=patience, min_delta=0)

    if retrain_model == True:
        #model.load_state_dict(torch.load(retrain_model_name))
        checkpoint = torch.load(retrain_model_name, map_location = device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    logger.info("model: %s", str(model))
    logger.info("device: %s", str(device))
    
    torch.cuda.empty_cache()
    gc.collect()

    if text2graph_type == 'cooc':
        return train_cooc_graph(model, criterion, optimizer, train_loader, val_loader, epoch_num, device, metrics, early_stopper)
    if text2graph_type == 'hetero':
        all_loader.to(device)
        return train_hetero_graph(model, criterion, optimizer, all_loader, epoch_num, device, metrics, early_stopper)

        #model, optimizer, metrics, train_embeddings, val_embeddings = train_hetero_graph(model, criterion, optimizer, all_loader, epoch_num, device, metrics, early_stopper)
        train_embeddings_data = {'doc_id': None,'labels': all_loader.y[all_loader.train_mask], 'embedding': train_embeddings}



def train_hetero_graph(model, criterion, optimizer, all_loader, epoch_num, device, metrics, early_stopper):    
    val_acc_dict, test_acc_dict, val_f1_score_macro_dict, test_f1_score_macro_dict = {}, {}, {}, {}
    best_model, best_embeddings = [], []
    best_convs_layers = []
    best_val_loss = 100000
    best_val_acc = 0
    best_val_macro_f1score = 0
    best_epoch = 1

    for epoch in range(1, epoch_num+1):
        #loss_train, _, loss_val, val_acc, val_f1_score_macro, val_f1_score_micro, _, _, _, _ = train_node(model, optimizer, criterion, all_loader)
        
        model.train()
        optimizer.zero_grad()
        out, embeddings, convs_layers = model(all_loader.x, all_loader.edge_index, all_loader.edge_attr, None)
        #out, embeddings, convs_layers = model(all_loader.x, all_loader.edge_index, None, None)

        #print(out[all_loader.train_mask], all_loader.y[all_loader.train_mask])
        loss_train = criterion(out[all_loader.train_mask], all_loader.y[all_loader.train_mask])
        loss_train.backward()
        optimizer.step()
        _, convs_layers, preds_test, loss_val, val_acc, val_f1_score_macro, val_f1_score_micro, loss_test, test_acc, test_f1_score_macro, test_f1_score_micro = test_node(model, criterion, all_loader)

        if (epoch == 1) or (epoch % 1 == 0):
            print(f'Epoch: {epoch:03d}, Train_Loss: {loss_train:.4f}, Test_Acc: {test_acc:.4f}, Test_F1_Macro: {test_f1_score_macro:.4f}, Val_Acc: {val_acc:.4f}, Val_F1_Macro: {val_f1_score_macro:.4f}, Val_Loss {loss_val:.4f}')
            #print("preds_test: ", sorted(Counter(preds_test).items()))
            val_acc_dict[epoch] = val_acc
            test_acc_dict[epoch] = test_acc
            val_f1_score_macro_dict[epoch] = val_f1_score_macro
            test_f1_score_macro_dict[epoch] = test_f1_score_macro

        if val_f1_score_macro > best_val_macro_f1score:
            best_epoch = epoch
            best_model = model
            best_val_loss = loss_val
            best_val_acc = val_acc
            best_val_macro_f1score = val_f1_score_macro
            best_embeddings = embeddings

        if early_stopper.early_stop(loss_val):
            print(f'Epoch: {epoch:03d}, Train_Loss: {loss_train:.4f}, Test_Acc: {test_acc:.4f}, Test_F1_Macro: {test_f1_score_macro:.4f}, Val_Acc: {val_acc:.4f}, Val_F1_Macro: {val_f1_score_macro:.4f}, Val_Loss {loss_val:.4f}')
            print('Early stopping fue to not improvement!')
            break

    print(f'Best_Epoch: {best_epoch:03d}, Best_Loss: {best_val_loss:.4f}, Best_Val_Acc: {best_val_acc:.4f}, Val_F1_Score_Macro: {best_val_macro_f1score:.4f}')
    return best_model, optimizer, metrics, best_embeddings[all_loader.train_mask], best_embeddings[all_loader.val_mask]


def test_node(model, criterion, data):
    model.eval()
    with torch.no_grad():
        out, embeddings, convs_layers = model(data.x, data.edge_index, data.edge_attr, None)
        #out, embeddings, convs_layers = model(data.x, data.edge_index, None, None)

        pred = out.argmax(dim=1)
        #*** VAL
        loss_val = criterion(out[data.val_mask], data.y[data.val_mask])
        truth_val = data.y[data.val_mask].cpu().detach().numpy().tolist()
        pred_val = pred[data.val_mask].cpu().detach().numpy().tolist()
        val_acc = accuracy_score(truth_val, pred_val)
        val_f1_score_macro = f1_score(truth_val, pred_val, average='macro')
        val_f1_score_micro = f1_score(truth_val, pred_val, average='micro')

        #*** TEST
        loss_test = criterion(out[data.test_mask], data.y[data.test_mask])
        truth_test = data.y[data.test_mask].cpu().detach().numpy().tolist()
        preds_test = pred[data.test_mask].cpu().detach().numpy().tolist()
        test_acc = accuracy_score(truth_test, preds_test)
        test_f1_score_macro = f1_score(truth_test, preds_test, average='macro')
        test_f1_score_micro = f1_score(truth_test, preds_test, average='micro')

    return embeddings, convs_layers, preds_test, loss_val, val_acc, val_f1_score_macro, \
            val_f1_score_micro, loss_test, test_acc, test_f1_score_macro, test_f1_score_micro


def train_cooc_graph(model, criterion, optimizer, train_loader, val_loader, epoch_num, device, metrics, early_stopper):    
    best_train_embeddings, best_val_embeddings, best_model = None, None, None
    best_val_score, best_f1_score, best_epoch_score, avg_val_score, epochs_cnt =  0, 0, 0, 0, 0
    try:
        for epoch in range(1, epoch_num):
            epochs_cnt += 1
            model, train_acc, train_loss, train_embeddings, _ = train_graph(model, criterion, optimizer, train_loader, epoch, device=device)
            #_, train_acc, _, _ = test_graph(model, criterion, train_loader, epoch, type='train', device=device)
            val_loss, val_f1, val_acc, val_prec, val_rec, val_embeddings, _ = test_graph(model, criterion, val_loader , epoch, type='valid', device=device)
            print(f'Epoch: {epoch:03d} | Train Loss {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | Val Loss: {val_loss:.4f} ')
            avg_val_score += val_acc
            if val_acc > best_val_score:
                best_val_score = val_acc
                best_f1_score = val_f1
                best_epoch_score = epoch
                best_train_embeddings = train_embeddings
                best_val_embeddings = val_embeddings
                best_model = model
            
            metrics['_epoch_stop'] = epoch
            metrics['_train_loss'] = train_loss
            metrics['_val_loss'] = val_loss
            metrics['_train_acc'] = train_acc
            metrics['_val_last_acc'] = val_acc

            #for key in metrics.keys():
            #    mlflow.log_metric(key=key, value=metrics[key], step=epoch)

            if early_stopper.early_stop(val_loss): 
                print('Early stopping fue to not improvement!')            
                break
    except Exception as err:
        #print("Traning GNN Error: ", str(err))
        #print("traceback: ", str(traceback.format_exc()))
        raise Exception(str(err))
    else:
        #metrics['_best_metrics'] = {'best_val_score': best_val_score,'best_epoch_score': best_epoch_score, 'avg_val_score': avg_val_score/epochs_cnt}
        metrics['_val_best_acc'] = best_val_score
        metrics['_val_best_epoch_acc'] = best_epoch_score
        metrics['_val_avg_acc'] = avg_val_score/epochs_cnt
        print(f'-----> Best Val-Score: {best_val_score:.4f} and F1-score {best_f1_score:.4f} in epoch: {best_epoch_score} | Avg Val Score: {avg_val_score/epochs_cnt:.4f}')
        return best_model, optimizer, metrics, train_embeddings, val_embeddings


def train_graph(model, criterion, optimizer, loader, epoch, device='cpu'):
    model.train()
    train_loss = 0.0
    steps = 0
    embeddings_data = []
    all_preds, all_labels = [], []
    for step, data in enumerate(loader):  # Iterate in batches over the training dataset.
        data.to(device) 
        #print('training batch...', step)
        out, embeddings, convs_layers = model.forward(data.x, data.edge_index, data.edge_attr, data.batch)  # Perform a single forward pass.
        embeddings_data.append({'batch': step, 'doc_id': data.context['id'], 'labels': data.y, 'embedding': embeddings})
        loss = criterion(out, data.y)  # Compute the loss.
        optimizer.zero_grad()  # Clear gradients.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        train_loss += loss.item()
        steps += 1

        all_preds.append(out.argmax(dim=1).cpu().detach().numpy())
        all_labels.append(data.y.cpu().detach().numpy())
    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    calculate_metrics(all_preds, all_labels, epoch, "train")
    return model, accuracy_score(all_labels, all_preds), train_loss / steps, embeddings_data, loader


def test_graph(model, criterion, loader, epoch, type, device='cpu'):
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
            out, embeddings, convs_layers = model(data.x, data.edge_index, data.edge_attr, data.batch)  
            embeddings_data.append({'batch': step, 'doc_id': data.context['id'], 'labels': data.y, 'embedding': embeddings})
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
        f1, acc, prec, rec = calculate_metrics(all_preds, all_labels, epoch, type)
        #log_conf_matrix(all_preds, all_labels, epoch)
        return test_loss / steps, f1, acc, prec, rec, embeddings_data, pred_loader  # Derive ratio of correct predictions.


def graph_neural_network(
        exp_file_name,
        dataset_partition, 
        exp_file_path, 
        graph_trans, 
        nfi, 
        cut_off_dataset,
        text2graph_type, 
        t2g_instance, 
        train_text_docs, 
        val_text_docs,
        test_text_docs,
        all_text_docs,
        device,
        edge_features,
        llm_finetuned_name,
        edge_dim=2,
        num_classes=2,
        num_features=256,
        batch_size_gnn=32,
        build_dataset = True,
        save_data = True

    ):
    
    if not edge_features:
        edge_dim = None

    if graph_trans == True: 
        if text2graph_type == 'cooc':
            # Text 2 Graph train data
            graphs_train_data = utils.t2g_transform(train_text_docs, t2g_instance)
            utils.save_data(graphs_train_data, path=f'{utils.OUTPUT_DIR_PATH}graphs/', file_name=f'graphs_train_{dataset_partition}')

            # Text 2 Graph val data
            graphs_val_data = utils.t2g_transform(val_text_docs, t2g_instance)
            utils.save_data(graphs_val_data, path=f'{utils.OUTPUT_DIR_PATH}graphs/', file_name=f'graphs_val_{dataset_partition}')
            
            # Text 2 Graph test data
            graphs_test_data = utils.t2g_transform(test_text_docs, t2g_instance)
            utils.save_data(graphs_test_data, path=f'{utils.OUTPUT_DIR_PATH}graphs/', file_name=f'graphs_test_{dataset_partition}')
            
            # Feat Init - Word2vect Model
            #model_w2v = node_feat_init.w2v_train(graph_data=graphs_train_data, num_features=num_features)
            #utils.save_data(model_w2v, path=f'{utils.OUTPUT_DIR_PATH}w2v_models/', file_name=f'model_w2v_{dataset_partition}')
        
        if text2graph_type == 'hetero':
            graphs_all_data = utils.t2g_transform(all_text_docs, t2g_instance)
            utils.save_data(graphs_all_data, path=f'{utils.OUTPUT_DIR_PATH}graphs/', file_name=f'graphs_all_data_{dataset_partition}')

    elif graph_trans == False:
        if text2graph_type == 'cooc':
            graphs_train_data = utils.load_data(path=f'{utils.OUTPUT_DIR_PATH}graphs/', file_name=f'graphs_train_{dataset_partition}')  
            graphs_val_data = utils.load_data(path=f'{utils.OUTPUT_DIR_PATH}graphs/', file_name=f'graphs_val_{dataset_partition}')  
            graphs_test_data = utils.load_data(path=f'{utils.OUTPUT_DIR_PATH}graphs/', file_name=f'graphs_test_{dataset_partition}')  
            #model_w2v = utils.load_data(path=f'{utils.OUTPUT_DIR_PATH}w2v_models/', file_name=f'model_w2v_{dataset_partition}')
        if text2graph_type == 'hetero':
            graphs_all_data = utils.load_data(path=f'{utils.OUTPUT_DIR_PATH}graphs/', file_name=f'graphs_all_data_{dataset_partition}')  

    else:
        ...
    
    #print("graphs_train_data: ", len(graphs_train_data))
    #print("graphs_val_data: ", len(graphs_val_data))
    print('device: ', device)

    #for g in graphs_train_data:
        #print(g['text'])
        #print(g['graph'])
        #print(g['graph'].nodes)
        #break
    #return

    # SET VARS
    all_loader = None
    train_loader = None
    val_loader = None
    
    #******************* TRAIN and GET GNN Embeddings
    if build_dataset == True:
        if not utils.is_dir_empty(dir_path=f'{exp_file_path}/embeddings_word_llm'):
            utils.delete_dir_files(f'{exp_file_path}/embeddings_word_llm')

        if text2graph_type == 'cooc':
            train_build_dataset = BuildDataset(graphs_train_data[:], train_text_docs, subset='train', device=device, edge_features=edge_features, 
                                               nfi=nfi, llm_finetuned_name=llm_finetuned_name, exp_file_path=exp_file_path, num_labels=num_classes, 
                                               dataset_partition=dataset_partition, num_features=num_features, text2graph_type=text2graph_type,
                                               avg_llm_found_tokens=True, avg_llm_not_found_tokens=True)
            train_dataset = train_build_dataset.process_dataset()
            val_build_dataset = BuildDataset(graphs_val_data[:], val_text_docs, subset='val', device=device, edge_features=edge_features, 
                                             nfi=nfi, llm_finetuned_name=llm_finetuned_name, exp_file_path=exp_file_path, num_labels=num_classes, 
                                             dataset_partition=dataset_partition, num_features=num_features, text2graph_type=text2graph_type,
                                             avg_llm_found_tokens=True, avg_llm_not_found_tokens=True)
            val_dataset = val_build_dataset.process_dataset()
            test_build_dataset = BuildDataset(graphs_test_data[:], test_text_docs, subset='test', device=device, edge_features=edge_features, 
                                              nfi=nfi, llm_finetuned_name=llm_finetuned_name, exp_file_path=exp_file_path, num_labels=num_classes, 
                                              dataset_partition=dataset_partition, num_features=num_features, text2graph_type=text2graph_type,
                                              avg_llm_found_tokens=True, avg_llm_not_found_tokens=True)
            test_dataset = test_build_dataset.process_dataset()

        if text2graph_type == 'hetero':
            text_data_lst = [{'id': d['context']['id'], 'label': d['context']['target'], 'text': d['doc']} for d in all_text_docs]
            doc_embs_hetero = node_feat_init_test.llm_get_embbedings_2(text_data_lst, subset='train', emb_type='llm_cls_2', device=device, output_path='', save_emb=False, llm_finetuned_name=llm_finetuned_name, num_labels=num_classes)
            set_idxs = {'train': len(train_text_docs), 'val': len(val_text_docs), 'test': len(test_text_docs)}

            all_build_dataset = BuildDataset(
                graphs_all_data[:], all_text_docs, subset='train', device=device, 
                edge_features=edge_features, nfi=nfi, llm_finetuned_name=llm_finetuned_name, 
                exp_file_path=exp_file_path, num_labels=num_classes, 
                dataset_partition=dataset_partition, num_features=num_features,
                text2graph_type=text2graph_type, avg_llm_found_tokens=True,
                avg_llm_not_found_tokens=True, doc_embs_hetero=doc_embs_hetero, set_idxs=set_idxs
            )
            all_dataset = all_build_dataset.process_dataset()
            all_loader = all_dataset[0]
            #data.to(device)
            print(all_loader)
            print(all_loader.y.shape)
            print(all_loader.train_mask.shape)
            print(all_loader.val_mask.shape)
            all_loader = all_loader.to(device)


    if not utils.is_dir_empty(dir_path=f'{exp_file_path}/embeddings_gnn'):
        utils.delete_dir_files(f'{exp_file_path}/embeddings_gnn')

    if text2graph_type == 'cooc':
        if not utils.is_dir_empty(dir_path=f'{exp_file_path}/embeddings_word_llm'):
            train_dataset_tensors = glob.glob(f'{exp_file_path}/embeddings_word_llm/data_train_*.pt')
            val_dataset_tensors = glob.glob(f'{exp_file_path}/embeddings_word_llm/data_val_*.pt')
            train_dataset, val_dataset = [], []
            for tensor_file in train_dataset_tensors:
                train_dataset.extend(torch.load(tensor_file))
            for tensor_file in val_dataset_tensors:
                val_dataset.extend(torch.load(tensor_file))

            print("train_dataset: ", len(train_dataset), "| val_dataset: ", len(val_dataset))
            cut_off_train = 100 #50
            cut_off_val = 100 #50
            train_dataset = train_dataset[ : int(len(train_dataset) * (int(cut_off_train) / 100)) ]
            val_dataset = val_dataset[ : int(len(val_dataset) * (int(cut_off_val) / 100))]
            print("train_dataset: ", len(train_dataset), "| val_dataset: ", len(val_dataset))

        train_loader = DataLoader(train_dataset, batch_size=batch_size_gnn, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size_gnn, shuffle=True)

        for step, data in enumerate(train_loader):
            print(data[0])
            break
    
    if text2graph_type == 'hetero':
        all_dataset = torch.load(f'{exp_file_path}/embeddings_word_llm/data_hetero_all.pt')
        all_loader = all_dataset[0]
        print(all_loader)
        print(all_loader.y.shape)
        print(all_loader.train_mask.shape)
        print(all_loader.val_mask.shape)


    init_metrics = {'_epoch_stop': 0,'_train_loss': 0,'_val_loss': 0,'_train_acc': 0,'_val_last_acc': 0,'_test_acc': 0,'_exec_time': 0,}

    torch.cuda.empty_cache()
    gc.collect()

    metrics = init_metrics.copy()
    train_model_args = {
        'text2graph_type': text2graph_type,
        'all_loader': all_loader, # just for hetero graph
        'train_loader': train_loader, 
        'val_loader': val_loader,  
        'metrics': metrics, 
        'device': device,
        'epoch_num': 100,
        'gnn_type': 'TransformerConv', # GCNConv, GATConv, TransformerConv, GraphConv
        'num_features': num_features,  # 64, 128, 256, 512, 768  (depend of nfi)
        'hidden_channels': 100, # size out embeddings: 64, 128, 256, 512, 768
        'learning_rate': 0.00001, # W2V: 0.0001 | LLM: 0.00001, 0.000001
        'gnn_dropout': 0.5,
        'gnn_pooling': 'gmeanp', # gmeanp, gaddp, gmaxp, topkp, sagp
        'gnn_batch_norm': 'BatchNorm1d', # None, BatchNorm1d
        'gnn_layers_convs': 1,
        'gnn_heads': 1, 
        'gnn_dense_nhid': 64, 
        'num_classes': num_classes,
        'edge_dim': edge_dim, # None, 2 
        'retrain_model_name': exp_file_path + 'model_GNN_test_autext24_all_100perc_50_p1.pt',
        'retrain_model': False,
        'patience': 10
    }

    model, optimizer, metrics, embeddings_train_gnn, embeddings_val_gnn = gnn_model(**train_model_args)

    #mlflow.pytorch.log_model(model, "model")
    for key in train_model_args.keys():
        if key in ['train_loader', 'val_loader', 'retrain_model_name', 'metrics']: 
            continue
        mlflow.log_param(key, train_model_args[key])

    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f'{exp_file_path}/model_GNN_{exp_file_name}_{dataset_partition}_states.pt')

    configs = {"train_model_args": str(train_model_args), "metrics": str(metrics), "model": str(model)}
    if save_data:
        torch.save(model, f'{exp_file_path}/model_GNN_{exp_file_name}_{dataset_partition}.pt')
        utils.save_llm_embedings(embeddings_data=embeddings_train_gnn, emb_type='gnn', file_path=f"{exp_file_path}/embeddings_gnn/autext_train_emb_batch_")
        utils.save_llm_embedings(embeddings_data=embeddings_val_gnn, emb_type='gnn', file_path=f"{exp_file_path}/embeddings_gnn/autext_val_emb_batch_")
        utils.save_json(configs, file_path=f'{exp_file_path}configs.json')

    logger.info("DONE GNN Process")



# *************************************** EXPERIMENTS IN BATCHES
def graph_neural_network_batch(autext_train_set, autext_val_set, autext_test_set, experiments_path_dir, experiments_path_file, exp_file_path, built_graph_dataset, num_classes=2):
    
    print('*** INIT EXPERIMENTS')
    experiments_data = utils.read_csv(f'{experiments_path_file}')
    print(experiments_data.info())

    train_text_docs = utils.process_dataset(autext_train_set)
    val_text_docs = utils.process_dataset(autext_val_set)
    test_text_docs = utils.process_dataset(autext_test_set)

    init_metrics = {'_epoch_stop': 0,'_train_loss': 0,'_val_loss': 0,'_train_acc': 0,'_val_last_acc': 0,'_test_acc': 0,'_exec_time': 0,}
    num_classes = num_classes
    #nfi = 'llm'
    #cuda_num = 0
    batch_size_gnn = 32 # 16 -> semeval | 64 -> autext
    first_running = True
    #exp_file_path = utils.OUTPUT_DIR_PATH + 'batch_experiment_autext23_20perc/'

    for index, row in experiments_data.iterrows():
        print("******************************************* Running experiment with ID: ", row['id'])
        start = time.time()

        if not utils.is_dir_empty(dir_path=f'{experiments_path_dir}/embeddings_gnn'):
            utils.delete_dir_files(f'{experiments_path_dir}/embeddings_gnn')
        
        device = torch.device(f"cuda:{row['cuda_num']}" if torch.cuda.is_available() else "cpu")
        print('device: ', device)

        cut_dataset_train = len(train_text_docs) * (int(row['dataset_percent']) / 100)
        train_text_docs_batch = train_text_docs[:int(cut_dataset_train)]
        cut_dataset_val = len(val_text_docs) * (100 / 100)
        val_text_docs_batch = val_text_docs[:int(cut_dataset_val)]
        cut_dataset_test = len(test_text_docs) * (33 / 100)
        test_text_docs_batch = test_text_docs[:int(cut_dataset_test)]

        if row['_done'] == True or row['_done'] == 'True':
            print('Experiment already DONE')
            continue

        edge_dim = 2
        if not row['gnn_edge_attr']:
            edge_dim = None

        t2g_instance = text2graph.Text2CoocGraph(
            graph_type = row['graph_edge_type'],
            window_size = row['window_size'], 
            apply_prep = True, 
            steps_preprocessing = {
                "to_lowercase": True,
                "handle_blank_spaces": True,
                "handle_html_tags": True,
                "handle_special_chars": row['prep_espcial_chars'],
                "handle_stop_words": row['prep_stop_words'],
            },
            language = 'en',
        )

        if not first_running:
            graph_instance_previous_values = {
                'graph_type': experiments_data.loc[index-1, 'graph_edge_type'],
                'window_size': experiments_data.loc[index-1, 'window_size'], 
                "handle_special_chars": experiments_data.loc[index-1, 'prep_espcial_chars'],
                "handle_stop_words": experiments_data.loc[index-1, 'prep_stop_words'],
                "gnn_edge_attr": experiments_data.loc[index-1, 'gnn_edge_attr'],
                "graph_node_feat_init": experiments_data.loc[index-1, 'graph_node_feat_init'],
                "dataset_percent": experiments_data.loc[index-1, 'dataset_percent'],
            } 

            graph_instance_current_values = {
                'graph_type': row['graph_edge_type'],
                'window_size': row['window_size'], 
                "handle_special_chars": row['prep_espcial_chars'],
                "handle_stop_words": row['prep_stop_words'],
                "gnn_edge_attr": row['gnn_edge_attr'],
                "graph_node_feat_init": row['graph_node_feat_init'],
                "dataset_percent": row['dataset_percent'],
            }

        #num_features = 768 
        #if row['graph_node_feat_init'] == 'llm':
        #    num_features = 768 
       
        #exp_file_path = utils.OUTPUT_DIR_PATH + 'batch_expriments/'
        try:
            metrics = init_metrics.copy() 
            #if True:
            if built_graph_dataset and (first_running or graph_instance_previous_values != graph_instance_current_values):
                print('GENERATING GRAPH AND BUILDING DATASET')
            
                utils.delete_dir_files(f'{experiments_path_dir}/embeddings_word_llm')
                    
                graphs_train_data = utils.t2g_transform(train_text_docs_batch, t2g_instance)
                graphs_val_data = utils.t2g_transform(val_text_docs_batch, t2g_instance)
                graphs_test_data = utils.t2g_transform(test_text_docs_batch, t2g_instance)

                train_build_dataset = BuildDataset(graphs_train_data[:], subset='train', device=device, edge_features=row['gnn_edge_attr'], nfi=row['graph_node_feat_init'], llm_finetuned_name=row['graph_node_feat_init_llm'], exp_file_path=experiments_path_dir, num_features=row['graph_nfi_embb_size'])
                train_dataset = train_build_dataset.process_dataset()
                val_build_dataset = BuildDataset(graphs_val_data[:], subset='val', device=device, edge_features=row['gnn_edge_attr'], nfi=row['graph_node_feat_init'], llm_finetuned_name=row['graph_node_feat_init_llm'], exp_file_path=experiments_path_dir, num_features=row['graph_nfi_embb_size'])
                val_dataset = val_build_dataset.process_dataset()
                test_build_dataset = BuildDataset(graphs_test_data[:], subset='test', device=device, edge_features=row['gnn_edge_attr'], nfi=row['graph_node_feat_init'], llm_finetuned_name=row['graph_node_feat_init_llm'], exp_file_path=experiments_path_dir, num_features=row['graph_nfi_embb_size'])
                test_dataset = test_build_dataset.process_dataset()
            
            exp_path = experiments_path_dir
            if built_graph_dataset == False:
                exp_path =  exp_file_path
             
            train_dataset_tensors = glob.glob(f'{exp_path}/embeddings_word_llm/data_train_*.pt')
            val_dataset_tensors = glob.glob(f'{exp_path}/embeddings_word_llm/data_val_*.pt')
            test_dataset_tensors = glob.glob(f'{exp_path}/embeddings_word_llm/data_test_*.pt')

            train_dataset, val_dataset, test_dataset = [], [], []
            for tensor_file in train_dataset_tensors:
                train_dataset.extend(torch.load(tensor_file))
            for tensor_file in val_dataset_tensors:
                val_dataset.extend(torch.load(tensor_file))
            for tensor_file in test_dataset_tensors:
                test_dataset.extend(torch.load(tensor_file))

            train_loader = DataLoader(train_dataset, batch_size=batch_size_gnn, shuffle=True, num_workers=4, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size_gnn, shuffle=True, num_workers=4, pin_memory=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size_gnn, shuffle=True, num_workers=4, pin_memory=True)

            train_model_args = {
                'train_loader': train_loader, 
                'val_loader': val_loader,  
                'metrics': metrics, 
                'device': device,
                'epoch_num': row['epoch_num'],
                'gnn_type': row['gnn_type'], # GCN, GAT, TransformerConv
                'num_features': row['graph_nfi_embb_size'], 
                'hidden_channels': row['gnn_nhid'], # size out embeddings
                'learning_rate': row['gnn_learning_rate'], # W2V: 0.0001 | LLM: 0.00001
                'gnn_dropout': row['gnn_dropout'],
                'gnn_pooling': row['gnn_pooling'], # gmeanp, gmaxp, topkp
                'gnn_batch_norm': row['gnn_batch_norm'], # None, BatchNorm1d
                'gnn_layers_convs': row['gnn_layers_convs'],
                'gnn_heads': row['gnn_heads'], 
                'gnn_dense_nhid': row['gnn_dense_nhid'],
                'num_classes': num_classes,
                'edge_dim': edge_dim, # None, 2 
            }

            model, optimizer, metrics, embeddings_train_gnn, embeddings_val_gnn = gnn_model(**train_model_args)

            #*** Eval TEST set
            correct = 0
            gnn_test_embeddings = []
            model.to(device)
            model.eval()
            with torch.no_grad():
                for step, data in enumerate(test_loader): 
                    data.to(device)
                    out, embeddings = model(data.x, data.edge_index, data.edge_attr, data.batch)  
                    y_pred = out.argmax(dim=1)  # Use the class with highest probability.
                    correct += int((y_pred == data.y).sum())  # Check against ground-truth labels.
                    gnn_test_embeddings.append({
                        'batch': step, 
                        'doc_id': data.context['id'],
                        "labels": data.y,
                        "y_pred": y_pred,
                        'embedding': embeddings
                    })

                preds_test_list = []  
                for emb_data in gnn_test_embeddings:
                    for doc_id, label, y_pred, embedding in zip(emb_data['doc_id'], emb_data['labels'], emb_data['y_pred'], emb_data['embedding']):
                        preds_test_list.append({
                            "doc_id": doc_id.cpu().detach().numpy().tolist(),
                            'label': label.cpu().detach().numpy().tolist(), 
                            'y_pred': y_pred.cpu().detach().numpy().tolist(), 
                        })
                        
                preds_test_df = pd.DataFrame(preds_test_list)
                y_true = preds_test_df['label'].values.tolist()
                y_pred = preds_test_df['y_pred'].values.tolist()
                
                print(preds_test_df.info()) 
                print(preds_test_df['label'].value_counts())
                print(preds_test_df['y_pred'].value_counts())

                test_accuracy = accuracy_score(y_true, y_pred)
                test_f1score = f1_score(y_true, y_pred , average='macro')

                print('-----> Accuracy:', test_accuracy)
                print('-----> F1Score:', test_f1score)

                test_accuracy = correct / len(test_loader.dataset)

                metrics['_test_acc'] = test_accuracy
                metrics['_test_f1score'] = test_f1score

            #utils.save_llm_embedings(embeddings_data=gnn_test_embeddings, emb_type='gnn',file_path=f"{experiments_path_dir}/embeddings_gnn/autext_{subset}_emb_batch_")

            metrics['_done'] = True
            del model
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
            first_running = False
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


    print('*** DONE EXPERIMENTS') 


def graph_neural_network_test_eval(autext_test_set, t2g_instance, nfi, exp_file_path, dataset_partition, llm_finetuned_name, edge_features, num_features, batch_size_gnn, device):

    #  Text 2 Graph test data and BuildDataset
    subset = 'test'
    #graphs_test_data = utils.t2g_transform(autext_test_set, t2g_instance)
    #utils.save_data(graphs_test_data, path=f'{utils.OUTPUT_DIR_PATH}graphs/', file_name=f'graphs_{subset}_{dataset_partition}')

    # check: tranformed in graph_neural_network_test
    #graphs_test_data = utils.load_data(path=f'{utils.OUTPUT_DIR_PATH}graphs/', file_name=f'graphs_{subset}_{dataset_partition}')  

    # check: generated in graph_neural_network_test
    #test_build_dataset = BuildDataset(graphs_test_data[:], subset=f'{subset}', device=device, edge_features=edge_features, nfi=nfi, llm_finetuned_name=llm_finetuned_name, exp_file_path=exp_file_path, dataset_partition=dataset_partition, num_features=num_features)
    #test_dataset = test_build_dataset.process_dataset()

    print("exp_file_path: ", exp_file_path)
    test_dataset_tensors = glob.glob(f'{exp_file_path}/embeddings_word_llm/data_{subset}_*.pt')
    test_dataset = []
    for tensor_file in test_dataset_tensors:
        test_dataset.extend(torch.load(tensor_file))

    test_loader = DataLoader(test_dataset, batch_size=batch_size_gnn, shuffle=True, num_workers=4, pin_memory=True)
    
    #  get GNN Emb
    gnn_model_name = f'model_GNN_test_{dataset_partition}'
    gnn_model = torch.load(f"{exp_file_path}{gnn_model_name}.pt", map_location = device)
    print(gnn_model)
    
    for step, data in enumerate(test_loader):
        #print(step, data['context']['id'], data['context']['target'], data['context']['lang'], data['context']['lang_confidence'])
        print(data[0])
        break

    gnn_test_embeddings = []
    gnn_model.to(device)
    gnn_model.eval()

    criterion = torch.nn.CrossEntropyLoss() # BCEWithLogitsLoss, CrossEntropyLoss, BCELoss
    test_loss, test_f1, test_acc, test_prec, test_rec, gnn_test_embeddings, _ = test_graph(model=gnn_model, criterion=criterion, loader=test_loader, epoch=0, type='test', device=device)
    print('-----> Accuracy:', test_acc)
    print('-----> F1Score:', test_f1)
    print('-----> Precision:', test_prec)
    print('-----> Recall:', test_rec)
    print('-----> Loss:', test_loss)
    
    mlflow.log_metric(key=f"F1Score-test", value=float(test_f1))
    mlflow.log_metric(key=f"Accuracy-test", value=float(test_acc))
    mlflow.log_metric(key=f"Precision-test", value=float(test_prec))
    mlflow.log_metric(key=f"Recall-test", value=float(test_rec))
    mlflow.log_metric(key=f"Loss-test", value=float(test_loss))

    '''
    with torch.no_grad():
        for step, data in enumerate(test_loader): 
            data.to(device)
            out, embeddings = gnn_model(data.x, data.edge_index, data.edge_attr, data.batch) 
            y_pred = out.argmax(dim=1)  # Use the class with highest probability.
            gnn_test_embeddings.append({
                'batch': step, 
                'doc_id': data.context['id'], 
                "labels": data.y,
                "y_pred": y_pred,
                'embedding': embeddings
            })

        #**************** TMP
        preds_test_list = []  
        for emb_data in gnn_test_embeddings:
            for doc_id, label, y_pred, embedding in zip(emb_data['doc_id'], emb_data['labels'], emb_data['y_pred'], emb_data['embedding']):
                preds_test_list.append({
                    "doc_id": doc_id.cpu().detach().numpy().tolist(),
                    'label': label.cpu().detach().numpy().tolist(), 
                    'y_pred': y_pred.cpu().detach().numpy().tolist(), 
                })

        preds_test_df = pd.DataFrame(preds_test_list)
        y_true = preds_test_df['label'].values.tolist()
        y_pred = preds_test_df['y_pred'].values.tolist()
        
        print(preds_test_df.info()) 
        print(preds_test_df['label'].value_counts())
        print(preds_test_df['y_pred'].value_counts())

        test_accuracy = accuracy_score(y_true, y_pred)
        test_f1score = f1_score(y_true, y_pred , average='macro')

        print('-----> Accuracy:', test_accuracy)
        print('-----> F1Score:', test_f1score)
        
        #**************** TMP
    '''

    utils.save_llm_embedings(embeddings_data=gnn_test_embeddings, emb_type='gnn',file_path=f"{exp_file_path}/embeddings_gnn/autext_{subset}_emb_batch_")
    

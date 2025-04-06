

import torch
import numpy as np
import random
import os
import nltk 
import re
import contractions
from torch_geometric.utils import degree
import numpy as np
import networkx as nx

from nltk.corpus import stopwords
nltk.download('stopwords')

import utils 

EXTERNAL_DISK_PATH = '/media/discoexterno/andric/data/experiments/' # hetero_graph, cooc_graph
ROOT_DIR = '/home/avaldez/projects/GraphDeepLearning'
DATASET_DIR = ROOT_DIR + '/datasets/'
OUTPUT_DIR_PATH = ROOT_DIR + '/outputs/'

def to_lowercase(text):
    return text.lower()

def handle_contraction_apostraphes(text):
    text = re.sub('([A-Za-z]+)[\'`]([A-Za-z]+)', r'\1'r'\2', text)
    return text

def handle_contraction(text):
  expanded_words = []
  for word in text.split():
    expanded_words.append(contractions.fix(word))
  return ' '.join(expanded_words)

def remove_blank_spaces(text):
    return re.sub(r'\s+', ' ', text).strip() # remove blank spaces

def remove_html_tags(text):
    return re.compile('<.*?>').sub(r'', text) # remove html tags

def remove_special_chars(text):
    text = re.sub('[^A-Za-z0-9]+ ', ' ', text) # remove special chars
    text = re.sub('\W+', ' ', text) # remove special chars
    text = text.replace('"'," ")
    text = text.replace('('," ")
    text = re.sub(r'\s+', ' ', text).strip() # remove blank spaces
    return text

def remove_stop_words(text):
    # remove stop words
    tokens = nltk.word_tokenize(text)
    without_stopwords = [word for word in tokens if not word.lower().strip() in set(stopwords.words('english'))]
    text = " ".join(without_stopwords)
    return text

def text_normalize(text, special_chars=False, stop_words=False):
    text = to_lowercase(text)
    text = handle_contraction(text)
    text = handle_contraction_apostraphes(text)
    text = remove_blank_spaces(text)
    text = remove_html_tags(text)
    if special_chars:
        text = remove_special_chars(text)
    if stop_words: 
        text = remove_stop_words(text)
    return text

def set_random_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)

def read_dataset(dataset_name):
    if dataset_name in ['semeval24', 'semeval24_s2']:
        
        #dataset_name = 'autext23' # autext23, autext23_s2
        if dataset_name == 'semeval24': # subtask1, subtask2
            subtask = 'subtask1' 
        if dataset_name == 'semeval24_s2':
            subtask = 'subtask2' #

        autext_train_set = utils.read_json(dir_path=f'{utils.DATASET_DIR}semeval2024/{subtask}/train_set.jsonl')
        autext_val_set = utils.read_json(dir_path=f'{utils.DATASET_DIR}semeval2024/{subtask}/dev_set.jsonl')
        autext_test_set = utils.read_json(dir_path=f'{utils.DATASET_DIR}semeval2024/{subtask}/test_set.jsonl')
        autext_train_set = autext_train_set.sample(frac=1).reset_index(drop=True)
        autext_val_set = autext_val_set.sample(frac=1).reset_index(drop=True)
        autext_test_set = autext_test_set.sample(frac=1).reset_index(drop=True)
        print("autext_train_set: ", autext_train_set.info())

        autext_train_set['word_len'] = autext_train_set['text'].str.split().str.len()
        autext_val_set['word_len'] = autext_val_set['text'].str.split().str.len()
        autext_test_set['word_len'] = autext_test_set['text'].str.split().str.len()
        print("\n min_max_avg_token Train: ", autext_train_set['word_len'].min(), autext_train_set['word_len'].max(), int(autext_train_set['word_len'].mean()))
        print("min_max_avg_token Val:   ", autext_val_set['word_len'].min(), autext_val_set['word_len'].max(),  int(autext_val_set['word_len'].mean()))
        print("min_max_avg_token Test:  ", autext_test_set['word_len'].min(), autext_test_set['word_len'].max(), int(autext_test_set['word_len'].mean()))
        print("total_distro_train_val_test: ", autext_train_set.shape, autext_val_set.shape, autext_test_set.shape)

        min_token_len = 1
        max_token_len = 5000
        autext_train_set = autext_train_set[(autext_train_set['word_len'] >= min_token_len) & (autext_train_set['word_len'] <= max_token_len)]
        autext_val_set = autext_val_set[(autext_val_set['word_len'] >= min_token_len) & (autext_val_set['word_len'] <= max_token_len)]
        #autext_train_set, autext_val_set = train_test_split(autext_train_set, test_size=0.3)
        print("label_distro_train_val_test: ", autext_train_set.value_counts('label'), autext_val_set.value_counts('label'), autext_test_set.value_counts('label'))

        print(autext_train_set.nlargest(5, ['word_len']) )
        #autext_val_set = pd.concat([autext_val_set, autext_val_set_2], axis=0)


        print("autext_train_set: ", autext_train_set.info())
        print("autext_val_set: ", autext_val_set.info())
        print("autext_test_set: ", autext_test_set.info())
        print(autext_train_set['model'].value_counts())
        print(autext_val_set['model'].value_counts())
        
        return autext_train_set, autext_val_set, autext_test_set


    # ****************************** READ DATASET AUTEXT 2023
    if dataset_name in ['autext23', 'autext23_s2']:
        
        #dataset_name = 'autext23' # autext23, autext23_s2
        if dataset_name == 'autext23': # subtask1, subtask2
            subtask = 'subtask1' 
        if dataset_name == 'autext23_s2':
            subtask = 'subtask2' #
        
        autext_train_set = utils.read_csv(file_path=f'{utils.DATASET_DIR}autext2023/{subtask}/train_set.csv') 
        autext_val_set = utils.read_csv(file_path=f'{utils.DATASET_DIR}autext2023/{subtask}/val_set.csv') 
        autext_test_set = utils.read_csv(file_path=f'{utils.DATASET_DIR}autext2023/{subtask}/test_set.csv') 
        print("autext_train_set: ", autext_train_set.info())
        print("autext_val_set: ", autext_val_set.info())
        print("autext_test_set: ", autext_test_set.info())
        print("total_distro_train_val_test: ", autext_train_set.shape, autext_val_set.shape, autext_test_set.shape)
        print("label_distro_train_val_test: ", autext_train_set.value_counts('label'), autext_val_set.value_counts('label'), autext_test_set.value_counts('label'))
        print("domain_distro_train_val_test: ", autext_train_set.value_counts('domain'), autext_val_set.value_counts('domain'), autext_test_set.value_counts('domain'))
        print("model_distro_train_val_test: ", autext_train_set.value_counts('model'), autext_val_set.value_counts('model'), autext_test_set.value_counts('model'))
        
        # Model distribution for each source
        print("Model distribution per source in Train set:\n", autext_train_set.groupby("domain")["model"].value_counts())
        print("Model distribution per source in Validation set:\n", autext_val_set.groupby("domain")["model"].value_counts())
        print("Model distribution per source in Test set:\n", autext_test_set.groupby("domain")["model"].value_counts())

        # Label distribution for each source
        print("Label distribution per source in Train set:\n", autext_train_set.groupby("domain")["label"].value_counts())
        print("Label distribution per source in Validation set:\n", autext_val_set.groupby("domain")["label"].value_counts())
        print("Label distribution per source in Test set:\n", autext_test_set.groupby("domain")["label"].value_counts())


        autext_train_set['word_len'] = autext_train_set['text'].str.split().str.len()
        autext_val_set['word_len'] = autext_val_set['text'].str.split().str.len()
        autext_test_set['word_len'] = autext_test_set['text'].str.split().str.len()
        print("min_max_avg_token Train: ", autext_train_set['word_len'].min(), autext_train_set['word_len'].max(), int(autext_train_set['word_len'].mean()))
        print("min_max_avg_token Val:   ", autext_val_set['word_len'].min(), autext_val_set['word_len'].max(),  int(autext_val_set['word_len'].mean()))
        print("min_max_avg_token Test:  ", autext_test_set['word_len'].min(), autext_test_set['word_len'].max(), int(autext_test_set['word_len'].mean()))
        
        return autext_train_set, autext_val_set, autext_test_set

    # ****************************** READ DATASET COLING 2024
    if dataset_name in ['coling24']:   
        #dataset_name = 'coling24' 
        autext_train_set = utils.read_json(dir_path=f'{utils.DATASET_DIR}coling2024/en_train.jsonl')
        autext_val_set = utils.read_json(dir_path=f'{utils.DATASET_DIR}coling2024/en_dev.jsonl')
        autext_test_set = utils.read_json(dir_path=f'{utils.DATASET_DIR}coling2024/test_set_en_with_label.jsonl')
        autext_train_set = autext_train_set.sample(frac=1).reset_index(drop=True)
        autext_val_set = autext_val_set.sample(frac=1).reset_index(drop=True)
        autext_test_set = autext_test_set.sample(frac=1).reset_index(drop=True)
        #autext_test_set = autext_test_set[['testset_id', 'label', 'text']]
        #print("distro_train_val_test: ", autext_train_set.shape, autext_val_set.shape, autext_test_set.shape)
        print("autext_train_set: ", autext_train_set.info())
        print("autext_val_set: ", autext_val_set.info())
        print("autext_test_set: ", autext_test_set.info())

        # Model distribution for each source
        print("Model distribution per source in Train set:\n", autext_train_set.groupby("source")["model"].value_counts())
        print("Model distribution per source in Validation set:\n", autext_val_set.groupby("source")["model"].value_counts())
        print("Model distribution per source in Test set:\n", autext_test_set.groupby("source")["model"].value_counts())

        # Label distribution for each source
        print("Label distribution per source in Train set:\n", autext_train_set.groupby("source")["label"].value_counts())
        print("Label distribution per source in Validation set:\n", autext_val_set.groupby("source")["label"].value_counts())
        print("Label distribution per source in Test set:\n", autext_test_set.groupby("source")["label"].value_counts())

        autext_train_set['word_len'] = autext_train_set['text'].str.split().str.len()
        autext_val_set['word_len'] = autext_val_set['text'].str.split().str.len()
        autext_test_set['word_len'] = autext_test_set['text'].str.split().str.len()
        print("min_max_avg_token Train: ", autext_train_set['word_len'].min(), autext_train_set['word_len'].max(), int(autext_train_set['word_len'].mean()))
        print("min_max_avg_token Val:   ", autext_val_set['word_len'].min(), autext_val_set['word_len'].max(),  int(autext_val_set['word_len'].mean()))
        print("min_max_avg_token Test:  ", autext_test_set['word_len'].min(), autext_test_set['word_len'].max(), int(autext_test_set['word_len'].mean()))
        print("total_distro_train_val_test: ", autext_train_set.shape, autext_val_set.shape, autext_test_set.shape)

        min_token_text = 1
        max_token_text = 1500
        autext_train_set = autext_train_set[(autext_train_set['word_len'] >= min_token_text) & (autext_train_set['word_len'] <= max_token_text)]
        autext_val_set = autext_val_set[(autext_val_set['word_len'] >= min_token_text) & (autext_val_set['word_len'] <= max_token_text)]
        print("label_distro_train_val_test: ", autext_train_set.value_counts('label'), autext_val_set.value_counts('label'), autext_test_set.value_counts('label'))
        #print(autext_train_set.nlargest(5, ['word_len']) )

        print("distro_train_val_test: ", autext_train_set.shape, autext_val_set.shape, autext_test_set.shape)
        #print(autext_train_set['model'].value_counts())
        #print(autext_val_set['model'].value_counts())
        
        return autext_train_set, autext_val_set, autext_test_set

    

def approximate_betweenness_centrality(edge_index, num_nodes, k=100):
    """
    Approximate betweenness centrality using sampling.
    :param edge_index: Tensor of shape [2, num_edges]
    :param num_nodes: Total number of nodes in the graph
    :param k: Number of nodes to sample
    :return: Approximate betweenness centrality as a tensor
    """

    # Convert to NetworkX graph
    G = nx.Graph()
    G.add_edges_from(edge_index.t().tolist())

    # Sample k nodes
    sampled_nodes = np.random.choice(G.nodes(), size=min(k, num_nodes), replace=False)

    # Compute betweenness centrality for sampled nodes
    betweenness = nx.betweenness_centrality(G, k=min(k, num_nodes), seed=42)
    betweenness = torch.tensor(list(betweenness.values()), dtype=torch.float).view(-1, 1)

    return betweenness

def approximate_closeness_centrality(edge_index, num_nodes, k=100):
    """
    Approximate closeness centrality using sampling.
    :param edge_index: Tensor of shape [2, num_edges]
    :param num_nodes: Total number of nodes in the graph
    :param k: Number of nodes to sample
    :return: Approximate closeness centrality as a tensor
    """

    # Convert to NetworkX graph
    G = nx.Graph()
    G.add_edges_from(edge_index.t().tolist())

    # Sample k nodes
    sampled_nodes = np.random.choice(G.nodes(), size=min(k, num_nodes), replace=False)

    # Compute closeness centrality for sampled nodes
    closeness = nx.closeness_centrality(G, u=sampled_nodes)
    closeness = torch.tensor(list(closeness.values()), dtype=torch.float).view(-1, 1)

    return closeness

def approximate_eigenvector_centrality(edge_index, num_nodes, max_iter=50):
    """
    Approximate eigenvector centrality using power iteration.
    :param edge_index: Tensor of shape [2, num_edges]
    :param num_nodes: Total number of nodes in the graph
    :param max_iter: Maximum number of iterations
    :return: Approximate eigenvector centrality as a tensor
    """

    # Convert to NetworkX graph
    G = nx.Graph()
    G.add_edges_from(edge_index.t().tolist())

    # Compute eigenvector centrality with limited iterations
    eigenvector = nx.eigenvector_centrality(G, max_iter=max_iter, tol=1e-3)
    eigenvector = torch.tensor(list(eigenvector.values()), dtype=torch.float).view(-1, 1)

    return eigenvector

def approximate_pagerank(edge_index, num_nodes, max_iter=50):
    """
    Approximate PageRank using power iteration.
    :param edge_index: Tensor of shape [2, num_edges]
    :param num_nodes: Total number of nodes in the graph
    :param max_iter: Maximum number of iterations
    :return: Approximate PageRank as a tensor
    """

    # Convert to NetworkX graph
    G = nx.Graph()
    G.add_edges_from(edge_index.t().tolist())

    # Compute PageRank with limited iterations
    pagerank = nx.pagerank(G, max_iter=max_iter, tol=1e-3)
    pagerank = torch.tensor(list(pagerank.values()), dtype=torch.float).view(-1, 1)

    return pagerank

def approximate_clustering_coefficient(edges, num_nodes, k=100):
    # Create a NetworkX graph from edges
    G = nx.Graph()
    G.add_edges_from(edges.t().tolist())

    # Sample k nodes randomly
    sampled_nodes = np.random.choice(list(G.nodes()), size=min(k, G.number_of_nodes()), replace=False)

    # Compute clustering coefficient for the sampled nodes
    clustering_coeff = nx.clustering(G, nodes=sampled_nodes)

    # Convert to a tensor of shape [num_nodes, 1]
    clustering_coeff_tensor = torch.zeros(num_nodes, 1, dtype=torch.float)
    for node, value in clustering_coeff.items():
        clustering_coeff_tensor[node] = value

    return clustering_coeff_tensor

    # Create a NetworkX graph from edges
    G = nx.Graph()
    G.add_edges_from(edges.t().tolist())

    # Compute clustering coefficient for all nodes
    clustering_coeff = nx.clustering(G)

    # Convert to a tensor of shape [num_nodes, 1]
    clustering_coeff_tensor = torch.zeros(num_nodes, 1, dtype=torch.float)
    for node, value in clustering_coeff.items():
        clustering_coeff_tensor[node] = value

    return clustering_coeff_tensor
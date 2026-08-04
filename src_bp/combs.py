import itertools
import pandas as pd
import utils

def get_template_comb_dict():
    return {
        'id': None,
        'dataset_name': None,
        'dataset_percent': None,
        'graph_type': None,
        'window_size': None,
        'prep_espcial_chars': None,
        'prep_stop_words': None,
        'graph_edge_type': None,
        'graph_build': True,
        'graph_node_feat_init_llm': None,
        'graph_node_feat_init': None,
        'graph_nfi_embb_size': None,
        'gnn_type': None,
        'gnn_heads': None,
        'gnn_layers_convs': None,
        'gnn_nhid': None,
        'gnn_dense_nhid': None,
        'gnn_learning_rate': None,
        'gnn_dropout': None,
        'gnn_pooling': None,
        'gnn_batch_norm': None,
        'gnn_edge_attr': None,
        'graph_direction': None,
        'epoch_num': None,
        'cuda_num': 0,
        '_epoch_stop': None,
        '_train_loss': None,
        '_val_loss': None,
        '_test_loss': None,
        '_val_acc': None,
        '_test_acc': None,
        '_val_f1_macro': None,
        '_test_f1_macro': None,
        '_best_test_acc': None,
        '_best_test_f1score': None,
        '_epoch_best_test_acc': None,
        '_exec_time': None,
        '_done': False,
        '_desc': ''
    } 

def build_comb(graph_config):
    combinations = []
    template_comb_dict = get_template_comb_dict()
    index = 1
    for comb in itertools.product(*graph_config):
        
        d = dict(template_comb_dict)
        d['id'] = index
        d['graph_type'] = comb[0]
        d['graph_node_feat_init'] = comb[1]
        d['graph_nfi_embb_size'] = comb[2]
        d['gnn_type'] = comb[3]
        d['gnn_heads'] = comb[4]
        d['gnn_layers_convs'] = comb[5]
        d['gnn_nhid'] = comb[6]
        d['gnn_dense_nhid'] = comb[7]
        d['gnn_learning_rate'] = comb[8]
        d['gnn_dropout'] = comb[9]
        d['gnn_pooling'] = comb[10]
        d['gnn_batch_norm'] = comb[11]
        d['epoch_num'] = comb[12]
        d['window_size'] = comb[13]
        d['gnn_edge_attr'] = comb[14]
        d['prep_espcial_chars'] = comb[15]
        d['prep_stop_words'] = comb[16]
        d['graph_edge_type'] = comb[17]
        d['dataset_percent'] = comb[18]
        d['graph_node_feat_init_llm'] = comb[19]
        d['graph_direction'] = comb[20]
        combinations.append(d)
        index += 1
    return combinations


def main(dataset, graph_type, dataset_percent):

    # ***** general config
    graph_config = [
        # 0 - graph type       [graph_type],
        [graph_type], 
        # 1 - graph_node_feat_init LLM, W2V  (see below)
        ['llm'], 
        # 2 - graph_nfi_embb_size size of LLM or W2V [64,128,256,512,768] 
        [768],  
        # 3 - gnn_type (see below)     [],   
        [],                                  
        # 4 - gnn_heads (see below)    [],       
        [],                              
        # 5 - gnn_layers_convs     [1,2,3,4]
        [1, 2, 3],  
        # 6 - gnn_nhid (# size out embeddings: hidden_channels)    [50, 100, 200]
        [100, 200],  
        # 7 - gnn_dense_nhid       [32,64,128]
        [32, 64], 
        # 8 - gnn_learning_rate    [0.0001, 0.00001]
        [0.0001],  
        # 9 - gnn_dropout      [0.5, 0.8]
        [0.5],  
        # 10 - gnn_pooling     ['gmeanp', 'gmaxp', 'topkp']
        ['gmeanp'], 
        # 11 - gnn_batch_norm      ['None', 'BatchNorm1d', 'LayerNorm']
        ['LayerNorm'], 
        # 12 - epoch_num       [100] 
        [100], 
        # 13 - graph cooc window_size      [2,5,10,20],
        [10],                              
        # 14 - graph edge attr cooc (see below)   [False, True],
        [],                          
        # 15 - graph handle special chars    [False, True],
        [False],                          
        # 16 - graph handle stop word    [False, True],
        [False],                          
        # 17 - graph edge type    ['Graph', 'DiGraph'], (conver in 20. graph_direction)
        ['Graph'],                          
        # 18 - dataset_percent  [20,50,100],
        [dataset_percent],               
        # 19 - graph_node_feat_init_llm
        ['microsoft/deberta-v3-base'],
        # 20 - graph_direction: directed or undirected
        ['directed', 'undirected']
    ]

    # ***** GCN specific config
    #graph_config[3] = ['GCNConv']   # gnn_type
    #graph_config[4] = [1]   # gnn_heads
    #graph_config[14] = [True]  # graph_edge_attr_cooc 
    #combinations = build_comb(*graph_config)
    
    # ***** GAT specific config
    #graph_config[3] = ['GATConv']   # gnn_type
    #graph_config[4] = [1, 2]   # gnn_heads [1,2,3] 
    #graph_config[14] = [False]    # graph_edge_attr_cooc : False, True
    #combinations = build_comb(graph_config)
    
    # ***** GAT specific config
    graph_config[3] = ['TransformerConv']   # gnn_type
    graph_config[4] = [1, 2]   # gnn_heads [1,2,3] 
    graph_config[14] = [False, True]    # graph_edge_attr_cooc : False, True
    combinations = build_comb(graph_config)


    # ***** NFI-W2v specific config
 
    
    # ***** Save data
    comb_df = pd.DataFrame(combinations).sample(frac=1)
    comb_df['dataset_name'] = dataset
    print(comb_df.info())
    comb_df.to_csv(utils.OUTPUT_DIR_PATH + f'experiments_{dataset}_{graph_type}_{utils.CURRENT_TIME}.csv', index=False)


if __name__ == '__main__':
    dataset = 'autext23_s2' # autext23, semeval24, coling25, autext23_s2, semeval24_s2
    dataset_percent = 10
    graph_type = 'hetero' # cooc, hetero, isg
    main(dataset, graph_type, dataset_percent) 
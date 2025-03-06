
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader, Data
from torch.nn import Linear, BatchNorm1d, ModuleList, LayerNorm, Sequential, Dropout, ReLU
from torch_geometric.nn import GCNConv, GATConv, TransformerConv, TopKPooling, GraphConv, SAGPooling, GINConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool


class GNN_3(torch.nn.Module):
    def __init__(self, gnn_type, hidden_channels, pooling='', n_layers=1, dropout=0.5, num_features=256, edge_dim=1, dense_nhid=64, heads=1, num_classes=2, task='cooc'):
        super().__init__()
        #torch.manual_seed(1234567)

        GNN_LAYER_BY_NAME = {
            "GCNConv": GCNConv,
            "GATConv": GATConv,
            "GraphConv": GraphConv,
            "TransformerConv": TransformerConv,
        }
        add_self_loops = True
        normalize = True
        self.task = task
        self.n_layers = n_layers
        self.edge_dim = edge_dim 
        self.dropout = dropout
        self.pooling = pooling
        self.dense_nhid = dense_nhid
        conv_layer = GNN_LAYER_BY_NAME[gnn_type]

        self.conv_layers = ModuleList([])
        self.bn_layers = ModuleList([])
        self.bn1 = BatchNorm1d(hidden_channels*heads)
        self.bn2 = BatchNorm1d(hidden_channels*heads)
        self.bn3 = BatchNorm1d(dense_nhid)

        if gnn_type in ['GATConv', 'TransformerConv']:
            self.conv1 = conv_layer(num_features, hidden_channels, heads=heads, dropout=0.0, concat=True, edge_dim=self.edge_dim)
            for i in range(self.n_layers):
                self.conv_layers.append(conv_layer(hidden_channels*heads, hidden_channels, heads=heads, dropout=0.0, concat=True, edge_dim=self.edge_dim))
                self.bn_layers.append(BatchNorm1d(hidden_channels*heads))
        else:
            self.conv1 = conv_layer(num_features, hidden_channels, add_self_loops=add_self_loops, normalize=normalize)
            for i in range(self.n_layers):
                self.conv_layers.append(conv_layer(hidden_channels, hidden_channels, add_self_loops=add_self_loops, normalize=normalize))
                self.bn_layers.append(BatchNorm1d(hidden_channels))


        self.linear1 = Linear(hidden_channels*heads, dense_nhid)
        self.linear2 = Linear(dense_nhid, int(dense_nhid)//2)
        self.linear3 = Linear(int(dense_nhid)//2, num_classes)
        #self.linear4 = Linear(int(dense_nhid)//4, num_classes)

    def forward(self, x, edge_index, edge_attr, batch):
        #print(edge_attr)
        convs_layers = []
        x = self.conv1(x, edge_index, edge_attr)
        convs_layers.append(x)
        x = x.relu()
        #x = self.bn1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        for i in range(self.n_layers):
            x = self.conv_layers[i](x, edge_index, edge_attr)
            convs_layers.append(x)
            x = x.relu()
            #x = self.bn_layers[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)


        if self.task == 'cooc':
            if self.pooling == 'gaddp':
                x = global_add_pool(x, batch)
            elif self.pooling == 'gmaxp':
                x = global_max_pool(x, batch)
            else:
                x = global_mean_pool(x, batch)

        #x = self.bn2(x)
        #x = F.dropout(x, p=self.dropout, training=self.training)
        
        out = torch.relu(self.linear1(x))
        out = torch.relu(self.linear2(out))
        out = torch.relu(self.linear3(out))
        #out = torch.relu(self.linear4(out))

        return out, x, convs_layers
    

class GNN_2(torch.nn.Module):
    def __init__(self, gnn_type, hidden_channels, pooling='', n_layers=1, dropout=0.5, num_features=256, edge_dim=1, dense_nhid=64, heads=1, num_classes=2, task='cooc'):
        super().__init__()
        torch.manual_seed(1234567)

        GNN_LAYER_BY_NAME = {
            "GCNConv": GCNConv,
            "GATConv": GATConv,
            "GraphConv": GraphConv,
            "TransformerConv": TransformerConv,
        }
        self.task = task
        self.n_layers = n_layers
        self.dropout = dropout
        self.pooling = pooling
        self.dense_nhid = dense_nhid
        self.edge_dim = edge_dim
        conv_layer = GNN_LAYER_BY_NAME[gnn_type]

        self.conv1 = conv_layer(num_features, hidden_channels, heads, edge_dim=self.edge_dim)
        self.bn1 = BatchNorm1d(hidden_channels*heads)
        self.conv_layers = ModuleList([])
        self.bn_layers = ModuleList([])
        for i in range(self.n_layers):
            self.conv_layers.append(conv_layer(hidden_channels*heads, hidden_channels, heads, edge_dim=self.edge_dim))
            self.bn_layers.append(BatchNorm1d(hidden_channels*heads))

        self.bn2 = BatchNorm1d(hidden_channels*heads)
        self.linear1 = Linear(hidden_channels*heads, dense_nhid)
        self.linear2 = Linear(dense_nhid, int(dense_nhid//2))
        self.linear3 = Linear(int(dense_nhid//2), num_classes)
        #self.linear4 = Linear(int(dense_nhid//4), num_classes)
        
    def forward(self, x, edge_index, edge_attr, batch):
        #x = self.conv1(x, edge_index)
        #x = x.relu()
        convs_layers = []
        x = self.conv1(x, edge_index, edge_attr)
        convs_layers.append(x)
        x = x.relu()
        x = self.bn1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        for i in range(self.n_layers):
            x = self.conv_layers[i](x, edge_index, edge_attr)
            convs_layers.append(x)
            x = x.relu()
            x = self.bn_layers[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.task == 'cooc':
            if self.pooling == 'gaddp':
                x = global_add_pool(x, batch)
            elif self.pooling == 'gmaxp':
                x = global_max_pool(x, batch)
            else:
                x = global_mean_pool(x, batch)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.bn2(x)

        out = torch.relu(self.linear1(x))
        out = torch.relu(self.linear2(out))
        out = torch.relu(self.linear3(out))
        #out = torch.relu(self.linear4(out))

        #out = self.dense_layer.forward(x)
        return out, x, convs_layers
    

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
        edge_dim=None
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
        self.sigmoid = torch.nn.Sigmoid()

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
        out = self.linear3(out)
        
        out = self.sigmoid(out)
        return out, x, []
        #return x


class GNNStanford(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, dense_hidden_dim, output_dim, dropout, num_layers, gnn_type='GCNConv', heads=1, task='hetero'):
        super(GNNStanford, self).__init__()
        self.task = task
        self.heads = heads
        self.gnn_type = gnn_type
        self.convs = ModuleList()
        self.convs.append(self.build_conv_model(input_dim, hidden_dim))
        self.lns = ModuleList()
        self.lns.append(LayerNorm(hidden_dim*heads))
        self.lns.append(LayerNorm(hidden_dim*heads))
        for l in range(2):
            self.convs.append(self.build_conv_model(hidden_dim*heads, hidden_dim))

        # post-message-passing
        self.post_mp = Sequential(
            Linear(hidden_dim*heads, dense_hidden_dim),
            Dropout(dropout),
            Linear(dense_hidden_dim, output_dim)
        )

        if not (self.task == 'hetero' or self.task == 'cooc'):
            raise RuntimeError('Unknown task.')

        self.dropout = dropout
        self.num_layers = num_layers
        self.sigmoid = torch.nn.Sigmoid()

    def build_conv_model(self, input_dim, hidden_dim):
        # refer to pytorch geometric nn module for different implementation of GNNs.
        if self.gnn_type == 'GCNConv':
            return GCNConv(input_dim, hidden_dim, self.heads)
        if self.gnn_type == 'GATConv':
            return GATConv(input_dim, hidden_dim, self.heads)
        if self.gnn_type == 'TransformerConv':
            return TransformerConv(input_dim, hidden_dim, self.heads)
        #else: # cooc
        #    return GINConv(Sequential(Linear(input_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim)))

    def forward(self, x, edge_index, edge_attr, batch):
        #x, edge_index, batch = data.x, data.edge_index, data.batch
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            emb = x
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if not i == self.num_layers - 1:
                x = self.lns[i](x)

        if self.task == 'cooc':
            x = global_mean_pool(x, batch)

        x = self.post_mp(x)
        #self.sigmoid(x)
        #F.log_softmax(x, dim=1)

        return self.sigmoid(x), emb, [] 

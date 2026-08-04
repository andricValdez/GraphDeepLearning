import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter_add, scatter_mean, scatter_max
from torch_geometric.utils import softmax as pyg_segment_softmax
from torch_geometric.nn import Set2Set


# ----------------------------
# Norm selector
# ----------------------------
def make_norm(norm_type: Optional[str], dim: int):
    if norm_type is None or norm_type.lower() == "none":
        return nn.Identity()
    if norm_type.lower() == "layernorm":
        return nn.LayerNorm(dim)
    if norm_type.lower() == "batchnorm":
        # BatchNorm1d expects [N, C]
        return nn.BatchNorm1d(dim)
    raise ValueError(f"Unknown norm_type: {norm_type}")



# Pooling heads
class AttentionPooling(nn.Module):
    """Graph-level attention pooling."""
    def __init__(self, in_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, in_dim)
        self.vec = nn.Linear(in_dim, 1, bias=False)

    def forward(self, x, batch):
        # score_i = a^T tanh(W h_i)
        s = torch.tanh(self.proj(x))
        s = self.vec(s).squeeze(-1)  # [N]
        alpha = pyg_segment_softmax(s, batch)  # [N]
        pooled = scatter_add(alpha.unsqueeze(-1) * x, batch, dim=0)  # [B, D]
        return pooled


def graph_pool(x, batch, how="mean", attn_pool: Optional[AttentionPooling] = None,
               set2set_op: Optional[Set2Set] = None):
    how = (how or "mean").lower()
    if how == "mean":
        return scatter_mean(x, batch, dim=0)
    if how == "sum":
        return scatter_add(x, batch, dim=0)
    if how == "max":
        out, _ = scatter_max(x, batch, dim=0)
        return out
    if how == "attention":
        assert attn_pool is not None, "Attention pooling requires attn_pool module"
        return attn_pool(x, batch)
    if how == "set2set":
        assert set2set_op is not None, "Set2Set pooling requires set2set_op module"
        return set2set_op(x, batch)
    raise ValueError(f"Unknown pooling_type: {how}")

# FeedForward block (Transformer style)
class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.1):
        super().__init__()
        inner = dim * mult
        self.net = nn.Sequential(
            nn.Linear(dim, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class HierarchicalGlobalPooling(nn.Module):
    """
    Create a virtual document-level node that attends to all nodes
    """
    def __init__(self, dim: int):
        super().__init__()
        self.doc_query = nn.Parameter(torch.randn(1, dim))
        self.attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        
    def forward(self, x, batch):
        """
        x: [N, dim]
        batch: [N] - which graph each node belongs to
        """
        unique_graphs = torch.unique(batch)
        global_contexts = []
        
        for graph_id in unique_graphs:
            mask = (batch == graph_id)
            graph_nodes = x[mask]  # [n_i, dim]
            
            # Attention: doc_query attends to all nodes
            query = self.doc_query.expand(1, -1, -1)  # [1, 1, dim]
            context, _ = self.attn(
                query,
                graph_nodes.unsqueeze(0),
                graph_nodes.unsqueeze(0)
            )  # [1, 1, dim]
            
            global_contexts.append(context.squeeze())
        
        return torch.stack(global_contexts, dim=0)  # [B, dim]



# *****************************************************************  
#                   ISG_VanillaGTN (baseline)
# *****************************************************************  

class ISGAwareMultiHeadAttention(nn.Module):
    """
    Extends your MultiHeadGraphAttentionEdge with ISG-specific enhancements
    """
    def __init__(self, dim: int, heads: int, attn_dropout: float = 0.1,
                 edge_attr_dim: Optional[int] = None,
                 use_token_distance: bool = True):
        super().__init__()
        assert dim % heads == 0, "hidden_dim must be divisible by heads"
        self.dim = dim
        self.heads = heads
        self.dh = dim // heads
        self.scale = 1.0 / math.sqrt(self.dh)
        self.use_token_distance = use_token_distance

        self.Wq = nn.Linear(dim, dim, bias=False)
        self.Wk = nn.Linear(dim, dim, bias=False)
        self.Wv = nn.Linear(dim, dim, bias=False)
        self.Wo = nn.Linear(dim, dim, bias=False)

        # Edge attribute bias (dependency embeddings)
        if edge_attr_dim is not None and edge_attr_dim > 0:
            self.edge_bias = nn.Linear(edge_attr_dim, heads, bias=False)
        else:
            self.edge_bias = None

        # Token distance gating (NEW)
        if use_token_distance:
            self.distance_gate = nn.Sequential(
                nn.Linear(1, heads),
                nn.Sigmoid()
            )
        
        self.drop_attn = nn.Dropout(attn_dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                token_distance: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [N, dim]
        edge_index: [2, E]
        edge_attr: [E, dep_emb_dim] - dependency embeddings
        token_distance: [E] - normalized inverse distances
        """
        N = x.size(0)
        src, dst = edge_index

        Q = self.Wq(x).view(N, self.heads, self.dh)
        K = self.Wk(x).view(N, self.heads, self.dh)
        V = self.Wv(x).view(N, self.heads, self.dh)

        q_i = Q[dst]
        k_j = K[src]
        v_j = V[src]

        # Base attention
        e = (q_i * k_j).sum(dim=-1) * self.scale  # [E, H]

        # Add dependency relation bias
        if self.edge_bias is not None and edge_attr is not None:
            e = e + self.edge_bias(edge_attr)  # [E, H]

        # Add token distance weighting (closer tokens = higher attention)
        if self.use_token_distance and token_distance is not None:
            dist_gate = self.distance_gate(token_distance.unsqueeze(-1))  # [E, H]
            e = e * dist_gate  # Multiplicative gating

        # Softmax per destination node
        alphas = []
        for h in range(self.heads):
            alpha_h = pyg_segment_softmax(e[:, h], dst)
            alphas.append(alpha_h)
        alpha = torch.stack(alphas, dim=1)  # [E, H]
        alpha = self.drop_attn(alpha)

        # Aggregate
        out_heads = torch.zeros((N, self.heads, self.dh), device=x.device, dtype=x.dtype)
        out_heads.index_add_(0, dst, alpha.unsqueeze(-1) * v_j)

        out = out_heads.reshape(N, self.dim)
        out = self.Wo(out)
        return out


class ISGGraphTransformerLayer(nn.Module):
    def __init__(self, dim: int, heads: int, attn_dropout: float,
                 ffn_dropout: float, ffn_mult: int, norm_type: str,
                 edge_attr_dim: Optional[int] = None,
                 use_token_distance: bool = True):
        super().__init__()
        self.pre_attn = make_norm(norm_type, dim)
        self.attn = ISGAwareMultiHeadAttention(
            dim=dim, heads=heads, attn_dropout=attn_dropout,
            edge_attr_dim=edge_attr_dim,
            use_token_distance=use_token_distance
        )
        self.pre_ffn = make_norm(norm_type, dim)
        self.ffn = FeedForward(dim, mult=ffn_mult, dropout=ffn_dropout)

    def forward(self, x, edge_index, edge_attr=None, token_distance=None, sentence_ids=None):
        h = self.pre_attn(x)
        h = self.attn(h, edge_index, edge_attr, token_distance)
        x = x + h

        h = self.pre_ffn(x)
        h = self.ffn(h)
        x = x + h
        return x


class ISG_VanillaGTN(nn.Module):
    """
    Enhanced VanillaGTN with ISG-specific features
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 dense_hidden_dim: int,
                 output_dim: int,
                 dropout: float,
                 num_layers: int,
                 use_edge_attr: bool = False,
                 use_token_distance: bool = True,  # NEW
                 heads: int = 4,
                 norm_type: str = "layernorm",
                 pooling_type: str = "mean",
                 post_mp_layers: int = 2,
                 edge_attr_dim: Optional[int] = None,
                 ffn_mult: int = 4,
                 attn_dropout: float = 0.1,
                 use_global_pooling=False
            ):
        super().__init__()

        self.pooling_type = pooling_type
        self.use_edge_attr = use_edge_attr
        self.use_token_distance = use_token_distance

        self.in_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()

        # Use ISG-aware layers
        self.layers = nn.ModuleList([
            ISGGraphTransformerLayer(
                dim=hidden_dim,
                heads=heads,
                attn_dropout=attn_dropout,
                ffn_dropout=dropout,
                ffn_mult=ffn_mult,
                norm_type=norm_type,
                edge_attr_dim=(edge_attr_dim if use_edge_attr else None),
                use_token_distance=use_token_distance
            ) for _ in range(num_layers)
        ])

        self.attn_pool = AttentionPooling(hidden_dim) if (pooling_type and pooling_type.lower() == "attention") else None
        self.set2set_op = Set2Set(hidden_dim, processing_steps=3) if (pooling_type and pooling_type.lower() == "set2set") else None

        # Post-MP MLP
        mlp_dims = [hidden_dim] + [dense_hidden_dim] * max(0, post_mp_layers - 1) + [output_dim]
        mlp = []
        for li, (din, dout) in enumerate(zip(mlp_dims[:-1], mlp_dims[1:])):
            mlp.append(nn.Linear(din, dout))
            if li < len(mlp_dims) - 2:
                mlp.append(nn.GELU())
                mlp.append(nn.Dropout(dropout))
        self.graph_head = nn.Sequential(*mlp)

        if use_global_pooling:
            self.global_pool = HierarchicalGlobalPooling(hidden_dim)


    def forward(self, x, edge_index, edge_attr, batch, token_distance=None, sentence_ids=None):
        h = self.in_proj(x)

        for layer in self.layers:
            h = layer(h, edge_index, edge_attr, token_distance)

        # Standard pooling
        pooled = graph_pool(h, batch,
                            how=self.pooling_type,
                            attn_pool=self.attn_pool,
                            set2set_op=self.set2set_op)
        
        # Add global context if enabled
        if hasattr(self, 'global_pool'):
            global_context = self.global_pool(h, batch)
            pooled = pooled + global_context  # Residual

        logits = self.graph_head(pooled)
        return logits
    



# *****************************************************************  
#                   VanillaGTNWithEdgeConcat (baseline)
# *****************************************************************  

class VanillaGTNWithEdgeConcat(nn.Module):
    """
    Vanilla GTN from the image (right diagram)
    Edge features are concatenated to node features before Q,K,V projection
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 dense_hidden_dim: int,
                 output_dim: int,
                 dropout: float,
                 num_layers: int,
                 use_edge_attr: bool = False,
                 heads: int = 4,
                 norm_type: str = "layernorm",
                 pooling_type: str = "mean",
                 post_mp_layers: int = 2,
                 edge_attr_dim: Optional[int] = None,
                 ffn_mult: int = 4,
                 attn_dropout: float = 0.1):
        super().__init__()

        self.pooling_type = pooling_type
        self.use_edge_attr = use_edge_attr
        self.edge_attr_dim = edge_attr_dim

        # Input projection to hidden (if needed)
        self.in_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()

        # Stack of GT layers with edge concatenation
        self.layers = nn.ModuleList([
            VanillaGTNLayerWithEdgeConcat(
                dim=hidden_dim,
                heads=heads,
                attn_dropout=attn_dropout,
                ffn_dropout=dropout,
                ffn_mult=ffn_mult,
                norm_type=norm_type,
                edge_attr_dim=(edge_attr_dim if use_edge_attr else None)
            ) for _ in range(num_layers)
        ])

        # Pooling modules
        self.attn_pool = AttentionPooling(hidden_dim) if (pooling_type and pooling_type.lower() == "attention") else None
        self.set2set_op = Set2Set(hidden_dim, processing_steps=3) if (pooling_type and pooling_type.lower() == "set2set") else None

        # Post-MP MLP
        mlp_dims = [hidden_dim] + [dense_hidden_dim] * max(0, post_mp_layers - 1) + [output_dim]
        mlp = []
        for li, (din, dout) in enumerate(zip(mlp_dims[:-1], mlp_dims[1:])):
            mlp.append(nn.Linear(din, dout))
            if li < len(mlp_dims) - 2:
                mlp.append(nn.GELU())
                mlp.append(nn.Dropout(dropout))
        self.graph_head = nn.Sequential(*mlp)

    def forward(self, x, edge_index, edge_attr, batch, token_distance=None, sentence_ids=None):
        h = self.in_proj(x)

        ea = edge_attr if (self.use_edge_attr and edge_attr is not None) else None

        for layer in self.layers:
            h = layer(h, edge_index, ea)

        # Graph readout
        pooled = graph_pool(h, batch,
                            how=self.pooling_type,
                            attn_pool=self.attn_pool,
                            set2set_op=self.set2set_op)

        logits = self.graph_head(pooled)
        return logits


class VanillaGTNLayerWithEdgeConcat(nn.Module):
    """
    Single layer that concatenates edge features to source nodes
    This matches the RIGHT diagram in the image
    """
    def __init__(self, dim: int, heads: int, attn_dropout: float,
                 ffn_dropout: float, ffn_mult: int, norm_type: str,
                 edge_attr_dim: Optional[int] = None):
        super().__init__()
        self.pre_attn = make_norm(norm_type, dim)
        self.attn = VanillaAttentionWithEdgeConcat(
            dim=dim, heads=heads, attn_dropout=attn_dropout,
            edge_attr_dim=edge_attr_dim
        )
        self.pre_ffn = make_norm(norm_type, dim)
        self.ffn = FeedForward(dim, mult=ffn_mult, dropout=ffn_dropout)

    def forward(self, x, edge_index, edge_attr=None):
        h = self.pre_attn(x)
        h = self.attn(h, edge_index, edge_attr)
        x = x + h

        h = self.pre_ffn(x)
        h = self.ffn(h)
        x = x + h
        return x


class VanillaAttentionWithEdgeConcat(nn.Module):
    """
    Vanilla multi-head attention where edge features are concatenated to node features
    This is the approach shown in the RIGHT diagram of the image
    """
    def __init__(self, dim: int, heads: int, attn_dropout: float = 0.1,
                 edge_attr_dim: Optional[int] = None):
        super().__init__()
        assert dim % heads == 0, "hidden_dim must be divisible by heads"
        self.dim = dim
        self.heads = heads
        self.dh = dim // heads
        self.scale = 1.0 / math.sqrt(self.dh)
        self.edge_attr_dim = edge_attr_dim
        
        # For destinations (queries) - no edge concat
        self.Wq = nn.Linear(dim, dim, bias=False)
        
        # For sources (keys, values) - may include edge concat
        if edge_attr_dim is not None and edge_attr_dim > 0:
            # Concatenate edge features to source node features
            self.Wk = nn.Linear(dim + edge_attr_dim, dim, bias=False)
            self.Wv = nn.Linear(dim + edge_attr_dim, dim, bias=False)
        else:
            self.Wk = nn.Linear(dim, dim, bias=False)
            self.Wv = nn.Linear(dim, dim, bias=False)
        
        self.Wo = nn.Linear(dim, dim, bias=False)
        self.drop_attn = nn.Dropout(attn_dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [N, dim]
        edge_index: [2, E] (src=j=row0, dst=i=row1)
        edge_attr: [E, edge_attr_dim] or None
        """
        N = x.size(0)
        src, dst = edge_index

        # Query from destination nodes (no edge info)
        Q = self.Wq(x).view(N, self.heads, self.dh)  # [N, H, dh]
        q_i = Q[dst]  # [E, H, dh]

        # Key and Value from source nodes (WITH edge info if available)
        if self.edge_attr_dim is not None and edge_attr is not None:
            # Concatenate edge features to source node features
            x_src = x[src]  # [E, dim]
            x_src_concat = torch.cat([x_src, edge_attr], dim=-1)  # [E, dim + edge_attr_dim]
            
            K = self.Wk(x_src_concat).view(-1, self.heads, self.dh)  # [E, H, dh]
            V = self.Wv(x_src_concat).view(-1, self.heads, self.dh)  # [E, H, dh]
        else:
            # No edge features - standard attention
            K = self.Wk(x).view(N, self.heads, self.dh)
            V = self.Wv(x).view(N, self.heads, self.dh)
            K = K[src]  # [E, H, dh]
            V = V[src]  # [E, H, dh]

        # Attention scores
        e = (q_i * K).sum(dim=-1) * self.scale  # [E, H]

        # Softmax per destination node, per head
        alphas = []
        for h in range(self.heads):
            alpha_h = pyg_segment_softmax(e[:, h], dst)
            alphas.append(alpha_h)
        alpha = torch.stack(alphas, dim=1)  # [E, H]
        alpha = self.drop_attn(alpha)

        # Aggregate
        out_heads = torch.zeros((N, self.heads, self.dh), device=x.device, dtype=x.dtype)
        out_heads.index_add_(0, dst, alpha.unsqueeze(-1) * V)

        out = out_heads.reshape(N, self.dim)
        out = self.Wo(out)
        return out
    

# *****************************************************************  
#                   HybridISG_GTN
# *****************************************************************  

class HybridISG_GTN(nn.Module):
    """
    Hybrid architecture combining:
    1. Edge concatenation (from Vanilla GTN paper - RIGHT diagram)
    2. ISG-specific enhancements (token distance, structural PEs, hierarchical pooling)
    
    Key Design:
    - Edge features concatenated to source nodes (proven approach)
    - PLUS: Token distance gating on attention scores
    - PLUS: Additional dependency relation bias
    - PLUS: Hierarchical global pooling
    - PLUS: ISG positional encodings
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 dense_hidden_dim: int,
                 output_dim: int,
                 dropout: float,
                 num_layers: int,
                 use_edge_attr: bool = False,
                 use_token_distance: bool = True,
                 use_edge_bias: bool = True,
                 use_sentence_aware: bool = False,
                 heads: int = 4,
                 norm_type: str = "layernorm",
                 pooling_type: str = "mean",
                 post_mp_layers: int = 2,
                 edge_attr_dim: Optional[int] = None,
                 ffn_mult: int = 4,
                 attn_dropout: float = 0.1,
                 use_global_pooling: bool = False):
        super().__init__()

        self.pooling_type = pooling_type
        self.use_edge_attr = use_edge_attr
        self.use_token_distance = use_token_distance
        self.use_edge_bias = use_edge_bias
        self.use_sentence_aware = use_sentence_aware

        # Input projection
        self.in_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()

        # Shared kwargs for both layer types
        _layer_kwargs = dict(
            dim=hidden_dim, heads=heads, attn_dropout=attn_dropout,
            ffn_dropout=dropout, ffn_mult=ffn_mult, norm_type=norm_type,
            edge_attr_dim=(edge_attr_dim if use_edge_attr else None),
            use_token_distance=use_token_distance, use_edge_bias=use_edge_bias,
        )
        if use_sentence_aware:
            # SentenceAwareTransformerLayer already has correct Pre-LN
            self.layers = nn.ModuleList([
                SentenceAwareTransformerLayer(**_layer_kwargs, use_sentence_aware=True)
                for _ in range(num_layers)
            ])
        else:
            # HybridISGTransformerLayer (Pre-LN fixed)
            self.layers = nn.ModuleList([
                HybridISGTransformerLayer(**_layer_kwargs)
                for _ in range(num_layers)
            ])

        # Pooling modules
        self.attn_pool = AttentionPooling(hidden_dim) if (pooling_type and pooling_type.lower() == "attention") else None
        self.set2set_op = Set2Set(hidden_dim, processing_steps=3) if (pooling_type and pooling_type.lower() == "set2set") else None

        # ISG Enhancement: Hierarchical global pooling
        if use_global_pooling:
            self.global_pool = HierarchicalGlobalPooling(hidden_dim)

        # Post-MP MLP
        mlp_dims = [hidden_dim] + [dense_hidden_dim] * max(0, post_mp_layers - 1) + [output_dim]
        mlp = []
        for li, (din, dout) in enumerate(zip(mlp_dims[:-1], mlp_dims[1:])):
            mlp.append(nn.Linear(din, dout))
            if li < len(mlp_dims) - 2:
                mlp.append(nn.GELU())
                mlp.append(nn.Dropout(dropout))
        self.graph_head = nn.Sequential(*mlp)

    def forward(self, x, edge_index, edge_attr, batch, token_distance=None, sentence_ids=None):
        h = self.in_proj(x)

        for layer in self.layers:
            h = layer(h, edge_index, edge_attr, token_distance, sentence_ids)

        # Standard pooling
        pooled = graph_pool(h, batch,
                            how=self.pooling_type,
                            attn_pool=self.attn_pool,
                            set2set_op=self.set2set_op)

        if hasattr(self, 'global_pool'):
            pooled = pooled + self.global_pool(h, batch)

        logits = self.graph_head(pooled)
        return logits


class HybridISGTransformerLayer(nn.Module):
    """
    Single transformer layer with hybrid approach
    """
    def __init__(self, dim: int, heads: int, attn_dropout: float,
                 ffn_dropout: float, ffn_mult: int, norm_type: str,
                 edge_attr_dim: Optional[int] = None,
                 use_token_distance: bool = True,
                 use_edge_bias: bool = True):
        super().__init__()
        self.pre_attn = make_norm(norm_type, dim)
        self.attn = HybridISGAttention(
            dim=dim, 
            heads=heads, 
            attn_dropout=attn_dropout,
            edge_attr_dim=edge_attr_dim,
            use_token_distance=use_token_distance,
            use_edge_bias=use_edge_bias
        )
        self.pre_ffn = make_norm(norm_type, dim)
        self.ffn = FeedForward(dim, mult=ffn_mult, dropout=ffn_dropout)

    def forward(self, x, edge_index, edge_attr=None, token_distance=None, sentence_ids=None):
        h = self.attn(self.pre_attn(x), edge_index, edge_attr, token_distance)
        x = x + h

        h = self.ffn(self.pre_ffn(x))
        x = x + h
        return x


class HybridISGAttention(nn.Module):
    """
    Hybrid multi-head attention combining:
    1. Edge feature concatenation (Vanilla GTN approach)
    2. Dependency bias on attention scores (ISG approach)
    3. Token distance gating (ISG approach)
    
    This gives us the best of both worlds:
    - Rich node representations via edge concat
    - Fine-grained attention control via bias + gating
    """
    def __init__(self, dim: int, heads: int, attn_dropout: float = 0.1,
                 edge_attr_dim: Optional[int] = None,
                 use_token_distance: bool = True,
                 use_edge_bias: bool = True):
        super().__init__()
        assert dim % heads == 0, "hidden_dim must be divisible by heads"
        self.dim = dim
        self.heads = heads
        self.dh = dim // heads
        self.scale = 1.0 / math.sqrt(self.dh)
        self.edge_attr_dim = edge_attr_dim 
        self.use_token_distance = use_token_distance
        self.use_edge_bias = use_edge_bias
        
        # Query projection (destination nodes - no edge info)
        self.Wq = nn.Linear(dim, dim, bias=False)
        
        # Key and Value projections (source nodes - WITH edge concat)
        if edge_attr_dim is not None and edge_attr_dim > 0:
            # Concatenate edge features to source nodes (Vanilla GTN approach)
            self.Wk = nn.Linear(dim + edge_attr_dim, dim, bias=False)
            self.Wv = nn.Linear(dim + edge_attr_dim, dim, bias=False)
            
            # ISG Enhancement 1: Additional dependency bias on attention scores
            if use_edge_bias:
                self.edge_bias = nn.Linear(edge_attr_dim, heads, bias=False)
            else:
                self.edge_bias = None
        else:
            self.Wk = nn.Linear(dim, dim, bias=False)
            self.Wv = nn.Linear(dim, dim, bias=False)
            self.edge_bias = None
        
        # ISG Enhancement 2: Token distance gating
        if use_token_distance:
            self.distance_gate = nn.Sequential(
                nn.Linear(1, heads),
                nn.Sigmoid()
            )
        
        self.Wo = nn.Linear(dim, dim, bias=False)
        self.drop_attn = nn.Dropout(attn_dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                token_distance: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [N, dim]
        edge_index: [2, E] (src, dst)
        edge_attr: [E, edge_attr_dim] - dependency embeddings
        token_distance: [E] - normalized inverse distances
        """
        N = x.size(0)
        src, dst = edge_index

        # Query from destination nodes (no edge info)
        Q = self.Wq(x).view(N, self.heads, self.dh)  # [N, H, dh]
        q_i = Q[dst]  # [E, H, dh]

        # Key and Value from source nodes WITH edge concatenation
        if self.edge_attr_dim is not None and edge_attr is not None:
            # Vanilla GTN approach: Concatenate edge to source node
            x_src = x[src]  # [E, dim]
            x_src_concat = torch.cat([x_src, edge_attr], dim=-1)  # [E, dim + edge_attr_dim]
            
            K = self.Wk(x_src_concat).view(-1, self.heads, self.dh)  # [E, H, dh]
            V = self.Wv(x_src_concat).view(-1, self.heads, self.dh)  # [E, H, dh]
        else:
            # No edge features
            K = self.Wk(x).view(N, self.heads, self.dh)
            V = self.Wv(x).view(N, self.heads, self.dh)
            K = K[src]
            V = V[src]

        # Base attention scores
        e = (q_i * K).sum(dim=-1) * self.scale  # [E, H]

        # ISG Enhancement 1: Add dependency relation bias
        # This gives additional control even though edges are already in K,V
        if self.edge_bias is not None and edge_attr is not None:
            e = e + self.edge_bias(edge_attr)  # [E, H]

        # ISG Enhancement 2: Token distance gating
        if self.use_token_distance and token_distance is not None:
            dist_gate = self.distance_gate(token_distance.unsqueeze(-1))  # [E, H]
            e = e * dist_gate  # Multiplicative gating

        # Softmax per destination node, per head
        alphas = []
        for h in range(self.heads):
            alpha_h = pyg_segment_softmax(e[:, h], dst)
            alphas.append(alpha_h)
        alpha = torch.stack(alphas, dim=1)  # [E, H]
        alpha = self.drop_attn(alpha)

        # Aggregate
        out_heads = torch.zeros((N, self.heads, self.dh), device=x.device, dtype=x.dtype)
        out_heads.index_add_(0, dst, alpha.unsqueeze(-1) * V)

        out = out_heads.reshape(N, self.dim)
        out = self.Wo(out)
        return out
    
 

# *****************************************************************  
#                   Sentence_Aware
# *****************************************************************  

class SentenceAwareHybridAttention(nn.Module):
    """
    Atención híbrida que diferencia entre relaciones intra vs inter-oracionales
    
    Combina:
    1. Edge concatenation (Vanilla GTN)
    2. Edge bias (ISG)
    3. Distance gating (ISG)
    4. Sentence-level gating (NUEVO)
    """
    def __init__(self, dim: int, heads: int, attn_dropout: float = 0.1,
                 edge_attr_dim: Optional[int] = None,
                 use_token_distance: bool = True,
                 use_edge_bias: bool = True,
                 use_sentence_aware: bool = True):  # NUEVO
        super().__init__()
        assert dim % heads == 0, "hidden_dim must be divisible by heads"
        self.dim = dim
        self.heads = heads
        self.dh = dim // heads
        self.scale = 1.0 / math.sqrt(self.dh)
        self.edge_attr_dim = edge_attr_dim
        self.use_token_distance = use_token_distance
        self.use_edge_bias = use_edge_bias
        self.use_sentence_aware = use_sentence_aware
        
        # Query projection (destination nodes - no edge info)
        self.Wq = nn.Linear(dim, dim, bias=False)
        
        # Key and Value projections (source nodes - WITH edge concat)
        if edge_attr_dim is not None and edge_attr_dim > 0:
            self.Wk = nn.Linear(dim + edge_attr_dim, dim, bias=False)
            self.Wv = nn.Linear(dim + edge_attr_dim, dim, bias=False)
            
            if use_edge_bias:
                self.edge_bias = nn.Linear(edge_attr_dim, heads, bias=False)
            else:
                self.edge_bias = None
        else:
            self.Wk = nn.Linear(dim, dim, bias=False)
            self.Wv = nn.Linear(dim, dim, bias=False)
            self.edge_bias = None
        
        # ISG Enhancement: Token distance gating
        if use_token_distance:
            self.distance_gate = nn.Sequential(
                nn.Linear(1, heads),
                nn.Sigmoid()
            )
        
        # NUEVO: Sentence-level gating
        if use_sentence_aware:
            # Gate para modular atención intra vs inter-oracional
            self.sentence_gate = nn.Sequential(
                nn.Linear(dim * 2, heads),  # concat de src y dst features
                nn.Sigmoid()
            )
            # Bias aprendible diferente para intra vs inter
            self.intra_sentence_bias = nn.Parameter(torch.zeros(heads))
            self.inter_sentence_bias = nn.Parameter(torch.zeros(heads))
        
        self.Wo = nn.Linear(dim, dim, bias=False)
        self.drop_attn = nn.Dropout(attn_dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                token_distance: Optional[torch.Tensor] = None,
                sentence_ids: Optional[torch.Tensor] = None) -> torch.Tensor:  # NUEVO parámetro
        """
        x: [N, dim]
        edge_index: [2, E] (src, dst)
        edge_attr: [E, edge_attr_dim] - dependency embeddings
        token_distance: [E] - normalized inverse distances
        sentence_ids: [N] - sentence ID for each node (NUEVO)
        """
        N = x.size(0)
        src, dst = edge_index

        # Query from destination nodes (no edge info)
        Q = self.Wq(x).view(N, self.heads, self.dh)  # [N, H, dh]
        q_i = Q[dst]  # [E, H, dh]

        # Key and Value from source nodes WITH edge concatenation
        if self.edge_attr_dim is not None and edge_attr is not None:
            x_src = x[src]  # [E, dim]
            x_src_concat = torch.cat([x_src, edge_attr], dim=-1)  # [E, dim + edge_attr_dim]
            
            K = self.Wk(x_src_concat).view(-1, self.heads, self.dh)  # [E, H, dh]
            V = self.Wv(x_src_concat).view(-1, self.heads, self.dh)  # [E, H, dh]
        else:
            K = self.Wk(x).view(N, self.heads, self.dh)
            V = self.Wv(x).view(N, self.heads, self.dh)
            K = K[src]
            V = V[src]

        # Base attention scores
        e = (q_i * K).sum(dim=-1) * self.scale  # [E, H]

        # ISG Enhancement 1: Add dependency relation bias
        if self.edge_bias is not None and edge_attr is not None:
            e = e + self.edge_bias(edge_attr)  # [E, H]

        # ISG Enhancement 2: Token distance gating
        if self.use_token_distance and token_distance is not None:
            dist_gate = self.distance_gate(token_distance.unsqueeze(-1))  # [E, H]
            e = e * dist_gate

        # NUEVO Enhancement 3: Sentence-aware modulation
        if self.use_sentence_aware and sentence_ids is not None:
            # Determinar si src y dst están en la misma oración
            same_sentence = (sentence_ids[src] == sentence_ids[dst]).float()  # [E]
            
            # Aplicar bias diferencial
            # intra-sentence: bias positivo (fomentar atención)
            # inter-sentence: bias negativo (moderar atención)
            sentence_bias = (
                same_sentence.unsqueeze(-1) * self.intra_sentence_bias.view(1, -1) +
                (1 - same_sentence.unsqueeze(-1)) * self.inter_sentence_bias.view(1, -1)
            )  # [E, H]
            
            e = e + sentence_bias
            
            # Gate adaptativo basado en features de src y dst
            x_src_dst = torch.cat([x[src], x[dst]], dim=-1)  # [E, dim*2]
            sentence_gate = self.sentence_gate(x_src_dst)  # [E, H]
            
            # Modular scores: intra-sentence tiene gate más alto
            # El gate aprende cuánto atenuar las conexiones inter-sentence
            e = e * (same_sentence.unsqueeze(-1) + (1 - same_sentence.unsqueeze(-1)) * sentence_gate)

        # Softmax per destination node, per head
        alphas = []
        for h in range(self.heads):
            alpha_h = pyg_segment_softmax(e[:, h], dst)
            alphas.append(alpha_h)
        alpha = torch.stack(alphas, dim=1)  # [E, H]
        alpha = self.drop_attn(alpha)

        # Aggregate
        out_heads = torch.zeros((N, self.heads, self.dh), device=x.device, dtype=x.dtype)
        out_heads.index_add_(0, dst, alpha.unsqueeze(-1) * V)

        out = out_heads.reshape(N, self.dim)
        out = self.Wo(out)
        return out


class SentenceAwareTransformerLayer(nn.Module):
    """
    Transformer layer con atención sentence-aware
    """
    def __init__(self, dim: int, heads: int, attn_dropout: float,
                 ffn_dropout: float, ffn_mult: int, norm_type: str,
                 edge_attr_dim: Optional[int] = None,
                 use_token_distance: bool = True,
                 use_edge_bias: bool = True,
                 use_sentence_aware: bool = True):
        super().__init__()
        self.pre_attn = make_norm(norm_type, dim)
        self.attn = SentenceAwareHybridAttention(
            dim=dim, 
            heads=heads, 
            attn_dropout=attn_dropout,
            edge_attr_dim=edge_attr_dim,
            use_token_distance=use_token_distance,
            use_edge_bias=use_edge_bias,
            use_sentence_aware=use_sentence_aware
        )
        self.pre_ffn = make_norm(norm_type, dim)
        self.ffn = FeedForward(dim, mult=ffn_mult, dropout=ffn_dropout)

    def forward(self, x, edge_index, edge_attr=None, token_distance=None, sentence_ids=None):
        h = self.pre_attn(x)
        h = self.attn(h, edge_index, edge_attr, token_distance, sentence_ids)
        x = x + h

        h = self.pre_ffn(x)
        h = self.ffn(h)
        x = x + h
        return x


class EnhancedHybridISG_GTN(nn.Module):
    """
    Arquitectura híbrida mejorada con:
    1. Edge concatenation (Vanilla GTN)
    2. Edge bias (ISG)
    3. Distance gating (ISG)
    4. Sentence-aware attention (NUEVO)
    5. Dependency frequency features (NUEVO)
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 dense_hidden_dim: int,
                 output_dim: int,
                 dropout: float,
                 num_layers: int,
                 use_edge_attr: bool = False,
                 use_token_distance: bool = True,
                 use_edge_bias: bool = True,
                 use_sentence_aware: bool = True,  # NUEVO
                 heads: int = 4,
                 norm_type: str = "layernorm",
                 pooling_type: str = "mean",
                 post_mp_layers: int = 2,
                 edge_attr_dim: Optional[int] = None,
                 ffn_mult: int = 4,
                 attn_dropout: float = 0.1,
                 use_global_pooling: bool = False):
        super().__init__()

        self.pooling_type = pooling_type
        self.use_edge_attr = use_edge_attr
        self.use_token_distance = use_token_distance
        self.use_edge_bias = use_edge_bias
        self.use_sentence_aware = use_sentence_aware

        # Input projection
        self.in_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()

        # Stack of Enhanced layers
        self.layers = nn.ModuleList([
            SentenceAwareTransformerLayer(
                dim=hidden_dim,
                heads=heads,
                attn_dropout=attn_dropout,
                ffn_dropout=dropout,
                ffn_mult=ffn_mult,
                norm_type=norm_type,
                edge_attr_dim=(edge_attr_dim if use_edge_attr else None),
                use_token_distance=use_token_distance,
                use_edge_bias=use_edge_bias,
                use_sentence_aware=use_sentence_aware
            ) for _ in range(num_layers)
        ])

        # Pooling modules
        self.attn_pool = AttentionPooling(hidden_dim) if (pooling_type and pooling_type.lower() == "attention") else None
        self.set2set_op = Set2Set(hidden_dim, processing_steps=3) if (pooling_type and pooling_type.lower() == "set2set") else None

        # ISG Enhancement: Hierarchical global pooling
        if use_global_pooling:
            self.global_pool = HierarchicalGlobalPooling(hidden_dim)

        # Post-MP MLP
        mlp_dims = [hidden_dim] + [dense_hidden_dim] * max(0, post_mp_layers - 1) + [output_dim]
        mlp = []
        for li, (din, dout) in enumerate(zip(mlp_dims[:-1], mlp_dims[1:])):
            mlp.append(nn.Linear(din, dout))
            if li < len(mlp_dims) - 2:
                mlp.append(nn.GELU())
                mlp.append(nn.Dropout(dropout))
        self.graph_head = nn.Sequential(*mlp)

    def forward(self, x, edge_index, edge_attr, batch, token_distance=None, sentence_ids=None):
        h = self.in_proj(x)

        for layer in self.layers:
            h = layer(h, edge_index, edge_attr, token_distance, sentence_ids)

        # Standard pooling
        pooled = graph_pool(h, batch,
                            how=self.pooling_type,
                            attn_pool=self.attn_pool,
                            set2set_op=self.set2set_op)
        
        # ISG Enhancement: Add global context if enabled
        if hasattr(self, 'global_pool'):
            global_context = self.global_pool(h, batch)
            pooled = pooled + global_context

        logits = self.graph_head(pooled)
        return logits
    

    
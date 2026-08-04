"""
ablation_configs.py
Complete ablation study configurations for comparing:
- Vanilla GTN (no edges)
- Vanilla GTN (edge concat)
- ISG-GTN (original)
- Hybrid-ISG-GTN (combining both approaches)
"""

def get_ablation_configs(dataset_name='autext23'):
    """
    Generate comprehensive ablation study configurations
    
    Categories:
    A. Baseline experiments
    B. Vanilla GTN comparisons
    C. ISG feature contributions
    D. Hybrid architecture experiments
    E. Ablation studies (remove one component)
    """
    
    # Base configuration (shared parameters)
    base_config = {
        'graph_type': 'undirected',
        'dataset_name': dataset_name,
        'cut_off_dataset': '100_100_100',
        'cuda_num': 0,
        'build_graph': False,
        'balance_dataset': True,
        'max_features': 20000,
        'min_df': 3,
        'max_df': 1.0,
        'stop_words': False,
        'special_chars': False,
        'patience': 10,
        'hidden_gnn_dim': 300,
        'dense_hidden_gnn_dim': 64,
        'num_gnn_layers': 1,
        'heads_gnn': 1,
        'dropout': 0.5,
        'lr': 0.00001,
        'norm_type': 'batchnorm',
        'post_mp_layers': 3,
        'lang_model_name': 'microsoft/deberta-v3-base',
        'leave_out_sources': True,
        'reduce_dim_emb': False,
        'project_after_concat': True,
        'reduced_dim': 256,
        'add_domain_feat': False,
    }
    
    experiments = {}
    
    experiments['A0_build_graph_extract_feat'] = {
        **base_config,
        'gnn_type': 'ISG-VanillaGTN',
        'edge_attr': True,
        'add_pos_feat': True,
        'use_lap_pe': True,
        'lap_k': 8,
        'undirected_for_lap': True,
        'use_degree_pe': True,
        'use_rw_pe': False,
        'use_isg_structural_pe': True,
        'use_token_distance_attn': True,
        'use_global_pooling': True,
        'pooling_type': 'mean',
        'description': 'Hybrid Full - build_graph_extract_feat'
    }
    
    # ==================== A. BASELINE ====================
    experiments['A1_minimal_baseline'] = {
        **base_config,
        'gnn_type': 'Vanilla-GTN-NoEdge',
        'edge_attr': False,
        'add_pos_feat': False,
        'use_lap_pe': False,
        'use_degree_pe': False,
        'use_rw_pe': False,
        'use_isg_structural_pe': False,
        'use_token_distance_attn': False,
        'use_global_pooling': False,
        'pooling_type': 'mean',
        'description': 'Minimal Baseline: LLM embeddings only, no edges, no PEs'
    }
    
    experiments['A2_baseline_with_lap_pe'] = {
        **base_config,
        'gnn_type': 'Vanilla-GTN-NoEdge',
        'edge_attr': False,
        'add_pos_feat': False,
        'use_lap_pe': True,
        'lap_k': 8,
        'undirected_for_lap': True,
        'use_degree_pe': False,
        'use_rw_pe': False,
        'use_isg_structural_pe': False,
        'use_token_distance_attn': False,
        'use_global_pooling': False,
        'pooling_type': 'mean',
        'description': 'Baseline + Laplacian PE (like paper LEFT diagram)'
    }
    
    # ==================== B. VANILLA GTN COMPARISONS ====================
    experiments['B1_vanilla_gtn_no_edges'] = {
        **base_config,
        'gnn_type': 'Vanilla-GTN-NoEdge',
        'edge_attr': False,
        'add_pos_feat': False,
        'use_lap_pe': True,
        'lap_k': 8,
        'undirected_for_lap': True,
        'use_degree_pe': False,
        'use_rw_pe': False,
        'use_isg_structural_pe': False,
        'use_token_distance_attn': False,
        'use_global_pooling': False,
        'pooling_type': 'mean',
        'description': 'Vanilla GTN LEFT: No edge features, Lap PE only'
    }
    
    experiments['B2_vanilla_gtn_edge_concat'] = {
        **base_config,
        'gnn_type': 'Vanilla-GTN-EdgeConcat',
        'edge_attr': True,
        'add_pos_feat': False,
        'use_lap_pe': True,
        'lap_k': 8,
        'undirected_for_lap': True,
        'use_degree_pe': False,
        'use_rw_pe': False,
        'use_isg_structural_pe': False,
        'use_token_distance_attn': False,
        'use_global_pooling': False,
        'pooling_type': 'mean',
        'description': 'Vanilla GTN RIGHT: Edge concat to source nodes'
    }
    
    # ==================== C. ISG FEATURE CONTRIBUTIONS ====================
    experiments['C1_pos_tags_only'] = {
        **base_config,
        'gnn_type': 'ISG-VanillaGTN',
        'edge_attr': True,
        'add_pos_feat': True,
        'use_lap_pe': False,
        'use_degree_pe': False,
        'use_rw_pe': False,
        'use_isg_structural_pe': False,
        'use_token_distance_attn': False,
        'use_global_pooling': False,
        'pooling_type': 'mean',
        'description': 'ISG: POS tags + dependency embeddings only'
    }
    
    experiments['C2_token_distance_only'] = {
        **base_config,
        'gnn_type': 'ISG-VanillaGTN',
        'edge_attr': True,
        'add_pos_feat': False,
        'use_lap_pe': False,
        'use_degree_pe': False,
        'use_rw_pe': False,
        'use_isg_structural_pe': False,
        'use_token_distance_attn': True,
        'use_global_pooling': False,
        'pooling_type': 'mean',
        'description': 'ISG: Token distance gating only'
    }
    
    experiments['C3_isg_structural_pe_only'] = {
        **base_config,
        'gnn_type': 'ISG-VanillaGTN',
        'edge_attr': True,
        'add_pos_feat': False,
        'use_lap_pe': False,
        'use_degree_pe': False,
        'use_rw_pe': False,
        'use_isg_structural_pe': True,
        'use_token_distance_attn': False,
        'use_global_pooling': False,
        'pooling_type': 'mean',
        'description': 'ISG: Structural PEs only (root_depth, betweenness, clustering, dep_diversity)'
    }
    
    experiments['C4_all_standard_pe'] = {
        **base_config,
        'gnn_type': 'ISG-VanillaGTN',
        'edge_attr': True,
        'add_pos_feat': False,
        'use_lap_pe': True,
        'lap_k': 8,
        'undirected_for_lap': True,
        'use_degree_pe': True,
        'use_rw_pe': False,
        'use_isg_structural_pe': False,
        'use_token_distance_attn': False,
        'use_global_pooling': False,
        'pooling_type': 'mean',
        'description': 'ISG: All standard PEs (Lap + Degree)'
    }
    
    experiments['C5_all_pe_combined'] = {
        **base_config,
        'gnn_type': 'ISG-VanillaGTN',
        'edge_attr': True,
        'add_pos_feat': False,
        'use_lap_pe': True,
        'lap_k': 8,
        'undirected_for_lap': True,
        'use_degree_pe': True,
        'use_rw_pe': False,
        'use_isg_structural_pe': True,
        'use_token_distance_attn': False,
        'use_global_pooling': False,
        'pooling_type': 'mean',
        'description': 'ISG: All PEs (Standard + ISG Structural)'
    }
    
    experiments['C6_pooling_attention'] = {
        **base_config,
        'gnn_type': 'ISG-VanillaGTN',
        'edge_attr': True,
        'add_pos_feat': False,
        'use_lap_pe': True,
        'lap_k': 8,
        'undirected_for_lap': True,
        'use_degree_pe': False,
        'use_rw_pe': False,
        'use_isg_structural_pe': False,
        'use_token_distance_attn': False,
        'use_global_pooling': False,
        'pooling_type': 'attention',
        'description': 'ISG: Attention pooling instead of mean'
    }
    
    experiments['C7_pooling_global'] = {
        **base_config,
        'gnn_type': 'ISG-VanillaGTN',
        'edge_attr': True,
        'add_pos_feat': False,
        'use_lap_pe': True,
        'lap_k': 8,
        'undirected_for_lap': True,
        'use_degree_pe': False,
        'use_rw_pe': False,
        'use_isg_structural_pe': False,
        'use_token_distance_attn': False,
        'use_global_pooling': True,
        'pooling_type': 'mean',
        'description': 'ISG: Hierarchical global pooling'
    }
    
    experiments['C8_isg_full_original'] = {
        **base_config,
        'gnn_type': 'ISG-VanillaGTN',
        'edge_attr': True,
        'add_pos_feat': True,
        'use_lap_pe': True,
        'lap_k': 8,
        'undirected_for_lap': True,
        'use_degree_pe': True,
        'use_rw_pe': False,
        'use_isg_structural_pe': True,
        'use_token_distance_attn': True,
        'use_global_pooling': True,
        'pooling_type': 'attention',
        'description': 'ISG-GTN FULL: All ISG features enabled'
    }
    
    # ==================== D. HYBRID ARCHITECTURE EXPERIMENTS ====================
    experiments['D1_hybrid_edge_concat_only'] = {
        **base_config,
        'gnn_type': 'Hybrid-ISG-GTN',
        'edge_attr': True,
        'add_pos_feat': False,
        'use_lap_pe': True,
        'lap_k': 8,
        'undirected_for_lap': True,
        'use_degree_pe': False,
        'use_rw_pe': False,
        'use_isg_structural_pe': False,
        'use_token_distance_attn': False,
        'use_global_pooling': False,
        'pooling_type': 'mean',
        'description': 'Hybrid: Edge concat only (like Vanilla RIGHT)'
    }
    
    experiments['D2_hybrid_plus_pos'] = {
        **base_config,
        'gnn_type': 'Hybrid-ISG-GTN',
        'edge_attr': True,
        'add_pos_feat': True,
        'use_lap_pe': True,
        'lap_k': 8,
        'undirected_for_lap': True,
        'use_degree_pe': False,
        'use_rw_pe': False,
        'use_isg_structural_pe': False,
        'use_token_distance_attn': False,
        'use_global_pooling': False,
        'pooling_type': 'mean',
        'description': 'Hybrid: Edge concat + POS tags + Lap PE'
    }
    
    experiments['D3_hybrid_plus_token_distance'] = {
        **base_config,
        'gnn_type': 'Hybrid-ISG-GTN',
        'edge_attr': True,
        'add_pos_feat': True,
        'use_lap_pe': True,
        'lap_k': 8,
        'undirected_for_lap': True,
        'use_degree_pe': False,
        'use_rw_pe': False,
        'use_isg_structural_pe': False,
        'use_token_distance_attn': True,
        'use_global_pooling': False,
        'pooling_type': 'mean',
        'description': 'Hybrid: Edge concat + POS + Lap PE + Token distance'
    }
    
    experiments['D4_hybrid_plus_structural_pe'] = {
        **base_config,
        'gnn_type': 'Hybrid-ISG-GTN',
        'edge_attr': True,
        'add_pos_feat': True,
        'use_lap_pe': True,
        'lap_k': 8,
        'undirected_for_lap': True,
        'use_degree_pe': True,
        'use_rw_pe': False,
        'use_isg_structural_pe': True,
        'use_token_distance_attn': True,
        'use_global_pooling': False,
        'pooling_type': 'mean',
        'description': 'Hybrid: Edge concat + All PEs + Token distance'
    }
    
    experiments['D5_hybrid_full'] = {
        **base_config,
        'gnn_type': 'Hybrid-ISG-GTN',
        'edge_attr': True,
        'add_pos_feat': True,
        'use_lap_pe': True,
        'lap_k': 8,
        'undirected_for_lap': True,
        'use_degree_pe': True,
        'use_rw_pe': False,
        'use_isg_structural_pe': True,
        'use_token_distance_attn': True,
        'use_global_pooling': True,
        'pooling_type': 'attention',
        'description': 'Hybrid FULL: Everything enabled (edge concat + all ISG features)'
    }
    
    # ==================== E. ABLATION STUDIES (Remove Components) ====================
    experiments['E1_hybrid_minus_pos'] = {
        **base_config,
        'gnn_type': 'Hybrid-ISG-GTN',
        'edge_attr': True,
        'add_pos_feat': False,
        'use_lap_pe': True,
        'lap_k': 8,
        'undirected_for_lap': True,
        'use_degree_pe': True,
        'use_rw_pe': False,
        'use_isg_structural_pe': True,
        'use_token_distance_attn': True,
        'use_global_pooling': True,
        'pooling_type': 'attention',
        'description': 'Hybrid Full - POS tags'
    }
    
    experiments['E2_hybrid_minus_token_distance'] = {
        **base_config,
        'gnn_type': 'Hybrid-ISG-GTN',
        'edge_attr': True,
        'add_pos_feat': True,
        'use_lap_pe': True,
        'lap_k': 8,
        'undirected_for_lap': True,
        'use_degree_pe': True,
        'use_rw_pe': False,
        'use_isg_structural_pe': True,
        'use_token_distance_attn': False,
        'use_global_pooling': True,
        'pooling_type': 'attention',
        'description': 'Hybrid Full - Token distance'
    }
    
    experiments['E3_hybrid_minus_structural_pe'] = {
        **base_config,
        'gnn_type': 'Hybrid-ISG-GTN',
        'edge_attr': True,
        'add_pos_feat': True,
        'use_lap_pe': True,
        'lap_k': 8,
        'undirected_for_lap': True,
        'use_degree_pe': True,
        'use_rw_pe': False,
        'use_isg_structural_pe': False,
        'use_token_distance_attn': True,
        'use_global_pooling': True,
        'pooling_type': 'attention',
        'description': 'Hybrid Full - ISG structural PEs'
    }
    
    experiments['E4_hybrid_minus_global_pooling'] = {
        **base_config,
        'gnn_type': 'Hybrid-ISG-GTN',
        'edge_attr': True,
        'add_pos_feat': True,
        'use_lap_pe': True,
        'lap_k': 8,
        'undirected_for_lap': True,
        'use_degree_pe': True,
        'use_rw_pe': False,
        'use_isg_structural_pe': True,
        'use_token_distance_attn': True,
        'use_global_pooling': False,
        'pooling_type': 'attention',
        'description': 'Hybrid Full - Global pooling'
    }
    
    experiments['E5_hybrid_minus_edge_concat'] = {
        **base_config,
        'gnn_type': 'ISG-VanillaGTN',
        'edge_attr': True,
        'add_pos_feat': True,
        'use_lap_pe': True,
        'lap_k': 8,
        'undirected_for_lap': True,
        'use_degree_pe': True,
        'use_rw_pe': False,
        'use_isg_structural_pe': True,
        'use_token_distance_attn': True,
        'use_global_pooling': True,
        'pooling_type': 'attention',
        'description': 'Hybrid Full - Edge concat (use ISG model instead)'
    }
    
    return experiments


def get_quick_ablation_configs(dataset_name='autext23'):
    """
    Quick ablation study (7 key experiments)
    Run time: ~2-3 hours
    """
    base_config = {
        'graph_type': 'undirected',
        'dataset_name': dataset_name,
        'cut_off_dataset': '50_50_50',
        'cuda_num': 1,
        'build_graph': True,
        'balance_dataset': True,
        'max_features': 20000,
        'min_df': 3,
        'max_df': 1.0,
        'stop_words': False,
        'special_chars': False,
        'patience': 10,
        'hidden_gnn_dim': 300,
        'dense_hidden_gnn_dim': 64,
        'num_gnn_layers': 1,
        'heads_gnn': 1,
        'dropout': 0.5,
        'lr': 0.00001,
        'norm_type': 'batchnorm',
        'post_mp_layers': 3,
        'lang_model_name': 'microsoft/deberta-v3-base',
        'leave_out_sources': True,
        'reduce_dim_emb': False,
        'project_after_concat': True,
        'reduced_dim': 256,
        'add_domain_feat': False,
    }
    
    quick_experiments = {
        'Q0_hybrid_full': {
            **base_config,
            'gnn_type': 'Hybrid-ISG-GTN',
            'edge_attr': True,
            'add_pos_feat': True,
            'use_lap_pe': True,
            'lap_k': 8,
            'undirected_for_lap': True,
            'use_degree_pe': True,
            'use_rw_pe': False,
            'use_isg_structural_pe': True,
            'use_token_distance_attn': True,
            'use_global_pooling': True,
            'pooling_type': 'attention',
            'description': 'Hybrid: Full configuration (BEST)'
        },
        
        'Q1_vanilla_edge_concat': {
            **base_config,
            'gnn_type': 'Vanilla-GTN-EdgeConcat',
            'edge_attr': True,
            'add_pos_feat': False,
            'use_lap_pe': True,
            'lap_k': 8,
            'undirected_for_lap': True,
            'use_degree_pe': False,
            'use_rw_pe': False,
            'use_isg_structural_pe': False,
            'use_token_distance_attn': False,
            'use_global_pooling': False,
            'pooling_type': 'mean',
            'description': 'Vanilla GTN: Edge concatenation'
        },
        
        'Q2_isg_basic': {
            **base_config,
            'gnn_type': 'ISG-VanillaGTN',
            'edge_attr': True,
            'add_pos_feat': True,
            'use_lap_pe': True,
            'lap_k': 8,
            'undirected_for_lap': True,
            'use_degree_pe': True,
            'use_rw_pe': False,
            'use_isg_structural_pe': True,
            'use_token_distance_attn': True,
            'use_global_pooling': False,
            'pooling_type': 'mean',
            'description': 'ISG-GTN: Core ISG features'
        },
        
        'Q3_isg_full': {
            **base_config,
            'gnn_type': 'ISG-VanillaGTN',
            'edge_attr': True,
            'add_pos_feat': True,
            'use_lap_pe': True,
            'lap_k': 8,
            'undirected_for_lap': True,
            'use_degree_pe': True,
            'use_rw_pe': False,
            'use_isg_structural_pe': True,
            'use_token_distance_attn': True,
            'use_global_pooling': True,
            'pooling_type': 'attention',
            'description': 'ISG-GTN: Full configuration'
        },
        
        'Q4_hybrid_basic': {
            **base_config,
            'gnn_type': 'Hybrid-ISG-GTN',
            'edge_attr': True,
            'add_pos_feat': True,
            'use_lap_pe': True,
            'lap_k': 8,
            'undirected_for_lap': True,
            'use_degree_pe': True,
            'use_rw_pe': False,
            'use_isg_structural_pe': True,
            'use_token_distance_attn': True,
            'use_global_pooling': False,
            'pooling_type': 'mean',
            'description': 'Hybrid: Edge concat + ISG features'
        },
        
        'Q5_baseline': {
            **base_config,
            'gnn_type': 'Vanilla-GTN-NoEdge',
            'edge_attr': False,
            'add_pos_feat': False,
            'use_lap_pe': True,
            'lap_k': 8,
            'undirected_for_lap': True,
            'use_degree_pe': False,
            'use_rw_pe': False,
            'use_isg_structural_pe': False,
            'use_token_distance_attn': False,
            'use_global_pooling': False,
            'pooling_type': 'mean',
            'description': 'Baseline: Vanilla GTN no edges'
        },
    }
    
    return quick_experiments


def get_custom_configs(dataset_name, experiment_names):
    """
    Get specific experiments by name
    
    Args:
        dataset_name: Dataset to use
        experiment_names: List of experiment names (e.g., ['B1_vanilla_gtn_no_edges', 'D5_hybrid_full'])
    """
    all_configs = get_ablation_configs(dataset_name)
    return {name: all_configs[name] for name in experiment_names if name in all_configs}


if __name__ == "__main__":
    # Test configurations
    configs = get_ablation_configs('autext23')
    print(f"Total experiments: {len(configs)}")
    print("\nExperiment names:")
    for name, config in configs.items():
        print(f"  {name}: {config['description']}")
    
    print(f"\n\nQuick ablation: {len(get_quick_ablation_configs('autext23'))} experiments")
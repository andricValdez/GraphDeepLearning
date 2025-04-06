import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Example results: Replace with actual experiment results
results = [
    {'graph_direction': 'undirected', 'add_edge_attr': True, 'gnn_type': 'TransformerConv', 'num_layers': 1, 'accuracy': 0.85, 'f1_macro': 0.83},
    {'graph_direction': 'directed', 'add_edge_attr': False, 'gnn_type': 'GCNConv', 'num_layers': 2, 'accuracy': 0.82, 'f1_macro': 0.80},
    {'graph_direction': 'undirected', 'add_edge_attr': False, 'gnn_type': 'GATConv', 'num_layers': 3, 'accuracy': 0.87, 'f1_macro': 0.86},
    {'graph_direction': 'directed', 'add_edge_attr': True, 'gnn_type': 'TransformerConv', 'num_layers': 2, 'accuracy': 0.84, 'f1_macro': 0.83},
    {'graph_direction': 'undirected', 'add_edge_attr': True, 'gnn_type': 'GCNConv', 'num_layers': 3, 'accuracy': 0.86, 'f1_macro': 0.85}
]

# Convert to DataFrame
df = pd.DataFrame(results)

# Set visualization style
sns.set(style="whitegrid")

# Plot performance comparison
plt.figure(figsize=(12, 6))
sns.barplot(x='gnn_type', y='accuracy', hue='graph_direction', data=df)

# Customize plot
plt.title("Performance Comparison of Co-Occurrence Graph Configurations")
plt.xlabel("GNN Type")
plt.ylabel("Accuracy")
plt.legend(title="Graph Direction")
plt.xticks(rotation=30)

# Show plot
plt.show()

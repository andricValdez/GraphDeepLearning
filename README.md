# GraphDeepLearning

This repository provides code and experiments for **Graph Deep Learning on textual data**, focusing on transforming documents into graph structures and applying **Graph Neural Networks (GNNs)** for classification tasks such as **AI-generated text detection** and **authorship analysis**.

---


## Key Features

- Support for **co-occurrence graphs** and **heterogeneous text graphs**
- Graph construction using a sliding window (default window size = 10)
- Node features initialized from **random**, **Word2Vec**, or **pretrained language models** (e.g., BERT)
- Graph Transformer Network (GTN) and standard GNN architectures
- Baseline comparisons with fine-tuned models like **BERT** and **RoBERTa**
- Scalable training with PyTorch Geometric on large datasets (e.g., COLING, SemEval)

---

## Installation

```bash
git clone https://github.com/andricValdez/GraphDeepLearning.git
cd GraphDeepLearning
pip install -r requirements.txt
```
---
See requirements.txt for all dependencies. Key packages include:

torch

torch_geometric

transformers

networkx

scikit-learn

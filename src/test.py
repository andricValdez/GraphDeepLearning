import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
import spacy
from sklearn.model_selection import train_test_split

import utils
import test_utils




nlp = spacy.load("en_core_web_sm")

text = ("Neural networks can detect patterns in complex data. "
        "They are often used in image recognition tasks. "
        "Training deep models requires significant computational resources.")

doc = nlp(text)

records = []
for sent in doc.sents:
    for token in sent:
        records.append({
            "Sentence": sent.text,
            "Token": token.text,
            "Lemma": token.lemma_,
            "POS": token.pos_,
            "Tag": token.tag_,
            "Dep": token.dep_,
            "Head": token.head.text
        })

df = pd.DataFrame(records)
print(df)


'''
text = "Neural networks can detect patterns in complex data. They are often used in image recognition tasks. Training deep models requires significant computational resources."

text = test_utils.text_normalize(text, special_chars=False, stop_words=False) 
print(text)

nlp = spacy.load("en_core_web_sm")
doc = nlp(text)

for token in doc:
    print(f"'{token.text}_{token.pos_}'", end=' | ')
print("\n")
for sent in doc.sents:
    print(sent)

'''

'''
print(40*'*', 'Dataset Distro-Partition')
subtask = 'subtask1' # subtask1, subtask2

# atext23: symanto/autextification2023
# atext24: Genaios/iberautextification 
#dataset = load_dataset("symanto/autextification2023, 'detection_en') # ['detection_en', 'attribution_en', 'detection_es', 'attribution_es']
dataset = load_dataset("Genaios/iberautextification", 'detection') # ['detection', 'attribution']

train_set = pd.DataFrame(dataset['train'])
autext_test_set = pd.DataFrame(dataset['test'])
autext_train_set, autext_val_set = train_test_split(train_set, test_size=0.3)

print("autext_train_set: ", autext_train_set.info())
print("Label distribution per source in Train set:\n", autext_train_set.groupby("domain")["label"].value_counts())

print("autext_val_set:  ", autext_val_set.info())
print("Label distribution per source in Val set:\n", autext_val_set.groupby("domain")["label"].value_counts())

print("autext_test_set:  ", autext_test_set.info())
print("Label distribution per source in Test set:\n", autext_test_set.groupby("domain")["label"].value_counts())

#autext_train_set.to_csv(f'{utils.DATASET_DIR}autext2024/{subtask}/train_set.csv')
#autext_val_set.to_csv(f'{utils.DATASET_DIR}autext2024/{subtask}/val_set.csv')
#autext_test_set.to_csv(f'{utils.DATASET_DIR}autext2024/{subtask}/test_set.csv')
'''


'''
import os
import yaml

experiment_path = "mlruns/240013343155012552"  # Adjust for your experiment ID

for run_id in os.listdir(experiment_path):
    meta_path = os.path.join(experiment_path, run_id, "meta.yaml")
    try:
        with open(meta_path) as f:
            meta = yaml.safe_load(f)
            assert isinstance(meta, dict)
    except Exception:
        print(f"Corrupted run: {run_id}")
        # os.remove(meta_path)  # or delete full run dir manually
        # 
'''
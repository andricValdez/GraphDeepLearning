import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from tqdm import tqdm  # Import tqdm for progress bar

import test_utils



class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Initialize tokenization with tqdm progress bar
        self.encodings = []
        for text in tqdm(self.texts, desc="Tokenizing texts", leave=False):  # Add tqdm here
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            self.encodings.append(encoding)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.encodings[idx]
        label = self.labels[idx]
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }

def evaluate(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # Get raw logits
            preds = torch.argmax(logits, dim=1)  # Convert logits to class predictions

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")

    return acc, f1


# ************************************************** autext23
# andricValdez/roberta-base-finetuned-autext23
    #Test Accuracy: 0.6184                                                                                                                                                                                                                           
    #Test F1 Score (Macro): 0.5435
# andricValdez/deberta-v3-base-finetuned-autext23

# ************************************************** semeval24
# andricValdez/roberta-base-finetuned-semeval24
    # Test Accuracy: 0.8425                                                                                        
    # Test F1 Score (Macro): 0.8418
# andricValdez/deberta-v3-base-finetuned-semeval24     
    # Test Accuracy: 0.7090                                                                                       
    # Test F1 Score (Macro): 0.6728
    # {'test_loss': 1.1749110221862793, 'test_accuracy': 0.8346361996392062, 'test_f1': 0.8345944012408657, 'test_runtime': 78.9471, 'test_samples_per_second': 63.194, 'test_steps_per_second': 3.952}

# ************************************************** coling24
# andricValdez/roberta-base-finetuned-coling24
    # Test Accuracy: 0.7323                                                                                        
    # Test F1 Score (Macro): 0.7080
    # From shared task (roberta-FT) F1-macro:0.7342
# andricValdez/deberta-v3-base-finetuned-coling24      
    # Test Accuracy: 0.6793                                                                                                                                                                                                                           
    # Test F1 Score (Macro): 0.6374   
    

MODEL_NAME = "andricValdez/deberta-v3-base-finetuned-semeval24"  # Change to "microsoft/deberta-v3-base" if needed
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)  # Binary Classification
batch_size = 64

cuda_num = 1
device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# autext23, semeval24, coling24, autext23_s2, semeval24_s2
dataset_name = 'semeval24'
cut_off_dataset = 100
train_text_set, val_text_set, test_text_set  = test_utils.read_dataset(dataset_name)

# *** TEST
cut_dataset_test = len(test_text_set) * (int(cut_off_dataset) / 100)
test_set = test_text_set[:int(cut_dataset_test)]

print("distro_test: ", len(test_set))
print("label_distro_test: ", test_set.value_counts('label'))
        
test_texts = list(test_set['text'])[:]
test_labels = list(test_set['label'])[:]

test_dataset = TextDataset(test_texts, test_labels, tokenizer)
print(test_dataset)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print(test_loader)

print("MODEL_NAME: ", MODEL_NAME)
print("dataset_name: ", dataset_name)
print("cut_off_dataset: ", cut_off_dataset)

# -------------------------------
# Run Evaluation
# -------------------------------
accuracy, f1_macro = evaluate(model, test_loader, device)
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test F1 Score (Macro): {f1_macro:.4f}")

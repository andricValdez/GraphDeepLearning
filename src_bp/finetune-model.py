from datasets import Dataset
import pandas as pd
import evaluate
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, AutoTokenizer, set_seed
import os
import argparse
import logging
from sklearn.metrics import accuracy_score, f1_score

import utils
import test_utils

def preprocess_function(examples, **fn_kwargs):
    return fn_kwargs['tokenizer'](examples["text"], truncation=True)


def get_data(train_path, dev_path, test_path, random_seed):
    """
    function to read dataframe with columns
    """

    train_df = pd.read_json(train_path, lines=True)
    val_df = pd.read_json(dev_path, lines=True)
    test_df = pd.read_json(test_path, lines=True)
    
    return train_df, val_df, test_df

def compute_metrics(eval_pred):

    f1_metric = evaluate.load("f1")

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    results = {}
    results.update(f1_metric.compute(predictions=predictions, references = labels, average="micro"))

    return results

def llm_compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}


def fine_tune(train_df, valid_df, id2label, label2id, model, model_name):

    # pandas dataframe to huggingface Dataset
    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)
    
    # get tokenizer and model from huggingface
    tokenizer = AutoTokenizer.from_pretrained(model)     
    model = AutoModelForSequenceClassification.from_pretrained(
       model, num_labels=len(label2id), id2label=id2label, label2id=label2id 
    )
    
    # tokenize data for train/valid
    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True, fn_kwargs={'tokenizer': tokenizer})
    tokenized_valid_dataset = valid_dataset.map(preprocess_function, batched=True,  fn_kwargs={'tokenizer': tokenizer})
    

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


    # create Trainer 
    training_args = TrainingArguments(
        output_dir=utils.OUTPUT_DIR_PATH + 'finetuned_hf_models/' + model_name,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=llm_compute_metrics,
    )

    trainer.train()

    # Optional: Push to Hub
    model.push_to_hub(model_name)
    tokenizer.push_to_hub(model_name)

    return model, tokenizer
 


def test(test_df, model_name, id2label, label2id):
    
    # load tokenizer from saved model 
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # load best model
    model = AutoModelForSequenceClassification.from_pretrained(
       model_name, num_labels=len(label2id), id2label=id2label, label2id=label2id
    )
            
    test_dataset = Dataset.from_pandas(test_df)

    tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True,  fn_kwargs={'tokenizer': tokenizer})
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # create Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=llm_compute_metrics,
    )
    # get logits from predictions and evaluate results using classification report
    predictions = trainer.predict(tokenized_test_dataset)

    #preds = np.argmax(predictions.predictions, axis=-1)
    #metric = evaluate.load("bstrai/classification_report")
    #results = metric.compute(predictions=preds, references=predictions.label_ids)
        
    # return dictionary of classification report
    return predictions

# python baseline.py --train_file_path en_train.jsonl --dev_file_path en_dev.jsonl --test_file_path en_devtest.jsonl --model roberta-base --prediction_file_path en_prediction.jsonl
# python baseline.py --train_file_path data/multilingual_train.jsonl --dev_file_path data/multilingual_dev.jsonl --test_file_path data/multilingual_devtest.jsonl --model xlm-roberta-base --prediction_file data/multilingual_prediction.jsonl
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    dataset_name = "autext24"
    model = 'microsoft/deberta-v3-base' # FacebookAI/roberta-base | microsoft/deberta-v3-base
    model_name = f"{model.split('/')[1]}-finetuned-{dataset_name}"   

    train_df, valid_df, test_df = test_utils.read_dataset(dataset_name)
    train_df = train_df[train_df['language'] == 'en'].reset_index(drop=True)
    valid_df = valid_df[valid_df['language'] == 'en'].reset_index(drop=True)
    test_df = test_df[test_df['language'] == 'en']

    leave_out_sources = ["news", "literary"] 
    train_keep = train_df[~train_df['source'].isin(leave_out_sources)]
    train_swap = train_df[train_df['source'].isin(leave_out_sources)]
    val_keep = valid_df[valid_df['source'].isin(leave_out_sources)]
    val_swap = valid_df[~valid_df['source'].isin(leave_out_sources)]
    train_text_set = pd.concat([train_keep, val_swap], ignore_index=True)
    val_text_set = pd.concat([val_keep, train_swap], ignore_index=True)
    print("Train set distro:\n", train_text_set.groupby("source")["label"].value_counts())
    print("Val set distro:\n", val_text_set.groupby("source")["label"].value_counts())

    
    id2label = {0: "human", 1: "machine"}
    label2id = {"human": 0, "machine": 1}

    set_seed(42)

    # train detector model
    fine_tune(train_df, valid_df, id2label, label2id, model, model_name)

    # test detector model
    model = 'andricValdez/' + model_name
    predictions = test(test_df, model, id2label, label2id)
    print(predictions.metrics)
    

# SVM/Linear - full dataset - YES cross domain train
#Matriz Confusion val: 
#[[5307  517]
# [2533 3831]]
#Accuracy test: 0.6610646978822615
#F1Score test: 0.6597121035355125

# w2vect - full dataset - YES cross domain train
# Precision: 0.639 / Recall: 0.838 / Accuracy: 0.661 / F1-Score: 0.642

# BertGCN - full dataset - YES cross domain train
# Epoch: 38  Train acc: 0.9224 loss: 0.1251  Val acc: 0.8012 loss: 0.6520  Test acc: 0.6280 Test f1score: 0.6247 loss: 0.9427

# roberta - full dataset - YES cross domain train ["news", "literary"] - NO finetune 
#{'test_loss': 0.7092229723930359, 'test_accuracy': 0.4668739071303672, 'test_f1': 0.29719152287000594, 'test_runtime': 141.4253, 'test_samples_per_second': 72.788, 'test_steps_per_second': 4.554}

# roberta - full dataset - NO cross domain train - finetune
# {'test_loss': 1.4936410188674927, 'test_accuracy': 0.6923450553720614, 'test_f1': 0.655443984391313, 'test_runtime': 79.6058, 'test_samples_per_second': 129.312, 'test_steps_per_second': 8.09}

# roberta - full dataset - YES cross domain train ["news", "literary"] - finetune
# {'test_loss': 0.9188358187675476, 'test_accuracy': 0.7624829998057121, 'test_f1': 0.7565716481706146, 'test_runtime': 94.2855, 'test_samples_per_second': 109.179, 'test_steps_per_second': 6.83}

# deberta - full dataset - NO cross domain train - finetune
# {'test_loss': 1.77083420753479, 'test_accuracy': 0.6452302312026423, 'test_f1': 0.5839551673145649, 'test_runtime': 112.9236, 'test_samples_per_second': 91.159, 'test_steps_per_second': 5.703}

# deberta - full dataset - YES cross domain train ["news", "literary"] - finetune
#{'test_loss': 0.9443115592002869, 'test_accuracy': 0.7875461433844958, 'test_f1': 0.7797360766274593, 'test_runtime': 113.3359, 'test_samples_per_second': 90.827, 'test_steps_per_second': 5.682}

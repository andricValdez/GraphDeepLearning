
from datasets import load_dataset, Dataset, DatasetDict
from transformers import logging
from transformers import AutoTokenizer, AutoModel, Trainer, AutoModelForSequenceClassification, TrainingArguments
from transformers import TrainingArguments, Trainer
from transformers import get_scheduler
import numpy as np
import evaluate
from torch.utils.data import DataLoader 
from torch.optim import AdamW
import torch
from collections import defaultdict
from tqdm.auto import tqdm
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import evaluate
import pandas as pd
import gc
import gensim
import re 

import utils

logging.set_verbosity_warning()
INDEX = 0

# google-bert/bert-base-cased
# FacebookAI/roberta-base
# google-bert/bert-base-multilingual-cased      multiligual
# FacebookAI/xlm-roberta-base                   multiligual
# intfloat/multilingual-e5-large                multiligual
LLM_HF_NAME = "FacebookAI/xlm-roberta-base"

# andricValdez/multilingual-e5-large-finetuned-autext24
# andricValdez/bert-base-multilingual-cased-finetuned-autext24
# andricValdez/bert-base-multilingual-cased-finetuned-autext24-subtask2
# andricValdez/multilingual-e5-large-finetuned-autext24-subtask2
LLM_HF_FINETUNED_NAME = "andricValdez/bert-base-multilingual-cased-finetuned-autext24"


def llm_tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], padding=True, truncation=True, return_tensors='pt', max_length=512)
 

def llm_compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}


def llm_fine_tuning(model_name, train_set_df, val_set_df, device, llm_to_finetune=LLM_HF_NAME, mode='finetune', num_labels=2):

    model_name = f"{llm_to_finetune}-finetuned-{model_name}"   
    dataset = DatasetDict({
        "train": Dataset.from_dict(train_set_df),
        "validation": Dataset.from_dict(pd.DataFrame(data=val_set_df))
    })
    print(dataset)
    tokenizer = AutoTokenizer.from_pretrained(llm_to_finetune) 
    tokenized_dataset = dataset.map(llm_tokenize_function, batched=True, fn_kwargs={"tokenizer": tokenizer})
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")
    print(tokenized_dataset)

    batch_size = 16
    model = (AutoModelForSequenceClassification.from_pretrained(llm_to_finetune, num_labels=num_labels).to(device))
    logging_steps = len(tokenized_dataset["train"]) // batch_size
    
    training_args = TrainingArguments(
        output_dir=utils.OUTPUT_DIR_PATH + 'finetuned_hf_models/' + model_name,
        num_train_epochs=5,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        disable_tqdm=False,
        logging_steps=logging_steps,
        push_to_hub=True,
        log_level="error"
    )

    trainer = Trainer(
        model=model, 
        args=training_args,
        compute_metrics=llm_compute_metrics,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer
    )
    if mode == 'finetune':
        trainer.train()
        preds_output = trainer.predict(tokenized_dataset["validation"])
        print(preds_output.metrics)
        trainer.push_to_hub()
    else:
        trainer.evaluate()
        preds_output = trainer.predict(tokenized_dataset["validation"])
        print(preds_output.metrics)
        y_preds = np.argmax(preds_output.predictions, axis=1)
        print(y_preds[:10])


def llm_get_embbedings(dataset, subset, emb_type='llm_cls', device='cpu', output_path='', save_emb=False, llm_finetuned_name=LLM_HF_FINETUNED_NAME, num_labels=2):

    dataset = Dataset.from_dict(pd.DataFrame(data=dataset))
    dataset = dataset.with_format("torch", device=device)
    dataset = DatasetDict({"dataset": dataset})

    print(f"NFI -> device: {device} | subset: {subset} | emb_type: {emb_type} | save_emb: {save_emb} ")
    torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(llm_finetuned_name) 
    tokenized_datasets = dataset.map(llm_tokenize_function, batched=True, fn_kwargs={"tokenizer": tokenizer})
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets.set_format("torch")

    dataset_loader = DataLoader(tokenized_datasets['dataset'], batch_size=utils.LLM_GET_EMB_BATCH_SIZE_DATALOADER)
    model = AutoModelForSequenceClassification.from_pretrained(llm_finetuned_name, num_labels=num_labels)
    model.to(device)

    global INDEX 
    INDEX = 0
    emb_output_model = []
    for step, batch in enumerate(tqdm(dataset_loader)):

        with torch.no_grad():
            outputs_model = model(batch['input_ids'].to(device), output_hidden_states=True)
            last_hidden_state = outputs_model.hidden_states[-1]

            embeddings_lst = llm_extract_emb(
                last_hidden_state=last_hidden_state, 
                batch=batch, batch_step=step, 
                subset=subset, tokenizer=tokenizer, 
                tokenized_datasets=tokenized_datasets, 
                emb_type=emb_type
            )
            if save_emb == True:
                utils.save_llm_embedings(embeddings_data=embeddings_lst, emb_type=emb_type, batch_step=step, file_path=output_path)
            else:    
                return embeddings_lst
                #emb_output_model += embeddings_lst

        torch.cuda.empty_cache()
        gc.collect()

    return emb_output_model


def llm_extract_emb(last_hidden_state, batch, batch_step, subset, tokenizer, tokenized_datasets, emb_type):
    global INDEX
    #INDEX = 0
    embeddings_dict_lst = []
    if emb_type == 'llm_cls': # llm_doc
        embeddings = last_hidden_state[:,0,:]
        embeddings_dict_lst.append({"batch": batch_step, "subset": subset, "doc_id": batch['id'], "labels": batch['label'], "embedding": embeddings})
        return embeddings_dict_lst
    elif emb_type == 'llm_word': # llm_word
        embeddings_dict = {}
        for i in range(0, len(last_hidden_state)):
            raw_tokens = [tokenizer.decode([token_id]) for token_id in tokenized_datasets['dataset'][INDEX]['input_ids']]
            doc_id = tokenized_datasets['dataset'][INDEX]['id'].cpu().detach().numpy()
            label = tokenized_datasets['dataset'][INDEX]['label']
            #d = {"doc_id": doc_id, "doc_index": INDEX, 'label': label, 'embedding': {}} # ver forma de obtener ID de docu
            embeddings_dict[str(doc_id)] = {"doc_index": INDEX, 'label': label, 'embedding': {}} # ver forma de obtener ID de docu
            
            for token, embedding in zip(raw_tokens, last_hidden_state[i]):
                #d['embedding'][token] = embedding.cpu().detach().numpy().tolist()
                embeddings_dict[str(doc_id)]['embedding'][token] = embedding.cpu().detach().numpy().tolist()
            #embeddings_dict_lst.append(d) 
            INDEX += 1

        return embeddings_dict
    else:
        ... # ERROR

def llm_get_embbedings_2(dataset, subset, emb_type='llm_cls', device='cpu', output_path='', save_emb=False, llm_finetuned_name=LLM_HF_FINETUNED_NAME, num_labels=2):

    tokenizer = AutoTokenizer.from_pretrained(llm_finetuned_name)
    model = AutoModel.from_pretrained(llm_finetuned_name)
    model = model.to(device)
    model.eval()

    if emb_type == 'llm_cls':
        
        embeddings_lst = []
        for row in tqdm(dataset):
            with torch.no_grad():
                inputs = tokenizer(row["text"], return_tensors='pt', padding=True, truncation=True, max_length=512)
                inputs.to(device)
                outputs_model = model(**inputs, output_hidden_states=True)
                last_hidden_state = outputs_model.hidden_states[-1]
                embedding = last_hidden_state[0,0,:].cpu().detach().numpy().tolist()
                embeddings_lst.append({'doc_id': row['id'], 'label': row['label'], "embedding": embedding})
        
        #print(len(raw_tokens), len(last_hidden_state))
        #print(len(embedding), row['id'], row['label'])
        #print(embeddings_lst)
        utils.save_jsonl(embeddings_lst, output_path)
 
    if emb_type == 'llm_word':
        embeddings_word_dict = {}
        with torch.no_grad():
            for row in dataset:
                inputs = tokenizer(row["text"], return_tensors='pt', padding=True, truncation=True, max_length=512)
                inputs.to(device)
                outputs_model = model(**inputs, output_hidden_states=True)
                last_hidden_state = outputs_model.hidden_states[-1]

                embeddings_word_dict[str(row['id'])] = {"doc_id": row['id'], 'label': row['label'], 'embedding': {}} # ver forma de obtener ID de docu
                for i in range(0, len(last_hidden_state)):
                    raw_tokens = [tokenizer.decode([token_id]) for token_id in inputs['input_ids'][i]]
                    #print(len(raw_tokens), len(last_hidden_state[i]))
                    for token, embedding in zip(raw_tokens, last_hidden_state[i]):
                        #print(str(token).strip())
                        embeddings_word_dict[str(row['id'])]['embedding'][str(token).strip()] = embedding.cpu().detach().numpy().tolist()
                        #embeddings_word_dict[str(token).strip()] = embedding.cpu().detach().numpy().tolist()
        return embeddings_word_dict
        #utils.save_jsonl([embeddings_word_dict], utils.OUTPUT_DIR_PATH + 'word_emb_test.jsonl')

    if emb_type == 'llm_cls_2':
        with torch.no_grad():
            embeddings_lst = []
            for row in tqdm(dataset):
                inputs = tokenizer(row["text"], return_tensors='pt', padding=True, truncation=True, max_length=512)
                inputs.to(device)
                outputs_model = model(**inputs, output_hidden_states=True)
                last_hidden_state = outputs_model.hidden_states[-1]
                embedding = last_hidden_state[0,0,:].cpu().detach().numpy().tolist()
                embeddings_lst.append(embedding)
            return embeddings_lst
      
def fasttext_train(graph_data, num_features):
    sent_tokens = []
    for g in graph_data:
        sent_tokens.append(list(g['graph'].nodes))
    model_fasttext = gensim.models.FastText(vector_size=num_features, window=3, min_count=1)  
    model_fasttext.build_vocab(corpus_iterable=sent_tokens)
    model_fasttext.train(corpus_iterable=sent_tokens, total_examples=len(sent_tokens), epochs=10)
    return model_fasttext


def w2v_train(graph_data, num_features):
    sent_tokens = []
    for g in graph_data:
        sent_tokens.append(list(g['graph'].nodes))
    model_w2v = gensim.models.Word2Vec(sent_tokens, min_count=1,vector_size=num_features, window=3)
    return model_w2v


def w2v_train_v2(train_text_docs, val_text_docs, test_text_docs, num_features=256):

    X_train, X_val, X_test = [], [], []

    #tokenize_pattern = '(?:[\\(){}[\]=&|^+<>/*%;.\'"?!~-]|(?:\w+|\d+))'
    tokenize_pattern = "[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+"

    for d in train_text_docs: 
        text_doc = utils.text_normalize(d['doc'], tokenize_pattern)
        X_train.append(re.findall(tokenize_pattern, text_doc))
    for d in val_text_docs:
        text_doc = utils.text_normalize(d['doc'], tokenize_pattern)
        X_val.append(re.findall(tokenize_pattern, text_doc))
    for d in test_text_docs:
        text_doc = utils.text_normalize(d['doc'], tokenize_pattern)
        X_test.append(re.findall(tokenize_pattern, text_doc))

    model_w2v = gensim.models.Word2Vec(X_train, min_count=1, vector_size=num_features, window=3)
    words = set(model_w2v.wv.index_to_key)
    print("X_train_val_test: ", len(X_train), len(X_val), len(X_test))
    print("words: ", len(words), model_w2v)

    X_train_vect = [[model_w2v.wv[i] for i in ls if i in words] for ls in X_train]
    X_val_vect = [[model_w2v.wv[i] for i in ls if i in words] for ls in X_val]
    X_test_vect = [[model_w2v.wv[i] for i in ls if i in words] for ls in X_test]

    X_train_vect_avg = []
    for vect in X_train_vect:
        vect = np.array(vect)
        if vect.size:
            X_train_vect_avg.append(vect.mean(axis=0))
        else:
            X_train_vect_avg.append(np.zeros(num_features, dtype=float))

    X_val_vect_avg = []
    for vect in X_val_vect:
        vect = np.array(vect)
        if vect.size:
            X_val_vect_avg.append(vect.mean(axis=0))
        else:
            X_val_vect_avg.append(np.zeros(num_features, dtype=float))

    X_test_vect_avg = []
    for vect in X_test_vect:
        vect = np.array(vect)
        if vect.size:
            X_test_vect_avg.append(vect.mean(axis=0))
        else:
            X_test_vect_avg.append(np.zeros(num_features, dtype=float))


    return model_w2v, X_train_vect_avg, X_val_vect_avg, X_test_vect_avg

def get_document_embedding_w2v(doc, w2v_model, embedding_dim=300):

    word_embeddings = [w2v_model.wv[word] for word in doc if word in w2v_model.wv]

    if len(word_embeddings) > 0:
        return np.mean(word_embeddings, axis=0)  # Average of word embeddings
    else:
        return np.zeros(embedding_dim)  # Zero vector for OOV words


def llm_get_embbedings_3(dataset, device, model, tokenizer, avg_llm_found_tokens=True, text2graph_type='cooc'):
    model.eval()

    embeddings_word_dict = {}
    token_freq = defaultdict(int)

    iter = enumerate(dataset)
    if text2graph_type == 'hetero':
        iter = enumerate(tqdm(dataset))

    for idx, text in iter: # hetero
        text = utils.text_normalize_v2(text['doc'])
        with torch.no_grad():
            encoded_text = tokenizer.encode_plus(text, return_tensors="pt", padding=True, truncation=True)
            encoded_text.to(device)
            outputs_model = model(**encoded_text, output_hidden_states=True)
            last_hidden_state = outputs_model.hidden_states[-1]

            for i in range(0, len(last_hidden_state)):
                raw_tokens = [tokenizer.decode([token_id]) for token_id in encoded_text['input_ids'][i]]
                for token, embedding in zip(raw_tokens, last_hidden_state[i]):
                    token = str(token).strip()
                    token_freq[token] += 1
                    current_emb = embedding.cpu().detach().numpy().tolist()

                    if avg_llm_found_tokens: # avg token
                        if token not in embeddings_word_dict.keys():
                            embeddings_word_dict[token] = current_emb
                        else:
                            embeddings_word_dict[token] = np.add.reduce([embeddings_word_dict[token], current_emb])
                    else: # get last token emb found
                        embeddings_word_dict[token] = current_emb

    if avg_llm_found_tokens:
        for token, freq in token_freq.items():
            embeddings_word_dict[token] = np.divide(embeddings_word_dict[token], freq).tolist()

    return embeddings_word_dict


def get_emb_word_nodes(embeddings_word_dict, graph, tokenizer, avg_llm_not_found_tokens=True, text2graph_type='cooc'):
    embeddings_word_dict_avg = {}
    not_found_tokens_dic = {}
    nodes = list(graph['graph'].nodes)
    cnt_not_found_nodes = 0

    iter = nodes
    if text2graph_type == 'hetero':
        iter = tqdm(nodes)

    for node in iter:
        node = str(node).strip()
        if node.startswith("D-"):
            continue
        if node in embeddings_word_dict.keys():
            embeddings_word_dict_avg[node] = embeddings_word_dict[node]
        else:
            cnt_not_found_nodes += 1
            node_tokens = tokenizer.encode_plus(node, return_tensors="pt", padding=True, truncation=True)
            raw_tokens = [tokenizer.decode([token_id]) for token_id in node_tokens['input_ids'][0]]
            not_found_tokens_dic[node] = raw_tokens[1:-1]

    if avg_llm_not_found_tokens == True:
        iter = not_found_tokens_dic.keys()
        if text2graph_type == 'hetero':
            iter = tqdm(not_found_tokens_dic.keys())

        for node in iter: # hetero
            #print(node, not_found_tokens_dic[node])
            if str(node).startswith("D-"):
                continue
            if str(node) in embeddings_word_dict_avg.keys():
                continue
            try:
                avg_emb = []
                for w_emb in not_found_tokens_dic[node]:
                    if w_emb not in embeddings_word_dict.keys():
                        continue
                    avg_emb.append(embeddings_word_dict[w_emb])
                embeddings_word_dict_avg[str(node).strip()] = np.mean(avg_emb, axis=0).flatten().tolist()
                #print(len(embeddings_word_dict_avg[node]))
            except Exception as err:
                print("****** err: ", err)
                ...

    #print("embeddings_word_dict_avg: ", len(embeddings_word_dict_avg.keys()), embeddings_word_dict_avg.keys())
    return embeddings_word_dict_avg




        



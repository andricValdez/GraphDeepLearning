import logging
import os
import sys
import glob
import pandas as pd
import numpy as np
from statistics import mean 
import joblib
from sklearn.datasets import fetch_20newsgroups
import json
from sklearn.utils import shuffle
from datetime import datetime
from joblib import Parallel, delayed
import time
from polyglot.detect import Detector
from polyglot.detect.base import logger as polyglot_logger

from stylometric import StyloCorpus
import node_feat_init

polyglot_logger.setLevel("ERROR")

#************************************* CONFIGS
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s; - %(levelname)s; - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ROOT_DIR = os.path.dirname(os.path.dirname(__file__)) + '/GraphDeepLearning'
DATASET_DIR = ROOT_DIR + '/datasets/'
OUTPUT_DIR_PATH = ROOT_DIR + '/outputs/'
INPUT_DIR_PATH = ROOT_DIR + '/inputs/'
CUT_PERCENTAGE_DATASET = 100
TODAY_DATE = datetime.today().strftime('%Y-%m-%d')
CURRENT_TIME = datetime.today().strftime('%Y%m%d%H%M%S')
LANGUAGE = 'en' #es, en, fr

LLM_GET_EMB_BATCH_SIZE_DATALOADER = 128


#************************************* UTILS

def read_csv(file_path):
  df = pd.read_csv(file_path)
  return df

def read_json(dir_path):
    logger.debug("*** Using dataset: %s", dir_path)
    return pd.read_json(path_or_buf=dir_path, lines=True)

def save_csv(dataframe, file_path):
    dataframe.to_csv(file_path, encoding='utf-8', index=False)
  
def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def save_jsonl(data, file_path):
    with open(file_path, "w") as outfile:
        for element in data:  
            json.dump(element, outfile)  
            outfile.write("\n")  

def save_data(data, file_name, path=OUTPUT_DIR_PATH, format_file='.pkl', compress=False):
    logger.info('Saving data: %s', file_name)
    path_file = os.path.join(path, file_name + format_file)
    joblib.dump(data, path_file, compress=compress)

def load_data(file_name, path=INPUT_DIR_PATH, format_file='.pkl', compress=False):
    logger.info('Loading data: %s', file_name)
    path_file = os.path.join(path, file_name + format_file)
    return joblib.load(path_file)

def cut_dataset(corpus_text_docs, cut_percentage_dataset):
  cut_dataset = len(corpus_text_docs) * (int(cut_percentage_dataset) / 100)
  return corpus_text_docs[:int(cut_dataset)]

def delete_dir_files(dir_path):
    if os.path.exists(dir_path):
        files = glob.glob(dir_path + '/*')
        for f in files:
            os.remove(f)

def create_dir(dir_path):
  if not os.path.exists(dir_path):
    os.makedirs(dir_path)

def is_dir_empty(dir_path):
    if len(os.listdir(dir_path)) == 0:
        return True
    else:    
        return False

def text_metrics(dataframe):
    text_lens, max_text_len, min_text_len = 0, 0, 1000000000
    for index, row in dataframe.iterrows():
        text_i_len = len(row['text'].split())
        text_lens += text_i_len
        if text_i_len > max_text_len:
            max_text_len = text_i_len
        if text_i_len < min_text_len:
            min_text_len = text_i_len
    print("Text AVG Tokens: ", text_lens/len(dataframe))
    print("Text Max Tokens: ", max_text_len)
    print("Text Min Tokens: ", min_text_len)
    

def process_dataset(corpus_text_docs):
    text_data_lst = []
    #corpus_text_docs = shuffle(corpus_text_docs)
    corpus_text_docs_dict = corpus_text_docs.to_dict('records')
    for instance in corpus_text_docs_dict:
        #if len(instance['text'].split()) < 20:
        #    continue
        doc = {
            "id": instance['id'], 
            "doc": instance['text'][:], 
            "label": instance['label'], 
            #"model": instance['model'], 
            "context": {
                "id": instance['id'], "target": instance['label'], 
                #"lang": instance['lang'], "lang_code": instance['lang_code'], 'lang_confidence': instance['lang_confidence']
            }
        }
        text_data_lst.append(doc)
    return text_data_lst

 
def t2g_transform(corpus_text_docs, t2g_instance, cut_off = 100):
    print("Init transform text to graph: ")
    # Apply t2g transformation
    cut_dataset = len(corpus_text_docs) * (int(cut_off) / 100)
    start_time = time.time() # time init
    graph_output = t2g_instance.transform(corpus_text_docs[:int(cut_dataset)])
    for corpus_text_doc in corpus_text_docs:
        for g in graph_output:
            if g['doc_id'] == corpus_text_doc['id']:
                g['context'] = corpus_text_doc['context']
                break
    end_time = (time.time() - start_time)
    print("\t * TOTAL TIME:  %s seconds" % end_time)
    return graph_output


def lang_identify(dataframe): # must contain 'text' column
    for index, row in dataframe[:].iterrows():
        #if len(row['text'].split(' ')) < 50:
        #    continue
        #if row['lang']:
        #    continue
        try: 
            lang = Detector(row['text'])
            #print(len(row['text'].split(' ')), lang.language)
            dataframe.loc[index, 'lang'] = lang.language.name
            dataframe.loc[index, 'lang_code'] = lang.language.code
            dataframe.loc[index, 'lang_confidence'] = lang.language.confidence
        except Exception as err:
            ...
            dataframe.loc[index, 'lang'] = 'inglés'
            dataframe.loc[index, 'lang_code'] = 'en'
            dataframe.loc[index, 'lang_confidence'] = 0
            #print("error detecting lang: ", str(err))
    return dataframe


def set_text_lang(dataset):
    dataset['lang'] = None
    dataset['lang_code'] = None
    dataset['lang_confidence'] = None
    dataset = lang_identify(dataframe=dataset) # must contain 'text' column
    return dataset
    
    #dataset_null_lang = dataset[dataset['lang'].isna()]
    #dataset.to_csv(OUTPUT_DIR_PATH + 'dataset.csv')
    #dataset_null_lang.to_csv(OUTPUT_DIR_PATH + 'autext_ddataset_null_langataset.csv')
    

def joblib_delayed(funct, params):
    return delayed(funct)(params)

def joblib_parallel(delayed_funct, process_name, num_proc, backend='loky', mmap_mode='c', max_nbytes=None):
    logger.info('Parallel exec for %s, num cpus used: %s', process_name, num_proc)
    return Parallel(
        n_jobs=num_proc,
        backend=backend,
        mmap_mode=mmap_mode,
        max_nbytes=max_nbytes
    )(delayed_funct)

def create_expriment_dirs(exp_file_path):
    create_dir(dir_path=exp_file_path)
    create_dir(dir_path=exp_file_path + 'embeddings_gnn/')
    create_dir(dir_path=exp_file_path + 'embeddings_cls_llm_1/')
    #create_dir(dir_path=exp_file_path + 'embeddings_cls_llm_2/')
    #create_dir(dir_path=exp_file_path + 'embeddings_cls_llm_3/')
    create_dir(dir_path=exp_file_path + 'embeddings_word_llm/')
    create_dir(dir_path=exp_file_path + 'graphs/')
    create_dir(dir_path=exp_file_path + 'preds/')
    create_dir(dir_path=exp_file_path + 'clf_models/')


def save_llm_embedings(embeddings_data, emb_type, batch_step=0, file_path=OUTPUT_DIR_PATH):
    if emb_type == 'llm_cls' or emb_type == 'gnn':
        for emb_data in embeddings_data:
            embeddings = []  
            for doc_id, label, embedding in zip(emb_data['doc_id'], emb_data['labels'], emb_data['embedding']):
                embeddings.append({
                    "doc_id": doc_id.cpu().detach().numpy().tolist(),
                    'label': label.cpu().detach().numpy().tolist(), 
                    "embedding": embedding.cpu().detach().numpy().tolist()
                })
                #d[doc_id] = doc_id.cpu().detach().numpy().tolist()
            save_jsonl(embeddings, file_path=f"{file_path}{emb_data['batch']}.jsonl")
    if emb_type == 'llm_word':
        embeddings = [] 
        for emb_data in embeddings_data:
            label = int(emb_data['label'].cpu().detach().numpy())
            #print(type(emb_data['id']), type(emb_data['label']), type(emb_data['embeddings']['[CLS]'][0]))
            embeddings.append({
                "doc_id": emb_data['doc_id'], 
                'label': label, 
                "embedding": emb_data['embedding']
            })

        save_jsonl(embeddings, file_path=f"{file_path}{batch_step}.jsonl")


def get_stylo_feat(exp_file_path, text_data, subset):
    print('Getting get_stylo_feat...')
    text_data_lst = [{'id': d['context']['id'], 'label': d['context']['target'], 'text': d['doc']} for d in text_data]
    stylo_feat_lst = extract_stylo_feat(pd.DataFrame(text_data_lst), output_path=exp_file_path, subset=subset)
    save_jsonl(stylo_feat_lst, file_path=f"{exp_file_path}stylometry_{subset}_feat.json")


def extract_stylo_feat(data, output_path, subset):
    train_corpus = StyloCorpus.from_glob_pattern(data)
    train_corpus.output_csv(f'{output_path}stylometry_{subset}.csv')
    stylo_data = pd.read_csv(f'{output_path}stylometry_{subset}.csv')
    stylo_feat_lst = []
    for i in range(0, len(stylo_data)):
        x = np.asarray(stylo_data.iloc[i].values[2:])
        x_norm = x/np.linalg.norm(x)
        #stylo_feat_dict[stylo_data.iloc[i][0]] = x_norm.tolist()
        stylo_feat_lst.append({"doc_id": stylo_data.iloc[i][0], 'label': int(stylo_data.iloc[i][1]),  'stylo_feat':  x_norm.tolist()})
    return stylo_feat_lst
    

def llm_get_embbedings(text_data, exp_file_path, subset, emb_type, device, save_emb, llm_finetuned_name, num_labels):
    print('Getting llm_embbedings...')
    text_data_lst = [{'id': d['context']['id'], 'label': d['context']['target'], 'text': d['doc']} for d in text_data]
    #output_train_path = f"{exp_file_path}/autext_{subset}_emb_batch_"
    #node_feat_init.llm_get_embbedings(text_data_lst, subset=subset, emb_type=emb_type, device=device, output_path=output_train_path, save_emb=True, llm_finetuned_name=llm_finetuned_name, num_labels=num_labels)
    output_train_path = f"{exp_file_path}/autext_{subset}_embeddings.jsonl"
    node_feat_init.llm_get_embbedings_2(text_data_lst, subset=subset, emb_type=emb_type, device=device, output_path=output_train_path, save_emb=True, llm_finetuned_name=llm_finetuned_name, num_labels=num_labels)


import pandas as pd
import numpy as np
import random
import json
import random
import glob
import pprint
import os
import joblib
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.metrics import f1_score

import utils
 

def get_metrics(predicted, labels, target_names, subset='val'):
    print(f'Accuracy {subset}:', np.mean(predicted == labels))  
    print(f'F1Score {subset}:', f1_score(labels, predicted , average='macro'))
    print(metrics.classification_report(labels, predicted, target_names=target_names))
    print(f"Matriz Confusion {subset}: ")
    print(metrics.confusion_matrix(labels, predicted))


def build_pipeline(model):
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(lowercase=True, analyzer='word', stop_words='english', ngram_range=(1,1))),
        ('clf', CalibratedClassifierCV(model())),
    ])
    return pipeline


def train(train_set, algo_ml):
    clf_models = {
        'LinearSVC': LinearSVC,
        'MultinomialNB': MultinomialNB,
        'LogisticRegression': LogisticRegression,
        'RandomForestClassifier': RandomForestClassifier,
        'SGDClassifier': SGDClassifier
    }   
    pipeline = build_pipeline(model = clf_models[algo_ml])
    pipeline.fit(train_set['text'], train_set['label'])
    return pipeline


def test(texts, labels, model, target_names, subset='val'):
    predicted = model.predict(texts)
    get_metrics(predicted, labels, target_names, subset=subset)


def main(train_set, val_set, test_set, target_names=['human', 'generated'], algo_ml='SGDClassifier'):
    print('training model...')
    print("train_set: ", len(train_set))
    print("val_set: ", len(val_set))
    print("test_set: ", len(test_set))
    print('\n')
    if algo_ml == 'xgboost':
        vectorizer = TfidfVectorizer()
        X_train = vectorizer.fit_transform(train_set['text'])
        X_val = vectorizer.transform(val_set['text'])
        X_test = vectorizer.transform(test_set['text'])

        print(X_train.shape, len(train_set['text']))
        print(X_train[0].shape, X_train[1].shape)
        print(X_val[0].shape, X_val[1].shape)
        print(X_test[0].shape, X_test[1].shape)
        print(X_train[0])

        model = xgb.XGBClassifier(n_jobs=-1)
        model.fit(X_train, train_set['label'])
        val_predicted = model.predict(X_val)
        test_predicted = model.predict(X_test)
        print('Accuracy val:', np.mean(val_predicted == val_set['label']))  
        print('F1Score val:', f1_score(val_set['label'], val_predicted , average='macro'))
        print('Accuracy test:', np.mean(test_predicted == test_set['label']))  
        print('F1Score test:', f1_score(test_set['label'], test_predicted , average='macro'))

    else:
        model = train(train_set, algo_ml)   
        test(texts=val_set['text'], labels=val_set['label'], model=model, target_names=target_names, subset='val')
        test(texts=test_set['text'], labels=test_set['label'], model=model, target_names=target_names, subset='test')
    
    #utils.save_data(data=model, file_name=algo_ml)

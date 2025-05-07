import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import spacy


import test_utils


text = "@NKANGEL74 your heading to bed &amp; I am just getting home from work as usual!! Go figure!! Knight girl!"

text = test_utils.text_normalize(text, special_chars=False, stop_words=False) 
print(text)

nlp = spacy.load("en_core_web_sm")
doc = nlp(text)

for token in doc:
    print(f"'{token.text}'", end=' | ')

for sent in doc.sents:
    print(sent)
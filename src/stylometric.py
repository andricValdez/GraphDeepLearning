import nltk
from nltk import sent_tokenize, word_tokenize, Text
from nltk.probability import FreqDist
import numpy as np
import random
import glob

DEFAULT_AUTHOR = "Unknown"

class StyloDocument:

    def __init__(self, doc, label, file_name, author=DEFAULT_AUTHOR):
        #with open(file_name, "r", encoding='utf-8', errors='ignore') as file:
        #    self.doc = file.read()
        self.doc = doc
        self.label = label
        self.author = author
        self.file_name = file_name
        self.tokens = word_tokenize(self.doc)
        self.text = Text(self.tokens)
        self.fdist = FreqDist(self.text)
        self.sentences = sent_tokenize(self.doc)
        self.sentence_chars = [len(sent) for sent in self.sentences]
        self.sentence_word_length = [len(sent.split()) for sent in self.sentences]
        self.paragraphs = [p for p in self.doc.split("\n\n") if len(p) > 0 and not p.isspace()]
        self.paragraph_word_length = [len(p.split()) for p in self.paragraphs]

    @classmethod
    def csv_header(cls):
        return (
            'id, label, LexicalDiversity,MeanWordLen,MeanSentenceLen,StdevSentenceLen,MeanParagraphLen,DocumentLen,'
            'Commas,Semicolons,Quotes,Exclamations,Colons,Dashes,Mdashes,'
            'Ands,Buts,Howevers,Ifs,Thats,Mores,Musts,Mights,This,Verys'
        )

    def term_per_thousand(self, term):
        return (self.fdist[term] * 1000) / self.fdist.N()

    def mean_sentence_len(self):
        return np.mean(self.sentence_word_length)

    def std_sentence_len(self):
        return np.std(self.sentence_word_length)

    def mean_paragraph_len(self):
        return np.mean(self.paragraph_word_length)
        
    def std_paragraph_len(self):
        return np.std(self.paragraph_word_length)

    def mean_word_len(self):
        words = set(word_tokenize(self.doc))
        word_chars = [len(word) for word in words]
        return sum(word_chars) / float(len(word_chars))

    def type_token_ratio(self):
        return (len(set(self.text)) / len(self.text)) * 100

    def unique_words_per_thousand(self):
        return self.type_token_ratio() / 100.0 * 1000.0 / len(self.text)

    def document_len(self):
        return sum(self.sentence_chars)

    def csv_output(self):
        return '"%s",%i,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g' % (
            self.author, 
            self.label, 
            self.type_token_ratio(), 
            self.mean_word_len(), 
            self.mean_sentence_len(),
            self.std_sentence_len(),
            self.mean_paragraph_len(), 
            self.document_len(),

            self.term_per_thousand(','),
            self.term_per_thousand(';'),
            self.term_per_thousand('"'),
            self.term_per_thousand('!'),
            self.term_per_thousand(':'),
            self.term_per_thousand('-'),
            self.term_per_thousand('--'),
            
            self.term_per_thousand('and'),
            self.term_per_thousand('but'),
            self.term_per_thousand('however'),
            self.term_per_thousand('if'),
            self.term_per_thousand('that'),
            self.term_per_thousand('more'),
            self.term_per_thousand('must'),
            self.term_per_thousand('might'),
            self.term_per_thousand('this'),
            self.term_per_thousand('very'),
        )

    def text_output(self):
        print("##############################################")
        print("")
        print("Name: ", self.file_name)
        print("")
        print(">>> Phraseology Analysis <<<")
        print("")
        print("Lexical diversity        :", self.type_token_ratio())
        print("Mean Word Length         :", self.mean_word_len())
        print("Mean Sentence Length     :", self.mean_sentence_len())
        print("STDEV Sentence Length    :", self.std_sentence_len())
        print("Mean paragraph Length    :", self.mean_paragraph_len())
        print("Document Length          :", self.document_len())
        print("")
        print(">>> Punctuation Analysis (per 1000 tokens) <<<")
        print("")
        print('Commas                   :', self.term_per_thousand(','))
        print('Semicolons               :', self.term_per_thousand(';'))
        print('Quotations               :', self.term_per_thousand('"'))
        print('Exclamations             :', self.term_per_thousand('!'))
        print('Colons                   :', self.term_per_thousand(':'))
        print('Hyphens                  :', self.term_per_thousand('-')) # m-dash or n-dash?
        print('Double Hyphens           :', self.term_per_thousand('--')) # m-dash or n-dash?
        print("")
        print(">>> Lexical Usage Analysis (per 1000 tokens) <<<")
        print("")
        print('and                      :', self.term_per_thousand('and'))
        print('but                      :', self.term_per_thousand('but'))
        print('however                  :', self.term_per_thousand('however'))
        print('if                       :', self.term_per_thousand('if'))
        print('that                     :', self.term_per_thousand('that'))
        print('more                     :', self.term_per_thousand('more'))
        print('must                     :', self.term_per_thousand('must'))
        print('might                    :', self.term_per_thousand('might'))
        print('this                     :', self.term_per_thousand('this'))
        print('very                     :', self.term_per_thousand('very'))
        print('')


class StyloCorpus:

    def __init__(self, documents_by_author):
        self.documents_by_author = documents_by_author

    @classmethod
    def from_glob_pattern(cls, data):
        documents_by_author = cls.get_dictionary_from_glob(data)
        return cls(documents_by_author)

    @classmethod
    def get_dictionary_from_glob(cls, data):
        documents_by_author = {}
        for index, row in data.iterrows():
            author=row['id']
            document = StyloDocument(doc=row['text'], label=row['label'], file_name=row['id'], author=author)
            if author not in documents_by_author:
                documents_by_author[author] = [document]
            else:
                documents_by_author[author].append(document)
        return documents_by_author
        
    def output_csv(self, out_file, author=None):
        csv_data = StyloDocument.csv_header() + '\n'
        if not author:
            for a in self.documents_by_author.keys():
                for doc in self.documents_by_author[a]:
                    csv_data += doc.csv_output() + '\n'
        else:
            for doc in self.documents_by_author[author]:
                csv_data += doc.csv_output() + '\n'
        if out_file:
            with open(out_file, 'w') as f:
                f.write(csv_data)
        return csv_data


from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer, BertTokenizerFast, BertModel
import networkx as nx
import networkx
from collections import defaultdict
import logging
import sys
import traceback 
import time
from joblib import Parallel, delayed
import warnings
import nltk
import os
import re
import string
import math
import codecs
import multiprocessing
from spacy.tokens import Doc
import spacy
from spacy.lang.xx import MultiLanguage
from spacy.cli import download
from spacy.tokenizer import Tokenizer
from spacy.util import compile_prefix_regex, compile_infix_regex, compile_suffix_regex
import contractions
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
import itertools
from math import log

import utils

#************************************* CONFIGS
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s; - %(levelname)s; - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
#tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
#TOKENIZER_FILE = "/home/avaldez/projects/Autextification2024/inputs/bert-base-uncased-vocab.txt"
#TOKENIZER_FILE = "/home/avaldez/projects/Autextification2024/inputs/bert-base-multilingual-cased.txt"
TOKENIZER_FILE = "/home/avaldez/projects/Autextification2024/inputs/bert-base-multilingual-cased-finetuned-autext24.txt"


def custom_tokenizer(nlp):
    #infix_re = re.compile(r'''[.\,\?\:\;\...\‘\’\`\“\”\"\'~]''')
    infix_re = re.compile(r'(?:[\\(){}[\]=&|^+<>/*%;.\'"?!~-]|(?:\w+|\d+))')
    prefix_re = compile_prefix_regex(nlp.Defaults.prefixes)
    suffix_re = compile_suffix_regex(nlp.Defaults.suffixes)

    return Tokenizer(nlp.vocab, prefix_search=prefix_re.search,
                                suffix_search=suffix_re.search,
                                infix_finditer=infix_re.finditer,
                                token_match=None)
'''
nlp = spacy.load('en_core_web_sm')
nlp.tokenizer = custom_tokenizer(nlp)

ingl_txt = u'I feel a great sense of obligation to warn fellow music enthusiasts about the profound disappointment this cd brought upon me. it was, in every sense of the word, a disaster. the track list seemed promising at first glance, but that"s as far as the allure went. the audio quality was abysmal, sounding as if it had been recorded in a tin can. the volume fluctuated dramatically from one track to another, forcing me to constantly adjust my stereo. as for the songs themselves, they were an uninspired mess. the lyrics lacked depth and originality, with repetitive themes and mundane metaphors. moreover, the melodies were disjointed and lacked any semblance of rhythm or harmony. it felt like a haphazardly thrown together mish-mash'
espa_txt = u'Estamos en este momento hospedados en el hotel, en general super bien, el personal en su mayorÃ­a excelente, pero se nota la preferencia hacia el pasajero de estados unidos, la habitaciÃ³n espaciosa, un poco antiguas y con un poco de oxido en el baÃ±o, pero en general salva hasta el momento, veamos como seguimos hasta el final de nuestra estancia, felicitaciones a susana que es nuestra relacionadora y encargada de que nuestra estancia sea la mejor, lamentablemente no pudimos tomar el paquete de clientes frecuentes por que no lo dejan contratar fuera del hotel, pero bueno nada'
gall_txt = u'Ã“scar de souto, nado en cabana de bergantiÃ±os o 23 de setembro de 1980, Ã© un escritor galego. traxectoria reside en arteixo dende o ano 1999. colaborador de varias publicaciÃ³ns periÃ³dicas, como a revista xente nova (1995-1999), airiÃ±os (2000-2009), el periÃ³dico de arteixo (2006-2010), e membro fundador da organizaciÃ³n literario-cultural amencer, de arteixo, e vicepresidente xove da asociaciÃ³n don bosco da coruÃ±a (2006-2010). en decembro de 2007 expÃ³n no centro cÃ­vico de arteixo xunto Ã¡ pintora uruguaia ana novo a obra mnemosyne, a voz da imaxe, onde se fusionan poesÃ­a e pintura abstracta. desa obra xorden os dous primeiros recitais poÃ©ticos coa organizaciÃ³n literario-cultural amencer (olca). participou no xurado do concurso literario manuel murguÃ­a de 2008, asÃ­ como en recitais de poesÃ­a en toda galicia. obra en galego poesÃ­a mÃ¡is alÃ¡ de min!, 2007, publicada en internet. bitÃ¡cora poÃ©tica, 2008, publicada en internet. destino de vinganza, 2008, publicada en internet. en ausencia, 2008, publicada en internet. laios d"alma, 2012. ten pendentes de publicaciÃ³n os libros lembranzas e terra (2006). carbÃ³n querubÃ­n , publicado en 2019, poesÃ­a en galego . narrativa rÃ­o arriba, 2008, publicado en internet. traduciÃ³ns se lo dije a la noche/dÃ­xenllo Ã¡ noite, de juan carlos garcÃ­a hoyuelos, 2011. aire, fuego y deseo/aire, lume e desexo, de juan carlos garcÃ­a hoyuelos, 2016. obras colectivas latexos, obra poÃ©tica. o lobo, unha carreira por sobrevivir, adega. versus, 2012, o. l. c. a. poesÃ­a'
port_txt = u'O efeito droste Ã© um fenÃ´meno visual que ocorre quando uma imagem contÃ©m uma versÃ£o menor de si mesma, criando uma ilusÃ£o de infinita regressÃ£o. o termo foi cunhado pelo matemÃ¡tico holandÃªs maurits cornelis escher, em referÃªncia ao nome da marca de cacau "droste", que possui uma lata com a imagem de uma enfermeira segurando uma bandeja com a mesma lata em suas mÃ£os. origem e desenvolvimento: o efeito droste tem origem na arte medieval, onde era utilizado em pinturas religiosas para representar o conceito de eternidade. no entanto, foi apenas no sÃ©culo xx que o fenÃ´meno ganhou maior destaque, principalmente atravÃ©s das obras do artista grÃ¡fico m.c. escher. escher ficou fascinado com o efeito e o utilizou em diversas de suas obras, como "ascensÃ£o e queda" e "casa de escalas". explicaÃ§Ã£o matemÃ¡tica: o efeito droste pode ser explicado pela geometria fractal, que Ã© a repetiÃ§Ã£o de um padrÃ£o infinitamente em diferentes escalas. quando aplicado em imagens, esse efeito Ã© criado atravÃ©s da inserÃ§Ã£o de uma versÃ£o menor da imagem original dentro dela mesma, seguindo uma proporÃ§Ã£o fixa. assim, a cada nova repetiÃ§Ã£o, a imagem se torna progressivamente menor, criando a ilusÃ£o de infinita regressÃ£o. aplicaÃ§Ãµes: alÃ©m do mundo das artes, o efeito droste tambÃ©m Ã© utilizado em Ã¡reas como publicidade, embalagens e design grÃ¡fico. a indÃºstria do entretenimento tambÃ©m se apropriou desse fenÃ´meno, utilizando-o em filmes, sÃ©ries e jogos. alÃ©m disso, o efeito droste tambÃ©m Ã© utilizado em matemÃ¡tica e informÃ¡tica, como na criaÃ§Ã£o de imagens fractais e em algoritmos de compressÃ£o de imagem. curiosidades: - a lata de cacau droste, que deu origem ao termo, foi produzida pela primeira vez em 1904 e ainda Ã© comercializada atÃ© hoje. - o termo "droste" possui uma origem curiosa. ele Ã© derivado do sobrenome da famÃ­lia que fundou a fÃ¡brica'
cata_txt = u'Mai havÃ­em provat un arrÃ²s tan bo. les racions sÃ³n molt generoses perÃ² ens el vam acabar tot. el preu Ã©s molt barat per la qualitat que menges. felicitats al cuiner! quims'
eusk_txt = u'Otso taldea elkarteak ikerketa hauek otsoen gorotzen dna aztertuaz eta talde ugaltzaileak somatuaz gauzatzen dituzte eta beren azken ondorioa, datuok ikusita, euskal herrian otsoa dagoeneko finkaturik dagoela litzateke. hala ere, animali basati hauen garapenak ikamika ugari sortzen ditu, batez ere abeltzainen eta otsoaren aldeko taldeen artean, lehenen ez baitituzte begi onez ikusten otsoa berreskuratzeko neurriak, izan ere beren lurretan kalte gehiegi eragiten dituztela saltu izan baitute behin eta berriro. otsoen aldekoek, aldiz, kalteak oso txikiak direla argudiatzen dute eta erakundeen diru-laguntzen alde azaltzen dira euskal autonomia erkidegoko mendilerroetan otsoen uluak entzuteko aukerak galdu ez baizik eta jarrai dezaten zein areagotu daitezen. 236'
doc = nlp(ingl_txt)
print([token.text for token in doc])
'''

class BTokenizerLLM:
    def __init__(self, vocab, vocab_file, lowercase=True):
        self.vocab = vocab
        self._tokenizer = BertWordPieceTokenizer(vocab_file, lowercase=lowercase)
        #self._tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')

    def __call__(self, text):
        tokens = self._tokenizer.encode(text)

        #tokens = self._tokenizer(text)
        #print(tokens)
        #print(self._tokenizer.decode(tokens['input_ids']))
        
        words = []
        spaces = []
        for i, (text, (start, end)) in enumerate(zip(tokens.tokens, tokens.offsets)):
            words.append(text)
            if i < len(tokens.tokens) - 1:
                # If next start != current end we assume a space in between
                next_start, next_end = tokens.offsets[i + 1]
                spaces.append(next_start > end)
            else:
                spaces.append(True)
        return Doc(self.vocab, words=words, spaces=spaces)

'''
tokenizer_file = "/home/avaldez/projects/Autextification2024/inputs/bert-base-multilingual-cased-finetuned-autext24.txt"
ingl_txt = u'I feel a great sense of obligation to warn fellow music enthusiasts about the profound disappointment this cd brought upon me. it was, in every sense of the word, a disaster. the track list seemed promising at first glance, but that"s as far as the allure went. the audio quality was abysmal, sounding as if it had been recorded in a tin can. the volume fluctuated dramatically from one track to another, forcing me to constantly adjust my stereo. as for the songs themselves, they were an uninspired mess. the lyrics lacked depth and originality, with repetitive themes and mundane metaphors. moreover, the melodies were disjointed and lacked any semblance of rhythm or harmony. it felt like a haphazardly thrown together mish-mash'
nlp = spacy.blank("en")
nlp.tokenizer = BTokenizerLLM(nlp.vocab, tokenizer_file)
doc = nlp(ingl_txt)
print(nlp.pipe_names)
print(doc.text)
print([token.text for token in doc])
'''

class Text2CoocGraph():
    def __init__(self, 
                graph_type, 
                apply_prep=True,
                parallel_exec=False, 
                window_size=1,
                language='en', 
                steps_preprocessing={},
                min_word_freq=1, 
                node_type='text'
            ):
        """Constructor method
        """
        self.apply_prep = apply_prep
        self.window_size = window_size
        self.graph_type = graph_type
        self.parallel_exec = parallel_exec
        self.language = language
        self.steps_prep = steps_preprocessing
        self.stemming = PorterStemmer()
        self.min_word_freq = min_word_freq
        self.node_type = node_type
        self.stopwords_lang = {
            "en": self._set_stopwords(utils.INPUT_DIR_PATH + '/stopwords_en.txt'),
            "es": self._set_stopwords(utils.INPUT_DIR_PATH + '/stopwords_es.txt'),
            #"pt": self._set_stopwords(utils.INPUT_DIR_PATH + '/stopwords_pt.txt'),
            #"ca": self._set_stopwords(utils.INPUT_DIR_PATH + '/stopwords_ca.txt'),
            #"eu": self._set_stopwords(utils.INPUT_DIR_PATH + '/stopwords_eu.txt'),
            #"gl": self._set_stopwords(utils.INPUT_DIR_PATH + '/stopwords_gl.txt'),
        }

        # scpay model
        #self.nlp = spacy.blank("xx")
        #self.nlp.tokenizer = BTokenizerLLM(self.nlp.vocab, TOKENIZER_FILE)    

        #exclude_modules = ["ner", "textcat", "tok2vec"]
        #self.nlp = spacy.load("en_core_web_sm", exclude=exclude_modules)
        #self.nlp.max_length = 10000000000

        exclude_modules = ["ner", "textcat"]
        self.nlp = spacy.load('en_core_web_sm', exclude=exclude_modules)
        self.nlp.tokenizer = custom_tokenizer(self.nlp)
    
    def _set_node_type(self, token):
        ntype = f'{token.text}'
        if self.node_type == 'lemma':
            ntype = f'{token.lemma_}'
        elif self.node_type == 'stem':
            ntype = f'{self.stemming.stem(token.text)}'
        elif self.node_type == 'text_pos':
            ntype = f'{token.text}_{token.pos_}'
        elif self.node_type == 'lemma_pos':
            ntype = f'{token.lemma_}_{token.pos_}'
        else:
            ntype = f'{token.text}'
        return str(ntype)


    def _get_entities(self, doc_instance, vocab) -> list:
        nodes = []
        for token in doc_instance:
            if token.text not in vocab:
                continue
            if token.text in ['[CLS]', '[SEP]', '[UNK]']:
                continue
            node = (f'{self._set_node_type(token)}', {'lemma_': token.lemma_, 'pos_tag': token.pos_}) # (word, {'node_attr': value}) | {'pos_tag': token.pos_} | token.lemma_ | token.text
            nodes.append(node)
        #print(nodes)
        return nodes


    def _get_relations(self, doc, vocab) -> list:
        d_cocc = defaultdict(int)
        text_doc_tokens, edges = [], []
        for token in doc:
            if token.text not in vocab:
                continue
            if token.text in ['[CLS]', '[SEP]', '[UNK]']:
                continue
            text_doc_tokens.append(f'{self._set_node_type(token)}') #  token.lemma_ | token.text
        for i in range(len(text_doc_tokens)):
            word = text_doc_tokens[i]
            next_word = text_doc_tokens[i+1 : i+1 + self.window_size]
            for t in next_word:
                key = (word, t)
                d_cocc[key] += 1

        unigram_freq = nltk.FreqDist(text_doc_tokens)
        bigram_freq = nltk.FreqDist(d_cocc)
        for words, value in d_cocc.items():
            pmi_val = self._pmi(words, unigram_freq, bigram_freq)
            edge = (words[0], words[1], {'freq': value, 'weight': round(pmi_val,4)})  # freq, pmi | (word_i, word_j, {'edge_attr': value})
            edges.append(edge)
        return edges

    def _pmi(self, words, unigram_freq, bigram_freq):
        prob_word1 = unigram_freq[words[0]] / float(sum(unigram_freq.values()))
        prob_word2 = unigram_freq[words[1]] / float(sum(unigram_freq.values()))
        prob_word1_word2 = bigram_freq[words] / float(sum(bigram_freq.values()))
        return math.log(prob_word1_word2/float(prob_word1*prob_word2),2) 


    def _set_stopwords(self, stoword_path):
        stopwords = []
        for line in codecs.open(stoword_path, encoding="utf-8"):
            # Remove black space if they exist
            stopwords.append(line.strip())
        return dict.fromkeys(stopwords, True)


    def _handle_stop_words(self, text, stop_words) -> str:
        tokens = nltk.word_tokenize(text)
        without_stopwords = [word for word in tokens if not word.lower().strip() in stop_words]
        return " ".join(without_stopwords)

    def _handle_contractions(self, text) -> str:
        text = re.sub('([A-Za-z]+)[\'`]([A-Za-z]+)', r'\1'r'\2', text)
        expanded_words = []
        tokens = nltk.word_tokenize(text)
        for token in tokens:
            expanded_words.append(contractions.fix(token))
        return ' '.join(expanded_words)

    def _text_normalize(self, text: str, lang_code: str) -> list:
        if self.apply_prep:
            if self.steps_prep['to_lowercase']:
                text = text.lower() # text to lower case
            if self.steps_prep['handle_blank_spaces']:
                text = re.sub(r'\s+', ' ', text).strip() # remove blank spaces
            if self.steps_prep['handle_html_tags']:
                text = re.compile('<.*?>').sub(r'', text) # remove html tags
            #if self.steps_prep['handle_contractions']:
            if True:
                text = self._handle_contractions(text) # handle contraction
            if self.steps_prep['handle_special_chars']:
                text = re.sub('[^A-Za-z0-9]+ ', ' ', text) # remove special chars
                text = re.sub('\W+ ',' ', text)
                text = text.replace('"'," ")
                text = re.sub(r'\s+', ' ', text).strip() # remove blank spaces
                text = self._handle_stop_words(text, stop_words=self.stopwords_lang[lang_code]) # remove stop words
        return text


    def _min_word_freq(self, doc):
        vocab = []
        for token in doc:
            vocab.append(self._set_node_type(token))
        vocab = { x: count for x, count in Counter(vocab).items() if count >= self.min_word_freq }
        #print(len(list(vocab.keys())), vocab)
        vocab = set(list(vocab.keys()))
        return vocab


    def _nlp_pipeline(self, docs: list, params = {'get_multilevel_lang_features': False}):
        doc_tuples = []
        Doc.set_extension("multilevel_lang_info", default=[], force=True)
        for doc, context in list(self.nlp.pipe(docs, as_tuples=True, n_process=4, batch_size=1000)):
            if params['get_multilevel_lang_features'] == True:
                doc._.multilevel_lang_info = self.get_multilevel_lang_features(doc)
            doc_tuples.append((doc, context))
        return doc_tuples


    def _build_graph(self, nodes: list, edges: list) -> networkx:
        if self.graph_type == 'DiGraph':
            graph = nx.DiGraph()
        else:
            graph = nx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        return graph


    def _transform_pipeline(self, doc_instance: tuple) -> list:
        output_dict = {
            'doc_id': doc_instance['id'], 
            'context': doc_instance['context'],
            'graph': None, 
            'number_of_edges': 0, 
            'number_of_nodes': 0, 
            'status': 'success'
        }
        try:
            # ******************************************** TEST - min_word_freq
            vocab = self._min_word_freq(doc_instance['doc'])

            # get_entities
            nodes = self._get_entities(doc_instance['doc'], vocab)
            # get_relations
            edges = self._get_relations(doc_instance['doc'], vocab)
            # build graph
            graph = self._build_graph(nodes, edges)
            output_dict['number_of_edges'] += graph.number_of_edges()
            output_dict['number_of_nodes'] += graph.number_of_nodes()
            output_dict['graph'] = graph
        except Exception as e:
            logger.error('Error: %s', str(e))
            logger.error('Error Detail: %s', str(traceback.format_exc()))
            output_dict['status'] = 'fail'
        finally:
            return output_dict
    

    def transform(self, corpus_texts) -> list:
        logger.info("Init transformations: Text to Co-Ocurrence Graph")
        logger.info("Transforming %s text documents...", len(corpus_texts))
        prep_docs, corpus_output_graph, delayed_func = [], [], []

        logger.debug("Preprocessing")
        
        for doc_data in corpus_texts:
            #lang_code = doc_data['context']['lang_code']
            lang_code = 'en'
            if self.apply_prep == True:
                doc_data['doc'] = self._text_normalize(doc_data['doc'], lang_code)
            prep_docs.append(
                (doc_data['doc'], {'id': doc_data['id'], "context": doc_data['context']})
            )

        logger.debug("Transform_pipeline")
        docs = self._nlp_pipeline(prep_docs)

        if self.parallel_exec == True: 
            for input_text in corpus_texts:
                logger.debug('--- Processing doc %s ', str(input_text['id'])) 
                delayed_func.append(
                    utils.joblib_delayed(funct=self._transform_pipeline, params=input_text) 
                )
            num_proc = multiprocessing.cpu_count() // 2
            corpus_output_graph = utils.joblib_parallel(delayed_func, num_proc=num_proc, process_name='transform_cooocur_graph')

        else:
            for doc, context in list(docs):
                corpus_output_graph.append(
                    self._transform_pipeline(
                        {
                            'id': context['id'], 
                            'doc': doc,
                            'context': context['context']
                        }
                    )
                )

            logger.info("Done transformations")
        
        return corpus_output_graph


class Text2HeteroGraph():
    def __init__(self, 
                 graph_type, 
                 apply_prep=True, 
                 parallel_exec=False, 
                 window_size=1, 
                 language='en', 
                 steps_preprocessing={}, 
                 min_word_freq=1, 
                 node_type='text'
            ):
        
        self.apply_prep = apply_prep
        self.window_size = window_size
        self.graph_type = graph_type
        self.parallel_exec = parallel_exec
        self.language = language
        self.steps_prep = steps_preprocessing
        #self.stop_words = set(stopwords.words('english'))
        self.stemming = PorterStemmer()
        self.min_word_freq = min_word_freq
        self.node_type = node_type
        self.stopwords_lang = {
            "en": self._set_stopwords(utils.INPUT_DIR_PATH + '/stopwords_en.txt'),
            "es": self._set_stopwords(utils.INPUT_DIR_PATH + '/stopwords_es.txt')
        }
        
        exclude_modules = ["ner", "textcat"]
        self.nlp = spacy.load('en_core_web_sm', exclude=exclude_modules)
        self.nlp.tokenizer = custom_tokenizer(self.nlp)

    def _set_node_type(self, token):
        ntype = f'{token.text}'
        if self.node_type == 'lemma':
            ntype = f'{token.lemma_}'
        elif self.node_type == 'stem':
            ntype = f'{self.stemming.stem(token.text)}'
        elif self.node_type == 'text_pos':
            ntype = f'{token.text}_{token.pos_}'
        elif self.node_type == 'lemma_pos':
            ntype = f'{token.lemma_}_{token.pos_}'
        else:
            ntype = f'{token.text}'
        return str(ntype)
    
    def _set_stopwords(self, stoword_path):
        stopwords = []
        for line in codecs.open(stoword_path, encoding="utf-8"):
            # Remove black space if they exist
            stopwords.append(line.strip())
        return dict.fromkeys(stopwords, True)
    
    def __get_windows(self, doc_words_list, window_size):
        word_window_freq = defaultdict(int)
        word_pair_count = defaultdict(int)
        len_doc_words_list = len(doc_words_list)
        len_windows = 0

        for i, doc in enumerate(doc_words_list):
            windows = []
            doc_words = doc['words']
            length = len(doc_words)

            if length <= window_size:
                windows.append(doc_words)
            else:
                for j in range(length - window_size + 1):
                    window = doc_words[j: j + window_size]
                    windows.append(list(set(window)))
            for window in windows:
                for word in window:
                    word_window_freq[word] += 1
                for word_pair in itertools.combinations(window, 2):
                    word_pair_count[word_pair] += 1
            len_windows += len(windows)

        return word_window_freq, word_pair_count, len_windows

    def __get_pmi(self, doc_words_list, window_size):
        word_window_freq, word_pair_count, len_windows = self.__get_windows(doc_words_list, window_size)
        word_to_word_pmi = []
        for word_pair, count in word_pair_count.items():
            word_freq_i = word_window_freq[word_pair[0]]
            word_freq_j = word_window_freq[word_pair[1]]
            pmi = log((1.0 * count / len_windows) / (1.0 * word_freq_i * word_freq_j/(len_windows * len_windows)))
            if pmi <= 0:
                continue
            word_to_word_pmi.append((word_pair[0], word_pair[1], {'weight': round(pmi, 2)}))
        return word_to_word_pmi

    def __get_tfidf(self, corpus_docs_list, vocab):
        vectorizer = TfidfVectorizer(vocabulary=vocab, norm=None, use_idf=True, smooth_idf=False, sublinear_tf=False, lowercase=False, tokenizer=None)
        tfidf = vectorizer.fit_transform(corpus_docs_list)
        words_docs_tfids = []
        len_tfidf = tfidf.shape[0]

        for ind, row in enumerate(tfidf):
            for col_ind, value in zip(row.indices, row.data):
                edge = ('D-' + str(ind+1), vocab[col_ind], {'weight': round(value, 2)})
                words_docs_tfids.append(edge)
        return words_docs_tfids

    def _handle_stop_words(self, text) -> str:
        tokens = nltk.word_tokenize(text)
        without_stopwords = [word for word in tokens if not word.lower().strip() in self.stop_words]
        return " ".join(without_stopwords)

    def _handle_contractions(self, text) -> str:
        text = re.sub('([A-Za-z]+)[\'`]([A-Za-z]+)', r'\1'r'\2', text)
        expanded_words = []
        tokens = nltk.word_tokenize(text)
        for token in tokens:
          expanded_words.append(contractions.fix(token))
        return ' '.join(expanded_words)

    def _nlp_pipeline(self, docs: list, params = {'get_multilevel_lang_features': False}):
        doc_tuples = []
        Doc.set_extension("multilevel_lang_info", default=[], force=True)
        for doc, context in list(self.nlp.pipe(docs, as_tuples=True, n_process=4, batch_size=1000)):
            if params['get_multilevel_lang_features'] == True:
                doc._.multilevel_lang_info = self.get_multilevel_lang_features(doc)
            doc_tuples.append((doc, context))
        return doc_tuples

    def _text_normalize(self, text: str, lang_code: str) -> list:
        if self.apply_prep:
            if self.steps_prep['to_lowercase']:
                text = text.lower() # text to lower case
            if self.steps_prep['handle_blank_spaces']:
                text = re.sub(r'\s+', ' ', text).strip() # remove blank spaces
            if self.steps_prep['handle_html_tags']:
                text = re.compile('<.*?>').sub(r'', text) # remove html tags
            if self.steps_prep['handle_contractions']:
                text = self._handle_contractions(text) # handle contractions
            if self.steps_prep['handle_special_chars']:
                text = re.sub('[^A-Za-z0-9]+ ', ' ', text) # remove special chars
                text = re.sub('\W+', ' ', text) # remove special chars
                text = text.replace('"'," ")
                text = text.replace('('," ")
                text = re.sub(r'\s+', ' ', text).strip() # remove blank spaces
            if self.steps_prep['handle_stop_words']:
                #text = self._handle_stop_words(text) # remove stop words
                text = self._handle_stop_words(text, stop_words=self.stopwords_lang[lang_code]) # remove stop words

        return text

    def _min_word_freq(self, docs):
        vocab, corpus_docs_list, doc_words_list = [], [], []
        for doc, context in docs:
            for token in doc:
                ntype = self._set_node_type(token)
                vocab.append(ntype)

        vocab = { x: count for x, count in Counter(vocab).items() if count >= self.min_word_freq }
        #print(len(list(vocab.keys())), vocab)
        vocab = set(list(vocab.keys()))

        for doc, context in docs:
            doc_tokens = []
            for token in doc:
                ntype = self._set_node_type(token)
                if ntype in vocab:
                    doc_tokens.append(ntype) # text,  lemma_, self.stemming.stem()

            corpus_docs_list.append(str(" ".join(doc_tokens)))
            doc_words_list.append({'doc': context['id'], 'words': doc_tokens})

        return vocab, corpus_docs_list, doc_words_list

    def __get_entities(self, doc_words_list: list) -> list:
        nodes = []
        for d in doc_words_list:
            node_doc =  ('D-' + str(d['doc']), {})
            nodes.append(node_doc)
            for word in d['words']:
                node_word = (str(word), {})
                nodes.append(node_word)
        return nodes

    def __get_relations(self, corpus_docs_list, doc_words_list, vocab) -> list:
        edges = []
        #tfidf
        word_to_doc_tfidf = self.__get_tfidf(corpus_docs_list, vocab)
        edges.extend(word_to_doc_tfidf)
        #pmi
        word_to_word_pmi = self.__get_pmi(doc_words_list, self.window_size)
        edges.extend(word_to_word_pmi)
        return edges

    def __build_graph(self, nodes: list, edges: list) -> networkx:
        if self.graph_type == 'DiGraph':
            graph = nx.DiGraph()
        else:
            graph = nx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        return graph

    def __transform_pipeline(self, corpus_docs: list) -> list:
        output_dict = {
            'doc_id': 1,
            'graph': None,
            'number_of_edges': 0,
            'number_of_nodes': 0,
            'status': 'success'
        }
        try:
            #1. text preprocessing
            corpus_docs_list = []
            doc_words_list = []
            len_corpus_docs = len(corpus_docs)
            vocab = set()
            delayed_func = []
            prep_docs = []
            lang_code = 'en'

            for doc_data in corpus_docs:
                if self.apply_prep == True:
                    doc_data['doc'] = self._text_normalize(doc_data['doc'], lang_code)
                prep_docs.append((doc_data['doc'], {'id': doc_data['id']}))

            docs = self._nlp_pipeline(prep_docs)

            # ******************************************** TEST - min_word_freq
            vocab, corpus_docs_list, doc_words_list = self._min_word_freq(docs)
            # ******************************************** TEST - min_word_freq

            '''
            for doc, context in docs:
                doc_tokens = [str(token.text) for token in doc] # text,  lemma_, self.stemming.stem()
                corpus_docs_list.append(str(" ".join(doc_tokens)))
                doc_words_list.append({'doc': context['id'], 'words': doc_tokens})
                vocab.update(set(doc_tokens))
            '''

            #2. get node/entities
            nodes = self.__get_entities(doc_words_list)
            #3. get edges/relations
            edges = self.__get_relations(corpus_docs_list, doc_words_list, list(vocab))
            #4. build graph
            graph = self.__build_graph(nodes, edges)
            output_dict['number_of_edges'] = graph.number_of_edges()
            output_dict['number_of_nodes'] = graph.number_of_nodes()
            output_dict['graph'] = graph
        except Exception as e:
            print('Error: %s', str(e))
            output_dict['status'] = 'fail'
        finally:
            corpus_docs_list = None
            doc_words_list = None
            vocab = None
            prep_docs = None
            nodes = None
            edges = None
            return output_dict

    def transform(self, corpus_docs: list) -> list:
        print("Init transformations: Text to Heterogeneous Graph")
        print("Transforming %s text documents...", len(corpus_docs))
        corpus_output_graph = [self.__transform_pipeline(corpus_docs)]
        print("Done transformations")
        return corpus_output_graph

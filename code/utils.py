from nltk import TweetTokenizer
import spacy
from sklearn.base import TransformerMixin
import pandas as pd
from gensim.models import KeyedVectors
import numpy as np
#import gensim.downloader as api


class Preprocessor(TransformerMixin):
    """preprocesses the data with Spacy"""
    def __init__(self, tokenize, lemmatize, spacy_pipeline, lexicon={}):
        self.processors = []
        self.tokens_from_lexicon = 0

        if tokenize:
            #self.processors.append(self.tokenize_with_spacy(spacy_pipeline))
            self.processors.append(self.tokenize_with_tweettokenizer())
        if lemmatize:
            self.processors.append(self.lemmatize_with_spacy(spacy_pipeline))
        if lexicon:
            self.processors.append(self.identify_in_lexicon(lexicon))

    def transform(self, data):
        for p in self.processors:
            data = p(data)
        return data

    def fit_transform(self, data, y=None):
        return self.transform(data)


    def tokenize_with_tweettokenizer(self):
        tokenizer = TweetTokenizer({'reduce_len': True, 'strip_handles': True, 'preserve_case': False})

        def tweet_tokenizer(data):
            print('Tokenize data...')
            return [' '.join(tokenizer.tokenize(tweet)) for tweet in data]
        
        return tweet_tokenizer


    def tokenize_with_spacy(self, spacy_pipeline):
        nlp = spacy.load(spacy_pipeline)

        def spacy_tokenizer(data):
            print('Tokenize data...')
            return [' '.join([token.text for token in nlp(tw)]) for tw in data]
        
        return spacy_tokenizer


    def lemmatize_with_spacy(self, spacy_pipeline):
        nlp = spacy.load(spacy_pipeline)

        def spacy_lemmatizer(data):
            print('Lemmatize data...')
            return [' '.join([token.lemma_ for token in nlp(tw)]) for tw in data]
        
        return spacy_lemmatizer
    

    def identify_in_lexicon(self, lexicon):
        """replaces words in a tweet by a label from a lexicon; defaults to 'NEUTRAL'"""

        def apply_lexicon(data):
            self.tokens_from_lexicon = 0
            processed = []
            for tw in data:
                processed_tweet = []
                for token in tw.split():
                    lex_id = 'neutral'
                    if token in lexicon:
                        lex_id = lexicon[token]
                        self.tokens_from_lexicon += 1
                    processed_tweet.append(lex_id.upper())
                processed.append(' '.join(t for t in processed_tweet))
            return processed

        return apply_lexicon


def hate_lexicon(lexicon_path):

    Dlex = {}
    lex_df = pd.read_csv(lexicon_path)

    for i, row in lex_df.iterrows():
        entry = lex_df.loc[i]['entry']
        label = lex_df.loc[i]['label']
        Dlex[entry] = label
    
    return Dlex


class Text2Embedding(TransformerMixin):

    def __init__(self, w2vfile):
        self.w2vfile = w2vfile

    def fit_transform(self, X, parameters=[]):
        #model = api.load(self.w2vfile)
        print('Convert text to word embeddings...')
        model = KeyedVectors.load_word2vec_format(self.w2vfile, binary=True)
        n_d = model.vector_size
        data = []
        for tweet in X:
            tokens = tweet.split()
            tweet_matrix = np.array([model[t] for t in tokens if t in model.key_to_index])
            if len(tweet_matrix) == 0:
                data.append(np.zeros(n_d))
            else:
                data.append(np.mean(tweet_matrix, axis=0))
        return np.array(data)

    def fit(self, X):
        return self.fit_transform(X)
    
    def transform(self, X):
        return self.fit_transform(X)
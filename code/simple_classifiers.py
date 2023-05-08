import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from utils import Preprocessor, hate_lexicon, Text2Embedding

def run(train_data, test_data, output_file, clf, frm,
        embed_model, tokenize, lemmatize, spacy_pipeline, lexicon):
    
    # LOAD DATA
    train_data = pd.read_csv(train_data)
    test_data = pd.read_csv(test_data)
    train_X, train_y = train_data['text'], train_data['label']
    test_X, test_y = test_data['text'], test_data['label']

    # SET PREPROCESSOR
    preprocessor = Preprocessor(tokenize, lemmatize, spacy_pipeline)

    # SET REPRESENTATIONS
    if frm == 'counts':
        representation = CountVectorizer()
    elif frm == 'tfidf':
        representation = TfidfVectorizer()
    elif frm == 'embedding':
        representation = Text2Embedding(embed_model)
    
    # SET CLASSIFIER
    if clf == 'svm':
        classifier = LinearSVC(max_iter=10000, dual=False, C=0.1)
    elif clf == 'naive_bayes':
        classifier = MultinomialNB()
    

    # SET PIPELINE
    if lexicon != None:
        lex_dic = hate_lexicon(lexicon)
        lex_prep = Preprocessor(tokenize, lemmatize, spacy_pipeline, lex_dic)
        lex_repr = CountVectorizer()
        combined_features = FeatureUnion([
            ('token_features', Pipeline([('prep1', preprocessor), ('repr1', representation)])),
            ('lexicon_features', Pipeline([('prep2', lex_prep), ('repr2', lex_repr)]))])
        pipeline = Pipeline([('features', combined_features), ('clf', classifier)])
    else:
        pipeline = Pipeline([('prep', preprocessor), ('frm', representation), ('clf', classifier)])

    # RUN PIPELINE  
    print('Train model...')
    pipeline.fit(train_X, train_y)

    if lexicon:
        n = pipeline.steps[0][1].transformer_list[1][1].steps[0][1].tokens_from_lexicon
        print(f"   -- Found {n} tokens in resource lexicon")

    # PREDICT AND EVALUATE TEST SET
    print('Predict test data...')
    preds = pipeline.predict(test_X)
    print(classification_report(test_y, preds))
    
    test_data['prediction'] = pd.Series(preds)
    test_data.to_csv(output_file, index=False)


if __name__ == '__main__':

    train_data = '' # path to csv with 'text' and 'label' columns
    test_data = '' # path to csv with 'text' and 'label' columns
    output_file = '' # path to csv
    clf = 'naive_bayes' # 'svm' or 'naive_bayes'
    frm = 'counts' # 'counts' or 'tfidf' or 'embedding'
    embed_model = None # if frm == 'embedding': a path to binary w2v-model 
    tokenize = True
    lemmatize = False
    spacy_pipeline = None # if lemmatize == True: e.g. 'de_core_news_sm' for German or 'en_core_web_sm' for English
    # ^^^ make sure that spacy pipeline is downloaded first via: python -m spacy download [spacy_pipeline]
    lexicon = None # path to csv with 'entry' and 'label' columns or None

    run(train_data, test_data, output_file, clf, frm,
        embed_model, tokenize, lemmatize, spacy_pipeline, lexicon)

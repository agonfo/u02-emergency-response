import sys
import sqlalchemy
import re
import pandas as pd
import numpy as np
import nltk
import pickle

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer


def load_data(database_filepath):
    '''
    load data from SQL
    INPUT: str - name of the file
    OUTPUT: features for ML pipeline
    '''
    # read in file
    engine = create_engine(f"sqlite:///{database_filepath}")

    # load to database
    df = pd.read_sql_table(database_filepath, engine)
    

    # define features and label arrays
    X = df.message.values
    Y = df.drop(columns=['message' , 'original' , 'id' , 'genre']).values
    category_names = df.drop(columns=['message' , 'original' , 'id' , 'genre']).columns.tolist()

    return X, Y , category_names


def tokenize(text):
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    stop_words = stopwords.words("english")

    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens :
        if tok not in stop_words:
            clean_tok = lemmatizer.lemmatize(tok).lower().strip()
            clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    machine learning model pipeline
    INPUT:none
    OUTPUT: Machine learning model
    Note: the best metaparameters after a GridSearchCV are : 
    {'clf__estimator__n_estimators': 50, 'vect__max_df': 0.75, 'vect__ngram_range': (1, 2)}
    but it takes more time to process and the accuaracy and f1-score are almost the same
    '''

    model_pipeline = Pipeline([
        ('vect' , CountVectorizer(tokenizer=tokenize)),
        ('tfidf' , TfidfTransformer()),             
        ('clf' , MultiOutputClassifier(RandomForestClassifier()))
    ])

    return model_pipeline


def evaluate_model(model, X_test, Y_test, category_names):

    Y_pred = model.predict(X_test)

    for i in range(len(np.transpose(Y_pred))):
        print(category_names[i] , classification_report(np.transpose(Y_test)[i], np.transpose(Y_pred)[i]))

    accuracy = (Y_pred == Y_test).mean()
    print("Accuracy:", accuracy)


def save_model(model, model_filepath):
    '''
    export model as a pickle file
    INPUT:  model:
            model_filepath : str - name of the model filepath - extention .sav
    OUTPUT: Machine learning model
    Note: the best metaparameters after a GridSearchCV are : 
    {'clf__estimator__n_estimators': 50, 'vect__max_df': 0.75, 'vect__ngram_range': (1, 2)}
    but it takes more time to process and the accuaracy and f1-score are almost the same
    '''
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
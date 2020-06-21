import json
import plotly
import pandas as pd
import joblib

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar , Pie
from sqlalchemy import create_engine


app = Flask(__name__)

# TODO: change names of the databes filename (.db) and the model name (.pkl)
database_filename = 'DisasterResponse.db'
model_name = 'classifier.pkl'

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine(f'sqlite:///../data/{database_filename}')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load(f"../models/{model_name}")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    od_columns = ['request' , 'offer']
    od_count = []

    cat_columns = df.drop(columns=['message' , 'original' , 'id' , 'genre' , 'related' , 'request' , 'offer' , 'aid_related', 'weather_related' , 'direct_report', 'other_aid']).columns.tolist()
    cat_count = []

    for col in od_columns:
        od_count.append(df.groupby(col).count()['id'][1])
    
    offer_demand_counts = pd.Series(od_count , index=od_columns)
    offer_demand_names = list(offer_demand_counts.index)

    for col in cat_columns:
        serie = df.groupby(col).count()['id']
        if serie.size >= 2:
            cat_count.append(serie[1])
        else:
            cat_count.append(0)

    categorie_counts = pd.Series(cat_count , index=cat_columns).sort_values(ascending=False)
    categorie_names = list(categorie_counts.index)

    
    # create visuals
    graphs = [
        {
            'data': [
                Pie(
                    labels=categorie_names,
                    values=categorie_counts
                )
            ],

            'layout': {
                'title': 'Distribution Categories'            
            }
        },
        {
            'data': [
                Bar(
                    x=offer_demand_names,
                    y=offer_demand_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Offer and Request',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': " "
                }
            }
        },
        {
            'data': [
                Bar(
                    x=categorie_names,
                    y=categorie_counts
                )
            ],

            'layout': {
                'title': 'Distribution Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': " "
                }
            }
        } 
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
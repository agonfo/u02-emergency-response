import sys
import pandas as pd
import sqlalchemy

from sqlalchemy import create_enginep


def load_data(messages_filepath, categories_filepath):
    '''
    Load data from csv files
    INPUT: str - data file path
    OUTPUT: DataFrame 
    '''
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')

    return df


def clean_data(df):
    '''
    reorganize and clean data
    INPUT: DataFrame
    OUTPUT: clean DataFrame 
    '''

    categories = df['categories'].str.split(pat=';' , expand=True)
    row = categories.iloc[0].tolist()
    category_colnames = [x.split('-',1)[0] for x in row]
    categories.columns = category_colnames

    for column in categories:
        categories[column] = categories[column].astype(str).str[slice(-1, -2 , -1)]    
        categories[column] = categories[column].astype(int)

    df = df.drop(columns=['categories'])
    df = df.join(categories).drop_duplicates()

    return df

def save_data(df, database_filename):
    '''
    save clean DataFrame as SQLite
    IMPUT:  df : DataFrame
            database_filename: str - database file name
    OUTPUT: none
    '''
    engine = create_engine(f"sqlite:///{database_filename}.db")
    df.to_sql(database_filename, engine, index=False) 

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
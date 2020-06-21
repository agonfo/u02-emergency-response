# Disaster Response Pipeline Project

### Table of Contents

1. [Installation and Development Environment](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Licensing, Authors, and Acknowledgements](#licensing)

## Instructions and Development Environment<a name="installation"></a>

The libraries used are the common - pandas and matplotlib - for more information please refer to `requiremenst.txt`
- The code should run with no issues using Python 3.7

NOTE: If necessary, please in line 18 and 19 of run.py , change the database_filenames and model_name used when calling process_data & train_classifier

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run the web app. (make sure you are in the app's directory)
    `python run.py`

3. Go to http://0.0.0.0:3001/ or http://localhost:3001/ to visualize the webpage


## Project Motivation<a name="motivation"></a>

This project is the second proyect for Udacity Data science nanodegree program. to create a EPL and a MachinLearning Pipeline:

1. Create a ETL 
2. Create a ML Pipeline
3. Webapp to deploy the program

## File Descriptions <a name="files"></a>

There are three main folders:
1. app - contain the web app for deployment (where run.py and html code is located )
2. data - folder where data and new data is store - with a process_data.py file to read the data source
3. models - model is located in train_classifier.py and the folder to save the model as .pkl

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Thanks to Udacity for the exercice and the base code to develop this proyect
Feel free to use the code here as you like! 

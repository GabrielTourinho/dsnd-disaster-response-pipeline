# Disaster Response Pipeline

### Table of Content
1. [Installation](#installation)
    1. [Instructions](#instructions)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>
| Package | Version |
|---------|---------|
| Python | 3.9 |
| Pandas | 1.3.3 |
| SQLAlchemy | 1.4.25 |
| NLTK | 3.6.3 |
| Scikit-learn | 1.0 |
| Flask | 2.0.1 |
| Plotly | 5.3.1 |
| Joblib | 1.0.1 |

### Instructions <a name="instructions"></a> 
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`   

3. Go to http://0.0.0.0:3001/
## Project Motivation <a name="motivation"></a>

This project has the aim of analyzing disaster data from [Figure Eight](https://appen.com/) to build a model for an API that classifies disaster messages.

## File Descriptions <a name="files"></a>
```
.
├── app
│   ├── run.py
│   └── templates
│       ├── go.html
│       └── master.html
├── data
│   ├── disaster_categories.csv
│   ├── disaster_messages.csv 
│   ├── DisasterResponse.db
│   └── process_data.py
├── models
│   ├── classifier.pkl*
│   └── train_classifier.py
└── README.md

* File will be generated once scripts are executed
```

## Licensing, Authors, and Acknowledgements <a name="licensing"></a>
- Project designed by [Udacity](https://udacity.com).
- Dataset from Figure Eight.
- Author [Gabriel Tourinho](https://github.com/GabrielTourinho)

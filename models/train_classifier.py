import sys
import pandas as pd
import re
import nltk
import pickle

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")


def load_data(database_filepath):
    """Function that loads data from database

    Args:
        database_filepath: database filepath

    Returns:
        X: Feature variables
        y: Target values

    """

    engine = create_engine("sqlite:///{}".format(database_filepath))
    df = pd.read_sql_table("DisasterMessages", engine)
    X = df["message"]
    y = df[df.columns[5:]]

    return X, y


def tokenize(text):
    """Function that tokenizes a given text

    Args:
        text: text to be tokenized

    Returns:
        tokens: tokenized words from text

    """

    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    #  normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    tokens = word_tokenize(text)

    # lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


def build_model():
    """Function that builds a machine learning model

    Args:
        None

    Returns:
        cv: classification model

    """

    # build machine learning pipeline
    pipeline = Pipeline(
        [
            ("vect", CountVectorizer(tokenizer=tokenize)),
            ("tfidf", TfidfTransformer()),
            ("clf", MultiOutputClassifier(RandomForestClassifier())),
        ]
    )

    parameters = {
        "vect__ngram_range": ((1, 1), (1, 2)),  # unigrams and bigrams
        "tfidf__use_idf": (True, False),
        "clf__estimator__n_estimators": [50, 100],
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """Function that evaluates a model using sklearn classification_report
    function

    Args:
        model: machine learning model
        X_test: test portion of the Feature variables
        Y_test: test portion of the classification
        category_names: category names

    Returns:
        None

    """

    y_pred = model.predict(X_test)

    print(classification_report(Y_test, y_pred, target_names=category_names))


def save_model(model, model_filepath):
    """Function that exports the model as a pickle file

    Args:
        model: machine learning model
        model_filepath: filepath

    Returns:
        None

    """

    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print("Loading data...\n    DATABASE: {}".format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print("Building model...")
        model = build_model()

        print("Training model...")
        model.fit(X_train, Y_train)

        print("Evaluating model...")
        evaluate_model(model, X_test, Y_test, category_names)

        print("Saving model...\n    MODEL: {}".format(model_filepath))
        save_model(model, model_filepath)

        print("Trained model saved!")

    else:
        print(
            "Please provide the filepath of the disaster messages database "
            "as the first argument and the filepath of the pickle file to "
            "save the model to as the second argument. \n\nExample: python "
            "train_classifier.py ../data/DisasterResponse.db classifier.pkl"
        )


if __name__ == "__main__":
    main()

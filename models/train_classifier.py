import sys
import pandas as pd
import re
import nltk
import pickle

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator, TransformerMixin
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
nltk.download("averaged_perceptron_tagger")

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """Starting Verb Extractor class

    This class extracts the starting verb of a sentence,
    creating a new feature for the classifier.

    """
    def starting_verb(self, text):
        text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def load_data(database_filepath):
    """Function that loads data from database

    Args:
        database_filepath: database filepath

    Returns:
        X: Feature variables
        y: Target values
        category_names: Name of each category

    """

    engine = create_engine("sqlite:///{}".format(database_filepath))
    df = pd.read_sql_table("DisasterResponseTable", engine)
    X = df["message"]
    y = df[df.columns[5:]]
    category_names = y.columns

    return X, y, category_names


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
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            ('starting_verb', StartingVerbExtractor())
        ])),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        "features__text_pipeline__tfidf__use_idf": (True, False),
        "clf__estimator__n_estimators": [50, 100],
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test):
    """Function that evaluates a model using sklearn classification_report
    function

    Args:
        model: machine learning model
        X_test: test portion of the Feature variables
        Y_test: test portion of the classification

    Returns:
        None

    """

    y_pred = model.predict(X_test)

    i = 0
    for col in Y_test:
        print('Category: {}'.format(col))
        print(classification_report(Y_test[col], y_pred[:, i]))
        i += 1

    accuracy = (y_pred == Y_test.values).mean()
    print('The model accuracy is {:.2f}'.format(accuracy))


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
        evaluate_model(model, X_test, Y_test)

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

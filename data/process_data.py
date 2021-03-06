import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Function that loads the data from csv files

    Args:
        messages_filepath: messages' dataset filepath
        categories_filepath:  categories' dataset filepath

    Returns:
        df: DataFrame from the merged messages and categories datasets

    """

    # load messages dataset
    messages = pd.read_csv(messages_filepath)

    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = pd.merge(messages, categories, how="outer", on="id")

    return df


def clean_data(df):
    """Function that cleans the DataFrame

    Args:
        df: DataFrame to be cleaned

    Returns:
        df: cleaned DataFrame

    """

    # create a dataframe of the 36 individual category columns
    categories = df["categories"].str.split(pat=";", expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # create a list of new column names for categories
    category_colnames = row.apply(lambda x: x.strip("-01"))

    # rename the columns of 'categories'
    categories.columns = category_colnames

    # convert category values to numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # make every category binary
    for col in categories.columns:
        categories.loc[(categories[col] > 1)] = 1

    # drop the original categories column from 'df'
    df = df.drop(columns="categories", axis=1)

    # concatenate the original dataframe with the new 'categories' dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    """Function that saves the data to a database

    Args:
        df: DataFrame
        database_filename:  database filepath

    Returns:
        None

    """

    engine = create_engine("sqlite:///{}".format(database_filename))
    df.to_sql("DisasterResponseTable", engine, index=False, if_exists="replace")


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(
            "Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}".format(
                messages_filepath, categories_filepath
            )
        )
        df = load_data(messages_filepath, categories_filepath)

        print("Cleaning data...")
        df = clean_data(df)

        print("Saving data...\n    DATABASE: {}".format(database_filepath))
        save_data(df, database_filepath)

        print("Cleaned data saved to database!")

    else:
        print(
            "Please provide the filepaths of the messages and categories "
            "datasets as the first and second argument respectively, as "
            "well as the filepath of the database to save the cleaned data "
            "to as the third argument. \n\nExample: python process_data.py "
            "disaster_messages.csv disaster_categories.csv "
            "DisasterResponse.db"
        )


if __name__ == "__main__":
    main()


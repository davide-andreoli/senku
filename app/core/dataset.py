from helpers.csv_loader import CSVLoader
import pandas as pd
import os


def summarize_data(df: pd.DataFrame):
    num_rows = df.shape[0]

    stats = f"Dataset loaded successfully!\n\n" f"- Rows (valid haikus): {num_rows}\n"

    sample = df.head()
    return stats, sample, True


def load_default_dataset():
    csv_loader = CSVLoader()
    csv_loader.load_default_dataset()
    df = pd.read_csv("dataset/haiku/valid-haikus.csv")
    return summarize_data(df)


def load_existing_data():
    if os.path.exists("dataset/haiku/valid-haikus.csv"):
        df = pd.read_csv("dataset/haiku/valid-haikus.csv")
        return summarize_data(df)
    return (
        "No data loaded yet. Please press 'Load default data'.",
        None,
        False,
    )

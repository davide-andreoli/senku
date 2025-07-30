import gradio as gr
from helpers.csv_loader import CSVLoader
import pandas as pd
import os


def summarize_data(df: pd.DataFrame):
    num_rows = df.shape[0]

    stats = f"Dataset loaded successfully!\n\n" f"- Rows (valid haikus): {num_rows}\n"

    sample = df.head()
    return stats, sample, gr.update(visible=True)


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
        gr.update(visible=False),
    )


with gr.Blocks() as dataset:
    with gr.Row():
        gr.Markdown("# Dataset")

    with gr.Row():
        gr.Markdown("""The dataset for Senku is a simple csv file containing three columns, one for each haiku row.
                    Data can be loaded from the default dataset, which as of right now is [Haiku Dataset](https://www.kaggle.com/datasets/hjhalani30/haiku-dataset/data) on Kaggle, or from a custom csv file.
                    When loaded the data will be validated using an haiku validator to make sure that the structure is consistent with the 5-7-5 syllable pattern.
                    All valid haikus will be collected into the `dataset/haiku/valid-haikus.csv` file, which will be used in training.
                    """)

    with gr.Row():
        gr.Markdown("## Load dataset")

    with gr.Row():
        load_button = gr.Button("Load default data")

    with gr.Row():
        gr.Markdown("## Dataset stats")

    status_output = gr.Markdown(label="Dataset stats")
    sample_output = gr.Dataframe(label="Sample data", visible=False)

    load_button.click(
        fn=load_default_dataset,
        inputs=[],
        outputs=[status_output, sample_output, sample_output],
    )

    dataset.load(
        fn=load_existing_data,
        inputs=[],
        outputs=[status_output, sample_output, sample_output],
    )

if __name__ == "__main__":
    dataset.launch()

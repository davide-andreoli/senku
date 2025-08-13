import gradio as gr
from app.core.dataset import load_default_dataset, load_existing_data


def gradio_load_default_dataset():
    stats, sample, visibility = load_default_dataset()
    return stats, sample, gr.update(visible=visibility)


def gradio_load_existing_data():
    stats, sample, visibility = load_existing_data()
    return stats, sample, gr.update(visible=visibility)


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
        fn=gradio_load_default_dataset,
        inputs=[],
        outputs=[status_output, sample_output, sample_output],
    )

    dataset.load(
        fn=gradio_load_existing_data,
        inputs=[],
        outputs=[status_output, sample_output, sample_output],
    )

if __name__ == "__main__":
    dataset.launch()

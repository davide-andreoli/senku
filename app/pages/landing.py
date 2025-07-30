import gradio as gr

with gr.Blocks() as landing:
    with gr.Row():
        gr.Markdown("# Senku")

    with gr.Row():
        gr.Markdown(
            "This is Senku, an app dedicated to experimenting with generative AI models on your laptop."
        )

    with gr.Row():
        gr.Markdown("## Project Overview")

    with gr.Row():
        gr.Markdown("""Senku is a lightweight transformer-based language model trained to generate haikus, short structured poems in 5-7-5 syllable format.
                    It is designed to be trained and run on a CPU, so the performances will be mediocre, but the aim is to explores minimal language model design, tokenization, training loops, and inference with a focus on tiny-scale LLM experimentation.""")

    with gr.Row():
        gr.Markdown("## App structure")

    with gr.Row():
        gr.Markdown("""In the tabs at the top of the page you will find:
                    - Home: this page, it contains general details about the project
                    - Dataset: in this page you can load the preset dataset or a custom dataset and explore the data it contains
                    - Training: in this page you can experiment with different model configurations and observe the training process
                    - Model: in this page you can play with the models that you trained before
                    """)


if __name__ == "__main__":
    landing.launch()

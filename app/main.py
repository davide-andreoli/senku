import gradio as gr

from pages.landing import landing
from pages.dataset import dataset
from pages.train import train

with gr.Blocks() as senku_app:
    with gr.Tab("Home"):
        landing.render()

    with gr.Tab("Dataset"):
        dataset.render()

    with gr.Tab("Train"):
        train.render()

if __name__ == "__main__":
    senku_app.launch()

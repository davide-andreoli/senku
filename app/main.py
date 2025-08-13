import gradio as gr

from app.pages.landing import landing
from app.pages.dataset import dataset
from app.pages.train import train
from app.pages.inference import inference

with gr.Blocks() as senku_app:
    with gr.Tab("Home"):
        landing.render()

    with gr.Tab("Dataset"):
        dataset.render()

    with gr.Tab("Train"):
        train.render()

    with gr.Tab("Inference"):
        inference.render()

if __name__ == "__main__":
    senku_app.launch()

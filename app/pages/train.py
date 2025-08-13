import gradio as gr
from app.core.train import validate_model, launch_training, resume_training
from app.core.common import list_available_checkpoints
from typing import Optional, List

stop_requested = False


def gradio_list_available_checkpoints():
    checkpoints = list_available_checkpoints()
    checkpoint_dict = {str(checkpoint): checkpoint for checkpoint in checkpoints}
    return checkpoint_dict


def gradio_resume_training(
    checkpoint_key: str, res_num_epochs: int, res_batch_size: int
):
    global stop_requested
    stop_requested = False

    checkpoint = gradio_list_available_checkpoints()[checkpoint_key]
    training_generator = resume_training(checkpoint, res_num_epochs, res_batch_size)
    for step in training_generator:
        if stop_requested:
            yield step[0], step[1], "Training manually stopped."
            break
        yield step


def gradio_validate_model(
    embedding_dimension: int = 128,
    context_length: int = 64,
    num_layers: int = 8,
    num_heads: int = 8,
    dropout: float = 0.1,
    bias: bool = False,
    tokenizer_strategy: str = "character",
):
    validation_output, validity = validate_model(
        embedding_dimension=embedding_dimension,
        context_length=context_length,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        bias=bias,
        tokenizer_strategy=tokenizer_strategy,
    )
    return validation_output, gr.update(visible=validity)


def load_model_details(checkpoint_key: str):
    checkpoint = gradio_list_available_checkpoints()[checkpoint_key]
    return checkpoint.checkpoint_details_string


def gradio_launch_training(
    embedding_dimension: int = 128,
    context_length: int = 64,
    num_layers: int = 8,
    num_heads: int = 8,
    dropout: float = 0.1,
    bias: bool = False,
    num_epochs: int = 50,
    batch_size: int = 32,
    tokenizer_strategy: str = "character",
    checkpoint_name: Optional[str] = None,
):
    global stop_requested
    stop_requested = False

    training_generator = launch_training(
        embedding_dimension=embedding_dimension,
        context_length=context_length,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        bias=bias,
        num_epochs=num_epochs,
        batch_size=batch_size,
        tokenizer_strategy=tokenizer_strategy,
        checkpoint_name=checkpoint_name,
    )

    for step in training_generator:
        if stop_requested:
            yield step[0], step[1], "Training manually stopped."
            break
        yield step


def reset_stop_flag():
    global stop_requested
    stop_requested = False


def set_stop_flag():
    global stop_requested
    stop_requested = True


with gr.Blocks() as train:
    with gr.Row():
        gr.Markdown("# Train")
    with gr.Tab("Existing model"):
        gr.Markdown("## Train an existing model")

        select_model_dropdown = gr.Dropdown(
            choices=list(gradio_list_available_checkpoints().keys()),
            label="Select model",
            interactive=True,
        )

        with gr.Column():
            gr.Markdown("## Selected model details")
            selected_model_output = gr.Markdown(label="Model details")

        with gr.Column():
            gr.Markdown("## Training settings")
            res_num_epochs = gr.Number(
                value=50,
                label="Number of epochs",
                info="The number of epochs to resume training for.",
            )
            res_batch_size = gr.Number(
                value=32,
                label="Batch size",
                info="The batch size to resume training with.",
            )
            resume_training_button = gr.Button("Resume training")

    with gr.Tab("New model"):
        gr.Markdown("## Train a new model")
        with gr.Column():
            gr.Markdown("## Model settings")
            # Model settings
            select_tokenizer_dropdown = gr.Dropdown(
                choices=["character", "syllable", "word"],
                label="Select tokenizer",
                interactive=True,
            )
            embedding_dimension = gr.Number(
                value=128,
                label="Embedding dimension",
                info="The size of the internal representation used to encode words or tokens. Higher values can capture more detail, but may require more memory and processing power.",
            )
            context_length = gr.Number(
                value=128,
                label="Context length",
                info="The maximum number of words or tokens the model looks at in one go. Longer context allows the model to consider more information when making predictions.",
            )
            num_layers = gr.Number(
                value=8,
                label="Number of layers",
                info="The number of processing steps (layers) in the model. More layers can improve understanding, but also make the model slower and more complex.",
            )
            num_heads = gr.Number(
                value=8,
                label="Number of attention heads",
                info="The number of ways the model splits its focus when understanding input. More heads allow the model to learn different types of patterns in parallel.",
            )
            dropout = gr.Number(
                value=0.1,
                label="Dropout",
                info="A method to prevent overfitting by randomly turning off parts of the model during training. A small value like 0.1 means 10% is turned off at each step.",
            )
            bias = gr.Checkbox(
                value=False,
                label="Use bias",
                info="Adds a small adjustable value to help the model learn better. Turning this on can slightly improve performance in some cases.",
            )

            validation_output = gr.Markdown(label="Model validation")

        with gr.Column():
            gr.Markdown("## Training settings")
            num_epochs = gr.Number(
                value=10,
                label="Number of epochs",
                info="The number of times the model will go through the entire training dataset. More epochs can help the model learn better, but too many might cause overfitting.",
            )
            batch_size = gr.Number(
                value=32,
                label="Batch size",
                info="The number of training examples the model processes at once before updating itself. Smaller batches use less memory, while larger batches can train faster but may need more resources.",
            )
            checkpoint_name = gr.Textbox(
                label="Checkpoint name",
                info="The name to give the checkpoint file. If left blank, a custom name will be used.",
            )
            start_training_button = gr.Button("Start training")

    with gr.Row():
        gr.Markdown("## Training progress")
    progress_bar_overall = gr.Slider(
        minimum=0,
        maximum=100,
        value=0,
        label="Overall Training Progress",
        interactive=False,
    )
    progress_bar_epoch = gr.Slider(
        minimum=0,
        maximum=100,
        value=0,
        label="Current Epoch Progress",
        interactive=False,
    )
    training_status = gr.Textbox(label="Training Status", interactive=False, lines=2)
    stop_training_button = gr.Button("Stop Training")

    model_settings: List[gr.Component] = [
        embedding_dimension,
        context_length,
        num_layers,
        num_heads,
        dropout,
        bias,
        select_tokenizer_dropdown,
    ]
    for component in model_settings:
        component.change(  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
            fn=gradio_validate_model,
            inputs=model_settings,
            outputs=[validation_output, start_training_button],
        )

    select_model_dropdown.change(
        fn=load_model_details,
        inputs=[select_model_dropdown],
        outputs=[selected_model_output],
    )
    train.load(
        fn=load_model_details,
        inputs=[select_model_dropdown],
        outputs=[selected_model_output],
    )

    start_training_button.click(
        fn=gradio_launch_training,
        inputs=[
            embedding_dimension,
            context_length,
            num_layers,
            num_heads,
            dropout,
            bias,
            num_epochs,
            batch_size,
            select_tokenizer_dropdown,
            checkpoint_name,
        ],
        outputs=[progress_bar_overall, progress_bar_epoch, training_status],
    )

    stop_training_button.click(
        fn=set_stop_flag,
        inputs=[],
        outputs=[],
    )

    resume_training_button.click(
        fn=gradio_resume_training,
        inputs=[select_model_dropdown, res_num_epochs, res_batch_size],
        outputs=[progress_bar_overall, progress_bar_epoch, training_status],
    )

if __name__ == "__main__":
    train.launch()

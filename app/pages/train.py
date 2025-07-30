import gradio as gr
from trainer.trainer import Trainer
from tokenizer.tokenizer import Tokenizer
from loaders.dataset import CSVListDataset
from models.gpt import GPTModel
import torch
import os


def get_model_and_config(
    embedding_dimension: int,
    context_length: int,
    num_layers: int,
    num_heads: int,
    dropout: float,
    bias: bool,
):
    tokenizer = Tokenizer()

    model_config = {
        "vocabulary_size": tokenizer.vocabulary_size,
        "embedding_dimension": embedding_dimension,
        "context_length": context_length,
        "number_of_layers": num_layers,
        "number_of_attention_heads": num_heads,
        "dropout": dropout,
        "bias": bias,
    }

    model = GPTModel(**model_config)
    print(model.checkpoint_name)
    return model, tokenizer


def validate_model(
    embedding_dimension: int = 128,
    context_length: int = 64,
    num_layers: int = 8,
    num_heads: int = 8,
    dropout: float = 0.1,
    bias: bool = False,
):
    is_invalid = False
    big_model = False
    checkpoint_exists = False

    invalid_lines = [
        "Model config is invalid.",
    ]

    if embedding_dimension % num_heads != 0:
        is_invalid = True
        invalid_lines.append(
            "Embedding dimension must be divisible by number of attention heads."
        )

    if is_invalid:
        return "\n\n".join(invalid_lines), gr.update(interactive=False)

    model, tokenizer = get_model_and_config(
        embedding_dimension, context_length, num_layers, num_heads, dropout, bias
    )

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, model.checkpoint_name)
    if os.path.exists(checkpoint_path):
        checkpoint_exists = True

    if model.total_size > 1024:
        big_model = True

    valid_lines = [
        "Model config is valid.\n",
        f"Total parameters: {model.total_parameters}.",
        f"Total size: {model.total_size} MB.",
    ]

    if big_model:
        valid_lines.append("This model might be too large to train on CPU.")

    if checkpoint_exists:
        valid_lines.append(f"Checkpoint found at: {model.checkpoint_name}.")
    else:
        valid_lines.append("No checkpoint found.")

    valid_string = "\n\n".join(valid_lines)

    return valid_string, gr.update(interactive=True)


def launch_training(
    embedding_dimension: int = 128,
    context_length: int = 64,
    num_layers: int = 8,
    num_heads: int = 8,
    dropout: float = 0.1,
    bias: bool = False,
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    reset: bool = False,
):
    torch.manual_seed(42)

    model, tokenizer = get_model_and_config(
        embedding_dimension, context_length, num_layers, num_heads, dropout, bias
    )
    checkpoint_dir = "checkpoints"

    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, model.checkpoint_name)

    dataset = CSVListDataset(
        file_path="dataset/haiku/valid-haikus.csv",
        tokenizer=tokenizer,
        context_length=context_length,
    )

    train_dataloader, validation_dataloader = dataset.get_train_validation_loader(
        batch_size=batch_size, num_workers=0
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
    )

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    trainer = Trainer(
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        model=model,
        optimizer=optimizer,
        loss=loss_fn,
        checkpoint_dir=checkpoint_path,
        reset_checkpoint=reset,
    )

    train_losses, val_losses = trainer.train(
        number_of_epochs=num_epochs,
        evaluation_mode="after_epoch",
    )

    # TODO: give visual feedback for the training
    return "Training complete!"


with gr.Blocks() as train:
    with gr.Row():
        gr.Markdown("# Train")
    with gr.Column():
        gr.Markdown("## Model settings")
        # Model settings
        embedding_dimension = gr.Number(value=128, label="Embedding dimension")
        context_length = gr.Number(value=128, label="Context length")
        num_layers = gr.Number(value=8, label="Number of layers")
        num_heads = gr.Number(value=8, label="Number of attention heads")
        dropout = gr.Number(value=0.1, label="Dropout")
        bias = gr.Checkbox(value=False, label="Use bias")
        validation_output = gr.Markdown(label="Model validation")

    with gr.Column():
        gr.Markdown("## Training settings")
        num_epochs = gr.Number(value=50, label="Number of epochs")
        batch_size = gr.Number(value=32, label="Batch size")
        reset = gr.Checkbox(value=False, label="Reset checkpoint")
        start_training_button = gr.Button("Start training")

    with gr.Row():
        gr.Markdown("## Training progress")

    training_output = gr.Markdown(
        "Training outcome will be shown here", label="Model validation"
    )

    model_settings = [
        embedding_dimension,
        context_length,
        num_layers,
        num_heads,
        dropout,
        bias,
    ]

    for component in model_settings:
        component.change(
            fn=validate_model,
            inputs=model_settings,
            outputs=[validation_output, start_training_button],
        )

    start_training_button.click(
        fn=launch_training,
        inputs=[
            embedding_dimension,
            context_length,
            num_layers,
            num_heads,
            dropout,
            bias,
            num_epochs,
            batch_size,
            reset,
        ],
        outputs=[training_output],
    )

if __name__ == "__main__":
    train.launch()

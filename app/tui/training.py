import questionary
from core.train import validate_model, launch_training
from rich import print


def train_model_flow():
    print("\nEnter model configuration:")
    emb = questionary.text("Embedding dimension", default="128").ask()
    ctx = questionary.text("Context length", default="128").ask()
    layers = questionary.text("Number of layers", default="8").ask()
    heads = questionary.text("Attention heads", default="8").ask()
    dropout = questionary.text("Dropout", default="0.1").ask()
    bias = questionary.confirm("Use bias?", default=False).ask()

    print("\nValidating config...")
    validation, is_valid = validate_model(
        embedding_dimension=int(emb),
        context_length=int(ctx),
        num_layers=int(layers),
        num_heads=int(heads),
        dropout=float(dropout),
        bias=bool(bias),
    )
    print(validation)
    if not is_valid:
        return

    epochs = questionary.text("Epochs", default="50").ask()
    batch = questionary.text("Batch size", default="32").ask()
    reset = questionary.confirm("Reset existing checkpoint?", default=False).ask()

    print("\nStarting training...\n")
    result = launch_training(
        embedding_dimension=int(emb),
        context_length=int(ctx),
        num_layers=int(layers),
        num_heads=int(heads),
        dropout=float(dropout),
        bias=bool(bias),
        num_epochs=int(epochs),
        batch_size=int(batch),
        reset=reset,
    )
    print(result)

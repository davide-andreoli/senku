import questionary
from core.common import list_available_checkpoints
from core.train import validate_model, launch_training, resume_training
from rich import print


def train_model_flow():
    selection = questionary.select(
        "What type of training would you lke to do?",
        choices=["Train an existing model", "Train a model from scratch"],
    ).ask()

    if selection == "Train an existing model":
        train_existing_model()
    elif selection == "Train a model from scratch":
        train_model_from_scratch_flow()


def train_existing_model():
    checkpoints = list_available_checkpoints()
    if not checkpoints:
        print("No checkpoints available. Train a model first.")
        return

    choices = [
        questionary.Choice(title=str(checkpoint), value=checkpoint)
        for checkpoint in checkpoints
    ]

    chosen_checkpoint = questionary.select(
        "Select a model checkpoint:", choices=choices
    ).ask()

    epochs = questionary.text("Epochs", default="50").ask()
    batch = questionary.text("Batch size", default="32").ask()

    result = resume_training(chosen_checkpoint, int(epochs), int(batch))
    print(result)


def train_model_from_scratch_flow():
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
    checkpoint_name = questionary.text(
        "Checkpoint name (leave blank to use a default name)", default=""
    ).ask()

    if checkpoint_name == "":
        checkpoint_name = None

    if checkpoint_name is not None and not checkpoint_name.endswith(".pt"):
        checkpoint_name += ".pt"

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
        checkpoint_name=checkpoint_name,
    )
    print(result)

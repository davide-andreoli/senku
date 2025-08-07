from trainer.trainer import Trainer
from loaders.dataset import CSVListDataset
from models.gpt import GPTModel
import torch
from helpers.checkpoint import SenkuCheckpoint
from helpers.classes import SenkuTokenizer
from typing import Optional, Dict, Any, cast


def get_model_and_config(
    embedding_dimension: int,
    context_length: int,
    num_layers: int,
    num_heads: int,
    dropout: float,
    bias: bool,
    tokenizer_strategy: str = "character",
):
    tokenizer = SenkuTokenizer.from_strategy(tokenizer_strategy)

    model_config: Dict[str, Any] = {
        "vocabulary_size": tokenizer.vocabulary_size,
        "embedding_dimension": embedding_dimension,
        "context_length": context_length,
        "number_of_layers": num_layers,
        "number_of_attention_heads": num_heads,
        "dropout": dropout,
        "bias": bias,
    }

    model = GPTModel(**model_config)
    return model, tokenizer


def validate_model(
    embedding_dimension: int = 128,
    context_length: int = 128,
    num_layers: int = 8,
    num_heads: int = 8,
    dropout: float = 0.1,
    bias: bool = False,
    tokenizer_strategy: str = "character",
):
    is_invalid = False
    big_model = False

    invalid_lines = [
        "Model config is invalid.",
    ]

    if embedding_dimension % num_heads != 0:
        is_invalid = True
        invalid_lines.append(
            "Embedding dimension must be divisible by number of attention heads."
        )

    if is_invalid:
        return "\n\n".join(invalid_lines), False

    model, _ = get_model_and_config(
        embedding_dimension,
        context_length,
        num_layers,
        num_heads,
        dropout,
        bias,
        tokenizer_strategy,
    )

    if model.total_size > 1024:
        big_model = True

    valid_lines = [
        "Model config is valid.\n",
        f"Total parameters: {model.total_parameters}.",
        f"Total size: {model.total_size} MB.",
    ]

    if big_model:
        valid_lines.append("This model might be too large to train on CPU.")

    valid_string = "\n\n".join(valid_lines)

    return valid_string, True


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
    checkpoint_name: Optional[str] = None,
    tokenizer_strategy: str = "character",
):
    torch.manual_seed(42)  # type: ignore[reportUnknownMemberType]

    model, tokenizer = get_model_and_config(
        embedding_dimension,
        context_length,
        num_layers,
        num_heads,
        dropout,
        bias,
        tokenizer_strategy,
    )

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
        checkpoint_name=checkpoint_name,
        tokenizer_strategy=tokenizer.strategy,
    )

    _, _ = trainer.train(
        number_of_epochs=num_epochs,
        evaluation_mode="after_epoch",
    )

    # TODO: give visual feedback for the training
    return "Training complete!"


def resume_training(
    checkpoint: SenkuCheckpoint,
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
):
    model = checkpoint.instantiate_model()
    tokenizer = checkpoint.instantiate_tokenizer()
    optimizer = checkpoint.instantiate_optimizer(model)
    dataset = CSVListDataset(
        file_path="dataset/haiku/valid-haikus.csv",
        tokenizer=tokenizer,
        context_length=cast(int, model.context_length),
    )

    train_dataloader, validation_dataloader = dataset.get_train_validation_loader(
        batch_size=batch_size, num_workers=0
    )

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    trainer = Trainer(
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        model=model,
        optimizer=optimizer,
        loss=loss_fn,
        checkpoint_name=checkpoint.checkpoint_name,
        epoch=checkpoint.epoch,
        tokenizer_strategy=tokenizer.strategy,
    )

    _, _ = trainer.train(
        number_of_epochs=num_epochs,
        evaluation_mode="after_epoch",
    )

    # TODO: give visual feedback for the training
    return "Training complete!"

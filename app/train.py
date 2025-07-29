from models.gpt import GPTModel
from tokenizer.tokenizer import Tokenizer
from loaders.dataset import CSVListDataset
from trainer.trainer import Trainer
import torch
import os
import argparse
import logging
import json
import matplotlib.pyplot as plt


def plot_training_curves(train_losses, val_losses, save_path="training_curves.png"):
    """Plot and save training curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss", alpha=0.7)
    plt.plot(val_losses, label="Validation Loss", alpha=0.7)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Training curves saved to {save_path}")


def save_model_config(config, path="model_config.json"):
    """Save model configuration for inference"""
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Model config saved to {path}")


def calculate_model_size(
    embedding_dim, context_length, num_layers, num_heads, vocab_size
):
    """Estimate model size for given parameters"""
    token_emb = vocab_size * embedding_dim
    pos_emb = context_length * embedding_dim

    layer_norm = 2 * embedding_dim * 2
    attention = 4 * embedding_dim * embedding_dim
    feedforward = (
        embedding_dim * (4 * embedding_dim) + (4 * embedding_dim) * embedding_dim
    )

    block_params = layer_norm + attention + feedforward
    total_transformer = num_layers * block_params

    final_norm = embedding_dim * 2
    output_head = embedding_dim * vocab_size

    total_params = token_emb + pos_emb + total_transformer + final_norm + output_head
    size_mb = (total_params * 4) / (1024 * 1024)

    return total_params, size_mb


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Trainer", description="Train a GPT model")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument(
        "--reset", action="store_true", help="Reset the model checkpoint"
    )
    parser.add_argument(
        "--dataset_type",
        choices=["csv", "folder"],
        default="csv",
        help="Dataset type to use",
    )
    parser.add_argument(
        "--data_path", default="dataset/haiku/valid-haikus.csv", help="Path to dataset"
    )

    parser.add_argument(
        "-log",
        "--logging_level",
        default="info",
        help="Logging level (debug, info, warning, error)",
    )
    parser.add_argument(
        "--plot_curves", action="store_true", help="Plot training curves after training"
    )

    parser.add_argument(
        "--num_epochs", type=int, default=50, help="Number of epochs to train"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-4, help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for regularization",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help="Number of warmup steps for learning rate",
    )

    parser.add_argument(
        "--embedding_dimension", type=int, default=128, help="Embedding dimension"
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=64,
        help="Context length (shorter for haikus)",
    )
    parser.add_argument(
        "--num_layers", type=int, default=8, help="Number of transformer layers"
    )
    parser.add_argument(
        "--num_heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--bias", action="store_true", help="Use bias in linear layers")

    parser.add_argument(
        "--evaluation_frequency",
        type=int,
        default=100,
        help="Steps between evaluations",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of data loading workers (0 for main process)",
    )
    parser.add_argument(
        "--checkpoint_dir", default="checkpoints", help="Directory to save checkpoints"
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=50,
        help="Max sequence length for folder dataset",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=25,
        help="Stride for sliding window in folder dataset",
    )

    args = parser.parse_args()

    logging.basicConfig(level=args.logging_level.upper())

    torch.manual_seed(42)
    # torch.set_num_threads(args.num_workers)
    tokenizer = Tokenizer()
    logging.info(f"Tokenizer vocabulary size: {tokenizer.vocabulary_size}")

    model_config = {
        "vocabulary_size": tokenizer.vocabulary_size,
        "embedding_dimension": args.embedding_dimension,
        "context_length": args.context_length,
        "number_of_layers": args.num_layers,
        "number_of_attention_heads": args.num_heads,
        "dropout": args.dropout,
        "bias": args.bias,
    }
    evaluation_frequency = args.evaluation_frequency

    dataset = CSVListDataset(
        file_path="dataset/haiku/valid-haikus.csv",
        tokenizer=tokenizer,
        context_length=args.context_length,
    )
    train_dataloader, validation_dataloader = dataset.get_train_validation_loader(
        batch_size=32, num_workers=args.num_workers
    )

    model = GPTModel(**model_config)

    logging.info(f"Total parameters: {model.total_parameters()}")
    logging.info(f"Total size: {model.total_size()} MB")

    save_model_config(
        model_config, os.path.join(args.checkpoint_dir, "model_config.json")
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint.pt")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),  # GPT-style betas
    )
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    if args.reset and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        logging.info("Checkpoint removed")

    if os.path.exists(checkpoint_path):
        logging.info("Checkpoint exists")
        model.load_checkpoint(checkpoint_path)

    if args.train:
        trainer = Trainer(
            train_dataloader=train_dataloader,
            validation_dataloader=validation_dataloader,
            model=model,
            optimizer=optimizer,
            loss=loss_fn,
            checkpoint_path=checkpoint_path,
            reset_checkpoint=args.reset,
        )
        train_losses, val_losses = trainer.train(
            number_of_epochs=args.num_epochs,
            evaluation_frequency=args.evaluation_frequency,
            evaluation_mode="after_epoch",
        )

    if args.plot_curves and len(train_losses) > 0:
        plot_path = os.path.join(args.checkpoint_dir, "training_curves.png")
        plot_training_curves(train_losses, val_losses, plot_path)

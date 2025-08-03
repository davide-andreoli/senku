import torch
import os
from models.gpt import GPTModel
from tokenizer.tokenizer import CharacterTokenizer
from typing import List


class SenkuCheckpoint:
    def __init__(self, checkpoint_path: str) -> None:
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        self.checkpoint_path = checkpoint_path
        self.checkpoint = torch.load(self.checkpoint_path, weights_only=True)

        if self.checkpoint["application"] != "senku":
            raise ValueError("The checkpoint was not made with Senku.")

        self.architecture = self.checkpoint["architecture"]
        self.tokenizer_class_name = self.checkpoint["tokenizer_class_name"]
        self._extract_metadata(architecture=self.architecture)

    def _extract_metadata(self, architecture: str):
        if architecture == "transformer":
            self.embedding_dimension = self.checkpoint["embedding_dimension"]
            self.vocabulary_size = self.checkpoint["vocabulary_size"]
            self.context_length = self.checkpoint["context_length"]
            self.number_of_attention_heads = self.checkpoint[
                "number_of_attention_heads"
            ]
            self.bias = self.checkpoint["bias"]
            self.number_of_layers = self.checkpoint["number_of_layers"]
            self.dropout = self.checkpoint["dropout"]

    def instantiate_model(self) -> GPTModel:
        if self.architecture == "transformer":
            model = GPTModel(
                self.vocabulary_size,
                self.embedding_dimension,
                self.context_length,
                self.number_of_layers,
                self.dropout,
                self.bias,
                self.number_of_attention_heads,
            )
            return model

    def instantiate_tokenizer(self) -> CharacterTokenizer:
        if self.tokenizer_class_name == "CharacterTokenizer":
            return CharacterTokenizer()


class SenkuCheckpointManager:
    def __init__(self, checkpoint_dir="checkpoints") -> None:
        self.checkpoint_dir = checkpoint_dir

    def list_checkpoints(self) -> List[SenkuCheckpoint]:
        senku_checkpoints = []
        for checkpoint in os.listdir(self.checkpoint_dir):
            try:
                senku_checkpoint = SenkuCheckpoint(
                    checkpoint_path=os.path.join(self.checkpoint_dir, checkpoint)
                )
                senku_checkpoints.append(senku_checkpoint)
            except (FileNotFoundError, ValueError):
                continue
        return senku_checkpoints

    # TODO: Make abstract classes for senku models and tokenizers
    def save_checkpoint(
        self,
        model: GPTModel,
        tokenizer: CharacterTokenizer,
        destination_path: str,
        epoch: int = 0,
    ) -> None:
        if model.architecture == "transformer":
            torch.save(
                {
                    "application": "senku",
                    "architecture": "transformer",
                    # TODO: maybe add an attribute into to Tokenizer rather then relying on the class name
                    "tokenizer": tokenizer.__class__.__name__,
                    "epoch": epoch,
                    "embedding_dimension": model.embedding_dimension,
                    "vocabulary_size": model.vocabulary_size,
                    "context_length": model.context_length,
                    "number_of_attention_heads": model.number_of_attention_heads,
                    "number_of_layers": model.number_of_layers,
                    "bias": model.bias,
                    "dropout": model.dropout,
                    "model_state_dict": model.state_dict(),
                },
                os.path.join(self.checkpoint_dir, destination_path),
            )

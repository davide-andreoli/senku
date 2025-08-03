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
        self.type = self.checkpoint["text"]
        self.model_state_dict = self.checkpoint["model_state_dict"]
        self.optimizer_state_dict = self.checkpoint["optimizer_state_dict"]
        self.epoch = self.checkpoint["epoch"]
        self._extract_metadata(architecture=self.architecture)

    def __repr__(self) -> str:
        name = f"{self.architecture.capitalize()} - "
        if self.architecture == "transformer":
            name += f"Tokenizer: {self.tokenizer_strategy} - "
            name += f"Embedding dimension: {self.embedding_dimension} - "
            name += f"Context length: {self.context_length} - "
            name += f"Number of attention heads: {self.number_of_attention_heads} - "
            name += f"Number of layers: {self.number_of_layers}"

        return name

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
            self.tokenizer_strategy = self.checkpoint["tokenizer_strategy"]

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
            model.load_state_dict(self.model_state_dict)
            return model

    def instantiate_tokenizer(self) -> CharacterTokenizer:
        if self.tokenizer_strategy == "character":
            return CharacterTokenizer()


class SenkuCheckpointManager:
    def __init__(self, checkpoint_dir="checkpoints") -> None:
        self.checkpoint_dir = checkpoint_dir

    def checkpoint_exists(self, checkpoint_name: str) -> bool:
        if os.path.isfile(os.path.join(self.checkpoint_dir, checkpoint_name)):
            return True
        else:
            return False

    def delete_checkpoint(self, checkpoint_name: str) -> None:
        os.remove(os.path.join(self.checkpoint_dir, checkpoint_name))

    def get_checkpoint(self, checkpoint_name: str) -> SenkuCheckpoint:
        return SenkuCheckpoint(os.path.join(self.checkpoint_dir, checkpoint_name))

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
        model_architecture: str,
        model_type: str,
        tokenizer_strategy: str,
        destination_path: str,
        model_state_dict: dict,
        optimizer_state_dict: dict,
        epoch: int = 0,
        **kwargs,
    ) -> None:
        if model_architecture == "transformer":
            torch.save(
                {
                    "application": "senku",
                    "architecture": "transformer",
                    "model_type": model_type,
                    # TODO: maybe add an attribute into to Tokenizer rather then relying on the class name
                    "tokenizer_strategy": tokenizer_strategy,
                    "epoch": epoch,
                    "model_state_dict": model_state_dict,
                    "optimizer_state_dict": optimizer_state_dict,
                    **kwargs,
                },
                os.path.join(self.checkpoint_dir, destination_path),
            )

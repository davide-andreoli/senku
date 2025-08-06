import torch
import os
from models.gpt import GPTModel
from tokenizer.tokenizer import CharacterTokenizer, SyllableTokenizer, WordTokenizer
from typing import List, Dict, Any
from helpers.classes import SenkuModel, SenkuTokenizer


class SenkuCheckpoint:
    def __init__(self, checkpoint_path: str) -> None:
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        if not checkpoint_path.endswith(".pt"):
            raise ValueError(f"Checkpoint not found at {checkpoint_path}")

        self.checkpoint_path = checkpoint_path
        self.checkpoint_name = os.path.split(checkpoint_path)[1]
        self.checkpoint = torch.load(self.checkpoint_path, weights_only=True)  # type: ignore[reportUnknownMemberType]

        if self.checkpoint.get("application", None) != "senku":
            raise ValueError("The checkpoint was not made with Senku.")

        self.architecture = self.checkpoint["architecture"]
        self.type = self.checkpoint["model_type"]
        self.model_state_dict = self.checkpoint["model_state_dict"]
        self.optimizer_state_dict = self.checkpoint["optimizer_state_dict"]
        self.epoch = self.checkpoint["epoch"]
        self._extract_metadata(architecture=self.architecture)

    def __repr__(self) -> str:
        name = f"{os.path.split(self.checkpoint_path)[1]} - "
        name += f"{self.architecture.capitalize()} - "
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

    def instantiate_model(self) -> SenkuModel:  # type: ignore[reportReturnType]
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

    def instantiate_tokenizer(self) -> SenkuTokenizer:
        if self.tokenizer_strategy == "character":
            return CharacterTokenizer()
        elif self.tokenizer_strategy == "syllable":
            return SyllableTokenizer()
        elif self.tokenizer_strategy == "word":
            return WordTokenizer()
        else:
            raise ValueError(f"Unknown tokenizer strategy: {self.tokenizer_strategy}")

    @property
    def checkpoint_details_string(self) -> str:
        details = f"Model architecture: {self.architecture}  \n"
        details += f"Model type: {self.type}  \n"
        details += f"Number of epochs: {self.epoch}  \n"
        details += f"Tokenizer strategy: {self.tokenizer_strategy}  \n"
        details += f"Embedding dimension: {self.embedding_dimension}  \n"
        details += f"Context length: {self.context_length}  \n"
        details += f"Number of attention heads: {self.number_of_attention_heads}  \n"
        details += f"Number of layers: {self.number_of_layers}  \n"
        details += f"Dropout: {self.dropout}  \n"
        return details

    # TODO: find a better way to handle this
    def instantiate_optimizer(
        self,
        model: SenkuModel,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
    ):
        optimizer = torch.optim.AdamW(
            model.parameters(),  # type: ignore[reportUnknownMemberType]
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )
        optimizer.load_state_dict(self.optimizer_state_dict)
        return optimizer


class SenkuCheckpointManager:
    def __init__(self, checkpoint_dir: str = "checkpoints") -> None:
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
        senku_checkpoints: List[SenkuCheckpoint] = []
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
        model_state_dict: Dict[str, Any],
        optimizer_state_dict: Dict[str, Any],
        epoch: int = 0,
        **kwargs: Dict[str, Any],
    ) -> None:
        if model_architecture == "transformer":
            torch.save(  # type: ignore[reportUnknownMemberType]
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

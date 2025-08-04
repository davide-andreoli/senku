from abc import ABC, abstractmethod
from typing import List, Dict, Any
import torch


class SenkuModel(ABC):
    @abstractmethod
    def __init__(self):
        self.architecture: str
        self.model_type: str

    @property
    @abstractmethod
    def keyword_arguments(self) -> Dict[str, Any]:
        pass

    @property
    @abstractmethod
    def total_parameters(self) -> int:
        pass

    @property
    @abstractmethod
    def total_size(self) -> float:
        pass


class SenkuTokenizer(ABC):
    @abstractmethod
    def __init__(self):
        self.vocabulary_size: int
        self.strategy: str
        self.encode_dict: Dict[str, int]
        self.decode_dict: Dict[int, str]

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        pass

    @abstractmethod
    def decode(self, encoded_text: List[int]) -> str:
        pass

    @property
    @abstractmethod
    def pad_token_id(self) -> int:
        pass

    @property
    @abstractmethod
    def eos_token_id(self) -> int:
        pass

    @abstractmethod
    def encode_to_tensor(self, text: str) -> torch.Tensor:
        pass

    @abstractmethod
    def decode_from_tensor(self, tensor: torch.Tensor) -> str:
        pass

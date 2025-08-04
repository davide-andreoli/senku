import string
import torch
from typing import List
from helpers.classes import SenkuTokenizer

VOCABULARY = list(
    string.ascii_letters + string.punctuation + string.digits + string.whitespace
)
SPECIAL_TOKENS = ["<UNK>", "<PAD>", "<EOS>"]


class CharacterTokenizer(SenkuTokenizer):
    def __init__(
        self,
        vocabulary: List[str] = VOCABULARY,
        special_tokens: List[str] = SPECIAL_TOKENS,
    ):
        self.strategy = "character"
        vocabulary.sort()
        special_tokens.sort()
        self.vocabulary = vocabulary + special_tokens
        self.vocabulary_size = len(self.vocabulary)
        self.encode_dict = {char: idx for idx, char in enumerate(self.vocabulary)}
        self.decode_dict = {idx: char for idx, char in enumerate(self.vocabulary)}

    def encode(self, text: str) -> List[int]:
        """Encode a string into a list of integers."""
        encoded_text = [
            self.encode_dict.get(char, self.encode_dict["<UNK>"]) for char in text
        ]
        encoded_text.extend([self.encode_dict["<EOS>"]])
        return encoded_text

    @property
    def pad_token_id(self):
        return self.encode_dict["<PAD>"]

    @property
    def eos_token_id(self):
        return self.encode_dict["<EOS>"]

    def decode(self, encoded_text: List[int]) -> str:
        """Decode a list of integers into a string."""
        decoded_text = "".join(
            [self.decode_dict.get(idx, "<UNK>") for idx in encoded_text]
        )
        return decoded_text

    def encode_to_tensor(self, text: str) -> torch.Tensor:
        encoded_text = self.encode(text)
        return torch.tensor(encoded_text).unsqueeze(0)

    def decode_from_tensor(self, tensor: torch.Tensor) -> str:
        encoded_text: List[int] = tensor.squeeze(0).tolist()  # type: ignore[reportUnknownMemberType]
        return self.decode(encoded_text)

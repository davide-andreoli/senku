import string
import torch
from typing import List, Set, cast
from helpers.classes import SenkuTokenizer
import pyphen  # pyright: ignore[reportMissingTypeStubs]
import csv
import re

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
        encoded_text: List[int] = tensor.squeeze(0).tolist()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        return self.decode(encoded_text)


class SyllableTokenizer(SenkuTokenizer):
    def __init__(
        self,
        dataset_path: str = "dataset/haiku/valid-haikus.csv",
        language: str = "en_US",
        special_tokens: List[str] = SPECIAL_TOKENS,
    ):
        self.strategy = "syllable"
        self.dataset_path = dataset_path
        self.language = language
        self.special_tokens = special_tokens
        self.dic = pyphen.Pyphen(lang=language)

        with open(self.dataset_path, "r", encoding="utf-8") as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)
            rows = list(csv_reader)
            haikus = [row[0] + "\n" + row[1] + "\n" + row[2] for row in rows]
            syllables: Set[str] = set()
            syllables.update(
                list(string.punctuation + string.digits + string.whitespace)
            )
            syllables.update(self.special_tokens)
            for haiku in haikus:
                for word in haiku.split():
                    parts = self.dic.inserted(word).split("-")  # pyright: ignore[reportUnknownMemberType]
                    syllables.update(parts)
            vocabulary = list(syllables)
            self.vocabulary = vocabulary
            self.vocabulary_size = len(self.vocabulary)
        self.encode_dict = {
            syllable: idx for idx, syllable in enumerate(self.vocabulary)
        }
        self.decode_dict = {
            idx: syllable for idx, syllable in enumerate(self.vocabulary)
        }

    def encode(self, text: str) -> List[int]:
        encoded_text: List[int] = []

        tokens = re.findall(r"\w+|[^\w\s]|\s", text)

        for token in tokens:
            if token.strip() == "":
                token_id = self.encode_dict.get(token, self.encode_dict.get("<UNK>"))
                encoded_text.append(cast(int, token_id))
            elif token.isalpha():
                syllables = self.dic.inserted(token).split("-")  # pyright: ignore[reportUnknownMemberType]
                for syllable in syllables:
                    token_id = self.encode_dict.get(
                        syllable, self.encode_dict.get("<UNK>")
                    )
                    encoded_text.append(cast(int, token_id))
            else:
                token_id = self.encode_dict.get(token, self.encode_dict.get("<UNK>"))
                encoded_text.append(cast(int, token_id))

        encoded_text.append(self.encode_dict["<EOS>"])
        return encoded_text

    @property
    def pad_token_id(self):
        return self.encode_dict["<PAD>"]

    @property
    def eos_token_id(self):
        return self.encode_dict["<EOS>"]

    def decode(self, encoded_text: List[int]) -> str:
        syllables = [self.decode_dict.get(idx, "<UNK>") for idx in encoded_text]
        return "".join(syllables).replace("<EOS>", "").strip()

    def encode_to_tensor(self, text: str) -> torch.Tensor:
        encoded_text = self.encode(text)
        return torch.tensor(encoded_text).unsqueeze(0)

    def decode_from_tensor(self, tensor: torch.Tensor) -> str:
        encoded_text: List[int] = tensor.squeeze(0).tolist()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        return self.decode(encoded_text)


class WordTokenizer(SenkuTokenizer):
    def __init__(
        self,
        dataset_path: str = "dataset/haiku/valid-haikus.csv",
        special_tokens: List[str] = SPECIAL_TOKENS,
    ):
        self.strategy = "word"
        self.dataset_path = dataset_path
        self.special_tokens = special_tokens

        with open(self.dataset_path, "r", encoding="utf-8") as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)
            rows = list(csv_reader)
            haikus = [row[0] + "\n" + row[1] + "\n" + row[2] for row in rows]

            tokens: Set[str] = set(special_tokens)

            for haiku in haikus:
                parts = re.findall(r"\w+|[^\w\s]|\s", haiku)
                tokens.update(parts)

        self.vocabulary = sorted(tokens)
        self.vocabulary_size = len(self.vocabulary)

        self.encode_dict = {token: idx for idx, token in enumerate(self.vocabulary)}
        self.decode_dict = {idx: token for token, idx in self.encode_dict.items()}

    def encode(self, text: str) -> List[int]:
        tokens = re.findall(r"\w+|[^\w\s]|\s", text)
        encoded: List[int] = []

        for token in tokens:
            token_id = self.encode_dict.get(token, self.encode_dict["<UNK>"])
            encoded.append(token_id)

        encoded.append(self.eos_token_id)
        return encoded

    def decode(self, encoded_text: List[int]) -> str:
        tokens = [self.decode_dict.get(idx, "<UNK>") for idx in encoded_text]
        return "".join(tokens).replace("<EOS>", "").strip()

    def encode_to_tensor(self, text: str) -> torch.Tensor:
        return torch.tensor(self.encode(text)).unsqueeze(0)

    def decode_from_tensor(self, tensor: torch.Tensor) -> str:
        return self.decode(tensor.squeeze(0).tolist())  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]

    @property
    def pad_token_id(self):
        return self.encode_dict["<PAD>"]

    @property
    def eos_token_id(self):
        return self.encode_dict["<EOS>"]

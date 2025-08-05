import torch
import os
from torch.utils.data import Dataset, DataLoader, Subset
import hashlib
import csv
from torch.nn.utils.rnn import pad_sequence
from typing import Any, List, Optional, Dict, Tuple
from helpers.classes import SenkuTokenizer


class TextDataset(Dataset[Any]):
    def __init__(
        self, text: str, tokenizer: SenkuTokenizer, max_length: int, stride: int
    ):
        self.input_ids: List[torch.Tensor] = []
        self.target_ids: List[torch.Tensor] = []
        token_ids = tokenizer.encode(text)
        print(f"Number of token ids: {len(token_ids)}")
        print(f"Max length: {max_length}")
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index: int):
        return self.input_ids[index], self.target_ids[index]

    def get_loader(
        self,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
        num_workers: int = 0,
    ):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
        )

    def get_train_validation_loader(
        self,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
        num_workers: int = 0,
        train_validation_ratio: float = 0.9,
    ):
        split_index = int(train_validation_ratio * len(self))
        train_indices = list(range(0, split_index))
        val_indices = list(range(split_index, len(self)))

        train_dataset = Subset(self, train_indices)
        validation_dataset = Subset(self, val_indices)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
        )

        return train_loader, validation_loader


class TextFolderDataset(Dataset[Any]):
    def __init__(
        self,
        folder_path: str,
        tokenizer: SenkuTokenizer,
        max_length: int,
        stride: int,
        cache_dir: Optional[str] = None,
    ):
        self.folder_path = folder_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.cache_dir = cache_dir or os.path.join(folder_path, ".token_cache")

        os.makedirs(self.cache_dir, exist_ok=True)

        self.chunk_metadata: List[Tuple[str, int]] = []
        self.file_lengths: Dict[str, int] = {}

        self._index_chunks()

    def _token_cache_path(self, file_path: str):
        fname_hash = hashlib.md5(file_path.encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, f"{fname_hash}.pt")

    def _tokenize_and_cache(self, file_path: str):
        cache_path = self._token_cache_path(file_path)
        if os.path.exists(cache_path):
            tokens = torch.load(cache_path)  # type: ignore [reportUnknownMemberType]
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            tokens = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
            torch.save(tokens, cache_path)  # type: ignore [reportUnknownMemberType]
        return tokens

    def _index_chunks(self):
        for fname in os.listdir(self.folder_path):
            if not fname.endswith(".txt"):
                continue
            fpath = os.path.join(self.folder_path, fname)
            tokens = self._tokenize_and_cache(fpath)
            num_tokens = tokens.size(0)
            self.file_lengths[fpath] = num_tokens

            for start in range(0, num_tokens - self.max_length - 1, self.stride):
                self.chunk_metadata.append((fpath, start))
        print(f"Total chunks: {len(self.chunk_metadata)}")

    def __len__(self):
        return len(self.chunk_metadata)

    def __getitem__(self, index: int):
        file_path, start = self.chunk_metadata[index]
        tokens = self._tokenize_and_cache(file_path)

        input_ids = tokens[start : start + self.max_length]
        target_ids = tokens[start + 1 : start + self.max_length + 1]

        return input_ids, target_ids

    def get_loader(
        self,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
        num_workers: int = 0,
    ):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
        )

    def get_train_validation_loader(
        self,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
        num_workers: int = 0,
        train_validation_ratio: float = 0.9,
    ):
        split_index = int(train_validation_ratio * len(self))
        train_indices = list(range(0, split_index))
        val_indices = list(range(split_index, len(self)))

        train_dataset = Subset(self, train_indices)
        validation_dataset = Subset(self, val_indices)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
        )

        return train_loader, validation_loader


class CSVListDataset(Dataset[Any]):
    def __init__(self, file_path: str, tokenizer: SenkuTokenizer, context_length: int):
        self.file_path = file_path
        self.tokenizer = tokenizer

        with open(self.file_path, "r", encoding="utf-8") as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)
            self.rows = list(csv_reader)

        self.samples: List[torch.Tensor] = []
        for row in self.rows:
            haiku = row[0] + "\n" + row[1] + "\n" + row[2]
            token_ids = tokenizer.encode(haiku)[:context_length]
            if len(token_ids) > 1:
                self.samples.append(torch.tensor(token_ids))

    def haiku_collate_fn(
        self, batch: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        batch: list of tuples (input_ids, target_ids), each a 1D tensor

        Returns:
            input_ids_padded: (batch_size, max_len) tensor
            target_ids_padded: (batch_size, max_len) tensor
            attention_mask: (batch_size, max_len) tensor with 1s on real tokens, 0s on padding
        """
        input_ids, target_ids = zip(*batch)

        input_ids_padded = pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        target_ids_padded = pad_sequence(
            target_ids, batch_first=True, padding_value=-100
        )  # -100 ignored by CrossEntropyLoss

        attention_mask = (input_ids_padded != self.tokenizer.pad_token_id).long()

        return (input_ids_padded, target_ids_padded, attention_mask)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        token_ids = self.samples[index]
        return (
            token_ids[:-1],
            token_ids[1:],
        )

    def get_loader(
        self,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
        num_workers: int = 0,
    ):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            collate_fn=self.haiku_collate_fn,
        )

    def get_train_validation_loader(
        self,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
        num_workers: int = 0,
        train_validation_ratio: float = 0.9,
    ):
        split_index = int(train_validation_ratio * len(self))
        train_indices = list(range(0, split_index))
        val_indices = list(range(split_index, len(self)))

        train_dataset = Subset(self, train_indices)
        validation_dataset = Subset(self, val_indices)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            collate_fn=self.haiku_collate_fn,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            collate_fn=self.haiku_collate_fn,
        )

        return train_loader, validation_loader

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import logging
from datetime import datetime
import uuid
from tqdm import tqdm
from helpers.checkpoint import SenkuCheckpointManager
from typing import Optional, Any, Tuple, List, Generator

trainer_logger = logging.getLogger("trainer")

# TODO: Make evaluation mode an enum
# TODO: Add support for early stopping


class Trainer:
    def __init__(
        self,
        train_dataloader: DataLoader[Any],
        validation_dataloader: DataLoader[Any],
        model: nn.Module,
        optimizer: Optimizer,
        loss: nn.Module,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        checkpoint_name: Optional[str] = None,
        epoch: int = 0,
        tokenizer_strategy: str = "character",
    ):
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.model = model.to(device)
        self.device = device
        self.optimizer = optimizer
        self.loss = loss
        self.checkpoint_manager = SenkuCheckpointManager()
        self.checkpoint_name = (
            checkpoint_name
            if checkpoint_name
            else f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}.pt"
        )
        if not self.checkpoint_name.endswith(".pt"):
            self.checkpoint_name += ".pt"
        self.epoch = epoch
        self.tokenizer_strategy = tokenizer_strategy

    def _process_batch(self, batch: torch.Tensor):
        """Handle different batch formats from different datasets"""
        if len(batch) == 2:
            # TextDataset or TextFolderDataset format
            input_batch, target_batch = batch
            attention_mask = None
        elif len(batch) == 3:
            # CSVListDataset format with attention mask
            input_batch, target_batch, attention_mask = batch
        else:
            raise ValueError(f"Unexpected batch format with {len(batch)} elements")

        return input_batch, target_batch, attention_mask

    def train(
        self,
        number_of_epochs: int,
        evaluation_frequency: int = 10,
        evaluation_mode: str = "after_epoch",
    ) -> Tuple[List[float], List[float]]:
        train_losses: List[float] = []
        validation_losses: List[float] = []
        steps = 0
        epoch_progress_bar = tqdm(
            total=number_of_epochs, desc="Training Progress", unit="epoch"
        )

        for epoch in range(self.epoch, self.epoch + number_of_epochs):
            trainer_logger.info(f"Epoch {epoch + 1}/{number_of_epochs + self.epoch}")
            self.model.train()
            epoch_train_losses: List[float] = []

            batch_progress_bar = tqdm(
                total=len(self.train_dataloader),
                desc=f"Epoch {epoch + 1}/{number_of_epochs + self.epoch}",
                unit="batch",
            )

            for _, batch in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()

                input_batch, target_batch, attention_mask = self._process_batch(batch)
                input_batch = input_batch.to(self.device)
                target_batch = target_batch.to(self.device)

                logits = self.model(input_batch)

                loss = self.loss(logits, target_batch, attention_mask)

                loss.backward()
                self.optimizer.step()

                epoch_train_losses.append(loss.item())
                steps += 1
                batch_progress_bar.update(1)
                batch_progress_bar.set_postfix({"loss": loss.item()})  # pyright: ignore[reportUnknownMemberType]

                trainer_logger.debug(f"Training loss at step {steps}: {loss.item()}")

                if evaluation_mode == "in_epoch" and steps % evaluation_frequency == 0:
                    val_loss = self._evaluate()
                    validation_losses.append(val_loss)
                    trainer_logger.info(f"Training loss at step {steps}: {loss.item()}")
                    trainer_logger.info(f"Validation loss at step {steps}: {val_loss}")

            batch_progress_bar.close()

            if evaluation_mode == "after_epoch":
                val_loss = self._evaluate()
                validation_losses.append(val_loss)
                trainer_logger.info(f"Validation loss at epoch {epoch + 1}: {val_loss}")

            epoch_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
            train_losses.append(epoch_train_loss)

            trainer_logger.info(
                f"Average training loss for epoch {epoch + 1}: {epoch_train_loss}"
            )

            self.save_checkpoint(epoch + 1)
            epoch_progress_bar.update(1)
            epoch_progress_bar.set_postfix(  # pyright: ignore[reportUnknownMemberType]
                {
                    "train_loss": epoch_train_loss,
                    "val_loss": validation_losses[-1] if validation_losses else 0.0,
                }
            )

        epoch_progress_bar.close()
        return train_losses, validation_losses

    def train_generator(
        self,
        number_of_epochs: int,
        evaluation_frequency: int = 10,
        evaluation_mode: str = "after_epoch",
    ) -> Generator[Tuple[float, float, str], None, None]:
        steps = 0

        for epoch in range(self.epoch, self.epoch + number_of_epochs):
            self.model.train()
            epoch_train_losses: List[float] = []

            total_batches = len(self.train_dataloader)

            for batch_idx, batch in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()
                input_batch, target_batch, attention_mask = self._process_batch(batch)
                input_batch = input_batch.to(self.device)
                target_batch = target_batch.to(self.device)

                logits = self.model(input_batch)

                loss = self.loss(logits, target_batch, attention_mask)

                loss.backward()
                self.optimizer.step()
                epoch_train_losses.append(loss.item())
                steps += 1

                progress_overall = (
                    ((epoch - self.epoch) + (batch_idx + 1) / total_batches)
                    / number_of_epochs
                    * 100
                )
                progress_epoch = ((batch_idx + 1) / total_batches) * 100
                status = (
                    f"Epoch {epoch + 1}/{self.epoch + number_of_epochs}, "
                    f"Batch {batch_idx + 1}/{total_batches}, "
                    f"Loss = {loss.item():.4f}"
                )

                if evaluation_mode == "in_epoch" and steps % evaluation_frequency == 0:
                    val_loss = self._evaluate()
                    status += f", Val Loss = {val_loss:.4f}"

                yield progress_overall, progress_epoch, status

            train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
            val_loss = self._evaluate()
            self.save_checkpoint(epoch + 1)

            status = f"Epoch {epoch + 1} done: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}"
            yield ((epoch + 1) / number_of_epochs * 100), 100.0, status

        yield 100.0, 100.0, "Training complete!"

    def _evaluate(self):
        """Run evaluation and return average validation loss"""
        self.model.eval()
        validation_loss = 0.0

        with torch.no_grad():
            for batch in self.validation_dataloader:
                input_batch, target_batch, attention_mask = self._process_batch(batch)
                input_batch = input_batch.to(self.device)
                target_batch = target_batch.to(self.device)

                logits = self.model(input_batch)

                loss = self.loss(logits, target_batch, attention_mask)

                validation_loss += loss.item()

        validation_loss /= len(self.validation_dataloader)
        self.model.train()
        return validation_loss

    def save_checkpoint(self, epoch: int = 0):
        self.checkpoint_manager.save_checkpoint(
            self.model.architecture,
            self.model.model_type,
            self.tokenizer_strategy,
            self.checkpoint_name,
            self.model.state_dict(),
            self.optimizer.state_dict(),
            epoch,
            **self.model.keyword_arguments,
        )

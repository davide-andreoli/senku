import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import logging
from datetime import datetime
import uuid
from tqdm import tqdm
from helpers.checkpoint import SenkuCheckpointManager

trainer_logger = logging.getLogger("trainer")

# TODO: Make evaluation mode an enum
# TODO: Add support for early stopping


class Trainer:
    def __init__(
        self,
        train_dataloader: DataLoader,
        validation_dataloader: DataLoader,
        model: nn.Module,
        optimizer: Optimizer,
        loss: nn.Module,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        checkpoint_name: str = None,
        epoch: int = 0,
        reset_checkpoint: bool = False,
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
        self.epoch = epoch

        if not reset_checkpoint and self.checkpoint_manager.checkpoint_exists(
            self.checkpoint_name
        ):
            print("-- Loading checkpoint --")
            self.load_checkpoint()

        if reset_checkpoint and self.checkpoint_manager.checkpoint_exists(
            self.checkpoint_name
        ):
            self.checkpoint_manager.delete_checkpoint(self.checkpoint_name)

    def _process_batch(self, batch):
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
    ):
        train_losses = []
        validation_losses = []
        steps = 0
        epoch_progress_bar = tqdm(
            total=number_of_epochs, desc="Training Progress", unit="epoch"
        )

        for epoch in range(self.epoch, self.epoch + number_of_epochs):
            trainer_logger.info(f"Epoch {epoch + 1}/{number_of_epochs + self.epoch}")
            self.model.train()
            epoch_train_losses = []

            batch_progress_bar = tqdm(
                total=len(self.train_dataloader),
                desc=f"Epoch {epoch + 1}/{number_of_epochs + self.epoch}",
                unit="batch",
            )

            for batch_index, batch in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()

                input_batch, target_batch, attention_mask = self._process_batch(batch)
                input_batch = input_batch.to(self.device)
                target_batch = target_batch.to(self.device)

                logits = self.model(input_batch)

                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = logits.view(-1, logits.size(-1))[active_loss]
                    active_labels = target_batch.view(-1)[active_loss]
                    loss = self.loss(active_logits, active_labels)
                else:
                    loss = self.loss(logits.flatten(0, 1), target_batch.flatten())

                loss.backward()
                self.optimizer.step()

                epoch_train_losses.append(loss.item())
                steps += 1
                batch_progress_bar.update(1)
                batch_progress_bar.set_postfix({"loss": loss.item()})

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
            epoch_progress_bar.set_postfix(
                {
                    "train_loss": epoch_train_loss,
                    "val_loss": validation_losses[-1] if validation_losses else 0.0,
                }
            )

        epoch_progress_bar.close()
        return train_losses, validation_losses

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

                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = logits.view(-1, logits.size(-1))[active_loss]
                    active_labels = target_batch.view(-1)[active_loss]
                    loss = self.loss(active_logits, active_labels)
                else:
                    loss = self.loss(logits.flatten(0, 1), target_batch.flatten())

                validation_loss += loss.item()

        validation_loss /= len(self.validation_dataloader)
        self.model.train()
        return validation_loss

    def save_checkpoint(self, epoch: int = None):
        self.checkpoint_manager.save_checkpoint(
            self.model.architecture,
            self.model.model_type,
            "character",  # TODO: investigate the best way to pass te tokenizer strategy here
            self.checkpoint_name,
            self.model.state_dict(),
            self.optimizer.state_dict(),
            epoch,
            **self.model.keyword_arguments,
        )

    def load_checkpoint(self, epoch: int = None):
        # TODO: This is probably non necessary since the checkpoint can instantiate the model directly
        checkpoint = self.checkpoint_manager.get_checkpoint(self.checkpoint_name)
        self.model.load_state_dict(checkpoint.model_state_dict)
        # TODO: Handle optimizers in the checkpoint s well
        self.optimizer.load_state_dict(checkpoint.optimizer_state_dict)
        self.epoch = checkpoint.epoch

import torch
from app.helpers.haiku_validator import HaikuValidator
from typing import Optional


class HaikuStructureLoss(torch.nn.Module):
    def __init__(
        self,
        base_loss: torch.nn.Module,
        tokenizer_strategy: str,
        newline_token_id: int,
        syllable_counts: torch.Tensor,
        structure_weight: float = 0.3,
        k: float = 10.0,
    ):
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.base_loss = base_loss
        self.tokenizer_strategy = tokenizer_strategy
        self.validator = HaikuValidator()
        self.structure_weight = structure_weight
        self.newline_token = newline_token_id
        self.k = k
        self.register_buffer("target_syllables", torch.tensor([5.0, 7.0, 5.0]))
        self.register_buffer("syllable_counts_per_token", syllable_counts)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if attention_mask is not None:
            active_loss = attention_mask.view(-1) == 1
            base_loss = self.base_loss(
                logits.view(-1, logits.size(-1))[active_loss],
                labels.view(-1)[active_loss],
            )
        else:
            base_loss = self.base_loss(
                logits.view(-1, logits.size(-1)), labels.view(-1)
            )

        structure_loss = self.calculate_structure_loss(logits, labels)

        return base_loss + self.structure_weight * structure_loss

    def calculate_structure_loss(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        # batch_size, seq_len, vocab_size = logits.size()
        probs = torch.softmax(logits, dim=-1)

        # Newline penalty
        newline_probs = probs[:, :, self.newline_token]
        expected_newlines = newline_probs.sum(dim=1)
        newline_penalty = (expected_newlines - 2) ** 2

        if self.tokenizer_strategy == "character":
            total_penalty = newline_penalty
        else:
            # Syllable penalty
            expected_syllables = torch.matmul(probs, self.syllable_counts_per_token)

            line_break_probs = torch.cumsum(newline_probs, dim=1)
            line1_weights = 1.0 - torch.sigmoid(self.k * (line_break_probs - 1.0))
            line2_weights = torch.sigmoid(
                self.k * (line_break_probs - 1.0)
            ) - torch.sigmoid(self.k * (line_break_probs - 2.0))
            line3_weights = torch.sigmoid(self.k * (line_break_probs - 2.0))

            syllable_penalty = 0.0
            for weights, target in zip(
                [line1_weights, line2_weights, line3_weights], self.target_syllables
            ):
                norm_weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
                line_syllables = (expected_syllables * norm_weights).sum(dim=1)
                syllable_penalty += (line_syllables - target) ** 2

            total_penalty = newline_penalty + syllable_penalty

        total_penalty = total_penalty.mean()
        mean_penalty = total_penalty.detach()
        normalized_penalty = total_penalty / (mean_penalty + 1e-8)
        return normalized_penalty

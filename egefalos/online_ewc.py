"""Online Elastic Weight Consolidation for continual learning.
Merged Fisher matrix prevents catastrophic forgetting across sequential tasks.
"""
import torch
import torch.nn as nn
from typing import Dict, Optional
import json
from pathlib import Path


class OnlineEWC:
    """Online EWC: merge all past-task Fisher matrices into one running total."""

    def __init__(self, model: nn.Module, gamma: float = 0.9):
        """
        Args:
            model: The model whose parameters we're protecting
            gamma: Decay rate for old Fisher info (0.9 = keep 90% of old, add 10% new)
        """
        self.model = model
        self.gamma = gamma

        # merged fisher: F_combined = gamma * F_old + F_new
        self.fisher_dict: Dict[str, torch.Tensor] = {}

        # Anchor weights (theta*) — parameters at last consolidation
        self.anchor_dict: Dict[str, torch.Tensor] = {}

        # Metadata
        self.task_count = 0
        self.consolidation_steps = 0

    def compute_fisher(self, dataloader, num_samples: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Compute Fisher Information Matrix on a data sample.

        Fisher[param] = E[(d log p / d param)^2]

        We approximate: Fisher ≈ (1/N) Σ(grad²) over N examples
        """
        device = next(self.model.parameters()).device
        fisher = {name: torch.zeros_like(param)
                  for name, param in self.model.named_parameters()
                  if param.requires_grad}

        total_samples = num_samples if num_samples is not None else len(dataloader) if hasattr(dataloader, '__len__') else 100
        samples_seen = 0

        self.model.train()
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx * (dataloader.batch_size if hasattr(dataloader, 'batch_size') else 1) >= total_samples:
                break

            # Handle both (input_ids, targets) tuples and dict-style batches
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                input_ids, targets = batch[0], batch[1]
            elif isinstance(batch, dict):
                input_ids = batch.get('input_ids', batch.get('input'))
                targets = batch.get('targets', batch.get('labels'))
            else:
                continue

            input_ids = input_ids.to(device) if torch.is_tensor(input_ids) else torch.tensor(input_ids).to(device)
            targets = targets.to(device) if torch.is_tensor(targets) else torch.tensor(targets).to(device)

            # Forward
            logits, loss, _ = self.model(input_ids, targets=targets)

            # Backward
            self.model.zero_grad()
            loss.backward()

            # Accumulate squared gradients
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.detach() ** 2

            samples_seen += input_ids.size(0)

        # Normalize by number of samples
        if samples_seen > 0:
            for name in fisher:
                fisher[name] /= samples_seen

        return fisher

    def merge_fisher(self, new_fisher: Dict[str, torch.Tensor]):
        """Merge new Fisher into running total using exponential decay.

        F_combined = gamma * F_old + (1 - gamma) * F_new
        """
        if not self.fisher_dict:
            # First task: just use the new Fisher
            self.fisher_dict = {k: v.clone() for k, v in new_fisher.items()}
        else:
            # Merge with decay
            for name in self.fisher_dict:
                if name in new_fisher:
                    self.fisher_dict[name] = (
                        self.gamma * self.fisher_dict[name] +
                        (1.0 - self.gamma) * new_fisher[name]
                    )

        self.task_count += 1

    def save_anchor_weights(self):
        """Save current model weights as anchor (theta*) for EWC penalty."""
        self.anchor_dict = {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

    def compute_ewc_penalty(self, lambda_ewc: float = 1000.0) -> torch.Tensor:
        """Compute EWC penalty term: λ/2 · Σᵢ Fᵢ · (θᵢ - θ*ᵢ)²"""
        if not self.fisher_dict or not self.anchor_dict:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)

        device = next(self.model.parameters()).device
        penalty = torch.tensor(0.0, device=device)
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.fisher_dict and name in self.anchor_dict:
                f = self.fisher_dict[name].to(device)
                theta_delta = param - self.anchor_dict[name].to(device)
                penalty += (f * theta_delta ** 2).sum()

        return (lambda_ewc / 2.0) * penalty

    def save(self, path):
        """Serialize Fisher matrix and anchor weights to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        device = next(self.model.parameters()).device

        # Move to CPU for storage
        fisher_cpu = {}
        for k, v in self.fisher_dict.items():
            fisher_cpu[k] = v.detach().cpu() if v.device != torch.device('cpu') else v.detach().clone()

        anchor_cpu = {}
        for k, v in self.anchor_dict.items():
            anchor_cpu[k] = v.detach().cpu() if v.device != torch.device('cpu') else v.detach().clone()

        torch.save({
            'fisher': fisher_cpu,
            'anchor': anchor_cpu,
            'task_count': self.task_count,
            'gamma': self.gamma,
            'consolidation_steps': self.consolidation_steps,
        }, path)

    def load(self, path):
        """Load Fisher matrix and anchor weights from disk."""
        path = Path(path)
        if not path.exists():
            print(f"  [EWC] No checkpoint at {path}. Starting fresh.")
            return

        device = next(self.model.parameters()).device
        checkpoint = torch.load(path, map_location=device, weights_only=True)

        self.fisher_dict = checkpoint['fisher']
        self.anchor_dict = checkpoint['anchor']
        self.task_count = checkpoint['task_count']
        self.gamma = checkpoint['gamma']
        self.consolidation_steps = checkpoint['consolidation_steps']

        # Move to model device
        for k in self.fisher_dict:
            self.fisher_dict[k] = self.fisher_dict[k].to(device)
        for k in self.anchor_dict:
            self.anchor_dict[k] = self.anchor_dict[k].to(device)

        print(f"  [EWC] Loaded {len(self.fisher_dict)} params, {self.task_count} tasks")

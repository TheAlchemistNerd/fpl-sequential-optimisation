"""
Q-Network — Ch. 9 §9.7.

Architecture:
    Input (757-dim) → Dense(512) → ReLU → LayerNorm → Dropout(0.1)
                   → Dense(256) → ReLU → LayerNorm → Dropout(0.1)
                   → Dense(128) → ReLU → LayerNorm
                   → Dense(|A|)

Key design choices (Ch. 9 §9.7):
    - LayerNorm: stabilises training for heterogeneous input features
    - Dropout: regularises, prevents overfitting to limited FPL data
    - Output dimension = number of pruned actions (~100)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    Deep Q-Network for FPL decision-making.

    Parameters
    ----------
    state_dim : int
        Input state dimensionality (757 for the belief-state MDP).
    action_dim : int
        Number of actions (pruned candidate set + no-transfer).
    """

    def __init__(
        self,
        state_dim: int = 757,
        action_dim: int = 101,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),

            nn.Linear(128, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        state : torch.Tensor
            Batch of state vectors, shape (batch, state_dim) or (state_dim,).

        Returns
        -------
        torch.Tensor — Q-values for all actions, shape (batch, action_dim).
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return self.network(state)

    def select_action(
        self,
        state: torch.Tensor,
        temperature: float = 1.0,
        rng: torch.Generator | None = None,
    ) -> int:
        """
        Select action via Boltzmann policy over Q-values.

        Parameters
        ----------
        state : torch.Tensor
            State vector.
        temperature : float
            Temperature for exploration.
        rng : torch.Generator

        Returns
        -------
        int — selected action index.
        """
        with torch.no_grad():
            q_values = self.forward(state).squeeze(0).cpu().numpy()

        if temperature <= 0:
            return int(q_values.argmax())

        # Boltzmann (softmax) selection
        logits = q_values / max(temperature, 1e-8)
        logits = logits - logits.max()  # numerical stability
        probs = torch.softmax(torch.tensor(logits), dim=0).numpy()

        if rng is None:
            rng = torch.Generator()

        action = torch.multinomial(torch.tensor(probs), 1, generator=rng)
        return int(action.item())


class TargetNetwork(QNetwork):
    """
    Target network for DDQN.

    A slowly-updated copy of the online network.
    Updated every C steps (hard update) or via exponential moving average (soft update).
    """

    def __init__(
        self,
        state_dim: int = 757,
        action_dim: int = 101,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(state_dim, action_dim, dropout)

    def hard_update_from(self, source: QNetwork) -> None:
        """Copy weights from the source network."""
        self.load_state_dict(source.state_dict())

    def soft_update_from(self, source: QNetwork, tau: float = 0.001) -> None:
        """
        Exponential moving average update.

        θ_target ← τ·θ_source + (1-τ)·θ_target
        """
        for target_param, source_param in zip(
            self.parameters(), source.parameters()
        ):
            target_param.data.copy_(
                tau * source_param.data + (1 - tau) * target_param.data
            )

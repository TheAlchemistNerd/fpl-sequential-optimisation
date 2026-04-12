"""
DDQN Evaluation — Season simulation and metrics.

Evaluates the trained DDQN agent by running a full 38-gameweek season
with greedy (deterministic) policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class EvaluationMetrics:
    """Metrics collected during evaluation."""
    total_points: float = 0.0
    gameweek_points: list[float] = field(default_factory=list)
    transfers_made: int = 0
    transfer_costs: float = 0.0
    captain_picks: list[int] = field(default_factory=list)
    squad_history: list[list[int]] = field(default_factory=list)

    @property
    def average_gw_points(self) -> float:
        if not self.gameweek_points:
            return 0.0
        return float(np.mean(self.gameweek_points))

    @property
    def std_gw_points(self) -> float:
        if len(self.gameweek_points) < 2:
            return 0.0
        return float(np.std(self.gameweek_points))


def evaluate_agent(
    env: "FPLEnvironment",  # type: ignore
    q_network: "QNetwork",  # type: ignore
    device: str = "cpu",
) -> EvaluationMetrics:
    """
    Evaluate a trained DDQN agent over a full season.

    Uses greedy (deterministic) policy: always selects argmax Q-value.

    Parameters
    ----------
    env : FPLEnvironment
        Reset environment.
    q_network : QNetwork
        Trained Q-network.
    device : str

    Returns
    -------
    EvaluationMetrics
    """
    metrics = EvaluationMetrics()

    state = env.reset()
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device)

    while not env.done:
        # Greedy action selection
        with torch.no_grad():
            q_values = q_network(state_tensor).squeeze(0).cpu().numpy()
        action = int(np.argmax(q_values))

        # Execute
        next_state, reward, done, info = env.step(action)
        state_tensor = torch.tensor(next_state, dtype=torch.float32, device=device)

        metrics.total_points += info.get("gw_points", 0)
        metrics.gameweek_points.append(info.get("gw_points", 0))
        metrics.transfers_made += 1 if info.get("transfer_cost", 0) > 0 else 0
        metrics.transfer_costs += info.get("transfer_cost", 0)
        metrics.squad_history.append(list(env.current_squad))

    return metrics

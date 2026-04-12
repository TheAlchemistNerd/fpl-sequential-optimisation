"""
Boltzmann Policy — Ch. 9 §9.8.

The Boltzmann (softmax) policy converts Q-values into a probability
distribution over actions. Derived from the Maximum Entropy RL objective:

    π(a|s) = exp(Q(s,a) / τ) / Σ_a' exp(Q(s,a') / τ)

The temperature τ controls exploration-exploitation:
    - High τ → near-uniform exploration
    - Low τ → greedy exploitation

τ is annealed over the season:
    τ_t = τ_init · (1 - t/T) + τ_final · (t/T)
"""

from __future__ import annotations

import numpy as np


def boltzmann_policy(
    q_values: np.ndarray,
    temperature: float = 1.0,
    rng: np.random.Generator | None = None,
) -> int:
    """
    Sample an action from the Boltzmann distribution over Q-values.

    Parameters
    ----------
    q_values : np.ndarray
        Q-values for all actions, shape (|A|,).
    temperature : float
        Temperature parameter. Higher → more exploration.
    rng : np.random.Generator

    Returns
    -------
    int — sampled action index.
    """
    if rng is None:
        rng = np.random.default_rng()

    if temperature <= 0:
        # Deterministic greedy
        return int(np.argmax(q_values))

    # Numerically stable softmax
    logits = q_values / max(temperature, 1e-8)
    logits = logits - np.max(logits)  # subtract max for stability
    probs = np.exp(logits)
    probs = probs / (probs.sum() + 1e-10)

    return int(rng.choice(len(q_values), p=probs))


def anneal_temperature(
    current_gameweek: int,
    total_gameweeks: int = 38,
    tau_init: float = 2.0,
    tau_final: float = 0.1,
) -> float:
    """
    Linearly anneal temperature over the season.

    τ_t = τ_0 · (1 - t/T) + τ_min · (t/T)

    Parameters
    ----------
    current_gameweek : int
        Current gameweek number (1-indexed).
    total_gameweeks : int
        Total number of gameweeks in the season.
    tau_init : float
        Initial temperature (high exploration).
    tau_final : float
        Final temperature (high exploitation).

    Returns
    -------
    float — annealed temperature.
    """
    fraction = current_gameweek / total_gameweeks
    return tau_init * (1 - fraction) + tau_final * fraction


def soft_value(q_values: np.ndarray, temperature: float = 1.0) -> float:
    """
    Compute the soft value function (LogSumExp operator).

    V_soft(s) = τ · log(Σ_a exp(Q(s,a) / τ))

    This is the smooth approximation of max_a Q(s,a) from
    entropy-regularised control (Ch. 9 §9.9.2).

    As τ → 0, converges to max_a Q(s,a).

    Parameters
    ----------
    q_values : np.ndarray
        Q-values for all actions.
    temperature : float

    Returns
    -------
    float — soft value.
    """
    if temperature <= 0:
        return float(np.max(q_values))

    logits = q_values / temperature
    # Numerically stable LogSumExp
    max_logit = np.max(logits)
    lse = max_logit + np.log(np.sum(np.exp(logits - max_logit)))
    return float(temperature * lse)

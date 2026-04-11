"""
JAX and Numpyro utility functions for FPL modelling.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def prepare_hmm_sequences(
    data: np.ndarray,
    player_ids: np.ndarray,
    actions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Groups tabular (N, D) data into padded (P, T_max, D) sequences for JAX.
    
    Returns
    -------
    padded_data : (num_players, max_timesteps, D)
    padded_actions : (num_players, max_timesteps, A)
    mask : (num_players, max_timesteps) — 1.0 for real data, 0.0 for padding.
    """
    unique_pids = np.unique(player_ids)
    num_players = len(unique_pids)
    D = data.shape[1]
    A = actions.shape[1]
    
    # Identify sequence lengths
    lengths = [np.sum(player_ids == pid) for pid in unique_pids]
    max_T = max(lengths)
    
    padded_data = np.zeros((num_players, max_T, D), dtype=np.float32)
    padded_actions = np.zeros((num_players, max_T, A), dtype=np.float32)
    mask = np.zeros((num_players, max_T), dtype=np.float32)
    
    for i, pid in enumerate(unique_pids):
        p_mask = (player_ids == pid)
        T_i = lengths[i]
        padded_data[i, :T_i] = data[p_mask]
        padded_actions[i, :T_i] = actions[p_mask]
        mask[i, :T_i] = 1.0
        
    return padded_data, padded_actions, mask

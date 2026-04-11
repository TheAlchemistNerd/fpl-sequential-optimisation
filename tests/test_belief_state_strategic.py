"""
Tests for strategic state assembly (Step C bridge).
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.belief_state import assemble_757_state, compute_belief_vecs_jax


def test_compute_belief_vecs_jax():
    """Verify JAX-native belief vector computation."""
    # S=10 samples, N=3 observations
    # Obs 0: assigned to state 0 nine times, state 1 once.
    # Obs 1: assigned to state 2 five times, state 3 five times.
    # Obs 2: assigned to state 0 always.
    z_samples = jnp.array([
        [0, 2, 0],
        [0, 2, 0],
        [0, 2, 0],
        [0, 2, 0],
        [0, 2, 0],
        [0, 3, 0],
        [0, 3, 0],
        [0, 3, 0],
        [0, 3, 0],
        [1, 3, 0],
    ])
    K = 30
    beliefs = compute_belief_vecs_jax(z_samples, num_states=K)
    
    assert beliefs.shape == (3, K)
    assert beliefs[0, 0] == pytest.approx(0.9)
    assert beliefs[0, 1] == pytest.approx(0.1)
    assert beliefs[1, 2] == pytest.approx(0.5)
    assert beliefs[1, 3] == pytest.approx(0.5)
    assert beliefs[2, 0] == pytest.approx(1.0)
    assert jnp.allclose(beliefs.sum(axis=1), 1.0)


def test_assemble_757_state_dimensions():
    """Verify the strategic state vector has correct dimensions and structure."""
    N = 720 # Estimated database size
    K = 30
    all_player_ids = list(range(1, N + 1))
    squad_ids = list(range(1, 16)) # Players 1-15
    
    # Mock belief states (only for squad members)
    belief_vecs = {
        pid: np.random.dirichlet([1.0]*K) for pid in squad_ids
    }
    
    budget = 101.5
    chips = {"wildcard": True, "bench_boost": False, "triple_captain": False, "free_hit": False}
    gw = 12
    ft = 1
    
    state = assemble_757_state(
        squad_ids=squad_ids,
        all_player_ids=all_player_ids,
        belief_vecs=belief_vecs,
        budget=budget,
        chips=chips,
        gameweek=gw,
        free_transfers=ft
    )
    
    # Target dimension: N + K + 4 (chips) + 3 (meta: budget, gw, ft)
    expected_dim = N + K + 4 + 3 # 720 + 30 + 4 + 3 = 757
    assert state.shape == (expected_dim,)
    
    # Check squad binary part
    assert state[0] == 1.0 # Player 1 is in squad
    assert state[15] == 0.0 # Player 16 is not
    
    # Check meta part (last 3 elements)
    assert state[-3] == pytest.approx(budget)
    assert state[-2] == float(gw)
    assert state[-1] == float(ft)
    
    # Check chips part (elements before meta)
    # [WC, BB, TC, FH]
    assert state[-7] == 1.0 # Wildcard active
    assert state[-6] == 0.0 # Bench boost inactive

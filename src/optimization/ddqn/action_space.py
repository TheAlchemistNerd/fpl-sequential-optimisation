"""
Action Space — Candidate Pruning.

The full FPL action space is combinatorially large (~8,775 possible single
transfers). The DDQN prunes to a candidate set of the top-N players by
expected points, reducing the output dimension to a tractable size.

Following Ch. 8 §8.8.3 and Ch. 9 §9.7:
    - Branching factor b ≈ 100 (top-ranked candidates)
    - Actions are encoded as: 0 = no transfer, 1..K = transfer in candidate k
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ActionSpace:
    """
    Pruned action space for DDQN.

    Parameters
    ----------
    candidate_ids : list[int]
        Player IDs of top-K candidates by expected points.
    current_squad : list[int]
        Current squad player IDs.
    no_transfer_action : int
        Action index for "no transfer" (typically 0).
    """

    candidate_ids: list[int] = field(default_factory=list)
    current_squad: list[int] = field(default_factory=list)
    no_transfer_action: int = 0

    @property
    def n_actions(self) -> int:
        """Total number of actions: 1 (no transfer) + len(candidates)."""
        return 1 + len(self.candidate_ids)

    @property
    def n_transfers(self) -> int:
        """Number of transfer actions (excluding no-transfer)."""
        return len(self.candidate_ids)

    def action_to_transfer(self, action: int) -> tuple[int, int] | None:
        """
        Decode an action index into a (player_in, player_out) pair.

        Action 0 = no transfer.
        Actions 1..K = transfer in candidate[action-1].

        The player to transfer out is selected as the lowest-value
        squad member not in the candidate set.

        Parameters
        ----------
        action : int
            Action index.

        Returns
        -------
        (player_in, player_out) or None for no transfer.
        """
        if action == self.no_transfer_action:
            return None

        candidate_idx = action - 1
        if candidate_idx >= len(self.candidate_ids):
            return None

        player_in = self.candidate_ids[candidate_idx]

        # Find lowest-value player to transfer out (not the incoming player)
        # In a full implementation, this would use price/value ranking
        # For now, transfer out the first squad member not in candidates
        candidate_set = set(self.candidate_ids)
        for pid in self.current_squad:
            if pid not in candidate_set and pid != player_in:
                return (player_in, pid)

        # Fallback: transfer out the last squad member
        if self.current_squad:
            return (player_in, self.current_squad[-1])

        return None

    def transfer_to_action(self, player_in: int) -> int:
        """
        Encode a transfer-in player as an action index.

        Parameters
        ----------
        player_in : int
            Player ID to transfer in.

        Returns
        -------
        int — action index (0 if player not in candidates).
        """
        if player_in in self.candidate_ids:
            return self.candidate_ids.index(player_in) + 1
        return self.no_transfer_action


def build_action_space(
    player_pool: dict[str, list],
    current_squad: list[int],
    top_k: int = 100,
) -> ActionSpace:
    """
    Build a pruned action space from the player pool.

    Parameters
    ----------
    player_pool : dict
        Must contain 'ids' and 'points' (expected points per player).
    current_squad : list[int]
        Current squad player IDs.
    top_k : int
        Number of top candidates to include.

    Returns
    -------
    ActionSpace with top-K candidates.
    """
    ids = player_pool.get("ids", [])
    points = player_pool.get("points", {})

    # Sort by expected points descending
    sorted_ids = sorted(ids, key=lambda pid: points.get(pid, 0), reverse=True)
    candidates = sorted_ids[:top_k]

    return ActionSpace(
        candidate_ids=candidates,
        current_squad=current_squad,
    )

"""
MCTS Tree Node — Chapter 8 §8.8.

Monte Carlo Tree Search node for FPL tactical planning.
Each node represents a (belief_state, squad_config) pair.
Edges represent transfer actions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MCTSNode:
    """
    A single node in the MCTS search tree.

    Attributes
    ----------
    state : dict
        The belief state + squad configuration at this node.
    action : Any
        The action that led to this node from its parent.
    parent : MCTSNode | None
        Parent node in the tree.
    children : dict
        Mapping from action → child MCTSNode.
    visit_count : int
        Number of times this node has been visited.
    total_reward : float
        Cumulative discounted reward from rollouts through this node.
    untried_actions : list
        Actions from this state that haven't been explored yet.
    """
    state: dict
    action: Any = None
    parent: "MCTSNode | None" = None
    children: dict[Any, "MCTSNode"] = field(default_factory=dict)
    visit_count: int = 0
    total_reward: float = 0.0
    untried_actions: list = field(default_factory=list)

    @property
    def is_fully_expanded(self) -> bool:
        """Whether all actions from this state have been explored."""
        return len(self.untried_actions) == 0

    @property
    def is_leaf(self) -> bool:
        """Whether this node has no children."""
        return len(self.children) == 0

    def ucb1(self, exploration_weight: float = 1.414) -> float:
        """
        Upper Confidence Bound 1 (UCB1) selection rule.

        UCB1(s, a) = Q(s, a) + c · √(log N(s) / N(s, a))

        where:
        - Q(s, a) = total_reward / visit_count (average rollout reward)
        - N(s) = parent.visit_count
        - N(s, a) = this node's visit_count
        - c = exploration_weight (default √2)

        Returns
        -------
        float — UCB1 value for this node.
        """
        if self.visit_count == 0:
            return float("inf")  # Unvisited nodes have infinite UCB1

        if self.parent is None or self.parent.visit_count == 0:
            return float("inf")

        exploitation = self.total_reward / self.visit_count
        exploration = exploration_weight * math.sqrt(
            math.log(self.parent.visit_count) / self.visit_count
        )
        return exploitation + exploration

    def select_best_child(self, exploration_weight: float = 1.414) -> "MCTSNode":
        """
        Select the child with the highest UCB1 value.

        Parameters
        ----------
        exploration_weight : float
            Controls exploration vs exploitation trade-off.

        Returns
        -------
        MCTSNode — the best child according to UCB1.
        """
        if not self.children:
            return self

        return max(
            self.children.values(),
            key=lambda child: child.ucb1(exploration_weight),
        )

    def expand(self, action: Any, child_state: dict) -> "MCTSNode":
        """
        Expand this node by adding a child for the given action.

        Parameters
        ----------
        action : Any
            The transfer action to take.
        child_state : dict
            The resulting belief state + squad configuration.

        Returns
        -------
        MCTSNode — the newly created child node.
        """
        self.untried_actions.remove(action)
        child = MCTSNode(
            state=child_state,
            action=action,
            parent=self,
        )
        self.children[action] = child
        return child

    def backpropagate(self, reward: float, discount: float = 0.9) -> None:
        """
        Backpropagate rollout reward up the tree.

        Parameters
        ----------
        reward : float
            The cumulative discounted reward from the rollout.
        discount : float
            Discount factor applied per level up the tree.
        """
        node = self
        current_reward = reward
        while node is not None:
            node.visit_count += 1
            node.total_reward += current_reward
            current_reward *= discount
            node = node.parent

    def __repr__(self) -> str:
        return (
            f"MCTSNode(visits={self.visit_count}, "
            f"reward={self.total_reward:.1f}, "
            f"children={len(self.children)}, "
            f"untried={len(self.untried_actions)})"
        )

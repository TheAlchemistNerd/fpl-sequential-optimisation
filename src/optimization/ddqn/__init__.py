"""DDQN Step C — Double Deep Q-Network with belief-state MDP input.

Implements Ch. 9 §9.5–9.8:
    - state_representation.py: 757-dim encoder (Ch. 9 §9.5.3)
    - q_network.py: Dense(512)→ReLU→LayerNorm→Dropout→...→Dense(|A|)
    - replay_buffer.py: Experience replay with uniform sampling
    - action_space.py: Candidate pruning (top-100)
    - boltzmann_policy.py: MaxEnt RL soft-Q updates (Ch. 9 §9.8)
    - fpl_environment.py: Gym-like env via FFBS simulation
    - training_loop.py: DDQN + HDP-HMM unified training (Ch. 9 §9.6.3)
    - evaluation.py: Season simulation, metrics
"""

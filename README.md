# RL-Foundation-Learning

<p align="center">
  <b>Comprehensive reinforcement learning fundamentals — from Q-Learning to Deep Q-Networks and beyond</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
  <img src="https://img.shields.io/github/last-commit/CHEN-taeo/RL-Foundation-Learning" alt="Last Commit">
  <img src="https://img.shields.io/badge/Topic-Reinforcement_Learning-orange" alt="Topic">
</p>

---

## Overview

A structured learning repository covering reinforcement learning fundamentals, classical algorithms, and deep RL implementations. Built as a personal study companion with practical code examples and visual comparisons.

## Core Concepts

### Q-Learning vs Deep Q-Networks (DQN)

A deep exploration of why table-based Q-Learning excels in discrete, low-dimensional environments but collapses when facing continuous, high-dimensional state spaces.

| Approach | Mechanism | Limitation |
|----------|-----------|------------|
| **Q-Learning** | Lookup table: `Q-table[state, action]` | Curse of dimensionality — only works for discrete states |
| **Deep Q-Network (DQN)** | Function approximation with neural networks | Requires careful training stability techniques |

**Key Insight: The Curse of Dimensionality**
- FrozenLake (4×4 grid, discrete): Q-Learning handles 16 states easily
- CartPole (continuous 4D): Q-Learning fails — infinite state combinations
- Solution: Neural network function approximation → DQN

## Learning Path

1. **State space complexity** — Understand dimensionality and its impact on algorithms
2. **Q-Learning fundamentals** — Table-based approach, Bellman equation
3. **Function approximation** — Why and when neural networks are necessary
4. **Deep Q-Networks** — Implementation, experience replay, target networks
5. **Advanced algorithms** — Policy Gradient, PPO, A3C, SAC

## Topics Covered

- Reinforcement Learning theory and mathematics
- Classic algorithm implementations (tabular methods)
- Deep RL with function approximation
- Visual comparisons and convergence analysis
- Practical code examples with PyTorch

## Getting Started

```bash
git clone https://github.com/CHEN-taeo/RL-Foundation-Learning.git
cd RL-Foundation-Learning
pip install torch numpy matplotlib gymnasium
```

## References

- Sutton & Barto — *Reinforcement Learning: An Introduction*
- DeepMind DQN Paper (Mnih et al., 2015)
- OpenAI Spinning Up in Deep RL

## License

MIT

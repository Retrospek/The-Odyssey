# Agent Mania

Agent Mania is a research-oriented reinforcement learning codebase focused on core decision-making algorithms, scalable experimentation, and clean PyTorch abstractions. It is designed to support algorithmic exploration in the same style used across applied ML research labs and quantitative decision systems: modular models, reusable training infrastructure, and a strong emphasis on policy optimization and value-based learning.

## Overview

This project brings together a compact set of foundational RL implementations built for experimentation, comparison, and extension. The codebase emphasizes reproducibility and maintainability, with shared model components and framework utilities that reduce boilerplate while keeping the algorithmic logic explicit.

The repository is particularly relevant for teams working in:

- reinforcement learning research and applied AI
- decision systems and automated control
- model development for sequential optimization problems
- benchmark-driven experimentation in dynamic environments
- quantitative research workflows requiring clear algorithmic structure

## Core capabilities

### Value-based reinforcement learning

Agent Mania includes implementations of classic value-function methods that form the foundation of modern deep RL:

- Deep Q-Network (DQN): a standard off-policy value-based approach using a target network and replay buffer to stabilize learning.
- Double DQN (DDQN): a refined variant that reduces value overestimation and improves learning stability in complex environments.

These implementations are structured around reusable model and buffer abstractions, making them suitable for experimentation with reward structures, network architectures, and training dynamics.

### Policy optimization

The project also includes a Trust Region Policy Optimization (TRPO) implementation, highlighting a more advanced policy-gradient approach grounded in constrained updates and first-order optimization theory.

TRPO is particularly relevant for settings where policy stability matters, and where researchers need to study the tradeoff between sample efficiency, update quality, and control performance in continuous or structured decision tasks.

### Shared research infrastructure

A key feature of the repository is its modular utility layer, including:

- base value and policy model classes
- experience replay and priority-based replay buffers
- transition container utilities for training loops
- checkpointing and update helpers for model lifecycle management

This infrastructure supports a clean separation between algorithmic logic and experimental scaffolding, which is valuable for both rapid prototyping and longer-term research iteration.

## Why this project matters

Agent Mania reflects a pragmatic research mindset: it combines foundational RL methods with reusable engineering patterns that are common in applied ML and quant-style experimentation. The code emphasizes understanding and control of learning dynamics rather than superficial abstraction, making it appropriate for technical discussions around model design, optimization behavior, and algorithmic tradeoffs.

For recruiters and research stakeholders, the repository communicates a strong profile in:

- deep reinforcement learning
- optimization and control
- PyTorch-based model development
- experimental research implementation
- robust algorithmic engineering

## Research direction

The project sits at the intersection of machine learning, optimization, and decision systems. It is well suited for work involving simulation environments, sequential control, and research-driven model development where performance depends on both statistical rigor and engineering quality.

Agent Mania is positioned as a clean, extensible foundation for exploring RL methods in a way that is both technically credible and relevant to applied research and quantitative modeling environments.

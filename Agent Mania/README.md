# Agent Mania

This directory is a from-scratch reinforcement-learning workspace. The
existing implementations cover **DQN**, **Double DQN (DDQN)**, and the
building blocks of **TRPO**. Use the original papers below as the algorithmic
source of truth; the code in this repository is a learning scaffold, not a
drop-in reproduction of every paper setting.

## What is here

| Algorithm  | Learning setting               | Main code                                                                                                                |
| ---------- | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------ |
| DQN        | Off-policy, discrete actions   | [`models/dqn/dqn_trainer.py`](models/dqn/dqn_trainer.py), [`single-agent/dqn/train.py`](single-agent/dqn/train.py)       |
| Double DQN | Off-policy, discrete actions   | [`models/ddqn/ddqn_trainer.py`](models/ddqn/ddqn_trainer.py), [`single-agent/ddqn/train.py`](single-agent/ddqn/train.py) |
| TRPO + GAE | On-policy, stochastic policies | [`models/trpo/trpo.py`](models/trpo/trpo.py), [`models/trpo/trpo_trainer.py`](models/trpo/trpo_trainer.py)               |

Shared pieces live in [`models/buffer.py`](models/buffer.py),
[`models/value_model.py`](models/value_model.py), and
[`models/pg_model.py`](models/pg_model.py).

## A good from-scratch workflow

1. Read the paper once for the objective and once for the algorithm box or
   pseudocode. Write down the tensors, their shapes, and which quantities must
   be detached from autograd.
2. Start with a small vector-observation environment and a deterministic seed.
   Confirm one batch produces the target, loss, and gradient you calculate by
   hand before attempting a full run.
3. Add one paper feature at a time. For DQN/DDQN: replay, target network, then
   exploration. For TRPO: rollouts, returns/advantages, value fitting, then
   the constrained policy update.
4. Log episodic return, loss, exploration/KL, gradient norms, and the exact
   configuration. Compare several seeds; a single successful run is not a
   reproduction.

The notation below uses a transition
`(s, a, r, s_next, done)`, discount `gamma`, online parameters `theta`, and
target/old parameters `theta_minus`.

## DQN

**Read first:**

- Mnih et al., _[Human-level control through deep reinforcement learning](https://doi.org/10.1038/nature14236)_, Nature, 2015 — the canonical DQN paper.
- Mnih et al., _[Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)_, 2013 — the earlier DQN description.
- Sutton and Barto, _[Reinforcement Learning: An Introduction, 2nd ed.](http://incompleteideas.net/book/the-book-2nd.html)_, Chapters 5–6 — policy evaluation and Q-learning foundations.

DQN approximates the action-value function with `Q_theta(s, a)`. Sample a
minibatch uniformly from replay and minimize the temporal-difference error:

```text
y = r + gamma * (1 - done) * max_a_next Q_theta_minus(s_next, a_next)
loss = mean(Huber(Q_theta(s, a), stop_gradient(y)))
```

Implementation checklist:

- Keep an **online** network for action selection and optimization and a
  **target** network for bootstrap values. Initialize the target as an exact
  copy and update it either periodically (hard update) or with
  `theta_minus <- tau * theta + (1 - tau) * theta_minus` (soft update).
- Store raw transitions in a fixed-capacity replay buffer; sample a random
  minibatch without removing it. Replay is what makes the data reusable and
  reduces correlation.
- Use epsilon-greedy exploration only while interacting with the environment;
  do not apply it when calculating the target. Mask terminal and truncated
  transitions correctly.
- Gather the value of the _recorded_ action, not the maximum online value, for
  the prediction term. The target must not receive gradients.

The current DQN trainer uses an MSE loss and configurable soft target updates;
those are deliberate choices to evaluate when comparing against the paper's
clipped-error and periodic-update setup.

## Double DQN (DDQN)

**Read first:**

- van Hasselt, Guez, and Silver, _[Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)_, AAAI, 2016 — the Deep Double Q-learning paper.
- van Hasselt, _[Double Q-learning](https://papers.nips.cc/paper_files/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html)_, NeurIPS, 2010 — the original tabular idea.
- The DQN references above — DDQN retains replay, target networks, and the
  rest of the DQN training loop.

DDQN changes only the bootstrap: the online network **selects** the next
action, while the target network **evaluates** it. This decoupling addresses
the overestimation caused by maximizing noisy value estimates.

```text
a_star = argmax_a_next Q_theta(s_next, a_next)
y = r + gamma * (1 - done) * Q_theta_minus(s_next, a_star)
loss = mean(Huber(Q_theta(s, a), stop_gradient(y)))
```

Implementation checklist:

- `action_net`/`policy_net` is the online network and is optimized; `q_net`/
  `target_net` is the frozen target network. Copy online weights into target
  before the first update.
- Compute `a_star` with the online network, then use `gather` on the target
  network. Reversing those roles silently turns the target back into DQN.
- Keep the target branch inside `torch.no_grad()` (or detach its result), and
  apply the same terminal mask and target-update policy as DQN.
- On a fixed batch, print DQN and DDQN targets. They should differ only when
  the online argmax is not the target argmax; that is a useful sanity check.

## TRPO and generalized advantage estimation

**Read first:**

- Schulman et al., _[Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)_, ICML, 2015 — constrained surrogate objective, Fisher-vector products, conjugate gradient, and backtracking line search.
- Schulman et al., _[High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)_, ICLR, 2016 — GAE's bias/variance trade-off.
- Kakade, _[A Natural Policy Gradient](https://proceedings.neurips.cc/paper_files/paper/2001/hash/4b86abe48d358ecf194c56c69108433e-Abstract.html)_, NeurIPS, 2001 — useful background for the natural-gradient connection.

TRPO is **on-policy**: collect a fresh rollout with an old stochastic policy
`pi_old`, then maximize its importance-weighted surrogate while constraining
how far the policy moves:

```text
maximize  mean[(pi_theta(a | s) / pi_old(a | s)) * advantage]
subject to mean[KL(pi_old(. | s) || pi_theta(. | s))] <= max_kl
```

For GAE, calculate TD residuals and advantages backward over a rollout:

```text
delta_t = r_t + gamma * (1 - done_t) * V(s_t+1) - V(s_t)
A_t     = delta_t + gamma * lambda * (1 - done_t) * A_t+1
return_t = A_t + V(s_t)
```

Implementation checklist:

- Snapshot and retain the **old** action log-probabilities (and, for discrete
  policies, the full old action distribution) during rollout collection.
  Do not define the old policy by detaching the current forward pass during
  the update; its KL is then always zero.
- Handle episode boundaries in GAE, bootstrap only when the final transition
  is nonterminal, and normalize advantages after computing them.
- Fit the value function to `return_t` separately from the policy update. The
  value optimizer should own only the value-network parameters if the actor
  and critic do not intentionally share all parameters.
- Implement the Fisher-vector product using autograd without materializing a
  Fisher matrix; solve the natural-gradient direction with conjugate gradient.
  Scale the step to meet `max_kl`, then use backtracking line search that
  accepts only an improving surrogate with an in-budget measured KL.
- Recollect data after every accepted policy update. Unlike DQN, do not train
  TRPO from a long-lived off-policy replay buffer.

The TRPO modules are best treated as a component-level starting point. Before
using them for experiments, verify the old-policy snapshot, KL calculation,
GAE boundary handling, and line-search acceptance against the paper.

## Minimal tests before a long training run

- A terminal transition has target `r` exactly; a nonterminal transition has
  the discounted bootstrap term.
- Altering target-network weights changes a DQN target but produces no target
  network gradients.
- For DDQN, force different online and target argmax actions and verify that
  selection comes from online while evaluation comes from target.
- GAE agrees with hand-calculated two- and three-step trajectories, including
  a terminal boundary.
- A TRPO update reports its surrogate objective and measured KL; reject a
  candidate step that exceeds `max_kl`.

## Paper-reading notes

Paper results are not directly comparable unless environment wrappers, action
repeats, network architecture, reward preprocessing, training frames, and
evaluation protocol also match. Start by reproducing the _algorithmic
invariants_ above on LunarLander, then record every intentional deviation from
the paper in the experiment configuration.

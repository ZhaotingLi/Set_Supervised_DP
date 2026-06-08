# Internals of `main-receding_horizon.py`

This document provides a high-level overview of the training pipeline implemented in **`main-receding_horizon.py`**.
It is intended for developers who want to understand or extend the code.



## Overview

`main-receding_horizon.py` is the **entry point for training and evaluation**.
It supports both a human-gated **Interactive Imitation Learning (IIL)** framework and an
**offline learning** framework that trains from pre-collected demonstrations or datasets
without online interventions.
It includes:

* Environment setup (Robosuite, MetaWorld, PushT, etc.)
* Agent initialization (Set-supervised DP/ DP/ DP-DPO/ Ambient DP/ CLIC/ IBC/ etc.)
* Teacher/oracle feedback for interactive runs
* Offline dataset loading and replay-based policy updates
* Training loop with either human/automatic corrections or offline batches
* Evaluation and logging

The workflow for interactive training is
```
Load Config → Select Env & Agent
       ↓
   Training Loop
       ├─ Agent acts
       ├─ Oracle feedback (optional)
       ├─ Agent update (Set-supervised DP/ DP/ DP-DPO/ Ambient DP/ CLIC/ IBC/ etc.)
       └─ Save progress
       ↓
 Periodic Evaluation
       ↓
 Save Models & Results
```

The workflow for offline training is
```
Load Config → Select Env & Agent → Load Offline Dataset
       ↓
   Offline Training Loop
       ├─ Sample batches from demonstrations/dataset
       ├─ Agent update (Set-supervised DP/ DP/ DP-DPO/ Ambient DP/ CLIC/ IBC/ etc.)
       └─ Save progress
       ↓
 Periodic Evaluation
       ↓
 Save Models & Results
```


## Main Components

### 1. **Environment & Agent Selection**

* `env_selector` → constructs the desired environment.
* `agent_selector` → builds the specified agent (CLIC, BC, Diffusion, etc.).

Each experiment is defined by a Hydra config (see `Files/src/config/`).



### 2. **Oracle & Feedback Handling**

Teacher policies provide corrective signals:

* `oracle_gimme_feedback`
* `oracle_feedback_HGDagger`
* `oracle_feedback_intervention_diff`

These generate **interventions** or **corrections** used by interactive imitation learning.



### 3. **Training Loop**

Core function: `train_interactive_learning_repetition(...)`

Steps per episode:

1. Reset environment (with optional randomized initial states).
2. Agent proposes an action.
3. Teacher/oracle may provide corrective feedback.
4. Agent updates policy:

   * If `use_CLIC_algorithm`: train with **contrastive losses** (policy/Q-value updates).
   * Otherwise: standard imitation or diffusion training.
5. Results (success/error/feedback count) are logged.


### 4. **Evaluation**

* Runs periodic evaluations (`evaluate_agent`).







## Extending

* **New Agents** → add to `agents/selector_policy.py`. See [this document](./Readme_New_Agent.md) for detailed instructions.
* **New Environments** → extend `env/env_selector.py`. See [this document](./Readme_New_Task.md) for detailed instructions.
* **Custom Feedback** → implement new oracle in `tools/oracle_feedback.py`.

# How to Add a New Agent

This guide provides instructions for integrating a new learning algorithm into this repository. Use `Files/src/agents/Set_Supervised_diffusion_policy_image.py` as a current example, especially for image-based agents.

-----

## AGENT CLASS STRUCTURE

Your new agent should be defined as a Python class under `Files/src/agents/`. The `__init__` method should set up the components needed for inference and training.

### Key Components to Initialize:

* **Model and Optimizer**: Define the policy model, observation encoder if needed, optimizer, and scheduler. For an image diffusion example, see `DiffusionUnetImagePolicy_Set_Supervised` in `Set_Supervised_diffusion_policy_image.py`.
* **Replay Buffer**: Initialize the buffer and batch loader used for training. The current diffusion agents use `build_replay_buffer_setup(...)`.
* **Hyperparameters**: Store agent-specific parameters such as `dim_a`, `dim_o`, `Ta`, learning rate, buffer sizes, and `shape_meta`.

For image agents, `shape_meta` should come from the task config and must match the observation keys returned by the environment.

-----

## REQUIRED METHODS

Your agent class should implement the following methods so it can be used by `main-receding_horizon.py`.

### 1. `action(observation)`

This method selects an action given the current observation.

* **Input**: processed observation from the environment. For image agents, this is usually a dict matching `shape_meta.obs`.
* **Output**: an action or action chunk as a NumPy array.
* **Example**: `Set_Supervised_diffusion_policy_image.py` encodes the image observation, runs diffusion sampling, clips the result, and returns the action chunk.

```python
def action(self, observation):
    # preprocess observation
    # run policy / sampling
    # return action as NumPy array
    return action
```

### 2. `collect_data_and_train(...)`

This is the main training entry point called by the training loop. New agents should implement this method instead of relying on older algorithm-specific function names.

```python
def collect_data_and_train(
    self, last_action, h, obs_proc, next_obs, t, done,
    agent_algorithm=None, agent_type=None, i_episode=None
):
    # store new data if feedback is available
    # sample from replay buffer
    # update the policy
    # optionally train at the end of the episode
    pass
```

In `Set_Supervised_diffusion_policy_image.py`, this method delegates to `TRAIN_Diffusion_with_Set_Supervised(...)`. That function:

* stores corrective feedback `h` in the replay buffer when feedback is non-zero;
* samples batches with `self.replay_batch_loader`;
* calls `compute_loss_Diffusion_Set_Supervised(...)`;
* performs extra updates at the end of an episode.

### 3. Loss Calculation and Network Update

Create a separate method for the actual gradient update. In the image set-supervised diffusion agent, this is `compute_loss_Diffusion_Set_Supervised(...)`.

```python
def update_networks(self, batch):
    # unpack batch
    # compute loss
    # optimizer step
    pass
```

### 4. `save_model()` and `load_model()`

Implement these if the agent has learnable parameters. The main pipeline calls them when saving or loading policies.

For example, `Set_Supervised_diffusion_policy_image.py` saves both the diffusion model and image observation encoder.

-----

## INTEGRATION

After creating your agent class, integrate it into the project.

### 1. Add to `selector_policy.py`

Add a new condition in `agent_selector(...)` to instantiate your agent and pass the needed config values.

```python
elif agent_type == 'Your_New_Agent':
    from agents.your_new_agent_file import YourNewAgentClass
    return YourNewAgentClass(
        dim_a=config_agent.dim_a,
        dim_o=config_agent.dim_o,
        shape_meta=config_agent.shape_meta,
        saved_dir=config_agent.saved_dir,
        load_dir=config_agent.load_dir,
        config_agent=config_agent,
        # other parameters
    )
```

For a current example, see the `Set_Supervised_Diffusion` branch in `selector_policy.py`, which creates `DiffusionUnetImagePolicy_Set_Supervised` when `config_agent.use_image` is true.

### 2. Create a Configuration File

Create or update a YAML config under `Files/src/config/`. The config should define the selected agent, algorithm, dimensions, replay-buffer settings, training hyperparameters, and save/load paths.

For image agents, also make sure:

* `use_image: ${task.use_image}`
* `shape_meta: ${task.shape_meta}`
* `image_crop_shape: ${task.crop_shape}`

-----

## LEGACY METHOD NAMES

Some older agents still expose names such as `TRAIN_Diffusion_withCLIC(...)` or `compute_loss_Diffusion_CLIC(...)`. In the current set-supervised image diffusion agent these are compatibility aliases. For new agents, prefer the public methods used by the main loop: `action(...)`, `collect_data_and_train(...)`, `save_model(...)`, and `load_model(...)`.

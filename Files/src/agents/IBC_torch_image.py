import logging
import numpy as np

from tools.buffer import Buffer

import pandas as pd

from typing import Iterable
import os

from tools.buffer import Buffer_uniform_sampling

import random
# for image based input
import cv2

from agents import mcmc_torch as mcmc


import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam

from agents.DP_model.common.normalizer import LinearNormalizer   # not used
from agents.DP_model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from agents.DP_model.common.pytorch_util import dict_apply
from agents.DP_model.diffusion.conditional_unet1d import ConditionalUnet1D, HumanFunctionModel, HumanFunctionModel_Resnet

from agents.Set_Supervised_diffusion_policy_image import BaseImagePolicy

from agents.Set_Supervised_diffusion_policy_image import collate_obs_dict

from tools.buffer import HDF5Buffer
from agents.DP_model.common.scheduler import CosineAnnealingWarmupRestarts

logger = logging.getLogger(__name__)

# MappedCategorical: maps class indices to specific action vectors
class MappedCategorical(Categorical):
    def __init__(self, logits=None, probs=None, mapped_values=None, validate_args=False):
        super().__init__(logits=logits, probs=probs, validate_args=validate_args)
        self.mapped_values = mapped_values

    def mode(self):
        # Index of the most probable class
        if self.logits is not None:
            idx = torch.argmax(self.logits, dim=-1)
        else:
            idx = torch.argmax(self.probs, dim=-1)
        # Map to action vectors
        return self.mapped_values[idx]

    def sample(self, sample_shape=torch.Size()):
        idx = super().sample(sample_shape)
        return self.mapped_values[idx]


def categorical_bincount(count: int, log_ps: torch.Tensor):
    """
    Args:
        count: number of draws per batch
        log_ps: tensor of shape [B, n] (unnormalized log-probabilities)
    Returns:
        counts: tensor of shape [B, n]
    """
    probs = F.softmax(log_ps, dim=-1)
    # Draw samples with replacement
    samples = torch.multinomial(probs, num_samples=count, replacement=True)
    B, n = log_ps.shape
    counts = torch.zeros((B, n), dtype=torch.int64, device=log_ps.device)
    for i in range(B):
        counts[i] = torch.bincount(samples[i], minlength=n)
    return counts


def iterative_dfo(network, batch_size, observations, action_samples,
                  policy_state, num_action_samples, min_actions, max_actions,
                  temperature=0.1, num_iterations=3, iteration_std=0.33,
                  training=False, late_fusion=False, tfa_step_type=()):
    """
    Port of TensorFlow iterative DFO to PyTorch.
    """
    device = action_samples.device
    if late_fusion:
        # Pre-encode observations
        obs_encodings = network.encode(observations, training=training)
        obs_encodings = obs_encodings.repeat(num_action_samples, 1)

    def update_selected_actions(samples, policy_state):
        if late_fusion:
            net_logits, new_state = network(observations, samples,
                                           observation_encoding=obs_encodings,
                                           training=training)
        else:
            net_logits = network(observations, samples)
            new_state = None
        # Reshape to [B, n]
        net_logits = net_logits.view(batch_size, num_action_samples)
        log_probs = net_logits / temperature
        # Count selections
        actions_selected = categorical_bincount(num_action_samples, log_probs)
        # Flatten back to [B*n]
        flat_counts = actions_selected.view(-1)
        # Repeat indices accordingly
        indices = torch.arange(batch_size * num_action_samples, device=device)
        repeat_idx = torch.repeat_interleave(indices, flat_counts)
        new_samples = samples[repeat_idx]
        return log_probs, new_samples, new_state

    log_probs, action_samples, new_state = update_selected_actions(
        action_samples, policy_state)

    for _ in range(num_iterations - 1):
        noise = torch.randn_like(action_samples) * iteration_std
        action_samples = torch.clamp(action_samples + noise, min_actions, max_actions)
        log_probs, action_samples, new_state = update_selected_actions(
            action_samples, new_state)
        iteration_std *= 0.5

    probs = F.softmax(log_probs, dim=1).view(-1)
    return probs, action_samples, new_state


def sample_action_from_Q_function(Q_value_network, state, batch_size,
                                  sample_actions_size, evaluation, dim_a, device, 
                                  mcmc_iteration=25, horizon=1):
    
    # Initial random actions
    total = batch_size * sample_actions_size
    # import pdb; pdb.set_trace()
    if horizon == 1:
        action_samples = torch.rand((total, dim_a), device=device) * 2 - 1
    else:
        action_samples = torch.rand((total, horizon, dim_a), device=device) * 2 - 1
    # Tile state
    # state_tiled = state.repeat(sample_actions_size, 1).detach()
    state_tiled = state.repeat(sample_actions_size, 1)
    # Langevin sampling (assumes mcmc functions implemented for PyTorch)
    action_samples = mcmc.langevin_actions_given_obs(
        Q_value_network, state_tiled, action_samples, num_iterations=mcmc_iteration,
        policy_state=None, min_actions=-1, max_actions=1,
        training=False, tfa_step_type=(), return_chain=False,
        grad_norm_type='inf', num_action_samples=sample_actions_size)
    # Second refinement
    action_samples = mcmc.langevin_actions_given_obs(
        Q_value_network, state_tiled, action_samples, num_iterations=mcmc_iteration,
        policy_state=None, min_actions=-1, max_actions=1,
        training=False, tfa_step_type=(), return_chain=False,
        grad_norm_type='inf', sampler_stepsize_init=1e-1,
        sampler_stepsize_final=1e-5, num_action_samples=sample_actions_size)
    # Compute probabilities
    probs = mcmc.get_probabilities(Q_value_network, batch_size,
                                sample_actions_size, state_tiled,
                                action_samples, training=False)

    dist = MappedCategorical(probs=probs, mapped_values=action_samples)
    action = dist.sample()
    return action


def sample_action_from_Q_function_greedy(last_action, Q_value_network,
                                          state, batch_size,
                                          sample_actions_size, evaluation,
                                          dim_a, device):
    total = batch_size * sample_actions_size
    action_samples = torch.rand((total, dim_a), device=device) * 2 - 1
    state_tiled = state.repeat(sample_actions_size, 1)
    probs, action_samples, _ = iterative_dfo(
        Q_value_network, batch_size, state_tiled, action_samples,
        None, sample_actions_size, -1, 1)

    dist = MappedCategorical(probs=probs, mapped_values=action_samples)
    # Greedy: pick highest-prob action
    idx = torch.argmax(probs)
    action = action_samples[idx].unsqueeze(0)
    # Optionally pick closest to last_action
    if last_action is not None:
        modes = dist.mode()
        dists = torch.norm(modes - last_action, dim=1)
        best = torch.argmin(dists)
        action = modes[best].unsqueeze(0)
    return action


def make_counter_example_actions(Q_value_network, observations,
                                 batch_size, dim_a,
                                 sample_closer_to_optimal_a=False,
                                 training=False,
                                 sample_actions_size=1024, horizon=1):
    total = batch_size * sample_actions_size
    if horizon == 1:
        random_actions = torch.rand((total, dim_a), device=observations.device) * 2 - 1
    else:
        random_actions = torch.rand((total, horizon, dim_a), device=observations.device) * 2 - 1
    # Langevin sampling
    samples = mcmc.langevin_actions_given_obs(
        Q_value_network, observations, random_actions,
        policy_state=None, min_actions=-1, max_actions=1,
        training=False, num_iterations=25,
        tfa_step_type=(), return_chain=False,
        grad_norm_type='inf', num_action_samples=sample_actions_size)
    if sample_closer_to_optimal_a:
        samples = mcmc.langevin_actions_given_obs(
            Q_value_network, observations, samples,num_iterations=10,
            policy_state=None, min_actions=-1, max_actions=1,
            training=False, tfa_step_type=(), return_chain=False,
            grad_norm_type='inf', sampler_stepsize_init=1e-1,
            sampler_stepsize_final=1e-5,
            num_action_samples=sample_actions_size)
    # Reshape to [sample_size, batch_size, dim_a]
    if horizon == 1:
        counter_examples = samples.view(sample_actions_size, batch_size, dim_a)
    else:
        counter_examples = samples.view(sample_actions_size, batch_size, horizon, dim_a)
    return counter_examples.detach(), None


def make_target_distribution_actions(Q_value_network, observations,
                                     batch_size, dim_a,
                                     action_data, h_human, normalized_random_hs,
                                     training=False,
                                     sample_actions_size=1024):
    total = batch_size * sample_actions_size
    random_actions = torch.rand((total, dim_a), device=observations.device) * 2 - 1
    samples = mcmc.langevin_actions_given_obs_with_corrective_measurement_model(
        Q_value_network, observations, random_actions,
        policy_state=None, min_actions=-1, max_actions=1,
        action_data=action_data, h_human=h_human,
        normalized_random_hs=normalized_random_hs,
        training=False, tfa_step_type=(), return_chain=False,
        grad_norm_type='inf', num_action_samples=sample_actions_size)
    counter_examples = samples.view(sample_actions_size, batch_size, dim_a)
    return counter_examples, None


def grad_penalty(energy_network, batch_size,
                 observations, combined_true_counter_actions,
                 grad_norm_type='inf', training=False,
                 only_apply_final_grad_penalty=True,
                 grad_margin=1.0, square_grad_penalty=True,
                 grad_loss_weight=1.0):
    # Compute gradients wrt actions
    de_dact, _ = mcmc.gradient_wrt_act(
        # energy_network, observations.detach(),
        energy_network, observations,
        combined_true_counter_actions.detach(), training=training,
        network_state=None, tfa_step_type=(), apply_exp=False)
    grad_norms = mcmc.compute_grad_norm(grad_norm_type, de_dact)
    grad_norms = grad_norms.reshape(batch_size, -1)
    if grad_margin is not None:
        grad_norms = torch.clamp(grad_norms - grad_margin, min=0.0)
    if square_grad_penalty:
        grad_norms = grad_norms**2
    grad_loss = grad_norms.mean(dim=1)
    return grad_loss * grad_loss_weight


def get_valid_sampled_actions(sampled_action, angle_condition, target_count=256):
    # sampled_action: [T, B, A], angle_condition: [T, B]
    # Sort by condition
    sorted_idx = torch.argsort(angle_condition.int(), dim=0, descending=True)
    T, B, A = sampled_action.shape
    # Gather
    batch_idx = torch.arange(B, device=sampled_action.device).unsqueeze(0).repeat(T, 1)
    indices = sorted_idx, batch_idx
    sorted_actions = sampled_action[indices]
    sorted_conditions = angle_condition[indices]
    # Mask where all true
    all_true = sorted_conditions.all(dim=1)
    valid = sorted_actions[all_true]
    count = valid.shape[0]
    if count >= target_count:
        return valid[:target_count]
    return valid


# Learning rate decay schedule using PyTorch LambdaLR
class TimeBasedDecay:
    def __init__(self, initial_lr, decay_rate):
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate

    def __call__(self, step):
        return self.initial_lr / (1 + self.decay_rate * step)



class IBC_Image(BaseImagePolicy):
    def __init__(self, 
                horizon,
                shape_meta, dim_a, dim_o, 
                 action_upper_limits, action_lower_limits,
                 buffer_min_size, buffer_max_size, buffer_sampling_rate,
                 buffer_sampling_size, train_end_episode,
                 policy_model_learning_rate,
                 saved_dir, load_dir, load_policy,
                 number_training_iterations,
                 sample_action_number,
                 obs_encoder_crop_shape = [76, 76],
                 n_action_steps = 1, nn_hidden_dim = None,
                 n_obs_steps=2, mcmc_iteration=25):
        
        super().__init__()
        
        # Initialize variables
        self.n_obs_steps = n_obs_steps
        self.h = None
        self.state_representation = None
        self.policy_action_label = None
        self.dim_a = dim_a
        self.dim_o = dim_o
        self.action_lower_limits = action_lower_limits
        self.count = 0
        self.buffer_sampling_rate = buffer_sampling_rate
        self.buffer_sampling_size = buffer_sampling_size
        self.train_end_episode = train_end_episode
        self.policy_model_learning_rate = policy_model_learning_rate
        self.buffer_max_size = buffer_max_size
        self.buffer_min_size = buffer_min_size
        
        self.device = torch.device("cuda:0")  

        # Define observation encoder. Its input is obs_dict, including images and low-level info
        self.obs_encoder = MultiImageObsEncoder(
            shape_meta=shape_meta,
            resize_shape=None,
            crop_shape=obs_encoder_crop_shape,
            random_crop=True,
            use_group_norm=True,
            share_rgb_model=False,
            imagenet_norm=False,
            use_spatial_softmax = True,
        ).to(self.device)
        self.obs_feature_dim = self.obs_encoder.output_shape()[0]

        input_dim = dim_a 
        global_cond_dim = self.obs_feature_dim * n_obs_steps
        if horizon > 1:
            from agents.DP_model.energy_nn import UnetEncoder_QModel
            self.action_value_model = UnetEncoder_QModel(
                input_dim=input_dim,
                local_cond_dim=None,
                down_dims=[256, 512, 1024],
                # down_dims=[512, 1024, 2048],
                kernel_size=5,
                n_groups=8,
                dim_o = global_cond_dim, 
                state_emb_dim = global_cond_dim,
                cond_predict_scale=True
            ).to(self.device)
        else:
            from agents.DP_model.energy_nn import ActionValueFunctionModel
            self.action_value_model = ActionValueFunctionModel(dim_a=dim_a, dim_o=self.obs_feature_dim* n_obs_steps, hidden_dims= nn_hidden_dim).to(self.device)

        
        
        self.optimizer_action_value_model = torch.optim.AdamW(
            self.parameters(),
            lr=self.policy_model_learning_rate,
            betas=(0.9, 0.999),
            eps=1.0e-7,
            weight_decay=1.0e-6
        )
        
        self.lr_scheduler = CosineAnnealingWarmupRestarts(
            self.optimizer_action_value_model,
            first_cycle_steps=162 * (500 + number_training_iterations),
            cycle_mult=1.0,
            max_lr=self.policy_model_learning_rate,
            min_lr=1e-5,
            warmup_steps=10,
            gamma=1.0,
        )


        self.buffer = Buffer_uniform_sampling(min_size=self.buffer_min_size,
                                             max_size=self.buffer_max_size)
        self.sample_action_number = sample_action_number
        self.mcmc_iteration = mcmc_iteration

        '''Hdf5 buffer used for real experiments with images (to save RAM space)'''
        # # field_shapes = {
        # #     'agentview_image':           (n_obs_steps, 3, 84, 84),
        # #     'robot0_eye_in_hand_image':  (n_obs_steps,3, 84, 84),
        # #     'robot0_eef_pos':            (n_obs_steps,3),
        # #     'robot0_eef_quat':           (n_obs_steps,4),
        # #     'robot0_gripper_qpos':       (n_obs_steps,2),
        # #     'teacher_action':                    (10,),
        # #     'robot_action':                    (10,),
        # # }

        # field_shapes = {}
        # # Observation fields: prepend n_obs_steps to each obs shape
        # for name, meta in shape_meta.get('obs', {}).items():
        #     dims = meta.get('shape', [])
        #     field_shapes[name] = (n_obs_steps, *tuple(dims))
        # # Action fields: both robot and teacher share the same action shape
        # field_shapes['robot_action']   = (dim_a,)
        # field_shapes['teacher_action'] = (dim_a,)

        # # import pdb; pdb.set_trace()

        # print("field_shapes: ", field_shapes)

        # self.buffer = HDF5Buffer(filename ='buffer.h5', field_shapes=field_shapes, min_size=self.buffer_min_size,
        #                                      max_size=self.buffer_max_size, dtype_map={})

        self.buffer_no_corrections = Buffer(min_size=self.buffer_min_size,
                                           max_size=self.buffer_max_size)
        self.policy_loss_list = []
        self.test_New_value_function_idea = False
        self.saved_dir = saved_dir
        self.load_dir = load_dir
        self.load_policy_flag = load_policy
        self.number_training_iterations = number_training_iterations
        self.time_step = None
        self.evaluation = False  # for the number of action spaces during inference
        self.evaluation_last = False
        self.list_action_tobeQueried = []
        self.last_action = True
        self.use_image = False
        self.sampled_action_last = None
        self.use_CLIC_algorithm = False
        self.e = 0.2 # used for relative correction data
        self.horizon = horizon

    def save_model(self):
        # Define the directory for saving model parameters
        network_saved_dir = self.saved_dir + 'network_params/'
        if not os.path.exists(network_saved_dir):
            os.makedirs(network_saved_dir)
        
        # Save the model state dictionary
        model_filename = network_saved_dir + 'action_value_model.pth'
        torch.save({'action_value_model_state_dict': self.action_value_model.state_dict()}, model_filename)
        # save the obs_encoder
        obs_enc_path = network_saved_dir + 'obs_encoder.pth'
        torch.save({'obs_encoder_state_dict': self.obs_encoder.state_dict()}, obs_enc_path)
        
        logger.info(f"action_value_model saved at {model_filename}")

    def load_model(self):
        model_dir = os.path.join(self.load_dir, 'network_params')

        # Load policy model
        model_path = os.path.join(model_dir, 'action_value_model.pth')
        if os.path.isfile(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.action_value_model.load_state_dict(checkpoint['action_value_model_state_dict'])
            logger.info(f"action_value_model loaded from {model_path}")
        else:
            logger.warning(f"action_value_model file not found at {model_path}, skipping.")

        # Load observation encoder
        obs_enc_path = os.path.join(model_dir, 'obs_encoder.pth')
        if os.path.isfile(obs_enc_path):
            checkpoint = torch.load(obs_enc_path, map_location=self.device)
            self.obs_encoder.load_state_dict(checkpoint['obs_encoder_state_dict'])
            logger.info(f"Obs encoder loaded from {obs_enc_path}")
        else:
            logger.warning(f"Obs encoder file not found at {obs_enc_path}, skipping.")

    def custom_l4_penalty(self, Q_values, lower_bound=-1.0, upper_bound=1.0):
        penalty = torch.clamp(Q_values - upper_bound, min=0.0)**2
        penalty += torch.clamp(Q_values - lower_bound, max=0.0)**2
        return torch.sum(penalty**2)

    def action_value_single_update_InfoNCE(self, observation, action, h_human, next_observation):
        self.train()
        self.action_value_model.training = True

        batch_size = h_human.shape[0]
        action_dim = h_human.shape[-1]  # (Batch, horizon, dim_a)
        sample_h_size = self.sample_action_number



        softmax_temperature = 1.0

        nobs_features = self.obs_encoder(observation)
        global_cond = nobs_features.reshape(batch_size, -1)   # just the observation, used to condition the action

        observation_tiled = global_cond.repeat(sample_h_size, 1)

        sampled_action, _ = make_counter_example_actions(
            self.action_value_model, observation_tiled,
            batch_size, action_dim,
            sample_actions_size=sample_h_size,
            horizon=self.horizon,
            sample_closer_to_optimal_a=True)
        # prepend true action and shifted action
        first = h_human.unsqueeze(0)
        
        sampled_action = torch.cat([first, sampled_action[1:]], dim=0)
        if self.horizon > 1:
            sampled_action = sampled_action.view(-1, self.horizon, action_dim)
        else:
            sampled_action = sampled_action.view(-1, action_dim)
        self.sampled_action_last = sampled_action


        # Q_s_a_target = self.action_value_model(observation_tiled, sampled_action).detach()
        # Q_s_a_target = Q_s_a_target.view(sample_h_size, batch_size).transpose(0, 1)

        # compute area-sector
        
        
        if self.horizon == 1:
            h_tiled = h_human.repeat(sample_h_size, 1)
            log_prob = -torch.sum((sampled_action - h_tiled) ** 2, dim=-1, keepdim=True)
        else:
            h_tiled = h_human.repeat(sample_h_size, 1, 1)
            log_prob = -torch.sum((sampled_action - h_tiled) ** 2, dim=(-2, -1), keepdim=True)
        
        threshold = 1e-16
        cond = (-log_prob < threshold).float()  
        cond = cond.view(sample_h_size, batch_size).transpose(0, 1) 
        obs_tiled = observation_tiled
        Q_s_a = self.action_value_model(obs_tiled, sampled_action)


        labels = cond
        labels = labels / (labels.sum(dim=1, keepdim=True) + 1e-8)
            
        # InfoNCE
        
       
        preds = Q_s_a.reshape(sample_h_size, batch_size)
        soft_p = F.softmax(preds/softmax_temperature, dim=0).transpose(0,1)
        
        
        loss_kl = F.kl_div(soft_p.log(), labels, reduction='batchmean')
        # loss_kl = F.kl_div(soft_p, labels_dynamic, reduction='batchmean')
        grad_pen = grad_penalty(self.action_value_model, batch_size, observation_tiled, sampled_action, training=True)
        grad_pen = grad_pen.mean()
        loss = loss_kl + 0.5*grad_pen
        logger.debug('IBC loss_Kl:  %s  grad_pen:  %s', loss_kl, grad_pen)
        self.optimizer_action_value_model.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.action_value_model.parameters(), 1.0)
        self.optimizer_action_value_model.step()
        self.lr_scheduler.step()


    def action_value_batch_update(self, batch):
        # state_batch = batch[0]
        # action_batch = batch[1]
        # h_human_batch = batch[2]
        # next_state_batch = batch[3]
        state_batch = [pair[0] for pair in batch]
        action_batch = [np.array(pair[2]) for pair in batch]  # robot action
        h_human_batch = [np.array(pair[1]) for pair in batch]  # human action
        batch_size = len(batch)
        # state = torch.tensor(np.stack(state_batch), dtype=torch.float32, device=self.device)
        # import pdb; pdb.set_trace()
        nobs = collate_obs_dict(state_batch)
        
        if self.horizon > 1:
            h_human_batch     = torch.tensor(np.reshape(h_human_batch, [batch_size, self.horizon, self.dim_a]), dtype=torch.float32, device=self.device)
            action_batch     = torch.tensor(np.reshape(action_batch, [batch_size, self.horizon, self.dim_a]), dtype=torch.float32, device=self.device)
        else:
            h_human_batch     = torch.tensor(np.reshape(h_human_batch, [batch_size, self.dim_a]), dtype=torch.float32, device=self.device)
            action_batch     = torch.tensor(np.reshape(action_batch, [batch_size, self.dim_a]), dtype=torch.float32, device=self.device)

        nobs = dict_apply(nobs, 
                    lambda x: torch.from_numpy(x).to(
                        device=self.device,  dtype=torch.float32))
        this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
        next_obs = None

       
        self.action_value_single_update_InfoNCE(
            this_nobs, action_batch, h_human_batch, next_obs
        )

    def action(self, obs_dict):
        if self.evaluation is True and self.evaluation_last is False:  # only set once
            self.evaluation_last = True
            # self.model.eval()  
            self.eval() # set self.training = False
            self.training = False
            logger.debug("set model.eval")

        

        obs_dict = dict_apply(obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=self.device,  dtype=torch.float32))
        obs_dict = dict_apply(obs_dict, 
                lambda x: x.unsqueeze(0))
        
        # import pdb
        # pdb.set_trace()

        this_nobs = dict_apply(obs_dict, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
        
        nobs_features = self.obs_encoder(this_nobs)
        state = nobs_features.reshape(1, -1) 

        action = sample_action_from_Q_function(self.action_value_model, state, 1, 512, True, self.dim_a, self.device, 
                                                horizon = self.horizon, mcmc_iteration=self.mcmc_iteration).detach()

        action = action.cpu().numpy()
        # print("action: ", action.shape)
        # import pdb; pdb.set_trace()
        '''test ramdom noise as kind of exploration'''
        # if not self.test_New_value_function_idea:
        #     if not self.evaluation:
        #         rand_number = random.random()
        #         if rand_number < 0.2:
        #             action = action + np.random.normal(loc=0.0, scale=0.2, size=action.shape)

        # out_action = []
        

        # for i in range(self.dim_a):
        #     action[i] = np.clip(action[i], -1, 1)
        #     out_action.append(action[i])
        numpy_action = np.clip(action, -1, 1)

        if self.horizon > 1:
            numpy_action = numpy_action[1:, :]

        return numpy_action
    
    def collect_data_and_train(self, last_action, h, obs_proc, next_obs, t, done, agent_algorithm=None, agent_type=None, i_episode=None):
        """Unified entry point used by main_IIL.py."""
        return self.TRAIN_Policy_with_Behavior_Cloning_Objective(last_action, t, done, i_episode, h, obs_proc)

    def TRAIN_Policy_with_Behavior_Cloning_Objective(self, action, t, done, i_episode, h, observation):
        next_observation = None # not implemented
        if np.any(h):  # if any element is not 0
            # 1. append  (o_t,  h_t, a_T) to D
           # h_t is the optimal action here and a_t is the robot action
            # print("action in TRAIN_Policy_with_Behavior_Cloning_Objective: ", action, " h: ", h)
            self.buffer.add([observation, h, action])
            self.latested_data_pair = [observation, h, action]
            # 4. Update Human model with a minibatch sampled from buffer D
            if self.buffer.initialized():

                self.train()  # set self.training = True
                self.training = True
                self.evaluation_last = False
                batch = self.buffer.sample(batch_size=self.buffer_sampling_size)
                # include the new data in this batch
                # batch[-1] = [observation, action, h, next_observation]
                batch[-1] = self.latested_data_pair
                self.action_value_batch_update(batch)

        # Train policy every k time steps from buffer
        elif self.buffer.initialized() and t % self.buffer_sampling_rate == 0:
            logger.info("train q value, IBC")
            #print('Train policy every k time steps from buffer')
            # update Human model
            for i in range(1):  # this is better than range(1), you should train the Q function a litte bit frequent than policy network
            # for i in range(1):
                batch = self.buffer.sample(batch_size=self.buffer_sampling_size)
                self.action_value_batch_update(batch)

            # # # Batch update of the policy with the Human Model
            # batch = self.buffer.sample(batch_size=self.buffer_sampling_size)
            # self._policy_batch_update_with_Q_value(batch)

        if done:
            self.last_action = None

        if self.buffer.initialized() and (self.train_end_episode and done):
            # self.buffer.reset_temp_buffer()
            # self.number_training_iterations = 300
            # self.number_training_iterations = 20
            self.train()  # set self.training = True
            self.training = True
            self.evaluation_last = False

            for i in range(self.number_training_iterations):
                if i % (self.number_training_iterations / 20) == 0:
                    logger.info("Progress Policy training: %i %%", i / self.number_training_iterations * 100)
                    logger.debug('buffer size:  %s', self.buffer.length())
                   
                for i in range(1):  # this is better than range(1), you should train the Q function a litte bit frequent than policy network
                # for i in range(1):
                    batch = self.buffer.sample(batch_size=self.buffer_sampling_size)
                    self.action_value_batch_update(batch)

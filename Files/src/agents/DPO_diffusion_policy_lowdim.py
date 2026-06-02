import logging
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from agents.DP_model.diffusion.mask_generator import LowdimMaskGenerator

import os
import numpy as np
from tools.buffer import Buffer, Buffer_uniform_sampling
import time
from torch.utils.tensorboard import SummaryWriter

import pdb

from agents.DP_model.common.normalizer import LinearNormalizer
from agents.DP_model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from agents.DP_model.common.pytorch_util import dict_apply

from tools.test_reflection import iterative_refelection_cone
from agents.DP_model.common.scheduler import CosineAnnealingWarmupRestarts


import copy

logger = logging.getLogger(__name__)

def collate_obs_dict(obs_list):
    """
    Collate a list of observation dictionaries into a single dictionary
    with batched tensors.
    
    Parameters:
        obs_list (List[Dict[str, Any]]): A list of dictionaries where each
            dictionary holds observations (e.g., 'rgb', 'lowdim').
            
    Returns:
        Dict[str, torch.Tensor]: A dictionary with the same keys as the input
            dictionaries. Each key maps to a tensor of shape (B, ...), where B
            is the number of observation dictionaries in the input list.
    """

    if not obs_list:
        raise ValueError("The list of observation dictionaries is empty.")
    
    # Assume all dictionaries have the same keys.
    keys = list(obs_list[0].keys())
    collated = {key: [] for key in keys}
    
    # Loop through each observation dictionary in the list.
    for obs in obs_list:
        for key in keys:
            data = obs[key]
            collated[key].append(data)
    
    # Stack tensors along a new batch dimension.
    for key in keys:
        collated[key] = np.stack(collated[key])
    
    return collated

class ModuleAttrMixin(nn.Module):
    def __init__(self):
        super().__init__()
        self._dummy_variable = nn.Parameter()

    def device(self):
        return next(iter(self.parameters())).device
    
    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


class BaseImagePolicy(ModuleAttrMixin):
    # init accepts keyword argument shape_meta, see config/task/*_image.yaml

    def action(self, obs_dict):
        """
        obs_dict:
            str: B,To,*
        return: B,Ta,Da
        """
        raise NotImplementedError()

    # reset state for stateful policies
    def reset(self):
        pass

    # ========== training ===========
    # no standard training interface except setting normalizer
    def set_normalizer(self, normalizer: LinearNormalizer):
        raise NotImplementedError()


class DiffusionUnetLowdimPolicy_DPO(BaseImagePolicy):
    def __init__(self, 
            noise_scheduler: DDPMScheduler,
            noise_scheduler_inference,
            horizon, obs_dim, action_dim, shape_meta,
            saved_dir, load_dir, load_pretrained_dir, load_policy, number_training_iterations,
            e_matrix, loss_weight_inverse_e, 
            desiredA_type, large_desiredA,
            sphere_alpha, sphere_gamma, 
            radius_ratio, 
            sample_action_number, sample_with_desiredA_reverse_start_t,
            buffer_min_size, buffer_max_size,
            buffer_sampling_rate, buffer_sampling_size, 
            policy_model_learning_rate,
            n_action_steps = 1, 
            n_obs_steps = 1 ,
            num_inference_steps=None,
            obs_as_local_cond=False,
            obs_as_global_cond=True,
            pred_action_steps_only=False,
            oa_step_convention=False, 
            diffusion_step_embed_dim = 128, unet_down_dims=[512, 1024, 2048],
            **kwargs):
        super().__init__()
        assert not (obs_as_local_cond and obs_as_global_cond)
        if pred_action_steps_only:
            assert obs_as_global_cond
        
        self.device = torch.device("cuda:0")  
      
        self.noise_scheduler = noise_scheduler
        self.noise_scheduler_inference = noise_scheduler_inference
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_local_cond or obs_as_global_cond) else obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_local_cond = obs_as_local_cond
        self.obs_as_global_cond = obs_as_global_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.oa_step_convention = oa_step_convention
        self.kwargs = kwargs

        self.e = np.diag(e_matrix)
        self.loss_weight_inverse_e = np.diag(loss_weight_inverse_e)
        
        self.desiredA_type = desiredA_type
        assert self.desiredA_type in ['Half', 'Circular']

        self.sphere_alpha = sphere_alpha    # parameter that adjusts where to sample counterexamples on the sphere
        self.sphere_gamma = sphere_gamma    # parameter that adjust the center of the sphere to sample counterexamples
        self.radius_ratio = radius_ratio    # parameter used in circular desired action space

        # For Ta>1, if true, when creating the desired action space, treat the whole action chunk as a single action. 
        self.large_desiredA = large_desiredA  
        self.sample_action_number = sample_action_number
        self.sample_with_desiredA_reverse_start_t = sample_with_desiredA_reverse_start_t

        self.policy_model_learning_rate = policy_model_learning_rate
        self.buffer_max_size = buffer_max_size
        self.buffer_sampling_size = buffer_sampling_size
        self.buffer_min_size = buffer_min_size
        self.buffer_sampling_rate = buffer_sampling_rate
        self.buffer = Buffer_uniform_sampling(min_size=self.buffer_min_size, max_size=self.buffer_max_size )
        self.use_CLIC_algorithm = True # use CLIC method
        self.dim_o = obs_dim
        self.dim_a = action_dim
        self.train_end_episode = True
    

        self.lambda_data = 0.75  # for loss

        self.sampled_action_during_training = None  # used for debugging
        self.sample_trajectories_list = [] # used for debugging

        

        self.evaluation = False  # for the number of action spaces during inference
        self.evaluation_last = False

        self.traning_count = 0
        self.number_training_iterations = number_training_iterations

        self.saved_dir = saved_dir  # used to save for the buffer & network models
        self.load_dir = load_dir
        self.load_pretrained_dir = load_pretrained_dir
        self.load_policy_flag = load_policy
        
        if horizon > 1:
            from agents.DP_model.diffusion.conditional_unet1d_original import ConditionalUnet1D
        else:
            from agents.DP_model.diffusion.conditional_unet1d import ConditionalUnet1D

        self.model = ConditionalUnet1D(
            input_dim=action_dim,
            local_cond_dim=None,
            global_cond_dim=obs_dim* n_obs_steps,
            cond_predict_scale=True
        ).to(self.device)


        self.DPO_use_ref_Model = False
        if self.DPO_use_ref_Model:

            # Start with the same weights as the current model
            self.ref_model = copy.deepcopy(self.model).to(self.device)
            # Freeze reference model parameters
            for p in self.ref_model.parameters():
                p.requires_grad = False
            self.ref_model.eval()

        if load_policy:
            # load model and buffer
            logger.info('self.load_policy_flag:  %s', self.load_policy_flag)
            # also load the buffer
            logger.info("LOAD buffer from load_dir")
            filename = self.load_dir + 'buffer_data.pkl'
            self.buffer.load_from_file(filename)
            logger.info('length of buffer:  %s', self.buffer.length())

            logger.info('LOAD MODELS')
            from hydra.utils import get_original_cwd
            project_root = get_original_cwd()
            full_path = os.path.join(project_root, self.load_dir)
            self.load_dir = full_path
            self.load_model()
                    

        self.count = 0 # count inside dpo loss
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.policy_model_learning_rate,
            betas=(0.9, 0.999),
            eps=1.0e-7,
            weight_decay=1.0e-6
        )

        self.lr_scheduler = CosineAnnealingWarmupRestarts(
            self.optimizer,
            first_cycle_steps=162 * (500 + self.number_training_iterations),
            cycle_mult=1.0,
            max_lr=self.policy_model_learning_rate,
            min_lr=1e-5,
            warmup_steps=10,
            gamma=1.0,
        )

        logger.debug('number of parameters in action decoder: %e', sum(p.numel() for p in self.model.parameters()))
        
        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
    

    def save_model(self):
        # Define the directory for saving model parameters
        network_saved_dir = self.saved_dir + 'network_params/'
        if not os.path.exists(network_saved_dir):
            os.makedirs(network_saved_dir)
        
        # Save the model state dictionary
        model_filename = network_saved_dir + 'diffusion_model.pth'
        torch.save({
            'model_state_dict': self.model.state_dict()
        }, model_filename)
        
        logger.info(f"diffusion model saved at {model_filename}")

    def load_model(self, model_name = None):
        model_dir = os.path.join(self.load_dir, 'network_params')

        # Load policy model
        if model_name is None:
            model_name = 'diffusion_model.pth'
        model_path = os.path.join(model_dir, model_name)
        if os.path.isfile(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Policy model loaded from {model_path}")
        else:
            logger.warning(f"Policy model file not found at {model_path}, skipping.")

    def update_ref_model(self):
        """Copy current model params into reference model (no grad, frozen)."""
        self.ref_model.load_state_dict(self.model.state_dict())
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler_inference
        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t.to(self.device), 
                local_cond=local_cond, global_cond=global_cond.to(self.device))

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory



    def action(self, state_representation):
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        if self.evaluation is True and self.evaluation_last is False:  # only set once
            self.evaluation_last = True
            self.model.eval()
            self.model.training = False
            logger.debug("set model.eval")
        with torch.no_grad():
            state_representation = torch.tensor(state_representation, dtype=self.dtype)
            state_representation = state_representation.unsqueeze(0)
            nobs = state_representation
            B, _, Do = nobs.shape
            To = self.n_obs_steps

            T = self.horizon
            Da = self.action_dim

            # build input
            device = self.device
            dtype = self.dtype

            # handle different ways of passing observation
            local_cond = None
            global_cond = nobs
            global_cond = nobs.reshape(nobs.shape[0], -1)

            
            
            shape = (B, T, Da)
            cond_data = torch.zeros(shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

            # run sampling
            nsample = self.conditional_sample(
                cond_data, 
                cond_mask,
                local_cond=local_cond,
                global_cond=global_cond,
                **self.kwargs)
            
            # unnormalize prediction
            naction_pred = nsample[...,:Da]
            action_pred = naction_pred

            action = action_pred
            if T == 1:
                numpy_action = naction_pred.detach().cpu().numpy().reshape(-1)
            else:
                # get action
                start = To - 1
                end = start + self.n_action_steps
                action = naction_pred[:,start:]
                numpy_action = action.detach().cpu().numpy().reshape(T-start, -1)


            # Clip the values within the range [-1, 1]
            numpy_action = np.clip(numpy_action, -1, 1)
        return numpy_action

    def compute_loss(self, batch):
        self.model.train()
        self.model.training = True
        state_batch = [np.array(pair[0]) for pair in batch]  # state(t) sequence
        h_human_batch = [np.array(pair[1]) for pair in batch]  # last
        batch_size = len(batch)
        # Convert the numpy array to a PyTorch tensor and reshape it
        obs = torch.tensor(np.reshape(state_batch, [batch_size,  self.dim_o]), dtype=torch.float32)
        action     = torch.tensor(np.reshape(h_human_batch, [batch_size, self.horizon, self.dim_a]), dtype=torch.float32)
        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = action.to(self.device)

        global_cond = obs   # just the observation, used to condition the action

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)

        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond.to(self.device))

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        # loss = loss * loss_mask.type(loss.dtype) 
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        logger.debug('DP loss:  %s', loss)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

    def compute_loss_DPO(self, batch, t=None, loss_source="intervention"):
        state_batch = [np.array(pair[0]) for pair in batch]
        h_human_batch = [np.array(pair[1]) for pair in batch]  # Preferred action
        action_negative_batch = [np.array(pair[2]) for pair in batch]  # Non-preferred action
        batch_size = len(batch)
        state = torch.tensor(np.reshape(state_batch, [batch_size,  self.dim_o]), dtype=torch.float32).to(self.device) # (batch, dim_o)
        

        action_preferred     = torch.tensor(np.reshape(h_human_batch, [batch_size, self.horizon, self.dim_a]), dtype=torch.float32).to(self.device)
        action_negative     = torch.tensor(np.reshape(action_negative_batch, [batch_size, self.horizon, self.dim_a]), dtype=torch.float32).to(self.device)
        
        # Sample timesteps
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps,
                                (action_preferred.shape[0],), device=self.device)

        noise_negative = torch.randn_like(action_negative)
        noisy_negative_action = self.noise_scheduler.add_noise(action_negative, noise_negative, timesteps)
        noise_positive = torch.randn_like(action_preferred)
        noisy_preferred_action = self.noise_scheduler.add_noise(action_preferred, noise_positive, timesteps)

        # Concatenate and continue with model prediction
        noisy_action = torch.cat([noisy_preferred_action, noisy_negative_action], dim=0)
        noise = torch.cat([noise_positive, noise_negative], dim=0)
        timesteps = torch.cat([timesteps, timesteps], dim=0)

        pred = self.model(noisy_action, timesteps, local_cond=None,
                        global_cond=torch.cat([state, state], dim=0))

        logger.debug('pred loss.shape:  %s', pred.shape) #  (Batch_size, T, dim_a)
        model_losses = (pred - noise).pow(2).mean(dim=2)
        logger.debug('model_losses:  %s', model_losses.shape)   #  # (2 * Batch_size, T)
        model_losses = model_losses.mean(dim = -1)  # traj-level is better than step-level (when combined with DP loss)
        model_losses_w = model_losses[:batch_size]   # preferred / winner
        model_losses_l = model_losses[batch_size:]   # negative / loser

        logger.debug('model_losses_w:  %s', model_losses_w.shape)   #  # (2 * Batch_size, T)
        logger.debug('model_losses_w:  %s  model_losses_l:  %s', model_losses_w.mean(), model_losses_l.mean())

        # ----------  reference model losses ----------
        if self.DPO_use_ref_Model:
            with torch.no_grad():
                ref_pred = self.ref_model(
                    noisy_action,
                    timesteps,
                    local_cond=None,
                    global_cond=torch.cat([state, state], dim=0)
                )
                ref_losses = (ref_pred - noise).pow(2).mean(dim=2)  # (2B, T)
                ref_losses = ref_losses.sum(dim=-1)                 # (2B,)
                ref_losses_w = ref_losses[:batch_size]
                ref_losses_l = ref_losses[batch_size:]

                logger.debug('ref_losses_w:  %s  ref_losses_l:  %s', ref_losses_w.mean(), ref_losses_l.mean())
            ref_diff = ref_losses_w - ref_losses_l                 # (B,)
        else:
            ref_diff = 0

        model_diff = model_losses_w - model_losses_l

        beta_dpo = 1
        preference_loss = -0.01 * F.logsigmoid(-0.5 * beta_dpo * (model_diff - ref_diff)).mean() # also works with DP loss
        winner_loss = model_losses_w.mean()
        loser_loss = model_losses_l.mean()
        loss = preference_loss + winner_loss
        logger.debug('DPO loss:  %s', loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        current_lr = self.optimizer.param_groups[0]["lr"]

        self._ensure_tensorboard_writer()
        loss_tag_prefix = f"Loss/{loss_source}"
        self.writer.add_scalar(f"{loss_tag_prefix}/preference_loss", preference_loss.detach(), self.global_step)
        self.writer.add_scalar(f"{loss_tag_prefix}/winner_loss", winner_loss.detach(), self.global_step)
        self.writer.add_scalar(f"{loss_tag_prefix}/loser_loss", loser_loss.detach(), self.global_step)
        self.writer.add_scalar(f"{loss_tag_prefix}/total_loss", loss.detach(), self.global_step)
        self.writer.add_scalar("Learning Rate", current_lr, self.global_step)
        self.global_step += 1

        return loss

    def _ensure_tensorboard_writer(self):
        if not hasattr(self, "writer"):
            self.writer = SummaryWriter(log_dir=os.path.join(self.saved_dir, "logs"))
        if not hasattr(self, "global_step"):
            self.global_step = 0
    
    def collect_data_and_train(self, last_action, h, obs_proc, next_obs, t, done, agent_algorithm=None, agent_type=None, i_episode=None):
        """Unified entry point used by main_IIL.py."""
        return self.TRAIN_Diffusion_withDPO(last_action, h, obs_proc, next_obs, t, done) 

    def TRAIN_Diffusion_withDPO(self, action, h, observation, next_observation,  t, done):
        # h: corrective feedback!
        if np.any(h):  # if any element is not 0
            # 1. append  (o_t, a_t, h_t) to D
            self.latested_data_pair =[observation, h, action] 
            self.buffer.add(self.latested_data_pair )  # state, a+, a-
            

            # 4. Update Human model with a minibatch sampled from buffer D
            if self.buffer.initialized():

                batch = self.buffer.sample(batch_size= int(self.buffer_sampling_size / 4))
                # include the new data in this batch
                batch[-1] = self.latested_data_pair
                self.compute_loss_DPO(
                    batch,
                    loss_source="intervention",
                )

        # Train policy every k time steps from buffer
        elif self.buffer.initialized() and t % self.buffer_sampling_rate == 0:
            for i in range(1):  # this is better than range(1), you should train the Q function a litte bit frequent than policy network
                batch = self.buffer.sample(batch_size= int(self.buffer_sampling_size / 4 ))
                self.compute_loss_DPO(
                    batch,
                    t,
                    loss_source="intervention",
                )

        if done:
            self.last_action = None

        if self.buffer.initialized() and (self.train_end_episode and done):
            self.count = 0
            self.train()  # set self.training = True
            self.model.training = True
            self.evaluation_last = False
            for i in range(self.number_training_iterations):
                if i % (self.number_training_iterations / 4) == 0:
                    if self.DPO_use_ref_Model:
                        self.update_ref_model()
                    logger.info("Progress Policy training: %i %%", i / self.number_training_iterations * 100)
                    logger.debug('buffer size:  %s', self.buffer.length())

                for i in range(1):  # this is better than range(1), you should train the Q function a litte bit frequent than policy network
                    batch = self.buffer.sample(batch_size=self.buffer_sampling_size)
                    self.compute_loss_DPO(
                        batch,
                        loss_source="intervention",
                    )


    def TRAIN_Policy_with_Behavior_Cloning_Objective(self, action, t, done, i_episode, h, observation):
        if np.any(h):  # if human teleoperates, also update the policy model
            # save the data pair to the buffer

            self.buffer.add([observation, h])
            self.latested_data_pair = [observation, h]
        
            if self.buffer.initialized():
                batch = self.buffer.sample(batch_size=10)
                batch[-1] = self.latested_data_pair
                self.compute_loss(batch)

        # Train policy every k time steps from buffer
        elif self.buffer.initialized() and t % self.buffer_sampling_rate == 0 or (self.buffer.initialized() and self.train_end_episode and done):
            batch = self.buffer.sample(batch_size=10)
            self.compute_loss(batch)

        if len(self.buffer.buffer) < self.buffer_sampling_size:
            self.traning_count = 0

        if len(self.buffer.buffer) > self.buffer_sampling_size and ( (self.train_end_episode and done)):
            for i in range(self.number_training_iterations):
                if i % (self.number_training_iterations / 20) == 0:
                    logger.info('number_training_iterations:  %s', self.number_training_iterations)
                    logger.info("train diffusion")
                    logger.info("Progress Policy training: %i %%", i / self.number_training_iterations * 100)
                    logger.debug('buffer size:  %s', self.buffer.length())

                batch = self.buffer.sample(batch_size=self.buffer_sampling_size)
                self.compute_loss(batch)

import logging
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

import os
import numpy as np
import time
import pdb
from agents.DP_model.common.normalizer import LinearNormalizer   # not used
from agents.DP_model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from agents.DP_model.common.pytorch_util import dict_apply
from agents.DP_model.diffusion.mask_generator import LowdimMaskGenerator
from agents.replay_batch_loader import collate_obs_dict
from agents.replay_buffer_setup import build_replay_buffer_setup

from tools.test_reflection import iterative_refelection_cone
from agents.DP_model.common.scheduler import CosineAnnealingWarmupRestarts
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

class ModuleAttrMixin(nn.Module):
    def __init__(self):
        super().__init__()
        self._dummy_variable = nn.Parameter()

    # @property
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

## used in 'project_and_reflect_trajectory' If we want to project the outsider randomly into desired action set.
def sample_uniform_in_ball(center, radius, eps=1e-8):
    """
    Uniformly sample points inside a D-dimensional ball.
    center: (..., D)
    radius: (..., 1) or (...,)  (broadcastable to center[..., :1])
    returns: (..., D)
    """
    D = center.shape[-1]

    # Random direction
    direction = torch.randn_like(center)
    direction = direction / (direction.norm(dim=-1, keepdim=True) + eps)

    # Random radius (uniform volume): r = R * U^(1/D)
    U = torch.rand(center.shape[:-1] + (1,), device=center.device, dtype=center.dtype)
    r = radius * U.pow(1.0 / D)

    return center + direction * r

# Helper function of 'project_and_reflect_trajectory'. 
# Used to check if a batch of action-chunks belong to the corresponding set of desired action-chunks
def check_if_inside_desiredA(
        sampled_action, action_negative, action_positive, 
        alpha, gamma, radius_ratio, lambda_data = 0.75,
        desiredA_type = 'Circular', use_lamda=False):  
    
    if desiredA_type == 'Circular':
        h_feedback = action_positive - action_negative

        log_prob_pi_a = -torch.sum(h_feedback ** 2, dim=2, keepdim=True)
        log_prob_pi_a_plus_eh = -torch.sum((sampled_action - action_positive) ** 2, dim=2, keepdim=True)

        condition_insideA = radius_ratio * radius_ratio * log_prob_pi_a - log_prob_pi_a_plus_eh < 0.0

    elif desiredA_type == 'Half':
        beta = alpha* 0.5 * np.pi / 180
        beta = torch.tensor(beta, dtype=torch.float32, device=action_negative.device)
        beta_cos = torch.cos(beta)  # assuming self.sphere_alpha is a tensor or scalar

        action_correction_middle_point = (1.0 - gamma) * action_negative + gamma *  action_positive

        # L2-normalize the difference between sampled_action and the middle point along the last dimension
        normalized_sampled_action_diff = F.normalize(sampled_action - action_correction_middle_point, dim=-1)
        # L2-normalize the difference between action_negative and action_preferred along the last dimension
        normalized_tile_h_human = F.normalize(action_positive - action_negative, dim=-1)
        # Compute the cosine similarity (dot product) along the last dimension
        cosine_angles = (normalized_sampled_action_diff * normalized_tile_h_human).sum(dim=-1)
        # Create a boolean tensor where True if cosine_angles >= beta_cos (i.e. angle <= beta), otherwise False
        angle_condition = cosine_angles >= beta_cos  # equivalent to tf.where(cosine_angles < beta_cos, False, True)
        
        # condition_insideA = angle_condition.unsqueeze(1)
        condition_insideA = angle_condition
    else:
        logger.warning("wrong desiredA_type")
    return condition_insideA

# This function enforces that the intermediate samples remain within the desired set during denoising.
# It corresponds to Line 10 of Algorithm 1 in the SDP paper. 
def project_and_reflect_trajectory(
        action_positive, action_negative, trajectory, check_if_inside_desiredA,
        alpha, gamma, radius_ratio,  
        desiredA_type='Circular'):
    """
    Projects trajectory points onto the line defined by action_positive and action_negative,
    then reflects them across action_positive if still outside the desired action space.

    Args:
        action_positive (torch.Tensor): Positive boundary action (shape: [batch, action_dim]).
        action_negative (torch.Tensor): Negative boundary action (shape: [batch, action_dim]).
        trajectory (torch.Tensor): Points to adjust (shape: [batch, action_dim]).
        check_if_inside_desiredA (callable): Function to check if trajectory is inside desired action space.

    Returns:
        torch.Tensor: Adjusted trajectory (shape: [batch, action_dim]).
    """
    if desiredA_type == 'Half':
        action_direction = action_positive - action_negative
        action_direction_norm = action_direction / (torch.norm(action_direction, dim=-1, keepdim=True) + 1e-8)

        # Identify trajectories outside the desired action space initially
        outside_desiredA_mask = ~check_if_inside_desiredA(
            trajectory, action_negative, action_positive, desiredA_type='Half',
              alpha=alpha, gamma=gamma, radius_ratio=radius_ratio).squeeze(-1)

        # Project these trajectories onto the line
        vector_from_negative = trajectory - action_negative
        proj_lengths = (vector_from_negative * action_direction_norm).sum(dim=-1, keepdim=True)
        projected_points = action_negative + proj_lengths * action_direction_norm

        # Replace only the outside points with their projected positions
        trajectory[outside_desiredA_mask] = projected_points[outside_desiredA_mask]

        # Check again and reflect points still outside
        still_outside_mask = ~check_if_inside_desiredA(
            trajectory, action_negative, action_positive, desiredA_type='Half',
              alpha=alpha, gamma=gamma, radius_ratio=radius_ratio).squeeze(-1)
        if still_outside_mask.any():
            reflected_points = 2 * action_positive - trajectory
            trajectory[still_outside_mask] = reflected_points[still_outside_mask]
    elif desiredA_type == 'Circular':
        
        reflected_points = action_positive  
        # Identify trajectories outside the desired action space initially
        outside_desiredA_mask = ~check_if_inside_desiredA(
            trajectory, action_negative, action_positive, desiredA_type='Circular',
            alpha=alpha, gamma=gamma, radius_ratio=radius_ratio).squeeze(-1)
        
        trajectory[ outside_desiredA_mask] = reflected_points[ outside_desiredA_mask] # project to a^+

        '''Ablation: Reflected to random points inside the desired action set'''
        # # radius per point: R = radius_ratio * ||action_positive - action_negative||
        # h = action_positive - action_negative                    # (B, T, D) or broadcastable
        # R = radius_ratio * h.norm(dim=-1, keepdim=True)          # (B, T, 1)

        # random_inside_points = sample_uniform_in_ball(action_positive, R)  # (B, T, D)

        # trajectory[outside_desiredA_mask] = random_inside_points[outside_desiredA_mask]
       
    else:
        logger.warning("wrong desiredA_type")
    return trajectory

'''Ablation: classifier guidance. The grad is defined as action_positive - sampled_action for single-step actions that are outside the desired action set'''
def obtain_target_distribution_derivative_desiredActionSpace(sampled_action, action, action_positive, lambda_data = 0.75,  desiredA_type='Circular'):
    '''
    Input:
        sampled_action: (sampled_a_size * B, dim_a)
        action: negative action (B, dim_a)
        h_human: (B, dim_a)
        normalized_random_hs: sampled implicit negative h_human (sample_implicit_size * B, dim_a)
    Output:
        d_D_normalized: Gradient tensor (sampled_h_size * B, dim_a)
    '''
    # gradient 
    # d_cos_theta_distance = 0.5 * ((action_positive - lambda_data * action) - (1-lambda_data) * sampled_action  )
    d_cos_theta_distance = action_positive - sampled_action
    angle_condition = check_if_inside_desiredA(sampled_action, action, action_positive, desiredA_type= desiredA_type, alpha=30, gamma=0.9, radius_ratio=0.1)
    d_cos_theta_distance = torch.where(angle_condition, torch.zeros_like(d_cos_theta_distance), d_cos_theta_distance)

    return d_cos_theta_distance

class DiffusionUnetImagePolicy_Set_Supervised(BaseImagePolicy):
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
            n_obs_steps = 2 ,
            obs_encoder_crop_shape = [76, 76],
            num_inference_steps=None,
            obs_as_local_cond=False,
            obs_as_global_cond=True,
            pred_action_steps_only=False,
            oa_step_convention=False, 
            no_negative_action = False, scale_no_negative_action = 0.01,
            use_hdf5_dataset = False,
            use_AutoEncoder_loss = False,
            diffusion_step_embed_dim = 128, unet_down_dims=[512, 1024, 2048],
            frozen_obs_encoder = False, config_agent = None,
            **kwargs):
        super().__init__()
        assert not (obs_as_local_cond and obs_as_global_cond)
        if pred_action_steps_only:
            assert obs_as_global_cond

        self.device = torch.device("cuda:0")  

        self.use_AutoEncoder_loss = use_AutoEncoder_loss
        if not use_AutoEncoder_loss:
            # Define observation encoder. Its input is obs_dict, including images and low-level info
            self.obs_encoder = MultiImageObsEncoder(
                shape_meta=shape_meta,
                resize_shape=None,
                crop_shape=obs_encoder_crop_shape,
                random_crop=True,
                use_group_norm=True,
                share_rgb_model=False,
                imagenet_norm=False,
                use_spatial_softmax = True, # only false for PushT?
            ).to(self.device)
        
        else:
            # Observation encoder used in real-robot experiment. 
            from agents.DP_model.vision.multi_image_obs_encoder_with_decoder_no_transformer import MultiImageObsEncoderWithDecoder
            
            self.obs_encoder = MultiImageObsEncoderWithDecoder(
                shape_meta=shape_meta,
                resize_shape=None,
                crop_shape=obs_encoder_crop_shape,
                random_crop=True,
                use_group_norm=True,
                share_rgb_model=False,
                imagenet_norm=False,
                use_spatial_softmax=False,
                use_global_bottleneck_for_policy=True,   # policy uses z: (B,256) per camera
                decode_from_global_z=True,               # recon from z -> seed -> decoder
                add_coord_channels_to_seed=False,        # set True to append (x,y) to seed
                bottleneck_dim=256,
            ).to(self.device)

        self.obs_feature_dim = self.obs_encoder.output_shape()[0]

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

        self.no_negative_action = no_negative_action  # if true, the agent doesn't have access to a-
        self.scale_no_negative_action =  scale_no_negative_action # a- = a+ + Gaussian(0, scale_no_negative_action^2)

        # For Ta>1, if true, when creating the desired action space, treat the whole action chunk as a single action. 
        self.large_desiredA = large_desiredA  
        self.sample_action_number = sample_action_number
        self.sample_with_desiredA_reverse_start_t = sample_with_desiredA_reverse_start_t

        self.policy_model_learning_rate = policy_model_learning_rate
        self.buffer_max_size = buffer_max_size
        self.buffer_sampling_size = buffer_sampling_size
        self.buffer_min_size = buffer_min_size
        self.buffer_sampling_rate = buffer_sampling_rate

        self.use_set_supervised_algorithm = True
        self.use_CLIC_algorithm = self.use_set_supervised_algorithm  # legacy alias
        self.dim_o = obs_dim
        self.dim_a = action_dim
        self.train_end_episode = True
    
        self.lambda_data = 0.75  # for loss

        self.sampled_action_during_training = None  # used for debugging
        self.sample_trajectories_list = [] # used for debugging

        self.evaluation = False  # for the number of action spaces during inference
        self.evaluation_last = False
        self.training_dataloader_num_workers = int(
            getattr(config_agent, "training_dataloader_num_workers", 4)
        )
        self.training_dataloader_pin_memory = bool(
            getattr(
                config_agent,
                "training_dataloader_pin_memory",
                self.device.type == "cuda",
            )
        )
        self.training_dataloader_prefetch_factor = int(
            getattr(config_agent, "training_dataloader_prefetch_factor", 2)
        )
        self.training_dataloader_traj_ref_num_workers = int(
            getattr(config_agent, "training_dataloader_traj_ref_num_workers", 0)
        )

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
        # Policy Decoder
        self.model = ConditionalUnet1D(
            input_dim=action_dim,
            local_cond_dim=None,
            global_cond_dim=self.obs_feature_dim * n_obs_steps,
            diffusion_step_embed_dim=diffusion_step_embed_dim, 
            kernel_size=5,
            cond_predict_scale=True, 
            down_dims= unet_down_dims,
        ).to(self.device)
        
        # Replay buffer init
        replay_setup = build_replay_buffer_setup(
            config_agent=config_agent,
            shape_meta=shape_meta,
            horizon=horizon,
            action_dim=action_dim,
            n_obs_steps=n_obs_steps,
            buffer_min_size=self.buffer_min_size,
            buffer_max_size=self.buffer_max_size,
            use_hdf5_dataset=use_hdf5_dataset,
            use_autoencoder_loss=use_AutoEncoder_loss,
            default_intervention_replay_buffer_type=(
                "traj_ref_buffer"
                if getattr(config_agent, "use_traj_ref_buffer", False)
                else "pickle_obs_action_buffer"
            ),
            allow_traj_ref_intervention=True,
            selfplay_replay_buffer_type="hdf5_obs_action_buffer",
            create_replay_batch_loader=True,
            replay_batch_loader_kwargs={
                "training_dataloader_num_workers": self.training_dataloader_num_workers,
                "training_dataloader_traj_ref_num_workers": self.training_dataloader_traj_ref_num_workers,
                "training_dataloader_pin_memory": self.training_dataloader_pin_memory,
                "training_dataloader_prefetch_factor": self.training_dataloader_prefetch_factor,
                "persistent_workers": False,
            },
        )
        self.buffer = replay_setup.buffer
        self.replay_batch_loader = replay_setup.replay_batch_loader
        if replay_setup.buffer_selfplay is not None:
            self.buffer_selfplay = replay_setup.buffer_selfplay
        self.intervention_replay_buffer_type = replay_setup.intervention_replay_buffer_type
        self.non_intervention_replay_buffer_type = replay_setup.non_intervention_replay_buffer_type
        self.use_hdf5_dataset = replay_setup.use_hdf5_dataset
        self.use_traj_ref_buffer_for_intervention_buffer = replay_setup.use_traj_ref_buffer_for_intervention_buffer
        self.use_traj_ref_buffer = replay_setup.use_traj_ref_buffer_for_intervention_buffer

        self.frozen_obs_encoder = frozen_obs_encoder
        self.optimizer = torch.optim.AdamW(
            # self.parameters(),
            self.model.parameters() if self.frozen_obs_encoder else self.parameters(),
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
        logger.info('number of parameters in obs encoder: %e', sum(p.numel() for p in self.obs_encoder.parameters()))

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
        torch.save({'model_state_dict': self.model.state_dict()}, model_filename)
        # save the obs_encoder
        obs_enc_path = network_saved_dir + 'obs_encoder.pth'
        torch.save({'obs_encoder_state_dict': self.obs_encoder.state_dict()}, obs_enc_path)
        
        logger.info(f"diffusion model saved at {model_filename}")

    def load_model(self):
        model_dir = os.path.join(self.load_dir, 'network_params')

        # Load policy model
        model_path = os.path.join(model_dir, 'diffusion_model.pth')
        if os.path.isfile(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Policy model loaded from {model_path}")
        else:
            logger.warning(f"Policy model file not found at {model_path}, skipping.")

        # Load observation encoder
        obs_enc_path = os.path.join(model_dir, 'obs_encoder.pth')
        if os.path.isfile(obs_enc_path):
            checkpoint = torch.load(obs_enc_path, map_location=self.device)
            self.obs_encoder.load_state_dict(checkpoint['obs_encoder_state_dict'])
            logger.info(f"Obs encoder loaded from {obs_enc_path}")
        else:
            logger.warning(f"Obs encoder file not found at {obs_enc_path}, skipping.")

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen_obs_encoder:
            self.obs_encoder.eval()  # keep frozen encoder in eval
        return self

    # ========= Generate target actions during training ============
    def conditional_sample_with_DesiredA(self, 
        condition_data, condition_mask, action_negative, action_positive,
        local_cond=None, global_cond=None,
        guidance_scale=1.0,
        reverse_start_t = 12,
        generator=None,
        **kwargs
        ):

        self.sample_trajectories_list = []
        with torch.no_grad():
            model = self.model
            scheduler = self.noise_scheduler
            trajectory = action_positive   # initialize A^{K_A} usng A^+

            scheduler.set_timesteps(self.num_inference_steps)

            for t in scheduler.timesteps:
                if t > reverse_start_t:
                    continue
                
                prev_trajectory = trajectory

                model_output = model(trajectory, t.to(self.device), 
                    local_cond=local_cond, global_cond=global_cond.to(self.device))

                model_output_guided = model_output 

                # compute previous step: x_t -> x_t-1
                trajectory = scheduler.step(
                    model_output_guided, t, trajectory, 
                    generator=generator,
                    **kwargs
                    ).prev_sample
                
                ## add reflection here, if some of trajectory is outside the desired action space, apply projection 
                if self.horizon > 1 and self.large_desiredA is True:
                    # Option 1: define a single desired action set from pair of positive and negative action-chunks
                    action_positive_A = action_positive.reshape(action_positive.shape[0], 1, -1)
                    action_negative_A = action_negative.reshape(action_positive.shape[0], 1, -1)
                    trajectory_A = trajectory.reshape(action_positive.shape[0], 1, -1)
                    trajectory_reflected_A = project_and_reflect_trajectory(action_positive_A, action_negative_A, trajectory_A, 
                                                                            check_if_inside_desiredA, desiredA_type=self.desiredA_type,
                                                                            alpha=self.sphere_alpha, gamma=self.sphere_gamma, radius_ratio=self.radius_ratio)
                    trajectory_reflected = trajectory_reflected_A.reshape(action_positive.shape[0], action_positive.shape[1], -1)
                else:
                    # Option 2: used in the SDP paper [section IV-B], define a sequence of single-step desired action sets
                    trajectory_reflected = project_and_reflect_trajectory(action_positive, action_negative, trajectory, 
                                                                          check_if_inside_desiredA, desiredA_type=self.desiredA_type,
                                                                          alpha=self.sphere_alpha, gamma=self.sphere_gamma, radius_ratio=self.radius_ratio)
                   

                if self.horizon > 1:
                    # for past actions, we donot need to do any sampling. Just set it to the data label
                    trajectory_reflected[:, 0, :] = action_positive[:, 0, :]

                # Optional final clamp to ensure boundaries
                trajectory = torch.clamp(trajectory_reflected, -1, 1)

            # caculate distance to action_positive
            trajectory[condition_mask] = condition_data[condition_mask]        
            return trajectory

    # ''' ========= Generate target actions during training with classifer guidacne, used in ablation study ============''' 
    # def conditional_sample_with_DesiredA(self, 
    #     condition_data, condition_mask, action_negative, action_positive,
    #     local_cond=None, global_cond=None,
    #     guidance_scale=10.0,
    #     reverse_start_t = 12,
    #     generator=None,
    #     **kwargs
    #     ):
    #     self.sample_trajectories_list = []
    #     with torch.no_grad():
    #         model = self.model
    #         scheduler = self.noise_scheduler

    #         trajectory = action_positive

    #         scheduler.set_timesteps(self.num_inference_steps)

    #         for t in scheduler.timesteps:
    #             if t > reverse_start_t:
    #                 continue
                
    #             prev_trajectory = trajectory

    #             # 2. predict model output
    #             model_output = model(trajectory, t.to(self.device), 
    #                 local_cond=local_cond, global_cond=global_cond.to(self.device))

    #             # # 3. compute guidance gradient
    #             grad_log_f = obtain_target_distribution_derivative_desiredActionSpace(
    #                 sampled_action =trajectory, action= action_negative,
    #                 action_positive=action_positive, lambda_data= self.lambda_data, desiredA_type= self.desiredA_type
    #             ).reshape_as(model_output)

    #             # # print("grad_log_f: ", grad_log_f)
    #             # # 4. apply guidance to model prediction
    #             diffusion_scale = 1.0
    #             model_output_guided = diffusion_scale * model_output - guidance_scale * grad_log_f

    #             # 5. compute previous step: x_t -> x_t-1
    #             trajectory = scheduler.step(
    #                 model_output_guided, t, trajectory, 
    #                 generator=generator,
    #                 **kwargs
    #                 ).prev_sample

    #             if self.horizon > 1:
    #                 # for past actions, we donot need to do any sampling. Just set it to the data label
    #                 trajectory[:, 0, :] = action_positive[:, 0, :]
    #             # trajectory = torch.clamp(trajectory, -1, 1)
    #             # trajectory_reflected = iterative_refelection_cone(x0= (1.0-gamma) * action_negative + gamma *  action_positive,
    #             #                                    v=action_positive- action_negative,
    #             #                                    alpha=torch.tensor(alpha * (3.141592653589793 / 180.0)),
    #             #                                    xs=prev_trajectory,
    #             #                                    x_out=trajectory)
                
    #             if t == 0:
    #                 if self.horizon > 1 and self.large_desiredA is True:
    #                     action_positive_A = action_positive.reshape(action_positive.shape[0], 1, -1)
    #                     action_negative_A = action_negative.reshape(action_positive.shape[0], 1, -1)
    #                     trajectory_A = trajectory.reshape(action_positive.shape[0], 1, -1)
    #                     trajectory_reflected_A = project_and_reflect_trajectory(action_positive_A, action_negative_A, trajectory_A, 
    #                                                                             check_if_inside_desiredA, desiredA_type=self.desiredA_type,
    #                                                                             alpha=self.sphere_alpha, gamma=self.sphere_gamma, radius_ratio=self.radius_ratio)
    #                     trajectory = trajectory_reflected_A.reshape(action_positive.shape[0], action_positive.shape[1], -1)
    #                 else:
    #                     trajectory = project_and_reflect_trajectory(action_positive, action_negative, trajectory, 
    #                                                                         check_if_inside_desiredA, desiredA_type=self.desiredA_type,
    #                                                                         alpha=self.sphere_alpha, gamma=self.sphere_gamma, radius_ratio=self.radius_ratio)
                        
    #                 if self.horizon > 1:
    #                     # for past actions, we donot need to do any sampling. Just set it to the data label
    #                     trajectory[:, 0, :] = action_positive[:, 0, :]    
                
    #             # self.sample_trajectories_list.append(trajectory_reflected)  # used for debugging in 2d
    #             # Optional final clamp to ensure boundaries
    #             trajectory = torch.clamp(trajectory, -1, 1)

    #         # caculate distance to action_positive
    #         trajectory[condition_mask] = condition_data[condition_mask]        
    #         return trajectory

    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler_inference  # Use scheduler defined for inference
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


    def sample_actions(self, global_cond, action_negative, action_positive, sampled_action_num, t):
        # global_cond is the feature of state/obs
        """
        Sample a batch of actions from the diffusion policy to represent the probability distribution.

        Args:
            state_representation: Input state representation tensor.
            num_samples (int): Number of action samples to generate.

        Returns:
            numpy_actions: Array of sampled actions, shape (num_samples, action_dim).
        """
        with torch.no_grad():
            
            B = global_cond.shape[0]
            To = self.n_obs_steps

            T = self.horizon
            Da = self.action_dim

            device = self.device
            dtype = self.dtype

            local_cond = None

            shape = (B, T, Da)
            cond_data = torch.zeros(shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

            nsample = self.conditional_sample_with_DesiredA(
                cond_data,
                cond_mask,
                action_negative=action_negative, action_positive=action_positive,
                local_cond=local_cond,
                global_cond=global_cond,
                reverse_start_t= self.sample_with_desiredA_reverse_start_t,  
                **self.kwargs)

            naction_pred = nsample[..., :Da]

            actions = torch.clamp(naction_pred, -1, 1)  # [batch * sampled_size, action_horizon, dim_a]
 
        return actions.detach()
    

    def action(self, obs_dict):
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        if self.evaluation is True and self.evaluation_last is False:  # only set once
            self.evaluation_last = True
            # self.model.eval()  
            self.eval() # set self.training = False
            self.model.training = False
            logger.debug("set model.eval")
        with torch.no_grad():

            obs_dict = dict_apply(obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=self.device,  dtype=torch.float32))
            
            obs_dict = dict_apply(obs_dict, 
                    lambda x: x.unsqueeze(0))
            nobs = obs_dict

            value = next(iter(nobs.values()))
            B = value.shape[0]
            To = self.n_obs_steps

            T = self.horizon
            Da = self.action_dim

            # build input
            device = self.device
            dtype = self.dtype

            # handle different ways of passing observation
            local_cond = None
            global_cond = None

            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            
            nobs_features = self.obs_encoder(this_nobs) # torch.Size([128, 66])
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1) # torch.Size([64, 132])        
            
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

            naction_pred = nsample[...,:Da]

            if T == 1:
                # numpy_action = naction_pred.detach().cpu().numpy().reshape(-1)
                numpy_action = naction_pred.detach().cpu().numpy().reshape(1, -1)
            else:
                # get action
                start = To - 1
                end = start + self.n_action_steps
                # action = naction_pred[:,start:end]
                action = naction_pred[:,start:]
                numpy_action = action.detach().cpu().numpy().reshape(T-start, -1)

            # Clip the values within the range [-1, 1]
            numpy_action = np.clip(numpy_action, -1, 1)
        return numpy_action

    ## SDP policy update, corresponding to Lines 2-14 of Algorithm 1 in the SDP paper.
    def compute_loss_Diffusion_Set_Supervised(self, batch, t=None, loss_source="intervention"):
        auxiliary_loss = None
        auxiliary_loss_selfplay = None
        pin_memory = self.training_dataloader_pin_memory and self.device.type == "cuda"
        if isinstance(batch, dict):
            nobs = dict_apply(
                batch["nobs"],
                lambda x: x.to(
                    device=self.device,
                    dtype=torch.float32,
                    non_blocking=pin_memory,
                ),
            )
            action_preferred = batch["action_preferred"].to(
                device=self.device,
                dtype=torch.float32,
                non_blocking=pin_memory,
            )
            batch_size = action_preferred.shape[0]
            if self.no_negative_action:
                vector_sample = torch.ones_like(action_preferred[0])
                vector_sample = vector_sample / (torch.norm(vector_sample) + 1e-8)
                action_negative = action_preferred + (
                    self.scale_no_negative_action * vector_sample
                )
            else:
                action_negative = batch["action_negative"].to(
                    device=self.device,
                    dtype=torch.float32,
                    non_blocking=pin_memory,
                )
        else:
            h_human_batch = [np.array(pair[1]) for pair in batch]  # Preferred action
            if self.no_negative_action: # if true, the agent doesn't have access to robot action (a-); instead, a- is created by a+ + noise
                vector_sample = np.ones_like(h_human_batch[0])
                vector_sample = vector_sample/np.linalg.norm(vector_sample)
                action_negative_batch = [action_positive + self.scale_no_negative_action * vector_sample
                                          for action_positive in h_human_batch] 
            else:
                action_negative_batch = [np.array(pair[2]) for pair in batch]  # Non-preferred action
            batch_size = len(batch)
            state_batch = [pair[0] for pair in batch]  # state(t) sequence
            nobs = collate_obs_dict(state_batch)

            nobs = dict_apply(nobs, 
                        lambda x: torch.from_numpy(x).to(
                            device=self.device ,  dtype=torch.float32))
            
            action_preferred = torch.tensor(
                np.reshape(h_human_batch, [batch_size, self.horizon, self.dim_a]),
                dtype=torch.float32,
            ).to(self.device)
            action_negative = torch.tensor(
                np.reshape(action_negative_batch, [batch_size, self.horizon, self.dim_a]),
                dtype=torch.float32,
            ).to(self.device)
        
        this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        state = nobs_features.reshape(batch_size, -1)   # the observation, used to condition the action

        ##########[Start] Sample desired action-chunks from target policy ##################################
        # Key difference from BC-based Diffusion Policy: instead of directly imitating
        # action_preferred/action_positive, the policy learns to imitate a desired action set.
        sample_action_number = self.sample_action_number
        state_repeated = state.repeat(sample_action_number, 1) # [batch * sampled_size, dim_o]
        action_preferred_repeated  = action_preferred.repeat(sample_action_number, 1,1)
        action_negative_repeated  = action_negative.repeat(sample_action_number, 1,1)

        start_time = time.time()
        sampled_action = self.sample_actions(state_repeated, action_negative=action_negative_repeated, 
                                             action_positive= action_preferred_repeated, sampled_action_num = sample_action_number, t= t) # [batch * sampled_size, 1, dim_a]
        end_time = time.time()
        logger.debug(f"total time for sampling actions: {end_time - start_time:.6f} seconds")
        ############[End] Sample desired action-chunks from target policy ##################################

        # # Sample timesteps
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps,
                                (sampled_action.shape[0],), device=self.device)
        noise_on_action_samples = torch.randn_like(sampled_action)
        noisy_action_samples = self.noise_scheduler.add_noise(sampled_action, noise_on_action_samples, timesteps)
        
        pred = self.model(noisy_action_samples, timesteps, 
            local_cond=None, global_cond=state_repeated.to(self.device))
     
        loss = (pred - noise_on_action_samples).pow(2).sum(dim=2)
        loss = loss.mean()
        action_loss = loss

        if self.use_AutoEncoder_loss and not self.frozen_obs_encoder:  # used in real-robot experiment
            total_loss_ae, per_key = self.obs_encoder.compute_autoencoder_loss(this_nobs, reduction='mean', return_per_key=True)
            auxiliary_loss = total_loss_ae
            loss = 0.2 * total_loss_ae  + loss
            logger.debug('loss:  %s  total_loss_ae:  %s', loss, total_loss_ae)
            if self.buffer_selfplay.initialized():
                batch_selfplay = self.buffer_selfplay.sample(batch_size=self.buffer_sampling_size)
                state_batch_selfplay = [pair[0] for pair in batch_selfplay]  # state(t) sequence
                nobs_selfplay = collate_obs_dict(state_batch_selfplay)
                nobs_selfplay = dict_apply(nobs_selfplay, 
                            lambda x: torch.from_numpy(x).to(
                                device=self.device ,  dtype=torch.float32))
                this_nobs_selfplay = dict_apply(nobs_selfplay, 
                        lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))

                total_loss_ae_selfplay, per_key = self.obs_encoder.compute_autoencoder_loss(this_nobs_selfplay, reduction='mean', return_per_key=True)
                auxiliary_loss_selfplay = total_loss_ae_selfplay

                loss = 0.2 * total_loss_ae_selfplay + loss
        
        logger.debug('loss:  %s', loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        current_lr = self.lr_scheduler.get_lr()[0]
        logger.debug('lr:  %s', self.lr_scheduler.get_lr())

        if not hasattr(self, "writer"):
            self.writer = SummaryWriter(log_dir=os.path.join(self.saved_dir, "logs"))
        if not hasattr(self, "global_step"):
            self.global_step = 0

        loss_tag_prefix = f"Loss/{loss_source}"
        self.writer.add_scalar(f"{loss_tag_prefix}/action_loss", action_loss.detach(), self.global_step)
        if auxiliary_loss is not None:
            self.writer.add_scalar(f"{loss_tag_prefix}/auxiliary_loss", auxiliary_loss.detach(), self.global_step)
        if auxiliary_loss_selfplay is not None:
            self.writer.add_scalar(f"{loss_tag_prefix}/auxiliary_loss_selfplay", auxiliary_loss_selfplay.detach(), self.global_step)
        self.writer.add_scalar(f"{loss_tag_prefix}/total_loss", loss.detach(), self.global_step)
        self.writer.add_scalar("Learning Rate", current_lr, self.global_step)
        self.global_step += 1

    def collect_data_and_train(self, last_action, h, obs_proc, next_obs, t, done, agent_algorithm=None, agent_type=None, i_episode=None):
        """Unified entry point used by main_IIL.py."""
        return self.TRAIN_Diffusion_with_Set_Supervised(last_action, h, obs_proc, next_obs, t, done, i_episode=i_episode)

    def TRAIN_Diffusion_with_Set_Supervised(self, action, h, observation, next_observation,  t, done, i_episode=None):
        # h: corrective feedback
        if np.any(h):  # if any element is not 0
            if self.replay_batch_loader.infer_replay_buffer_type(self.buffer) == "traj_ref_buffer":
                if i_episode is None:
                    raise ValueError(
                        "i_episode must be provided when self.buffer uses Buffer_uniform_refer_Traj_hdf5."
                    )
                self.latested_data_pair = [int(i_episode), int(t), h, action]
            else:
                self.latested_data_pair =[observation, h, action]  # action: [horizon, dim_a]
            self.buffer.add(self.latested_data_pair )  # state, a+, a-
            
            # Update Policy model with a minibatch sampled from buffer D
            if self.buffer.initialized():
                self.train()  # set self.training = True
                self.model.training = True
                self.evaluation_last = False
                batch, _ = self.replay_batch_loader.sample_intervention_batch(
                    batch_size=int(self.buffer_sampling_size / 4)
                )
                # include the new data in this batch
                if isinstance(batch, dict):
                    batch = self.replay_batch_loader.inject_sample_into_collated_batch(
                        batch,
                        self.latested_data_pair,
                    )
                else:
                    batch[-1] = self.latested_data_pair
                self.compute_loss_Diffusion_Set_Supervised(batch, loss_source="intervention")

        # Train policy every k time steps from buffer
        elif self.buffer.initialized() and t % self.buffer_sampling_rate == 0:
            for i in range(1): 
                batch, _ = self.replay_batch_loader.sample_intervention_batch(
                    batch_size=int(self.buffer_sampling_size / 4)
                )
                self.compute_loss_Diffusion_Set_Supervised(batch, loss_source="intervention")

        if done:
            self.last_action = None
            logger.debug('buffer size:  %s', self.buffer.length())
            
        if self.buffer.initialized() and (self.train_end_episode and done):
            self.train()  # set self.training = True
            self.model.training = True
            self.evaluation_last = False
            for i in range(self.number_training_iterations):
                if i % (self.number_training_iterations / 2) == 0:
                    logger.info("Progress Policy training: %i %%", i / self.number_training_iterations * 100)
                    logger.info('buffer size:  %s', self.buffer.length())
                for i in range(1): 
                    batch, _ = self.replay_batch_loader.sample_intervention_batch(
                        batch_size=self.buffer_sampling_size
                    )
                    self.compute_loss_Diffusion_Set_Supervised(batch, loss_source="intervention")

    # Backward-compatible alias: CLIC-Diffusion is the old name for Set_Supervised Diffusion.
    def TRAIN_Diffusion_withCLIC(self, *args, **kwargs):
        return self.TRAIN_Diffusion_with_Set_Supervised(*args, **kwargs)

    def compute_loss_Diffusion_CLIC(self, *args, **kwargs):
        return self.compute_loss_Diffusion_Set_Supervised(*args, **kwargs)


DiffusionUnetImagePolicy_CLIC = DiffusionUnetImagePolicy_Set_Supervised

__all__ = [
    "BaseImagePolicy",
    "DiffusionUnetImagePolicy_Set_Supervised",
    "DiffusionUnetImagePolicy_CLIC",
    "ModuleAttrMixin",
    "check_if_inside_desiredA",
    "collate_obs_dict",
    "obtain_target_distribution_derivative_desiredActionSpace",
    "project_and_reflect_trajectory",
    "sample_uniform_in_ball",
]

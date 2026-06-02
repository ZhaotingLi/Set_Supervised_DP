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

import pdb

from agents.DP_model.common.normalizer import LinearNormalizer   # not used
from agents.DP_model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from agents.DP_model.common.pytorch_util import dict_apply

from tools.test_reflection import iterative_refelection_cone
from agents.DP_model.common.scheduler import CosineAnnealingWarmupRestarts

from collections import deque
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class RunningStatsVec:
    """Exponentially weighted running mean/variance/covariance for vector data."""
    def __init__(
        self,
        dim: int,
        forgetting_factor: float = 0.95,
        mean_last_n_window: int = 0,
    ):
        self.n = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.mean_last_n = np.zeros(dim, dtype=np.float64)
        self.M2 = np.zeros(dim, dtype=np.float64)
        self.M2_cov = np.zeros((dim, dim), dtype=np.float64)
        self.forgetting_factor = float(forgetting_factor)
        self.weight_sum = 0.0
        self.weight_sq_sum = 0.0
        self.mean_last_n_window = max(0, int(mean_last_n_window))
        self.last_n_values = deque()
        self.last_n_sum = np.zeros(dim, dtype=np.float64)

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if self.n == 0 and self.mean.shape[0] != x.shape[0]:
            self.mean = np.zeros_like(x)
            self.mean_last_n = np.zeros_like(x)
            self.M2 = np.zeros_like(x)
            self.M2_cov = np.zeros((x.shape[0], x.shape[0]), dtype=np.float64)
            self.last_n_sum = np.zeros_like(x)

        beta = self.forgetting_factor
        if not (0.0 < beta <= 1.0):
            raise ValueError("forgetting_factor must be in (0, 1].")

        prev_weight_sum = self.weight_sum
        prev_mean = self.mean.copy()
        decayed_weight_sum = beta * prev_weight_sum
        self.weight_sum = decayed_weight_sum + 1.0
        self.weight_sq_sum = (beta * beta) * self.weight_sq_sum + 1.0

        self.n += 1
        delta = x - prev_mean
        self.mean = prev_mean + delta / self.weight_sum

        if decayed_weight_sum > 0.0:
            scatter_scale = decayed_weight_sum / self.weight_sum
            outer = np.outer(delta, delta)
            self.M2 = beta * self.M2 + scatter_scale * np.diag(outer)
            self.M2_cov = beta * self.M2_cov + scatter_scale * outer
        else:
            self.M2.fill(0.0)
            self.M2_cov.fill(0.0)

        if self.mean_last_n_window > 0:
            if len(self.last_n_values) >= self.mean_last_n_window:
                self.last_n_sum -= self.last_n_values.popleft()
            self.last_n_values.append(x)
            self.last_n_sum += x
            self.mean_last_n = self.last_n_sum / len(self.last_n_values)
        else:
            self.mean_last_n = self.mean.copy()

    def var(self) -> np.ndarray:
        if self.n < 2:
            return np.zeros_like(self.mean)
        denom = self.weight_sum - (self.weight_sq_sum / self.weight_sum)
        if denom <= 0.0:
            return np.zeros_like(self.mean)
        return self.M2 / denom

    def cov(self) -> np.ndarray:
        if self.n < 2:
            return np.zeros_like(self.M2_cov)
        denom = self.weight_sum - (self.weight_sq_sum / self.weight_sum)
        if denom <= 0.0:
            return np.zeros_like(self.M2_cov)
        return self.M2_cov / denom

    def scalar_var(self, reduce="mean") -> float:
        v = self.var()
        if reduce == "max":
            return float(np.max(v))
        if reduce == "sum":
            return float(np.sum(v))
        return float(np.mean(v))


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
            # # Convert to tensor if necessary.
            # if not torch.is_tensor(data):
            #     data = torch.tensor(data)
            collated[key].append(data)
    
    # Stack tensors along a new batch dimension.
    for key in keys:
        collated[key] = np.stack(collated[key])
    
    return collated

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


class BaseLowdimPolicy(ModuleAttrMixin):  
    # ========= inference  ============
    # also as self.device and self.dtype for inference device transfer
    # def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    def action(self, state_representation):
        """
        obs_dict:
            obs: B,To,Do
        return: 
            action: B,Ta,Da
        To = 3
        Ta = 4
        T = 6
        |o|o|o|
        | | |a|a|a|a|
        |o|o|
        | |a|a|a|a|a|
        | | | | |a|a|
        """
        raise NotImplementedError()

    # reset state for stateful policies
    def reset(self):
        pass

def sample_uniform_in_ball(center, radius, eps=1e-8):
    D = center.shape[-1]
    direction = torch.randn_like(center)
    direction = direction / (direction.norm(dim=-1, keepdim=True) + eps)
    U = torch.rand(center.shape[:-1] + (1,), device=center.device, dtype=center.dtype)
    r = radius * U.pow(1.0 / D)
    return center + direction * r

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
        outside_desiredA_mask = ~check_if_inside_desiredA(
            trajectory, action_negative, action_positive, desiredA_type='Circular',
            alpha=alpha, gamma=gamma, radius_ratio=radius_ratio).squeeze(-1)
        trajectory[ outside_desiredA_mask] = reflected_points[ outside_desiredA_mask]

    else:
        logger.warning("wrong desiredA_type")
    return trajectory


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
    sample_implicit_size = 128
    sample_h_size = 64

    # gradient 
    # d_cos_theta_distance = 0.5 * ((action_positive - lambda_data * action) - (1-lambda_data) * sampled_action  )
    d_cos_theta_distance = action_positive - sampled_action
    angle_condition = check_if_inside_desiredA(sampled_action, action, action_positive, lambda_data, desiredA_type= desiredA_type)
    d_cos_theta_distance = torch.where(angle_condition, torch.zeros_like(d_cos_theta_distance), d_cos_theta_distance)

    # TODO caculate the gradient of the posterior (with 1/(1+exp( )))

    # pdb.set_trace()
    return d_cos_theta_distance



class DiffusionUnetLowdimPolicy_Set_Supervised(BaseLowdimPolicy):
    def __init__(self, 
            # model: ConditionalUnet1D,
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
            no_negative_action = False,
            diffusion_step_embed_dim = 128, unet_down_dims=[512, 1024, 2048], config_agent = None,
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
        # self.normalizer = LinearNormalizer()
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

        self.traning_count = 0
        self.number_training_iterations = number_training_iterations

        self.saved_dir = saved_dir  # used to save for the buffer & network models
        self.load_dir = load_dir
        self.load_pretrained_dir = load_pretrained_dir
        self.load_policy_flag = load_policy

        self.policy_model_type = getattr(config_agent, "policy_model_type", "ConditionalUnet1D")

        policy_model_key = str(self.policy_model_type).strip().lower()

        if policy_model_key in ["conditionalunet1d", "conditional_unet1d", "unet"]:
            if horizon > 1:
                from agents.DP_model.diffusion.conditional_unet1d_original import ConditionalUnet1D
            else:
                from agents.DP_model.diffusion.conditional_unet1d import ConditionalUnet1D

            self.model = ConditionalUnet1D(
                input_dim=action_dim,
                local_cond_dim=None,
                global_cond_dim=obs_dim * n_obs_steps,
                cond_predict_scale=True,
                diffusion_step_embed_dim=diffusion_step_embed_dim,
                down_dims=unet_down_dims,
            ).to(self.device)
        elif policy_model_key in ["humanfunctionmodel", "human_function_model", "human"]:
            from agents.DP_model.diffusion.conditional_unet1d import HumanFunctionModel
            self.model = HumanFunctionModel(
                dim_a=self.dim_a * horizon,
                dim_o=self.dim_o,
                diffusion_step_embed_dim=diffusion_step_embed_dim,
            ).to(self.device)
        else:
            raise ValueError(
                "Unsupported policy_model_type "
                f"{self.policy_model_type!r}. Use 'ConditionalUnet1D' or 'HumanFunctionModel'."
            )
                    
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
    
        self.save_save_count = 0
    

    def save_model(self):
        # Define the directory for saving model parameters
        network_saved_dir = self.saved_dir + 'network_params/'
        if not os.path.exists(network_saved_dir):
            os.makedirs(network_saved_dir)
        
        # Save the model state dictionary
        # model_filename = network_saved_dir + 'diffusion_model' + str(self.save_save_count) + '.pth'
        model_filename = network_saved_dir + 'diffusion_model' + '.pth'
        torch.save({
            'model_state_dict': self.model.state_dict()
        }, model_filename)
        
        self.save_save_count = self.save_save_count + 1
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

    # ========= Generate target actions during training ============
    def conditional_sample_with_DesiredA(self, 
        condition_data, condition_mask, action_negative, action_positive,
        local_cond=None, global_cond=None,
        # guidance_scale=30.0,
        guidance_scale=1.0,
        # reverse_start_t = 20,
        reverse_start_t = 12,
        generator=None,
        **kwargs
        ):

        self.sample_trajectories_list = []
        with torch.no_grad():
            model = self.model
            scheduler = self.noise_scheduler

            '''TODO check which is better, starting from Gaussian noise or a+'''
            # trajectory = torch.randn(
            #     size=condition_data.shape, 
            #     dtype=condition_data.dtype,
            #     device=condition_data.device,
            #     generator=generator)
            
            

            # gaussian_noise = torch.randn(
            #     size=condition_data.shape, 
            #     dtype=condition_data.dtype,
            #     device=condition_data.device,
            #     generator=generator)

            trajectory = action_positive # + 0.1 * gaussian_noise

            scheduler.set_timesteps(self.num_inference_steps)
            # mcmc_steps_num = 5
            # mcmc_steps = torch.zeros(mcmc_steps_num, dtype=scheduler.timesteps.dtype, device=scheduler.timesteps.device)
            # timesteps_ = torch.cat([scheduler.timesteps, mcmc_steps], dim=0)
            # for t in timesteps_:
            
            for t in scheduler.timesteps:

                if t > reverse_start_t:
                    continue
                
                prev_trajectory = trajectory
                # condition_false =  ~check_if_inside_desiredA(prev_trajectory, action_negative, action_positive, desiredA_type=self.desiredA_type)
                # num_true = torch.sum(condition_false)
                # print("t: ", t, " outside desiredA: ", num_true, " total: ", trajectory.shape)

                # # 1. apply conditioning
                # trajectory[condition_mask] = condition_data[condition_mask]

                # 2. predict model output
                model_output = model(trajectory, t.to(self.device), 
                    local_cond=local_cond, global_cond=global_cond.to(self.device))

                # # 3. compute guidance gradient
                # grad_log_f = obtain_target_distribution_derivative_desiredActionSpace(
                #     sampled_action =trajectory, action= action_negative,
                #     action_positive=action_positive, lambda_data= self.lambda_data, desiredA_type= self.desiredA_type
                # ).reshape_as(model_output)

                # # print("grad_log_f: ", grad_log_f)
                # # 4. apply guidance to model prediction
                # diffusion_scale = 1.0
                # model_output_guided = diffusion_scale * model_output - guidance_scale * grad_log_f

                # # No guiding 
                model_output_guided = model_output 

                # 5. compute previous step: x_t -> x_t-1
                trajectory = scheduler.step(
                    model_output_guided, t, trajectory, 
                    generator=generator,
                    **kwargs
                    ).prev_sample
                
                ### also save the traj before reflection
                traj_numpy = trajectory.detach().cpu().numpy().reshape(trajectory.shape[0], -1, trajectory.shape[-1])
                self.sample_trajectories_list.append(traj_numpy)

                ## add reflection here, if some of trajectory is outside the desired action space, apply projection 
                # trajectory_reflected = project_and_reflect_trajectory(action_positive, action_negative, trajectory, check_if_inside_desiredA, desiredA_type=self.desiredA_type)

                if self.horizon > 1 and self.large_desiredA is True:
                    action_positive_A = action_positive.reshape(action_positive.shape[0], 1, -1)
                    action_negative_A = action_negative.reshape(action_positive.shape[0], 1, -1)
                    trajectory_A = trajectory.reshape(action_positive.shape[0], 1, -1)
                    trajectory_reflected_A = project_and_reflect_trajectory(action_positive_A, action_negative_A, trajectory_A, 
                                                                            check_if_inside_desiredA, desiredA_type=self.desiredA_type,
                                                                            alpha=self.sphere_alpha, gamma=self.sphere_gamma, radius_ratio=self.radius_ratio)
                    trajectory_reflected = trajectory_reflected_A.reshape(action_positive.shape[0], action_positive.shape[1], -1)
                else:
                    trajectory_reflected = project_and_reflect_trajectory(action_positive, action_negative, trajectory, 
                                                                          check_if_inside_desiredA, desiredA_type=self.desiredA_type,
                                                                          alpha=self.sphere_alpha, gamma=self.sphere_gamma, radius_ratio=self.radius_ratio)

                if self.horizon > 1:
                    # for past actions, we donot need to do any sampling. Just set it to the data label
                    trajectory_reflected[:, 0, :] = action_positive[:, 0, :]
                # trajectory = torch.clamp(trajectory, -1, 1)
                # trajectory_reflected = iterative_refelection_cone(x0= (1.0-gamma) * action_negative + gamma *  action_positive,
                #                                    v=action_positive- action_negative,
                #                                    alpha=torch.tensor(alpha * (3.141592653589793 / 180.0)),
                #                                    xs=prev_trajectory,
                #                                    x_out=trajectory)
                
                
                # self.sample_trajectories_list.append(trajectory_reflected)  # used for debugging in 2d

                # Optional final clamp to ensure boundaries


                trajectory = torch.clamp(trajectory_reflected, -1, 1)

                # Save trajectory snapshot
                traj_numpy = trajectory.detach().cpu().numpy().reshape(trajectory.shape[0], -1, trajectory.shape[-1])
                self.sample_trajectories_list.append(traj_numpy)

            # Final conditioning enforcement
            trajectory[condition_mask] = condition_data[condition_mask]

            # Also save the final result
            traj_numpy = trajectory.detach().cpu().numpy().reshape(trajectory.shape[0], -1, trajectory.shape[-1])
            self.sample_trajectories_list.append(traj_numpy)
            return trajectory
        
    def conditional_sample_with_DesiredA_with_saving_traj(self, 
        condition_data, condition_mask, action_negative, action_positive,
        local_cond=None, global_cond=None,
        # guidance_scale=30.0,
        guidance_scale=1.0,
        # reverse_start_t = 20,
        reverse_start_t = 12,
        generator=None,
        **kwargs
        ):

        self.sample_trajectories_list = []
        with torch.no_grad():
            model = self.model
            scheduler = self.noise_scheduler

            '''TODO check which is better, starting from Gaussian noise or a+'''
            # trajectory = torch.randn(
            #     size=condition_data.shape, 
            #     dtype=condition_data.dtype,
            #     device=condition_data.device,
            #     generator=generator)
            
            

            # gaussian_noise = torch.randn(
            #     size=condition_data.shape, 
            #     dtype=condition_data.dtype,
            #     device=condition_data.device,
            #     generator=generator)

            trajectory = action_positive # + 0.1 * gaussian_noise

            scheduler.set_timesteps(self.num_inference_steps)
            
            for t in scheduler.timesteps:

                if t > reverse_start_t:
                    continue
                
                prev_trajectory = trajectory

                model_output = model(trajectory, t.to(self.device), 
                    local_cond=local_cond, global_cond=global_cond.to(self.device))

                # # No guiding 
                model_output_guided = model_output 

                # 5. compute previous step: x_t -> x_t-1
                trajectory = scheduler.step(
                    model_output_guided, t, trajectory, 
                    generator=generator,
                    **kwargs
                    ).prev_sample
                
                ## add reflection here, if some of trajectory is outside the desired action space, apply projection 
                if self.horizon > 1 and self.large_desiredA is True:
                    action_positive_A = action_positive.reshape(action_positive.shape[0], 1, -1)
                    action_negative_A = action_negative.reshape(action_positive.shape[0], 1, -1)
                    trajectory_A = trajectory.reshape(action_positive.shape[0], 1, -1)
                    trajectory_reflected_A = project_and_reflect_trajectory(action_positive_A, action_negative_A, trajectory_A, 
                                                                            check_if_inside_desiredA, desiredA_type=self.desiredA_type,
                                                                            alpha=self.sphere_alpha, gamma=self.sphere_gamma, radius_ratio=self.radius_ratio)
                    trajectory_reflected = trajectory_reflected_A.reshape(action_positive.shape[0], action_positive.shape[1], -1)
                else:
                    trajectory_reflected = project_and_reflect_trajectory(action_positive, action_negative, trajectory, 
                                                                          check_if_inside_desiredA, desiredA_type=self.desiredA_type,
                                                                          alpha=self.sphere_alpha, gamma=self.sphere_gamma, radius_ratio=self.radius_ratio)

                if self.horizon > 1:
                    # for past actions, we donot need to do any sampling. Just set it to the data label
                    trajectory_reflected[:, 0, :] = action_positive[:, 0, :]


                trajectory = torch.clamp(trajectory_reflected, -1, 1)

                # Save trajectory snapshot
                traj_numpy = trajectory.detach().cpu().numpy().reshape(trajectory.shape[0], -1, trajectory.shape[-1])
                self.sample_trajectories_list.append(traj_numpy)

            # Final conditioning enforcement
            trajectory[condition_mask] = condition_data[condition_mask]

            # Also save the final result
            traj_numpy = trajectory.detach().cpu().numpy().reshape(trajectory.shape[0], -1, trajectory.shape[-1])
            self.sample_trajectories_list.append(traj_numpy)
   
            return trajectory

    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        # scheduler = self.noise_scheduler
        scheduler = self.noise_scheduler_inference
        # print("condition data device: ", condition_data.device)
        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            
            # if self.horizon == 1 and not self.evaluation:
            #     if t > 20:  #TODO use DDIM to reduce the sampling time during inference
            #         continue
            
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            # step_start_time = time.time()
            model_output = model(trajectory, t.to(self.device), 
                local_cond=local_cond, global_cond=global_cond.to(self.device))
            # step_end_time = time.time()
            # one_step_duration = step_end_time - step_start_time 
            # formatted_duration = "{:.6f}".format(one_step_duration)
            # print("duration model: ", formatted_duration, "seconds")


            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory
    

    def conditional_sample_with_saving_trajs(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)

        scheduler.set_timesteps(self.num_inference_steps)

        # For saving trajectories at each step
        saved_trajectories = []

        for t in scheduler.timesteps:
            # Enforce conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # Model prediction
            model_output = model(trajectory, t.to(self.device), 
                local_cond=local_cond, global_cond=global_cond.to(self.device))

            # Diffusion step
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
            ).prev_sample

            # Save trajectory snapshot
            traj_numpy = trajectory.detach().cpu().numpy().reshape(trajectory.shape[0], -1, trajectory.shape[-1])
            saved_trajectories.append(traj_numpy)

        # Final conditioning enforcement
        trajectory[condition_mask] = condition_data[condition_mask]

        # Also save the final result
        traj_numpy = trajectory.detach().cpu().numpy().reshape(trajectory.shape[0], -1, trajectory.shape[-1])
        saved_trajectories.append(traj_numpy)

        return trajectory, saved_trajectories


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
            # import pdb
            # pdb.set_trace()
            B, _, Do = nobs.shape
            To = self.n_obs_steps

            T = self.horizon
            Da = self.action_dim

            # build input
            device = self.device
            # print("device: ", device)
            dtype = self.dtype

            # handle different ways of passing observation
            local_cond = None
            global_cond = nobs
            global_cond = nobs.reshape(nobs.shape[0], -1)    
            
            shape = (B, T, Da)
            # shape = (B, Da)
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
            # action_pred = self.normalizer['action'].unnormalize(naction_pred)
            action_pred = naction_pred
            action = action_pred
            if T == 1:
                numpy_action = naction_pred.detach().cpu().numpy().reshape(-1)
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

    def compute_loss(self, batch): # original BC loss
        self.model.train()
        self.model.training = True

        state_batch = [np.array(pair[0]) for pair in batch]  # state(t) sequence
        h_human_batch = [np.array(pair[1]) for pair in batch]  # last
        # next_state_batch = [np.array(pair[3]) for pair in batch]
        #print("h_human_batch: ",h_human_batch)
        batch_size = len(batch)
        # Convert the numpy array to a PyTorch tensor and reshape it
        obs = torch.tensor(np.reshape(state_batch, [batch_size,  self.dim_o]), dtype=torch.float32)
        action     = torch.tensor(np.reshape(h_human_batch, [batch_size, self.horizon, self.dim_a]), dtype=torch.float32)

        local_cond = None
        global_cond = None

        trajectory = action.to(self.device)

        global_cond = obs   # the observation, used to condition the action

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
        # print("loss: ", loss)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    ## SDP policy update, corresponding to Lines 2-14 of Algorithm 1 in the SDP paper.
    def compute_loss_Diffusion_Set_Supervised(self, batch, t = None, loss_source="intervention"):
        state_batch = [np.array(pair[0]) for pair in batch]
        h_human_batch = [np.array(pair[1]) for pair in batch]  # Preferred action
        action_negative_batch = [np.array(pair[2]) for pair in batch]  # Non-preferred action
        batch_size = len(batch)
        # state = torch.tensor(np.stack(state_batch), dtype=torch.float32, device=self.device)
        state = torch.tensor(np.reshape(state_batch, [batch_size,  self.dim_o]), dtype=torch.float32).to(self.device) # (batch, dim_o)
        

        action_preferred     = torch.tensor(np.reshape(h_human_batch, [batch_size, self.horizon, self.dim_a]), dtype=torch.float32).to(self.device)
        action_negative     = torch.tensor(np.reshape(action_negative_batch, [batch_size, self.horizon, self.dim_a]), dtype=torch.float32).to(self.device)
        
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

        condition_false =  ~check_if_inside_desiredA(sampled_action, action_negative_repeated, action_preferred_repeated, desiredA_type=self.desiredA_type, 
                                                     alpha=self.sphere_alpha, gamma=self.sphere_gamma, radius_ratio=self.radius_ratio)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps,
                                (sampled_action.shape[0],), device=self.device)
        noise_on_action_samples = torch.randn_like(sampled_action)
        noisy_action_samples = self.noise_scheduler.add_noise(sampled_action, noise_on_action_samples, timesteps)
        
        
        pred = self.model(noisy_action_samples, timesteps, 
            local_cond=None, global_cond=state_repeated.to(self.device)) 

        loss = (pred - noise_on_action_samples).pow(2).sum(dim=2)
        loss = loss.mean()
        action_loss = loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        # print("lr: ", self.lr_scheduler.get_last_lr()[0])
        current_lr = self.lr_scheduler.get_lr()[0]
        logger.debug('lr:  %s', self.lr_scheduler.get_lr())

        self._ensure_tensorboard_writer()
        loss_tag_prefix = f"Loss/{loss_source}"
        self.writer.add_scalar(f"{loss_tag_prefix}/action_loss", action_loss.detach(), self.global_step)
        self.writer.add_scalar(f"{loss_tag_prefix}/total_loss", loss.detach(), self.global_step)
        self.writer.add_scalar("Learning Rate", current_lr, self.global_step)
        self.global_step += 1

    def _ensure_tensorboard_writer(self):
        if not hasattr(self, "writer"):
            self.writer = SummaryWriter(log_dir=os.path.join(self.saved_dir, "logs"))
        if not hasattr(self, "global_step"):
            self.global_step = 0

    def TRAIN_Diffusion_with_Set_Supervised(self, action, h, observation, next_observation,  t, done):
        # h: corrective feedback!
        if np.any(h):  # if any element is not 0
            # 1. append  (o_t, a_t, h_t) to D
            self.latested_data_pair =[observation, h, action]
            self.buffer.add(self.latested_data_pair )  # state, a+, a-
            
            # 4. Update Human model with a minibatch sampled from buffer D
            if self.buffer.initialized():
                batch = self.buffer.sample(batch_size= int(self.buffer_sampling_size / 4))
                batch[-1] = self.latested_data_pair
                self.compute_loss_Diffusion_Set_Supervised(
                    batch,
                    loss_source="intervention",
                )

        # Train policy every k time steps from buffer
        elif self.buffer.initialized() and t % self.buffer_sampling_rate == 0:
            for i in range(1): 
                batch = self.buffer.sample(batch_size= int(self.buffer_sampling_size / 4 ))
                self.compute_loss_Diffusion_Set_Supervised(
                    batch,
                    loss_source="intervention",
                )

        if done:
            self.last_action = None

        if self.buffer.initialized() and (self.train_end_episode and done):
            self.model.train()
            self.model.training = True
            self.evaluation_last = False
            for i in range(self.number_training_iterations):
                if i % (self.number_training_iterations / 2) == 0:
                    logger.info("Progress Policy training: %i %%", i / self.number_training_iterations * 100)
                    logger.info('buffer size:  %s', self.buffer.length())

                for i in range(1): 
                    batch = self.buffer.sample(batch_size=self.buffer_sampling_size)
                    self.compute_loss_Diffusion_Set_Supervised(
                        batch,
                        loss_source="intervention",
                    )


    def collect_data_and_train(self, last_action, h, obs_proc, next_obs, t, done, agent_algorithm=None, agent_type=None, i_episode=None):
        """Unified entry point used by main_IIL.py."""
        return self.TRAIN_Diffusion_with_Set_Supervised(last_action, h, obs_proc, next_obs, t, done)

    # Backward-compatible alias: CLIC-Diffusion is the old name for Set_Supervised Diffusion.
    def TRAIN_Diffusion_withCLIC(self, *args, **kwargs):
        return self.TRAIN_Diffusion_with_Set_Supervised(*args, **kwargs)

    def compute_loss_Diffusion_CLIC(self, *args, **kwargs):
        return self.compute_loss_Diffusion_Set_Supervised(*args, **kwargs)


DiffusionUnetLowdimPolicy_CLIC = DiffusionUnetLowdimPolicy_Set_Supervised

__all__ = [
    "BaseLowdimPolicy",
    "DiffusionUnetLowdimPolicy_Set_Supervised",
    "DiffusionUnetLowdimPolicy_CLIC",
    "ModuleAttrMixin",
    "RunningStatsVec",
    "check_if_inside_desiredA",
    "collate_obs_dict",
    "obtain_target_distribution_derivative_desiredActionSpace",
    "project_and_reflect_trajectory",
    "sample_uniform_in_ball",
]

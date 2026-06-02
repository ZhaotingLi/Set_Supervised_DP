"""
diffusion_unet_hybrid_image_policy_with_flow_matching.py

This is your DiffusionUnetHybridImagePolicy with TWO additions:
1) Training loss switch via config_agent.loss_type:
   - "diffusion" -> original DP diffusion loss (unchanged)
   - "flow_matching"/"flow"/"fm" -> flow-matching loss (new)

2) Inference sampling switch in action():
   - diffusion -> original conditional_sample (DDIM/DP scheduler)
   - flow_matching -> NEW conditional_sample_flow_matching (ODE integration)

Everything else is kept in the same style as your DP code.

Important notes:
- Flow-matching loss implemented as conditional flow matching with linear interpolation:
    z ~ N(0,I), t~U(0,1), x_t=(1-t)z + t x, target v = (x - z)
  Model predicts v(x_t, t, cond).
- Flow sampling integrates ODE: dx/dt = v_theta(x,t,cond) from t=0 to 1
  using Euler steps with self.num_inference_steps steps.

- Because your ConditionalUnet1D expects an integer timestep (like diffusion),
  we map continuous t∈[0,1] -> integer timestep in [0, num_train_timesteps-1]
  using noise_scheduler.config.num_train_timesteps when available.
"""
import logging

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from agents.DP_model.common.normalizer import LinearNormalizer   # not used
from agents.DP_model.diffusion.mask_generator import LowdimMaskGenerator
from agents.DP_model.vision.multi_image_obs_encoder import MultiImageObsEncoder

import os
import numpy as np
from agents.replay_buffer_setup import build_replay_buffer_setup
import time
from torch.utils.tensorboard import SummaryWriter

from agents.DP_model.common.scheduler import CosineAnnealingWarmupRestarts
from agents.DP_model.common.lr_scheduler import get_scheduler

from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.obs_core as rmbn
import agents.DP_model.vision.crop_randomizer as dmvc
from agents.DP_model.common.pytorch_util import dict_apply, replace_submodules


from robomimic.config import config_factory
import robomimic.scripts.generate_paper_configs as gpc
from robomimic.scripts.generate_paper_configs import (
    modify_config_for_default_image_exp,
    modify_config_for_default_low_dim_exp,
    modify_config_for_dataset,
)

logger = logging.getLogger(__name__)

def get_robomimic_config(
        algo_name='bc_rnn',
        hdf5_type='low_dim',
        task_name='square',
        dataset_type='ph'
    ):
    base_dataset_dir = '/tmp/null'
    filter_key = None

    modifier_for_obs = modify_config_for_default_image_exp
    if hdf5_type in ["low_dim", "low_dim_sparse", "low_dim_dense"]:
        modifier_for_obs = modify_config_for_default_low_dim_exp

    algo_config_name = "bc" if algo_name == "bc_rnn" else algo_name
    config = config_factory(algo_name=algo_config_name)
    config = modifier_for_obs(config)
    config = modify_config_for_dataset(
        config=config,
        task_name=task_name,
        dataset_type=dataset_type,
        hdf5_type=hdf5_type,
        base_dataset_dir=base_dataset_dir,
        filter_key=filter_key,
    )
    algo_config_modifier = getattr(gpc, f'modify_{algo_name}_config_for_dataset')
    config = algo_config_modifier(
        config=config,
        task_name=task_name,
        dataset_type=dataset_type,
        hdf5_type=hdf5_type,
    )
    return config


class ModuleAttrMixin(nn.Module):
    def __init__(self):
        super().__init__()
        # Your snippet had nn.Parameter() with no args; that errors in vanilla torch.
        # If your local file actually has a working line, keep it.
        self._dummy_variable = nn.Parameter(torch.zeros(1), requires_grad=True)

    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


class BaseImagePolicy_original(ModuleAttrMixin):
    def action(self, obs_dict):
        raise NotImplementedError()

    def reset(self):
        pass

    def set_normalizer(self, normalizer: LinearNormalizer):
        raise NotImplementedError()


def collate_obs_dict(obs_list):
    if not obs_list:
        raise ValueError("The list of observation dictionaries is empty.")
    keys = list(obs_list[0].keys())
    collated = {key: [] for key in keys}
    for obs in obs_list:
        for key in keys:
            collated[key].append(obs[key])
    for key in keys:
        collated[key] = np.stack(collated[key])
    return collated


class DiffusionUnetHybridImagePolicy(BaseImagePolicy_original):
    def __init__(
        self,
        noise_scheduler: DDPMScheduler,
        noise_scheduler_inference,
        horizon,
        obs_dim,
        action_dim,
        shape_meta,
        saved_dir,
        load_dir,
        load_pretrained_dir,
        load_policy,
        number_training_iterations,
        buffer_min_size,
        buffer_max_size,
        buffer_sampling_size,
        policy_model_learning_rate,
        n_action_steps=1,
        n_obs_steps=2,
        num_inference_steps=None,
        obs_as_local_cond=False,
        obs_as_global_cond=True,
        pred_action_steps_only=False,
        oa_step_convention=False,
        crop_shape=(76, 76),
        obs_encoder_group_norm=True,
        eval_fixed_crop=True,
        use_ambient_loss=False,
        ambient_k=3,
        use_hdf5_dataset=False,
        diffusion_step_embed_dim=128,
        unet_down_dims=[512, 1024, 2048],
        use_AutoEncoder_loss=False,
        frozen_obs_encoder=False,
        config_agent=None,
        **kwargs
    ):
        super().__init__()
        assert not (obs_as_local_cond and obs_as_global_cond)

        self.device = torch.device("cuda:0")

        # NEW: store config_agent & loss_type
        self.config_agent = config_agent
        self.loss_type = getattr(config_agent, "loss_type", "diffusion") if config_agent is not None else "diffusion"
        # self.loss_type = 'flow_matching'

        # --- original DP init code (kept) ---
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)

            typ = attr.get('type', 'low_dim')
            if typ == 'rgb':
                obs_config['rgb'].append(key)
            elif typ == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {typ}")

        self.use_AutoEncoder_loss = use_AutoEncoder_loss
        if not use_AutoEncoder_loss:
            self.obs_encoder = MultiImageObsEncoder(
                shape_meta=shape_meta,
                resize_shape=None,
                crop_shape=crop_shape,
                random_crop=True,
                use_group_norm=True,
                share_rgb_model=False,
                imagenet_norm=False,
                use_spatial_softmax=True,
            ).to(self.device)
        else:
            from agents.DP_model.vision.multi_image_obs_encoder_with_decoder_no_transformer import MultiImageObsEncoderWithDecoder
            self.obs_encoder = MultiImageObsEncoderWithDecoder(
                shape_meta=shape_meta,
                resize_shape=None,
                crop_shape=crop_shape,
                random_crop=True,
                use_group_norm=True,
                share_rgb_model=False,
                imagenet_norm=False,
                use_spatial_softmax=False,
                use_global_bottleneck_for_policy=True,
                decode_from_global_z=True,
                add_coord_channels_to_seed=False,
                bottleneck_dim=256,
            ).to(self.device)

        obs_feature_dim = self.obs_encoder.output_shape()[0]
        self.obs_feature_dim = obs_feature_dim

        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        if horizon > 1:
            from agents.DP_model.diffusion.conditional_unet1d_original import ConditionalUnet1D
        else:
            from agents.DP_model.diffusion.conditional_unet1d import ConditionalUnet1D
        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=unet_down_dims,
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True
        )

        self.model = model.to(self.device)
        self.noise_scheduler = noise_scheduler
        self.noise_scheduler_inference = noise_scheduler_inference
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        self.policy_model_learning_rate = policy_model_learning_rate
        self.buffer_max_size = buffer_max_size
        self.buffer_min_size = buffer_min_size
        self.buffer_sampling_size = buffer_sampling_size

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
            selfplay_replay_buffer_type="hdf5_obs_action_buffer",
            create_replay_batch_loader=True,
            replay_batch_loader_kwargs={
                "collate_hdf5_intervention_samples": False,
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

        self.use_CLIC_algorithm = False
        self.dim_o = obs_dim
        self.dim_a = action_dim
        self.train_end_episode = True
        self.buffer_sampling_rate = 5
        self.traning_count = 0
        self.number_training_iterations = number_training_iterations

        self.saved_dir = saved_dir
        self.load_dir = load_dir
        self.load_pretrained_dir = load_pretrained_dir
        self.load_policy_flag = load_policy

        self.evaluation = False
        self.use_ambient_loss = use_ambient_loss
        self.ambient_k = ambient_k

        self.e = 0.2
        self.frozen_obs_encoder = frozen_obs_encoder

        self.optimizer = torch.optim.AdamW(
            self.model.parameters() if self.frozen_obs_encoder else self.parameters(),
            lr=self.policy_model_learning_rate,
            betas=(0.95, 0.999),
            eps=1.0e-8,
            weight_decay=1.0e-6
        )

        self.lr_scheduler = CosineAnnealingWarmupRestarts(
            self.optimizer,
            first_cycle_steps=296 * (500 + self.number_training_iterations),
            cycle_mult=1.0,
            max_lr=self.policy_model_learning_rate,
            min_lr=1e-5,
            warmup_steps=10,
            gamma=1.0,
        )

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        logger.info("Diffusion params: %e", sum(p.numel() for p in self.model.parameters()))
        logger.info("Vision params: %e", sum(p.numel() for p in self.obs_encoder.parameters()))

    
    

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


    # -------------------------
    # NEW: helpers
    # -------------------------
    def _use_flow_matching(self) -> bool:
        lt = str(self.loss_type).lower()
        return lt in ["flow", "flow_matching", "fm", "flowmatching"]

    def _get_num_train_timesteps(self) -> int:
        # Use diffusion scheduler config as a convenient "timestep scale"
        if self.noise_scheduler is not None and hasattr(self.noise_scheduler, "config"):
            return int(getattr(self.noise_scheduler.config, "num_train_timesteps", 1000))
        return 1000

    def _t01_to_timesteps(self, t01: torch.Tensor) -> torch.Tensor:
        """
        Map continuous t in [0,1] to the integer timestep space expected by ConditionalUnet1D.
        """
        n = self._get_num_train_timesteps()
        t_int = torch.clamp((t01 * (n - 1)).round().long(), 0, n - 1)
        return t_int

    # -------------------------
    # ORIGINAL diffusion sampler
    # -------------------------
    def conditional_sample(
            self,
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

        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]
            model_output = model(
                trajectory, t,
                local_cond=local_cond, global_cond=global_cond
            )
            trajectory = scheduler.step(
                model_output, t, trajectory,
                generator=generator,
                **kwargs
            ).prev_sample

        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    # -------------------------
    # NEW: Flow-matching sampler
    # -------------------------
    def conditional_sample_flow_matching(
            self,
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            **kwargs
        ):
        """
        ODE sampling for flow matching:
          dx/dt = v_theta(x, t, cond), integrate t: 0 -> 1.
        We keep the same signature as conditional_sample so action() can switch cleanly.

        Conditioning behavior (for parity with diffusion sampler):
        - At every step, enforce x[mask] = condition_data[mask]
        """
        model = self.model

        x = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator
        )

        # integrate with Euler
        n_steps = int(self.num_inference_steps)
        if n_steps < 2:
            n_steps = 2
        ts = torch.linspace(0.0, 1.0, n_steps, device=x.device, dtype=torch.float32)
        dt = ts[1] - ts[0]

        for i in range(n_steps - 1):
            # enforce conditioning (like diffusion sampler)
            x[condition_mask] = condition_data[condition_mask]

            t01 = ts[i].expand(x.shape[0])  # (B,)
            timesteps = self._t01_to_timesteps(t01)

            v = model(
                x, timesteps,
                local_cond=local_cond, global_cond=global_cond
            )
            x = x + dt * v

        # final conditioning
        x[condition_mask] = condition_data[condition_mask]
        return x

    # -------------------------
    # action(): select sampler
    # -------------------------
    def action(self, obs_dict):
        if self.evaluation:
            self.eval()
            self.model.training = False

        with torch.no_grad():
            obs_dict = dict_apply(
                obs_dict,
                lambda x: torch.from_numpy(x).to(device=self.device, dtype=torch.float32)
            )
            obs_dict = dict_apply(obs_dict, lambda x: x.unsqueeze(0))
            nobs = obs_dict

            value = next(iter(nobs.values()))
            B = value.shape[0]
            Do = self.obs_feature_dim
            To = self.n_obs_steps
            T = self.horizon
            Da = self.action_dim

            device = self.device
            dtype = self.dtype

            local_cond = None
            global_cond = None

            if self.obs_as_global_cond:
                this_nobs = dict_apply(
                    nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:])
                )
                nobs_features = self.obs_encoder(this_nobs)
                global_cond = nobs_features.reshape(B, -1)

                cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
                cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            else:
                this_nobs = dict_apply(
                    nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:])
                )
                nobs_features = self.obs_encoder(this_nobs).reshape(B, To, -1)

                cond_data = torch.zeros(size=(B, T, Da + Do), device=device, dtype=dtype)
                cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
                cond_data[:, :To, Da:] = nobs_features
                cond_mask[:, :To, Da:] = True

            # sampler switch
            if self._use_flow_matching():
                nsample = self.conditional_sample_flow_matching(
                    cond_data, cond_mask,
                    local_cond=local_cond,
                    global_cond=global_cond,
                    **self.kwargs
                )
            else:
                nsample = self.conditional_sample(
                    cond_data, cond_mask,
                    local_cond=local_cond,
                    global_cond=global_cond,
                    **self.kwargs
                )

            naction_pred = nsample[..., :Da]

            if T == 1:
                numpy_action = naction_pred.detach().cpu().numpy().reshape(1, -1)
            else:
                start = To - 1
                action = naction_pred[:, start:]
                numpy_action = action.detach().cpu().numpy().reshape(T - start, -1)

            numpy_action = np.clip(numpy_action, -1, 1)
            return numpy_action

    # -------------------------
    # NEW: Flow Matching Loss
    # -------------------------
    def compute_loss_flow_matching(self, batch, loss_source="intervention"):
        auxiliary_loss = None
        auxiliary_loss_selfplay = None
        state_batch = [pair[0] for pair in batch]
        h_human_batch = [np.array(pair[1]) for pair in batch]

        batch_size = len(batch)

        nobs = collate_obs_dict(state_batch)
        trajectory = torch.tensor(
            np.reshape(h_human_batch, [batch_size, self.horizon, self.dim_a]),
            dtype=torch.float32
        ).to(self.device)

        nobs = dict_apply(
            nobs,
            lambda x: torch.from_numpy(x).to(device=self.device, dtype=torch.float32)
        )

        local_cond = None
        global_cond = None

        if self.obs_as_global_cond:
            this_nobs = dict_apply(
                nobs, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:])
            )
            nobs_features = self.obs_encoder(this_nobs)
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            nobs_features = nobs_features.reshape(batch_size, self.horizon, -1)
            trajectory = torch.cat([trajectory, nobs_features], dim=-1)

        z = torch.randn_like(trajectory)
        t01 = torch.rand((batch_size,), device=self.device, dtype=torch.float32)
        t_b = t01.view(batch_size, 1, 1)

        x_t = (1.0 - t_b) * z + t_b * trajectory
        v_target = trajectory - z

        timesteps = self._t01_to_timesteps(t01)

        v_pred = self.model(
            x_t, timesteps,
            local_cond=local_cond,
            global_cond=global_cond
        )

        loss = F.mse_loss(v_pred, v_target, reduction='none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean').mean()
        flow_matching_loss = loss

        if self.use_AutoEncoder_loss and not self.frozen_obs_encoder:
            total_loss_ae, per_key = self.obs_encoder.compute_autoencoder_loss(
                this_nobs, reduction='mean', return_per_key=True
            )
            auxiliary_loss = total_loss_ae

            batch_selfplay = self.buffer_selfplay.sample(batch_size=self.buffer_sampling_size)
            state_batch_selfplay = [pair[0] for pair in batch_selfplay]
            nobs_selfplay = collate_obs_dict(state_batch_selfplay)
            nobs_selfplay = dict_apply(
                nobs_selfplay,
                lambda x: torch.from_numpy(x).to(device=self.device, dtype=torch.float32)
            )
            this_nobs_selfplay = dict_apply(
                nobs_selfplay,
                lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:])
            )
            total_loss_ae_selfplay, per_key = self.obs_encoder.compute_autoencoder_loss(
                this_nobs_selfplay, reduction='mean', return_per_key=True
            )
            auxiliary_loss_selfplay = total_loss_ae_selfplay

            loss = 0.2 * total_loss_ae + 0.2 * total_loss_ae_selfplay + loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        current_lr = self.lr_scheduler.get_lr()[0]

        self._ensure_tensorboard_writer()
        loss_tag_prefix = f"Loss/{loss_source}"
        self.writer.add_scalar(f"{loss_tag_prefix}/flow_matching_loss", flow_matching_loss.detach(), self.global_step)
        if auxiliary_loss is not None:
            self.writer.add_scalar(f"{loss_tag_prefix}/auxiliary_loss", auxiliary_loss.detach(), self.global_step)
        if auxiliary_loss_selfplay is not None:
            self.writer.add_scalar(f"{loss_tag_prefix}/auxiliary_loss_selfplay", auxiliary_loss_selfplay.detach(), self.global_step)
        self.writer.add_scalar(f"{loss_tag_prefix}/total_loss", loss.detach(), self.global_step)
        self.writer.add_scalar("Learning Rate", current_lr, self.global_step)
        self.global_step += 1

    # -------------------------
    # compute_loss(): switch
    # -------------------------
    def compute_loss(self, batch, loss_source="intervention"):
        if self._use_flow_matching():
            return self.compute_loss_flow_matching(
                batch,
                loss_source=loss_source,
            )

        # ---- ORIGINAL DP diffusion loss (kept) ----
        auxiliary_loss = None
        auxiliary_loss_selfplay = None
        state_batch = [pair[0] for pair in batch]
        h_human_batch = [np.array(pair[1]) for pair in batch]

        batch_size = len(batch)
        nobs = collate_obs_dict(state_batch)
        action = torch.tensor(np.reshape(h_human_batch, [batch_size, self.horizon, self.dim_a]), dtype=torch.float32)
        local_cond = None
        global_cond = None

        nobs = dict_apply(
            nobs,
            lambda x: torch.from_numpy(x).to(device=self.device, dtype=torch.float32)
        )
        trajectory = action.to(self.device)

        if self.obs_as_global_cond:
            this_nobs = dict_apply(
                nobs,
                lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:])
            )
            nobs_features = self.obs_encoder(this_nobs)
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            nobs_features = nobs_features.reshape(batch_size, self.horizon, -1)
            cond_data = torch.cat([trajectory, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        condition_mask = self.mask_generator(trajectory.shape)

        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (bsz,), device=trajectory.device
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)

        pred = self.model(
            noisy_trajectory, timesteps,
            local_cond=local_cond, global_cond=global_cond
        )

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        action_loss = loss

        if self.use_AutoEncoder_loss and not self.frozen_obs_encoder:
            total_loss_ae, per_key = self.obs_encoder.compute_autoencoder_loss(
                this_nobs, reduction='mean', return_per_key=True
            )
            auxiliary_loss = total_loss_ae

            batch_selfplay = self.buffer_selfplay.sample(batch_size=self.buffer_sampling_size)
            state_batch_selfplay = [pair[0] for pair in batch_selfplay]
            nobs_selfplay = collate_obs_dict(state_batch_selfplay)
            nobs_selfplay = dict_apply(
                nobs_selfplay,
                lambda x: torch.from_numpy(x).to(device=self.device, dtype=torch.float32)
            )
            this_nobs_selfplay = dict_apply(
                nobs_selfplay,
                lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:])
            )

            total_loss_ae_selfplay, per_key = self.obs_encoder.compute_autoencoder_loss(
                this_nobs_selfplay, reduction='mean', return_per_key=True
            )
            auxiliary_loss_selfplay = total_loss_ae_selfplay

            logger.debug('DP loss:  %s  total_loss_ae:  %s  total_loss_ae_selfplay:  %s', loss, total_loss_ae, total_loss_ae_selfplay)
            loss = 0.2 * total_loss_ae + 0.2 * total_loss_ae_selfplay + loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        current_lr = self.lr_scheduler.get_lr()[0]

        self._ensure_tensorboard_writer()
        loss_tag_prefix = f"Loss/{loss_source}"
        self.writer.add_scalar(f"{loss_tag_prefix}/action_loss", action_loss.detach(), self.global_step)
        if auxiliary_loss is not None:
            self.writer.add_scalar(f"{loss_tag_prefix}/auxiliary_loss", auxiliary_loss.detach(), self.global_step)
        if auxiliary_loss_selfplay is not None:
            self.writer.add_scalar(f"{loss_tag_prefix}/auxiliary_loss_selfplay", auxiliary_loss_selfplay.detach(), self.global_step)
        self.writer.add_scalar(f"{loss_tag_prefix}/total_loss", loss.detach(), self.global_step)
        self.writer.add_scalar("Learning Rate", current_lr, self.global_step)
        self.global_step += 1


    def compute_loss_AmbientDiffusion(self, batch, loss_source="intervention"):
        state_batch = [pair[0] for pair in batch]  # state(t) sequence

        h_human_batch = [np.array(pair[1]) for pair in batch]  # last
        robot_action_batch = [np.array(pair[2]) for pair in batch]
        batch_size = len(batch)
        nobs = collate_obs_dict(state_batch)
        action     = torch.tensor(np.reshape(h_human_batch, [batch_size, self.horizon, self.dim_a]), dtype=torch.float32)
    
        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        # trajectory = nactions

        nobs = dict_apply(nobs, 
                    lambda x: torch.from_numpy(x).to(
                        device=self.device,  dtype=torch.float32))
        # action = action.unsqueeze(1)
        trajectory = action.to(self.device)
        
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
                
            nobs_features = self.obs_encoder(this_nobs) # torch.Size([2 * batch_size, obs_feature_dim])
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1) # torch.Size([batch_size, 2 * obs_feature_dim])
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size,  self.horizon, -1)
            cond_data = torch.cat([trajectory, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Attention: in the main, we change the input of action to optimal teacher action. 
        optimal_action     = torch.tensor(np.reshape(robot_action_batch, [batch_size, self.horizon, self.dim_a]), dtype=torch.float32)

        diff = torch.abs(action - optimal_action)               # [batch, horizon, dim_a
        # Add a small random perturbation to `diff` if it's all zeros
        if diff.sum() == 0:
            diff = diff + torch.randn_like(diff) * 1e-6  # Adding a tiny random noise
        ## caculate corruption matrix A (batch, dim_a, dim_a), diagonal matrix
        # For partial feedback, (I- A) * action = (I- A) * robot_action

        # k = 3  # or whatever number of dimensions you want to “keep”
        k = self.ambient_k # PushT
        _, idx = torch.topk(-diff, k, dim=-1)               # idx: [batch, k]
        A_diag = torch.zeros_like(diff)                    # [batch, dim]
        A_diag.scatter_(dim=-1, index=idx, src=torch.ones_like(idx, dtype=torch.float))                    # set the k smallest‐diff dims to 1                    # [batch, dim, dim]
        mask_corruption = A_diag.to(self.device)  # [batch, 1, dim]
        # import pdb; pdb.set_trace()
      
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

        loss = F.mse_loss(pred, target, reduction='none')  # (batch, T, dim_a)
        # import pdb; pdb.set_trace()
        loss = loss * mask_corruption.type(loss.dtype) 
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        ambient_loss = loss
        logger.debug('loss:  %s', loss)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        current_lr = self.optimizer.param_groups[0]["lr"]

        self._ensure_tensorboard_writer()
        loss_tag_prefix = f"Loss/{loss_source}"
        self.writer.add_scalar(f"{loss_tag_prefix}/ambient_loss", ambient_loss.detach(), self.global_step)
        self.writer.add_scalar(f"{loss_tag_prefix}/total_loss", loss.detach(), self.global_step)
        self.writer.add_scalar("Learning Rate", current_lr, self.global_step)
        self.global_step += 1

    def _ensure_tensorboard_writer(self):
        if not hasattr(self, "writer"):
            self.writer = SummaryWriter(log_dir=os.path.join(self.saved_dir, "logs"))
        if not hasattr(self, "global_step"):
            self.global_step = 0

    def compute_difference_to_action_labels(self, batch):
        self.eval()
        self.model.training = False
        state_batch = [pair[0] for pair in batch]  # state(t) sequence

        h_human_batch = [np.array(pair[1]) for pair in batch]  # last
        batch_size = len(batch)
        nobs = collate_obs_dict(state_batch)

        action     = np.reshape(h_human_batch, [batch_size, self.horizon, self.dim_a])
        nobs = collate_obs_dict(state_batch)

        nobs = dict_apply(nobs, 
                    lambda x: torch.from_numpy(x).to(
                        device=self.device ,  dtype=torch.float32))
        
        ''' v2: sampling noise from both positive and negative action'''
        this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        state = nobs_features.reshape(batch_size, -1)   # ju
        global_cond = state

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

            nsample = self.conditional_sample(
                cond_data, 
                cond_mask,
                local_cond=local_cond,
                global_cond=global_cond,
                **self.kwargs)

            naction_pred = nsample[..., :Da]
            actions = torch.clamp(naction_pred, -1, 1).cpu().numpy() # [batch * sampled_size, action_horizon, dim_a]

        logger.debug('actions shape:  %s', actions.shape)
        loss = np.linalg.norm(actions[:, :8]- action[:, :8])
        logger.debug('loss:  %s', loss)
        return loss

    def collect_data_and_train(self, last_action, h, obs_proc, next_obs, t, done, agent_algorithm=None, agent_type=None, i_episode=None):
        return self.TRAIN_Policy_with_Behavior_Cloning_Objective(last_action, t, done, i_episode, h, obs_proc)

    def _sample_intervention_batch(self, batch_size):
        batch, _ = self.replay_batch_loader.sample_intervention_batch(
            batch_size=batch_size
        )
        return batch

    def TRAIN_Policy_with_Behavior_Cloning_Objective(self, action, t, done, i_episode, h, observation):
        if np.any(h):
            self.buffer.add([observation, h, action])
            self.latested_data_pair = [observation, h, action]

            if self.buffer.initialized():
                self.train()
                self.model.training = True
                batch = self._sample_intervention_batch(
                    batch_size=int(self.buffer_sampling_size)
                )
                batch[-1] = self.latested_data_pair
                if self.use_ambient_loss:
                    self.compute_loss_AmbientDiffusion(
                        batch,
                        loss_source="intervention",
                    )
                else:
                    self.compute_loss(
                        batch,
                        loss_source="intervention",
                    )

        elif self.buffer.initialized() and t % self.buffer_sampling_rate == 0 or (self.buffer.initialized() and self.train_end_episode and done):
            batch = self._sample_intervention_batch(
                batch_size=int(self.buffer_sampling_size)
            )
            if self.use_ambient_loss:
                self.compute_loss_AmbientDiffusion(
                    batch,
                    loss_source="intervention",
                )
            else:
                self.compute_loss(
                    batch,
                    loss_source="intervention",
                )

        if not self.buffer.initialized():
            self.traning_count = 0

        if self.buffer.initialized() and ((self.train_end_episode and done)):
            self.train()
            self.model.training = True
            for i in range(self.number_training_iterations):
                if i % (self.number_training_iterations / 20) == 0:
                    logger.info('number_training_iterations:  %s', self.number_training_iterations)
                    logger.debug(f"train policy (loss_type={self.loss_type})")
                    logger.info("Progress Policy training: %i %%", i / self.number_training_iterations * 100)
                    logger.debug('buffer size:  %s', self.buffer.length())

                batch = self._sample_intervention_batch(
                    batch_size=self.buffer_sampling_size
                )
                if self.use_ambient_loss:
                    self.compute_loss_AmbientDiffusion(
                        batch,
                        loss_source="intervention",
                    )
                else:
                    self.compute_loss(
                        batch,
                        loss_source="intervention",
                    )

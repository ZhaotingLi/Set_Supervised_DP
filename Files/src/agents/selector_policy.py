# from agents.ibc import ibc
# from agents.CLIC_Ceiling import CLIC_Ceiling
# from agents.Implicit_CLIC_Ceiling import Implicit_CLIC_Ceiling

# from agents.diffusion_policy import DiffusionUnetLowdimPolicy
# # from agents.diffusion_policy_image import DiffusionUnetImagePolicy
# from agents.Set_Supervised_diffusion_policy import DiffusionUnetLowdimPolicy_Set_Supervised
# from agents.Set_Supervised_diffusion_policy_image import DiffusionUnetImagePolicy_Set_Supervised


import torch 
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
"""
Functions that selects the agent
"""


from omegaconf import DictConfig

def agent_selector(agent_type: str, config_agent: DictConfig):
    # CLIC family
    algorithm_type = config_agent.algorithm
    if agent_type == 'CLIC':
        if config_agent.use_image:
            from agents.CLIC_torch_image import CLIC
        else:
            from agents.CLIC_torch import CLIC

        return CLIC(
            shape_meta=config_agent.shape_meta,
            dim_a=config_agent.dim_a,dim_o=config_agent.dim_o,
            action_upper_limits=config_agent.action_upper_limits,
            action_lower_limits=config_agent.action_lower_limits,
            buffer_min_size=config_agent.buffer_min_size,
            buffer_max_size=config_agent.buffer_max_size,
            buffer_sampling_rate=config_agent.buffer_sampling_rate,
            buffer_sampling_size=config_agent.buffer_sampling_size,
            train_end_episode=config_agent.train_end_episode,
            policy_model_learning_rate=config_agent.policy_model_learning_rate,
            e_matrix=config_agent.e,
            loss_weight_inverse_e=config_agent.loss_weight_inverse_e,
            sphere_alpha=config_agent.sphere_alpha,
            sphere_gamma=config_agent.sphere_gamma,
            radius_ratio = config_agent.radius_ratio,
            saved_dir=config_agent.saved_dir,
            load_dir=config_agent.load_dir,
            load_policy=config_agent.load_policy,
            number_training_iterations=config_agent.number_training_iterations,
            obs_encoder_crop_shape = config_agent.image_crop_shape,
            desiredA_type=config_agent.desiredA_type,
            softmax_temperature_measurement=config_agent.softmax_temperature_measurement,
            softmax_temperature_policy=config_agent.softmax_temperature_policy,
            prob_weight=config_agent.prob_weight,
            nn_hidden_dim = config_agent.nn_hidden_dim,
            grad_pen_weight = config_agent.grad_pen_weight,
            config_agent = config_agent,
        )

    # Implicit CLIC Ceiling
    elif agent_type == 'CLIC_Ceiling_Implicit':
        from agents.CLIC_torch import Implicit_CLIC_Ceiling
        return Implicit_CLIC_Ceiling(
            dim_a=config_agent.dim_a,
            dim_o=config_agent.dim_o,
            action_upper_limits=config_agent.action_upper_limits,
            action_lower_limits=config_agent.action_lower_limits,
            buffer_min_size=config_agent.buffer_min_size,
            buffer_sampling_rate=config_agent.buffer_sampling_rate,
            buffer_sampling_size=config_agent.buffer_sampling_size,
            train_end_episode=config_agent.train_end_episode,
            policy_model_learning_rate=config_agent.policy_model_learning_rate,
            human_model_learning_rate=config_agent.human_model_learning_rate,
            e_matrix=config_agent.e,
            loss_weight_inverse_e=config_agent.loss_weight_inverse_e,
            sphere_alpha=config_agent.sphere_alpha,
            sphere_gamma=config_agent.sphere_gamma,
            action_limit=config_agent.action_limit,
            buffer_max_size=config_agent.buffer_max_size,
            saved_dir=config_agent.saved_dir,
            load_dir=config_agent.load_dir,
            load_policy=config_agent.load_policy,
            number_training_iterations=config_agent.number_training_iterations
        )

    # CLIC Ceiling
    elif agent_type == 'CLIC_Ceiling':
        from agents.CLIC_torch import CLIC_Ceiling
        return CLIC_Ceiling(
            dim_a=config_agent.dim_a,
            dim_o=config_agent.dim_o,
            action_upper_limits=config_agent.action_upper_limits,
            action_lower_limits=config_agent.action_lower_limits,
            buffer_min_size=config_agent.buffer_min_size,
            buffer_sampling_rate=config_agent.buffer_sampling_rate,
            buffer_sampling_size=config_agent.buffer_sampling_size,
            train_end_episode=config_agent.train_end_episode,
            policy_model_learning_rate=config_agent.policy_model_learning_rate,
            human_model_learning_rate=config_agent.human_model_learning_rate,
            e_matrix=config_agent.e,
            loss_weight_inverse_e=config_agent.loss_weight_inverse_e,
            sphere_alpha=config_agent.sphere_alpha,
            sphere_gamma=config_agent.sphere_gamma,
            action_limit=config_agent.action_limit,
            buffer_max_size=config_agent.buffer_max_size,
            saved_dir=config_agent.saved_dir,
            load_dir=config_agent.load_dir,
            load_policy=config_agent.load_policy,
            number_training_iterations=config_agent.number_training_iterations
        )

    # implicit IIL
    elif agent_type == 'Implicit_BC':
        # from agents.some_module import ibc
        if config_agent.use_image:
            from agents.IBC_torch_image import IBC_Image
            return IBC_Image(
                horizon = config_agent.Ta,
                shape_meta=config_agent.shape_meta,
                dim_a=config_agent.dim_a,dim_o=config_agent.dim_o,
                action_upper_limits=config_agent.action_upper_limits,
                action_lower_limits=config_agent.action_lower_limits,
                buffer_min_size=config_agent.buffer_min_size,
                buffer_max_size=config_agent.buffer_max_size,
                buffer_sampling_rate=config_agent.buffer_sampling_rate,
                buffer_sampling_size=config_agent.buffer_sampling_size,
                train_end_episode=config_agent.train_end_episode,
                policy_model_learning_rate=config_agent.policy_model_learning_rate,
                saved_dir=config_agent.saved_dir,
                sample_action_number = config_agent.sample_action_number,
                load_dir=config_agent.load_dir,
                load_policy=config_agent.load_policy,
                number_training_iterations=config_agent.number_training_iterations,
                obs_encoder_crop_shape = config_agent.image_crop_shape,
                nn_hidden_dim = config_agent.nn_hidden_dim,
                mcmc_iteration =  config_agent.mcmc_iteration,
            )
        else:
            return ibc(dim_a=config_agent.dim_a,
                         dim_o=config_agent.dim_o,
                         action_upper_limits=config_agent.action_upper_limits,
                         action_lower_limits=config_agent.action_lower_limits,
                         buffer_min_size=config_agent.buffer_min_size,
                         buffer_max_size=config_agent.buffer_max_size,
                         buffer_sampling_rate=config_agent.buffer_sampling_rate,
                        buffer_sampling_size=config_agent.buffer_sampling_size,
                         number_training_iterations = config_agent.number_training_iterations,
                         train_end_episode=config_agent.train_end_episode,
                         policy_model_learning_rate=config_agent.policy_model_learning_rate,
                        saved_dir=config_agent.saved_dir,
                         load_dir=config_agent.load_dir,
                        load_policy=config_agent.load_policy,)

    # Diffusion-based policies
    elif agent_type.startswith('Diffusion') or agent_type == 'Set_Supervised_Diffusion':
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        from diffusers.schedulers.scheduling_ddim import DDIMScheduler

        noise_scheduler_DDIM = DDIMScheduler(
            num_train_timesteps=config_agent.DDPM_num_train_timesteps,  # keep same T as training
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule='squaredcos_cap_v2',  # keep the schedule consistent
            clip_sample=True,
            prediction_type='epsilon',          # must match training
            set_alpha_to_one=True               # (diffusers default; fine to keep)
        )

        noise_scheduler_DDPM = DDPMScheduler(
            num_train_timesteps=config_agent.DDPM_num_train_timesteps, beta_start=0.0001, beta_end=0.02,
            beta_schedule='squaredcos_cap_v2', variance_type='fixed_small',
            clip_sample=True, prediction_type='epsilon'
        )
        
        if agent_type == 'Diffusion':
            from agents.diffusion_policy_image_original import DiffusionUnetHybridImagePolicy
            from agents.diffusion_policy import DiffusionUnetLowdimPolicy
            from agents.DP_model.diffusion.conditional_unet1d import ConditionalUnet1D, HumanFunctionModel
            if not config_agent.use_image:
                return DiffusionUnetLowdimPolicy(
                    # model=HumanFunctionModel(dim_o=config_agent.dim_o, dim_a=config_agent.dim_a),
                    noise_scheduler=noise_scheduler_DDPM,
                    noise_scheduler_inference = noise_scheduler_DDIM if config_agent.use_DDIM_during_inference else noise_scheduler_DDPM,
                    obs_dim=config_agent.dim_o,
                    action_dim=config_agent.dim_a,
                    horizon=config_agent.Ta,
                    saved_dir=config_agent.saved_dir,
                    load_dir=config_agent.load_dir,
                    load_pretrained_dir=config_agent.load_pretrained_dir,
                    load_policy=config_agent.load_policy,
                    number_training_iterations=config_agent.number_training_iterations,
                    buffer_min_size=config_agent.buffer_min_size,
                    buffer_max_size=config_agent.buffer_max_size,
                    buffer_sampling_size=config_agent.buffer_sampling_size,
                    policy_model_learning_rate=config_agent.policy_model_learning_rate,
                    use_ambient_loss = config_agent.use_ambient_loss,
                    ambient_k = config_agent.ambient_k, 
                    num_inference_steps=config_agent.num_inference_steps,
                    diffusion_step_embed_dim=config_agent.diffusion_step_embed_dim, unet_down_dims = config_agent.unet_down_dims,
                )
            else:
                return DiffusionUnetHybridImagePolicy(
                    noise_scheduler=noise_scheduler_DDPM,
                    noise_scheduler_inference = noise_scheduler_DDIM if config_agent.use_DDIM_during_inference else noise_scheduler_DDPM,
                    obs_dim=config_agent.dim_o,
                    action_dim=config_agent.dim_a, 
                    shape_meta=config_agent.shape_meta,
                    crop_shape= config_agent.image_crop_shape,
                    horizon=config_agent.Ta,
                    saved_dir=config_agent.saved_dir,
                    load_dir=config_agent.load_dir,
                    load_pretrained_dir=config_agent.load_pretrained_dir,
                    load_policy=config_agent.load_policy,
                    number_training_iterations=config_agent.number_training_iterations,
                    buffer_min_size=config_agent.buffer_min_size,
                    buffer_max_size=config_agent.buffer_max_size,
                    buffer_sampling_size=config_agent.buffer_sampling_size,
                    policy_model_learning_rate=config_agent.policy_model_learning_rate,
                    use_ambient_loss = config_agent.use_ambient_loss,
                    ambient_k = config_agent.ambient_k,
                    use_hdf5_dataset = getattr(config_agent, "use_hdf5_dataset", False),
                    num_inference_steps=config_agent.num_inference_steps,
                    diffusion_step_embed_dim=config_agent.diffusion_step_embed_dim, unet_down_dims = config_agent.unet_down_dims,
                    frozen_obs_encoder = config_agent.frozen_obs_encoder, use_AutoEncoder_loss = config_agent.use_AutoEncoder_loss,
                    config_agent = config_agent,
                )

        elif agent_type == 'Set_Supervised_Diffusion' and algorithm_type in ['Set_Supervised_Diffusion_intervention', 'Set_Supervised_Diffusion_relative']:
            from agents.DP_model.diffusion.conditional_unet1d import ConditionalUnet1D, HumanFunctionModel
            from agents.Set_Supervised_diffusion_policy_image import DiffusionUnetImagePolicy_Set_Supervised
            from agents.Set_Supervised_diffusion_policy import DiffusionUnetLowdimPolicy_Set_Supervised
            use_image = config_agent.use_image
            if use_image:
                return DiffusionUnetImagePolicy_Set_Supervised(
                    noise_scheduler=noise_scheduler_DDPM,
                    noise_scheduler_inference = noise_scheduler_DDIM if config_agent.use_DDIM_during_inference else noise_scheduler_DDPM,
                    obs_dim=config_agent.dim_o,
                    action_dim=config_agent.dim_a,
                    shape_meta=config_agent.shape_meta,
                    obs_encoder_crop_shape=config_agent.image_crop_shape,
                    horizon=config_agent.Ta,
                    desiredA_type = config_agent.desiredA_type,
                    large_desiredA = config_agent.large_desiredA,
                    sphere_alpha=config_agent.sphere_alpha,
                    sphere_gamma = config_agent.sphere_gamma,
                    radius_ratio = config_agent.radius_ratio,
                    sample_action_number = config_agent.sample_action_number,
                    sample_with_desiredA_reverse_start_t = config_agent.sample_with_desiredA_reverse_start_t,
                    saved_dir=config_agent.saved_dir,
                    load_dir=config_agent.load_dir,
                    load_pretrained_dir=config_agent.load_pretrained_dir,
                    load_policy=config_agent.load_policy,
                    number_training_iterations=config_agent.number_training_iterations,
                    buffer_min_size=config_agent.buffer_min_size,
                    buffer_sampling_rate=config_agent.buffer_sampling_rate,
                    buffer_sampling_size=config_agent.buffer_sampling_size,
                    policy_model_learning_rate=config_agent.policy_model_learning_rate,
                    e_matrix=config_agent.e,
                    loss_weight_inverse_e=config_agent.loss_weight_inverse_e,
                    buffer_max_size=config_agent.buffer_max_size,
                    no_negative_action = config_agent.no_negative_action, scale_no_negative_action= config_agent.scale_no_negative_action,
                    use_hdf5_dataset = getattr(config_agent, "use_hdf5_dataset", False),
                    num_inference_steps=config_agent.num_inference_steps,
                    use_AutoEncoder_loss = config_agent.use_AutoEncoder_loss,
                    diffusion_step_embed_dim=config_agent.diffusion_step_embed_dim, unet_down_dims = config_agent.unet_down_dims,
                    frozen_obs_encoder = config_agent.frozen_obs_encoder, 
                    config_agent = config_agent,
                )
            else:
                return DiffusionUnetLowdimPolicy_Set_Supervised(
                    noise_scheduler=noise_scheduler_DDPM,
                    noise_scheduler_inference = noise_scheduler_DDIM if config_agent.use_DDIM_during_inference else noise_scheduler_DDPM,
                    obs_dim=config_agent.dim_o,
                    action_dim=config_agent.dim_a,
                    shape_meta=config_agent.shape_meta,
                    horizon=config_agent.Ta,
                    desiredA_type = config_agent.desiredA_type,
                    large_desiredA = config_agent.large_desiredA,
                    sphere_alpha=config_agent.sphere_alpha,
                    sphere_gamma = config_agent.sphere_gamma,
                    radius_ratio = config_agent.radius_ratio,
                    sample_action_number = config_agent.sample_action_number,
                    sample_with_desiredA_reverse_start_t = config_agent.sample_with_desiredA_reverse_start_t,
                    saved_dir=config_agent.saved_dir,
                    load_dir=config_agent.load_dir,
                    load_pretrained_dir=config_agent.load_pretrained_dir,
                    load_policy=config_agent.load_policy,
                    number_training_iterations=config_agent.number_training_iterations,
                    buffer_min_size=config_agent.buffer_min_size,
                    buffer_sampling_rate=config_agent.buffer_sampling_rate,
                    buffer_sampling_size=config_agent.buffer_sampling_size,
                    policy_model_learning_rate=config_agent.policy_model_learning_rate,
                    e_matrix=config_agent.e,
                    loss_weight_inverse_e=config_agent.loss_weight_inverse_e,
                    buffer_max_size=config_agent.buffer_max_size,
                    no_negative_action = config_agent.no_negative_action,
                    num_inference_steps=config_agent.num_inference_steps,
                    diffusion_step_embed_dim=config_agent.diffusion_step_embed_dim, unet_down_dims = config_agent.unet_down_dims,
                    config_agent = config_agent,
                )

        elif agent_type == 'Set_Supervised_Diffusion' and algorithm_type == 'Diffusion_DPO':
            from agents.DPO_diffusion_policy_image import DiffusionUnetImagePolicy_DPO
            from agents.DPO_diffusion_policy_lowdim import DiffusionUnetLowdimPolicy_DPO
            use_image = config_agent.use_image
            if use_image:
                
                return DiffusionUnetImagePolicy_DPO(
                    noise_scheduler=noise_scheduler_DDPM,
                    noise_scheduler_inference = noise_scheduler_DDIM if config_agent.use_DDIM_during_inference else noise_scheduler_DDPM,
                    obs_dim=config_agent.dim_o,
                    action_dim=config_agent.dim_a,
                    shape_meta=config_agent.shape_meta,
                    obs_encoder_crop_shape=config_agent.image_crop_shape,
                    horizon=config_agent.Ta,
                    desiredA_type = config_agent.desiredA_type,
                    large_desiredA = config_agent.large_desiredA,
                    sphere_alpha=config_agent.sphere_alpha,
                    sphere_gamma = config_agent.sphere_gamma,
                    radius_ratio = config_agent.radius_ratio,
                    sample_action_number = config_agent.sample_action_number,
                    sample_with_desiredA_reverse_start_t = config_agent.sample_with_desiredA_reverse_start_t,
                    saved_dir=config_agent.saved_dir,
                    load_dir=config_agent.load_dir,
                    load_pretrained_dir=config_agent.load_pretrained_dir,
                    load_policy=config_agent.load_policy,
                    number_training_iterations=config_agent.number_training_iterations,
                    buffer_min_size=config_agent.buffer_min_size,
                    buffer_sampling_rate=config_agent.buffer_sampling_rate,
                    buffer_sampling_size=config_agent.buffer_sampling_size,
                    policy_model_learning_rate=config_agent.policy_model_learning_rate,
                    e_matrix=config_agent.e,
                    loss_weight_inverse_e=config_agent.loss_weight_inverse_e,
                    buffer_max_size=config_agent.buffer_max_size,
                    use_hdf5_dataset = getattr(config_agent, "use_hdf5_dataset", False),
                    num_inference_steps=config_agent.num_inference_steps,
                    use_AutoEncoder_loss = config_agent.use_AutoEncoder_loss,
                    diffusion_step_embed_dim=config_agent.diffusion_step_embed_dim, unet_down_dims = config_agent.unet_down_dims,
                    frozen_obs_encoder = config_agent.frozen_obs_encoder,
                    config_agent = config_agent,
                )
            else:
                return DiffusionUnetLowdimPolicy_DPO(
                    noise_scheduler=noise_scheduler_DDPM,
                    noise_scheduler_inference = noise_scheduler_DDIM if config_agent.use_DDIM_during_inference else noise_scheduler_DDPM,
                    obs_dim=config_agent.dim_o,
                    action_dim=config_agent.dim_a,
                    shape_meta=config_agent.shape_meta,
                    horizon=config_agent.Ta,
                    desiredA_type = config_agent.desiredA_type,
                    large_desiredA = config_agent.large_desiredA,
                    sphere_alpha=config_agent.sphere_alpha,
                    sphere_gamma = config_agent.sphere_gamma,
                    radius_ratio = config_agent.radius_ratio,
                    sample_action_number = config_agent.sample_action_number,
                    sample_with_desiredA_reverse_start_t = config_agent.sample_with_desiredA_reverse_start_t,
                    saved_dir=config_agent.saved_dir,
                    load_dir=config_agent.load_dir,
                    load_pretrained_dir=config_agent.load_pretrained_dir,
                    load_policy=config_agent.load_policy,
                    number_training_iterations=config_agent.number_training_iterations,
                    buffer_min_size=config_agent.buffer_min_size,
                    buffer_sampling_rate=config_agent.buffer_sampling_rate,
                    buffer_sampling_size=config_agent.buffer_sampling_size,
                    policy_model_learning_rate=config_agent.policy_model_learning_rate,
                    e_matrix=config_agent.e,
                    loss_weight_inverse_e=config_agent.loss_weight_inverse_e,
                    buffer_max_size=config_agent.buffer_max_size,
                    num_inference_steps=config_agent.num_inference_steps,
                    diffusion_step_embed_dim=config_agent.diffusion_step_embed_dim, unet_down_dims = config_agent.unet_down_dims,
                )

    else:
        raise NameError(f'Unknown agent type: {agent_type}')

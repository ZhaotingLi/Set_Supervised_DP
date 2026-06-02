import logging
import numpy as np

from tools.buffer import Buffer
import os

from tools.buffer import Buffer_uniform_sampling

from agents import mcmc_torch as mcmc

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam

from agents.DP_model.energy_nn import ActionValueFunctionModel, EBTQFunctionModel
from agents.Set_Supervised_diffusion_policy_image import BaseImagePolicy

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
                                  sample_actions_size, evaluation, dim_a, device):
    
    # Initial random actions
    total = batch_size * sample_actions_size
    action_samples = torch.rand((total, dim_a), device=device) * 2 - 1
    # Tile state
    state_tiled = state.repeat(sample_actions_size, 1)
    # Langevin sampling (assumes mcmc functions implemented for PyTorch)
    action_samples = mcmc.langevin_actions_given_obs(
        Q_value_network, state_tiled, action_samples,
        policy_state=None, min_actions=-1, max_actions=1,
        training=False, tfa_step_type=(), return_chain=False,
        grad_norm_type='inf', num_action_samples=sample_actions_size)
    # Second refinement
    action_samples = mcmc.langevin_actions_given_obs(
        Q_value_network, state_tiled, action_samples,
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
                                 sample_actions_size=1024):
    total = batch_size * sample_actions_size
    random_actions = torch.rand((total, dim_a), device=observations.device) * 2 - 1
    # Langevin sampling
    samples = mcmc.langevin_actions_given_obs(
        Q_value_network, observations, random_actions,
        policy_state=None, min_actions=-1, max_actions=1,
        training=False, num_iterations=25,
        tfa_step_type=(), return_chain=False,
        grad_norm_type='inf', num_action_samples=sample_actions_size)
    if sample_closer_to_optimal_a:
        samples = mcmc.langevin_actions_given_obs(
            Q_value_network, observations, samples,
            policy_state=None, min_actions=-1, max_actions=1,
            training=False, tfa_step_type=(), return_chain=False,
            grad_norm_type='inf', sampler_stepsize_init=1e-1,
            sampler_stepsize_final=1e-5,
            num_action_samples=sample_actions_size)
    # Reshape to [sample_size, batch_size, dim_a]
    counter_examples = samples.view(sample_actions_size, batch_size, dim_a)
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



class CLIC(BaseImagePolicy):
    def __init__(self, dim_a, dim_o, shape_meta, action_upper_limits, action_lower_limits, e_matrix,
                 loss_weight_inverse_e, sphere_alpha, sphere_gamma, radius_ratio,
                 buffer_min_size, buffer_max_size, buffer_sampling_rate,
                 buffer_sampling_size, train_end_episode,
                 policy_model_learning_rate,
                 saved_dir, load_dir, load_policy,
                 number_training_iterations,
                #  desiredA_type = 'Circular',
                desiredA_type = 'Half',
                 n_obs_steps=1, obs_encoder_crop_shape=None, config_agent = None):
        
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
        self.use_CLIC_algorithm = True # use CLIC method

        self.e = np.diag(e_matrix)
        self.loss_weight_inverse_e = np.diag(loss_weight_inverse_e)

        self.sphere_alpha = sphere_alpha    # parameter that adjusts where to sample counterexamples on the sphere
        self.sphere_gamma = sphere_gamma    # parameter that adjust the center of the sphere to sample counterexamples
        self.radius_ratio = radius_ratio    # parameter used in circular desired action space


        self.horizon = 1

        self.desiredA_type = desiredA_type
        assert self.desiredA_type in ['Half', 'Circular']
        
        self.device = torch.device("cuda:0")  

        # Define observation encoder. Its input is obs_dict, including images and low-level info

        # self.action_value_model = ActionValueFunctionModel(dim_a=dim_a, dim_o=dim_o* n_obs_steps).to(self.device)
        self.action_value_model = EBTQFunctionModel(dim_a=dim_a, dim_o=dim_o* n_obs_steps, hidden_dim=128).to(self.device)
        self.optimizer_action_value_model = Adam(
                    self.parameters(),
                    lr=self.policy_model_learning_rate)
        
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
        self.buffer_no_corrections = Buffer(min_size=self.buffer_min_size,
                                           max_size=self.buffer_max_size)
        self.last_metrics = None
        self.test_New_value_function_idea = False
        self.saved_dir = saved_dir
        self.load_dir = load_dir
        self.load_policy_flag = load_policy
        self.number_training_iterations = number_training_iterations
        self.time_step = None
        self.evaluation = False
        self.list_action_tobeQueried = []
        self.last_action = True
        self.use_image = False
        self.sampled_action_last = None

    def save_model(self):
        # Define the directory for saving model parameters
        network_saved_dir = self.saved_dir + 'network_params/'
        if not os.path.exists(network_saved_dir):
            os.makedirs(network_saved_dir)
        
        # Save the model state dictionary
        model_filename = network_saved_dir + 'action_value_model.pth'
        torch.save({'action_value_model_state_dict': self.action_value_model.state_dict()}, model_filename)
        
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



    def sample_orthogonal_vectors(self, h_human, sample_h_size):
        batch_human, dim_a = h_human.shape
        h_human_normalized = F.normalize(h_human, p=2, dim=-1)
        basis_vectors = []
        for i in range(dim_a - 1):
            random_vector = torch.randn((batch_human, dim_a), device=h_human.device)
            proj = (random_vector * h_human_normalized).sum(dim=1, keepdim=True)
            random_vector = random_vector - proj * h_human_normalized
            for bv in basis_vectors:
                proj_b = (random_vector * bv).sum(dim=1, keepdim=True)
                random_vector = random_vector - proj_b * bv
            random_vector = F.normalize(random_vector, p=2, dim=-1)
            basis_vectors.append(random_vector)
        basis_vectors = torch.stack(basis_vectors, dim=1)

        coeffs = torch.randn((sample_h_size, dim_a - 1), device=h_human.device)
        coeffs = F.normalize(coeffs, p=2, dim=1)
        coeffs = coeffs.view(1, sample_h_size, dim_a - 1).repeat(batch_human, 1, 1)

        sampled = torch.matmul(coeffs, basis_vectors)
        sampled = sampled.transpose(0, 1)

        h_exp = h_human.unsqueeze(0).repeat(sample_h_size, 1, 1)
        norm_h = h_exp.norm(dim=-1, keepdim=True)
        sampled = F.normalize(sampled, p=2, dim=-1) * norm_h
        sampled = sampled.view(-1, h_human.shape[-1])
        return sampled

    def sample_vectors_from_sphere(self, h_human, sample_h_size, alpha):
        batch_human, dim_a = h_human.shape
        if dim_a == 1:
            raise NotImplementedError("dim_a == 1 not supported")
        h_norm = F.normalize(h_human, p=2, dim=-1)
        basis_vectors = []
        for i in range(dim_a - 1):
            rv = F.one_hot(torch.tensor(i, device=h_human.device), dim_a).float()
            rv = rv.unsqueeze(0).repeat(batch_human, 1)
            proj = (rv * h_norm).sum(dim=1, keepdim=True)
            rv = rv - proj * h_norm
            for bv in basis_vectors:
                p = (rv * bv).sum(dim=1, keepdim=True)
                rv = rv - p * bv
            rv = F.normalize(rv, p=2, dim=-1)
            basis_vectors.append(rv)
        basis_vectors = torch.stack(basis_vectors, dim=1)


        if dim_a - 1 == 1:
            coeffs = self.sample_equally_on_1d(sample_h_size)
        else:
            coeffs = torch.randn((sample_h_size, dim_a - 1), device=h_human.device)
            coeffs = F.normalize(coeffs, p=2, dim=1)
        coeffs = coeffs.view(1, sample_h_size, dim_a - 1).repeat(batch_human, 1, 1)

        vecs = torch.matmul(coeffs, basis_vectors).transpose(0, 1)
        h_exp = h_human.unsqueeze(0).repeat(sample_h_size, 1, 1)
        norm_h = h_exp.norm(dim=-1, keepdim=True)
        vecs = F.normalize(vecs, p=2, dim=-1) * norm_h
        alpha = torch.as_tensor(alpha, device=h_human.device, dtype=h_human.dtype)
        samples = 0.5 * vecs * torch.sin(alpha) + 0.5 * h_exp * torch.cos(alpha)
        return samples
    
    def sample_equally_on_1d(self,sample_h_size):
        vectors = np.array([[1.0], [-1.0]])
        
        # Select a specific number of vectors to match the sample_h_size
        vectors = np.tile(vectors, (sample_h_size // 2, 1))
        
        # If sample_h_size is odd, append one more vector to match the required size
        if sample_h_size % 2 != 0:
            vectors = np.vstack([vectors, np.array([[1.0]])])
        return torch.as_tensor(vectors, dtype=torch.float32, device=self.device)

    def custom_l4_penalty(self, Q_values, lower_bound=-1.0, upper_bound=1.0):
        penalty = torch.clamp(Q_values - upper_bound, min=0.0)**2
        penalty += torch.clamp(Q_values - lower_bound, max=0.0)**2
        return torch.sum(penalty**2)

    def process_tensor_sample_h_size_batch(self, tensor, sample_h_size, sample_implicit_size, batch_size):
        t = tensor.view(sample_h_size, batch_size, -1)
        t = t.repeat(1, sample_implicit_size, 1)
        return t.view(sample_h_size * sample_implicit_size * batch_size, -1)

    def action_value_single_update_feasible_space_with_bayesian_implicit_areasector(self, observation, action, h_human, next_observation):
        self.action_value_model.train()
        self.action_value_model.training = True

        batch_size = h_human.shape[0]
        action_dim = h_human.shape[-1]
        sample_h_size = 512
        sample_implicit_size = 128

        alpha = self.sphere_alpha * np.pi / 180
        beta = self.sphere_alpha * 0.5 * np.pi / 180
        beta_cos = torch.cos(torch.tensor(beta, device=self.device))
        sphere_center_gamma = self.sphere_gamma

        softmax_temperature = 1.0

        global_cond = observation

        observation_tiled = global_cond.repeat(sample_h_size, 1)

        sampled_action, _ = make_counter_example_actions(
            self.action_value_model, observation_tiled,
            batch_size, action_dim,
            sample_actions_size=sample_h_size)
        # prepend true action and shifted action
        first = action.unsqueeze(0)
        second = (action + torch.matmul(h_human, torch.tensor(self.e,dtype=torch.float32, device=self.device))).unsqueeze(0)
        sampled_action = torch.cat([first, second, sampled_action[2:]], dim=0)
        sampled_action = sampled_action.view(-1, action_dim)
        self.sampled_action_last = sampled_action

        sampled_action_desired = None
        if sampled_action_desired is not None and sampled_action_desired.shape[0] > 0:
            sampled_action = torch.cat([sampled_action, sampled_action_desired], dim=0).detach()
            extra = sampled_action_desired.shape[0] // batch_size
            sample_h_size += extra
            observation_tiled = observation.repeat(sample_h_size, 1)

        Q_s_a_target = self.action_value_model(observation_tiled, sampled_action).detach()
        Q_s_a_target = Q_s_a_target.view(sample_h_size, batch_size).transpose(0, 1)

        test_bounded2N = False
        if self.dim_a > 1:
            rerun_count = 0
            rerun_limit = 10
            while rerun_count <= rerun_limit:
                if not test_bounded2N:
                    v_combined = self.sample_vectors_from_sphere(h_human, sample_implicit_size, alpha)
                if torch.isnan(v_combined).any():
                    rerun_count += 1
                    logger.warning(f"NaN detected, rerunning... (Attempt {rerun_count})")
                else:
                    break
            if torch.isnan(v_combined).any():
                logger.warning("NaN persists")
                return None
            normalized_random_hs = v_combined.reshape(-1, action_dim) * 2.0

        # compute area-sector
        action_tiled = action.repeat(sample_h_size, 1)
        h_tiled = h_human.repeat(sample_h_size, 1)
        diff = sampled_action - action_tiled
        coeff_exp = 0.1
        area_sector = -diff.norm(dim=-1, keepdim=True)
        shifted = diff - self.e[0,0]*sphere_center_gamma*h_tiled
        area_sector = area_sector + shifted.norm(dim=-1, keepdim=True)
        # area_sector = (area_sector / coeff_exp).exp().reciprocal().log1p().neg()
        area_sector = torch.log(1.0 / (1.0 + torch.exp(area_sector / coeff_exp)))
        area_sector = area_sector.reshape(sample_h_size, batch_size).transpose(0,1)

        if self.dim_a > 1:
            act_imp = action.repeat(sample_implicit_size,1)
            act_imp_neg = (action + self.e[0,0]*sphere_center_gamma*h_human).repeat(sample_implicit_size,1)
            implicit_neg = act_imp_neg + self.e[0,0]*(1-sphere_center_gamma)*normalized_random_hs
            imp_tiled = act_imp.repeat(sample_h_size,1)
            neg_tiled = implicit_neg.repeat(sample_h_size,1)
            samp_tiled = self.process_tensor_sample_h_size_batch(sampled_action, sample_h_size, sample_implicit_size, batch_size)
            e_h = self.e[0,0]*self.process_tensor_sample_h_size_batch(h_tiled, sample_h_size, sample_implicit_size, batch_size)
            a_imp_diff = -(samp_tiled - neg_tiled).norm(dim=-1,keepdim=True)
            b_imp_diff = (samp_tiled - imp_tiled - e_h).norm(dim=-1,keepdim=True)
            imp_sec = (a_imp_diff + b_imp_diff)/coeff_exp
            # imp_sec = imp_sec.exp().reciprocal().log1p().neg()
            imp_sec = torch.log(1.0 / (1.0 + torch.exp(imp_sec )))
            imp_sec = imp_sec.reshape(sample_h_size, sample_implicit_size, batch_size)
            sum_imp = imp_sec.sum(dim=1).transpose(0,1)
            weight = 0.3*sample_implicit_size
            # weight = 0.5*sample_implicit_size
            mean_area = (sum_imp + weight*area_sector) / (weight+sample_implicit_size)
        else:
            mean_area = area_sector

        combined_cond = mean_area
        labels_dynamic = F.softmax(Q_s_a_target/softmax_temperature + combined_cond/0.1, dim=-1)

        # InfoNCE
        obs_tiled = observation_tiled
        Q_s_a = self.action_value_model(obs_tiled, sampled_action)
        preds = Q_s_a.reshape(sample_h_size, batch_size)
        soft_p = F.softmax(preds/softmax_temperature, dim=0).transpose(0,1)
        loss_kl = F.kl_div(soft_p.log(), labels_dynamic, reduction='batchmean')
        # loss_kl = F.kl_div(soft_p, labels_dynamic, reduction='batchmean')
        grad_pen = grad_penalty(self.action_value_model, batch_size, observation_tiled, sampled_action, training=True)
        grad_pen = grad_pen.mean()
        loss = loss_kl + grad_pen
        logger.debug('%s  loss_Kl:  %s  grad_pen:  %s', self.desiredA_type, loss_kl, grad_pen)
        
        self.optimizer_action_value_model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.action_value_model.parameters(), 1.0)
        self.optimizer_action_value_model.step()
        self.lr_scheduler.step()

        with torch.no_grad():
            self.last_metrics = {
                "total_loss": float(loss.detach().item()),
                "kl_loss": float(loss_kl.detach().item()),
                "grad_penalty": float(grad_pen.detach().item()),
                "lr": float(self.optimizer_action_value_model.param_groups[0]["lr"]),
                # "Q_target_mean": float(Q_s_a_target.mean().item()),
                # "Q_target_std": float(Q_s_a_target.std().item()),
            }


    def action_value_single_update_Circular(self, observation, action, h_human, next_observation):
        self.action_value_model.train()
        self.action_value_model.training = True

        batch_size = h_human.shape[0]
        action_dim = h_human.shape[-1]  # (Batch, horizon, dim_a)
        # sample_h_size = self.sample_action_number  #TODO add to config
        # sample_h_size = 512
        sample_h_size = 128
        action_negative = action
        action_positive = action + h_human

        global_cond = observation
        observation_tiled = global_cond.repeat(sample_h_size, 1)
   
        sampled_action, _ = make_counter_example_actions(
            self.action_value_model, observation_tiled,
            batch_size, action_dim,
            sample_actions_size=sample_h_size,)
            # horizon=self.horizon)
        # prepend true action and shifted action
        first = action_positive.unsqueeze(0)
        
        sampled_action = torch.cat([first, sampled_action[1:]], dim=0)
        sampled_action = sampled_action.detach()
        # if self.horizon > 1:
        #     sampled_action = sampled_action.view(-1, self.horizon, action_dim)
        # else:
        #     sampled_action = sampled_action.view(-1, action_dim)  
        sampled_action = sampled_action.view(-1, action_dim)
        self.sampled_action_last = sampled_action

        Q_s_a_target = self.action_value_model(observation_tiled, sampled_action).detach()

        # if self.horizon == 1:
        #     action_tiled = action.repeat(sample_h_size, 1)
        #     action_positive_tiled = action_positive.repeat(sample_h_size, 1)
        #     log_prob = -torch.sum((sampled_action - action_positive_tiled) ** 2, dim=-1, keepdim=True)
        # else:
        #     action_positive_tiled = action_positive.repeat(sample_h_size, 1, 1)
        #     action_tiled = action.repeat(sample_h_size, 1, 1)
        #     log_prob = -torch.sum((sampled_action - action_positive_tiled) ** 2, dim=(-2, -1), keepdim=True)

        action_positive_tiled = action_positive.repeat(sample_h_size, 1)

        coefficient_exp = 0.05

        # distance between sampled_action and action_positive_tiled
        diff_pos = sampled_action - action_positive_tiled
        distance_a_a_plus = diff_pos.norm(dim=-1, keepdim=True)

        if action_negative is not None:
            action_negative_tiled = action_negative.repeat(sample_h_size, 1)

            # distance between negative and positive actions
            diff_neg = action_negative_tiled - action_positive_tiled
            # sigma = 0.5 * diff_neg.norm(dim=-1, keepdim=True)
            sigma = self.radius_ratio * diff_neg.norm(dim=-1, keepdim=True)

        # area_sector = (distance_a_a_plus - sigma) / sigma
        area_sector = (distance_a_a_plus - sigma)
        combined_condition_value = torch.log(1.0 / (1.0 + torch.exp(area_sector / coefficient_exp)))
        # softmax_temperature = 0.1
        softmax_temperature = 0.5 # better than 0.1
        # softmax_temperature = 1.0 
        labels = Q_s_a_target/softmax_temperature + combined_condition_value/0.1  # CT01
        labels = labels.reshape(sample_h_size, batch_size)
        labels = F.softmax(labels, dim=0).transpose(0,1).detach()
        
        # labels = cond   ## these two lines are IBC
        # labels = labels / (labels.sum(dim=1, keepdim=True) + 1e-8)
            
        obs_tiled = observation_tiled
        Q_s_a = self.action_value_model(obs_tiled, sampled_action)
        preds = Q_s_a.reshape(sample_h_size, batch_size)
        soft_p = F.softmax(preds/softmax_temperature, dim=0).transpose(0,1)
        
        
        loss_kl = F.kl_div(soft_p.log(), labels, reduction='batchmean')
        # loss_kl = F.kl_div(soft_p, labels_dynamic, reduction='batchmean')
        grad_pen = grad_penalty(self.action_value_model, batch_size, observation_tiled, sampled_action, training=True)
        grad_pen = grad_pen.mean()
        loss = loss_kl + grad_pen
        logger.debug('%s  loss_Kl:  %s  grad_pen:  %s', self.desiredA_type, loss_kl, grad_pen)
        
        self.optimizer_action_value_model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.action_value_model.parameters(), 1.0)
        self.optimizer_action_value_model.step()
        self.lr_scheduler.step()

        with torch.no_grad():
            self.last_metrics = {
                "total_loss": float(loss.detach().item()),
                "kl_loss": float(loss_kl.detach().item()),
                "grad_penalty": float(grad_pen.detach().item()),
                "lr": float(self.optimizer_action_value_model.param_groups[0]["lr"]),
                # "Q_target_mean": float(Q_s_a_target.mean().item()),
                # "Q_target_std": float(Q_s_a_target.std().item()),
            }


    def action_value_batch_update(self, batch):
        # state_batch = batch[0]
        # action_batch = batch[1]
        # h_human_batch = batch[2]
        # next_state_batch = batch[3]
        state_batch = [pair[0] for pair in batch]
        action_batch = [np.array(pair[1]) for pair in batch] 
        h_human_batch = [np.array(pair[2]) for pair in batch]  
        batch_size = len(batch)
        # state = torch.tensor(np.stack(state_batch), dtype=torch.float32, device=self.device)


        h_human_batch     = torch.tensor(np.reshape(h_human_batch, [batch_size, self.dim_a]), dtype=torch.float32, device=self.device)
        action_batch     = torch.tensor(np.reshape(action_batch, [batch_size, self.dim_a]), dtype=torch.float32, device=self.device)
        obs = torch.tensor(np.reshape(state_batch, [batch_size,  self.dim_o]), dtype=torch.float32, device=self.device)
        next_obs = None

        if self.desiredA_type == 'Half':
            self.action_value_single_update_feasible_space_with_bayesian_implicit_areasector(
                obs, action_batch, h_human_batch, next_obs
            )  
        elif self.desiredA_type == 'Circular':
            self.action_value_single_update_Circular(
                obs, action_batch, h_human_batch, next_obs
            )        
        else:
            raise ValueError(f"Unsupported desiredA type {self.desiredA_type}")


        

    def action(self, observation):

        observation = torch.tensor(observation, dtype=self.dtype, device=self.device)
        # observation = observation.unsqueeze(0)
        action = sample_action_from_Q_function(self.action_value_model, observation, 1, 512, True, self.dim_a, self.device).detach()

        action = action.cpu().numpy()

        out_action = []

        for i in range(self.dim_a):
            action[i] = np.clip(action[i], -1, 1) 
            out_action.append(action[i])

        return np.array(out_action)

    def collect_data_and_train(self, last_action, h, obs_proc, next_obs, t, done, agent_algorithm=None, agent_type=None, i_episode=None):
        """Unified entry point used by main_IIL.py """
        self.TRAIN_Q_value(last_action, h, obs_proc, next_obs, t, done, i_episode)  
        return  self.last_metrics
    
    def TRAIN_Q_value(self, action, h, observation, next_observation,  t, done, i_episode = None):
        if np.any(h):  # if any element is not 0
            # 1. append  (o_t, a_t, h_t) to D
            # self.h_to_buffer = tf.convert_to_tensor(np.reshape(h, [1, self.dim_a]), dtype=tf.float32)
            # action_tf = tf.convert_to_tensor(np.reshape(action, [1, self.dim_a]), dtype=tf.float32)

            # self.buffer.add([observation, action, h, next_observation])
            # self.buffer.add([observation, action, h, next_observation])
            self.latested_data_pair = [observation, action, (h - action) / self.e[0,0] , next_observation]
            self.buffer.add( self.latested_data_pair)
            # [!!!] next_observation is not used

            # 4. Update Human model with a minibatch sampled from buffer D
            if self.buffer.initialized():

                batch = self.buffer.sample(batch_size=self.buffer_sampling_size)
                # include the new data in this batch
                batch[-1] = [observation, action, h, next_observation]
                self.action_value_batch_update(batch)

        # Train policy every k time steps from buffer
        elif self.buffer.initialized() and t % self.buffer_sampling_rate == 0:
            logger.info("train q value")
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

        self.last_metrics = None # only save the metrics of end-of-episode

        number_training_iterations_ = self.number_training_iterations
        if self.buffer.initialized() and (self.train_end_episode and done):
            # self.buffer.reset_temp_buffer()
            # self.number_training_iterations = 300
            # self.number_training_iterations = 20
            for i in range(number_training_iterations_):
                if i % (number_training_iterations_ / 20) == 0:
                    logger.info("Progress Policy training: %i %%", i / number_training_iterations_ * 100)
                    logger.debug('buffer size:  %s', self.buffer.length())
                   
                for i in range(1):  # this is better than range(1), you should train the Q function a litte bit frequent than policy network
                # for i in range(1):
                    batch = self.buffer.sample(batch_size=self.buffer_sampling_size)
                    self.action_value_batch_update(batch)
                    

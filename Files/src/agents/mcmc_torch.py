# """MCMC algorithms to optimize samples from EBMs."""

# import collections
# from typing import Tuple

# import gin
# import numpy as np
# import tensorflow as tf
# from tf_agents.utils import nest_utils

# # This global makes it easier to switch on/off tf.range in this file.
# # Which I am often doing in order to debug anything in the binaries
# # that use this.
# my_range = tf.range

# # @tf.function
# def gradient_wrt_act(energy_network,
#                      observations,
#                      actions,
#                      training,
#                      network_state,
#                      tfa_step_type,
#                      apply_exp,
#                      obs_encoding=None):
#   """Compute dE(obs,act)/dact, also return energy."""
#   with tf.GradientTape() as g:
#     g.watch(actions)
#     if obs_encoding is not None:
#       energies, _ = energy_network((observations, actions),
#                                    training=training,
#                                    network_state=network_state,
#                                    step_type=tfa_step_type,
#                                    observation_encoding=obs_encoding)
#     else:
#       # energies, _ = energy_network((observations, actions),
#       #                              training=training,
#       #                              network_state=network_state,
#       #                              step_type=tfa_step_type)
#       energies = energy_network([observations, actions], training=training)
                                   
#     # If using a loss function that involves the exp(energies),
#     # should we apply exp() to the energy when taking the gradient?
#     if apply_exp:
#       energies = tf.math.exp(energies)
#   # My energy sign is flipped relative to Igor's code,
#   # so -1.0 here.
#   denergies_dactions = g.gradient(energies, actions) * -1.0
#   return denergies_dactions, energies


# def compute_grad_norm(grad_norm_type, de_dact):
#   """Given de_dact and the type, compute the norm."""
#   if grad_norm_type is not None:
#     grad_norm_type_to_ord = {'1': 1,
#                              '2': 2,
#                              'inf': np.inf}
#     grad_type = grad_norm_type_to_ord[grad_norm_type]
#     grad_norms = tf.linalg.norm(de_dact, axis=1, ord=grad_type)
#   else:
#     # It will be easier to manage downstream if we just fill this with zeros.
#     # Rather than have this be potentially a None type.
#     grad_norms = tf.zeros_like(de_dact[:, 0])
#   return grad_norms


# def langevin_step(energy_network,
#                   observations,
#                   actions,
#                   training,
#                   policy_state,
#                   tfa_step_type,
#                   noise_scale,
#                   grad_clip,
#                   delta_action_clip,
#                   stepsize,
#                   apply_exp,
#                   min_actions,
#                   max_actions,
#                   grad_norm_type,
#                   obs_encoding):
#   """Single step of Langevin update."""
#   l_lambda = 1.0
#   # Langevin dynamics step
#   de_dact, energies = gradient_wrt_act(energy_network,
#                                        observations,
#                                        actions,
#                                        training,
#                                        policy_state,
#                                        tfa_step_type,
#                                        apply_exp,
#                                        obs_encoding)

#   # This effectively scales the gradient as if the actions were
#   # in a min-max range of -1 to 1.
#   delta_action_clip = delta_action_clip * 0.5*(max_actions - min_actions)

#   # TODO(peteflorence): can I get rid of this copy, for performance?
#   # Times 1.0 since I don't trust tf.identity to make a deep copy.
#   unclipped_de_dact = de_dact * 1.0
#   grad_norms = compute_grad_norm(grad_norm_type, unclipped_de_dact)

#   if grad_clip is not None:
#     de_dact = tf.clip_by_value(de_dact, -grad_clip, grad_clip)
#   gradient_scale = 0.5  # this is in the Langevin dynamics equation.
#   de_dact = (gradient_scale * l_lambda * de_dact +
#              tf.random.normal(tf.shape(actions)) * l_lambda * noise_scale)
#   delta_actions = stepsize * de_dact

#   # Clip to box.
#   delta_actions = tf.clip_by_value(delta_actions, -delta_action_clip,
#                                    delta_action_clip)
#   # print("shape of delta_actions: ", delta_actions.shape)   # (B * sampled_action_size, dim_a)
#   # TODO(peteflorence): investigate more clipping to sphere:
#   # delta_actions = tf.clip_by_norm(
#   #  delta_actions, delta_action_clip, axes=[1])

#   actions = actions - delta_actions
#   actions = tf.clip_by_value(actions,
#                              min_actions,
#                              max_actions)

#   return actions, energies, grad_norms



# class ExponentialSchedule:
#   """Exponential learning rate schedule for Langevin sampler."""

#   def __init__(self, init, decay):
#     self._decay = decay
#     self._latest_lr = init

#   def get_rate(self, index):
#     """Get learning rate. Assumes calling sequentially."""
#     del index
#     self._latest_lr *= self._decay
#     return self._latest_lr


# class PolynomialSchedule:
#   """Polynomial learning rate schedule for Langevin sampler."""

#   def __init__(self, init, final, power, num_steps):
#     self._init = init
#     self._final = final
#     self._power = power
#     self._num_steps = num_steps

#   def get_rate(self, index):
#     """Get learning rate for index."""
#     return ((self._init - self._final) *
#             ((1 - (float(index) / float(self._num_steps-1))) ** (self._power))
#             ) + self._final


# def update_chain_data(num_iterations,
#                       step_index,
#                       actions,
#                       energies,
#                       grad_norms,
#                       full_chain_actions,
#                       full_chain_energies,
#                       full_chain_grad_norms):
#   """Helper function to keep track of data during the mcmc."""
#   # I really wish tensorflow made assignment-by-index easy.
#   # Then this function could just be:
#   # full_chain_actions[step_index] = actions
#   # full_chain_energies[step_index] = energies
#   # full_chain_grad_norms[step_index] = grad_norms

#   iter_onehot = tf.one_hot(step_index, num_iterations)[Ellipsis, None]
#   iter_onehot = tf.broadcast_to(iter_onehot, tf.shape(full_chain_energies))

#   energies = tf.squeeze(energies) # [LZT] add this to keep the dim right (change (8000, 1) to (8000,))
#   # print("shape of energies: ", energies.shape)
#   # print("iter_onehot shape: ", iter_onehot.shape)
#   new_energies = energies * iter_onehot
#   full_chain_energies += new_energies

#   new_grad_norms = grad_norms * iter_onehot
#   full_chain_grad_norms += new_grad_norms

#   iter_onehot = iter_onehot[Ellipsis, None]
#   iter_onehot = tf.broadcast_to(iter_onehot, tf.shape(full_chain_actions))
#   actions_expanded = actions[None, Ellipsis]
#   actions_expanded = tf.broadcast_to(actions_expanded, tf.shape(iter_onehot))
#   new_actions_expanded = actions_expanded * iter_onehot
#   full_chain_actions += new_actions_expanded
#   return full_chain_actions, full_chain_energies, full_chain_grad_norms


# # @tf.function
# # @gin.configurable
# def langevin_actions_given_obs(
#     energy_network,
#     observations,  # B*n x obs_spec or B x obs_spec if late_fusion
#     action_samples,  # B*n x act_spec
#     policy_state,
#     min_actions,
#     max_actions,
#     num_action_samples,
#     num_iterations=25,
#     # num_iterations=100,
#     training=False,
#     tfa_step_type=(),
#     sampler_stepsize_init=1e-1,
#     sampler_stepsize_decay=0.8,  # if using exponential langevin rate.
#     noise_scale=1.0,
#     grad_clip=None,
#     delta_action_clip=0.1,
#     stop_chain_grad=True,
#     apply_exp=False,
#     use_polynomial_rate=True,  # default is exponential
#     sampler_stepsize_final=1e-5,  # if using polynomial langevin rate.
#     sampler_stepsize_power=2.0,  # if using polynomial langevin rate.
#     return_chain=False,
#     grad_norm_type = 'inf',
#     late_fusion=False):
#   """Given obs and actions, use dE(obs,act)/dact to perform Langevin MCMC."""
#   stepsize = sampler_stepsize_init
#   actions = tf.identity(action_samples)

#   if use_polynomial_rate:
#     schedule = PolynomialSchedule(sampler_stepsize_init, sampler_stepsize_final,
#                                   sampler_stepsize_power, num_iterations)
#   else:  # default to exponential rate
#     schedule = ExponentialSchedule(sampler_stepsize_init,
#                                    sampler_stepsize_decay)

#   b_times_n = tf.shape(action_samples)[0]
#   act_dim = tf.shape(action_samples)[-1]

#   # Note 2: to work inside the tf.range, we have to initialize all these
#   # outside the loop.

#   # Note 1: for 1 step, there are [0, 1] points in the chain
#   # grad norms will be for [0, ... N-1]

#   # full_chain_actions is actually currently [1, ..., N]
#   full_chain_actions = tf.zeros((num_iterations, b_times_n, act_dim))
#   # full_chain_energies will also be for [0, ..., N-1]
#   full_chain_energies = tf.zeros((num_iterations, b_times_n))
#   # full_chain_grad_norms will be for [0, ..., N-1]
#   full_chain_grad_norms = tf.zeros((num_iterations, b_times_n))


#   obs_encoding = None

#   for step_index in my_range(num_iterations):
#     actions, energies, grad_norms = langevin_step(energy_network,
#                                                   observations,
#                                                   actions,
#                                                   training,
#                                                   policy_state,
#                                                   tfa_step_type,
#                                                   noise_scale,
#                                                   grad_clip,
#                                                   delta_action_clip,
#                                                   stepsize,
#                                                   apply_exp,
#                                                   min_actions,
#                                                   max_actions,
#                                                   grad_norm_type,
#                                                   obs_encoding)
#     # print("actions shape after langevin_step: ", actions.shape)
#     if stop_chain_grad:
#       actions = tf.stop_gradient(actions)
#     stepsize = schedule.get_rate(step_index + 1)  # Get it for the next round.

#     if return_chain:
#       (full_chain_actions, full_chain_energies,
#        full_chain_grad_norms) = update_chain_data(num_iterations, step_index,
#                                                   actions, energies, grad_norms,
#                                                   full_chain_actions,
#                                                   full_chain_energies,
#                                                   full_chain_grad_norms)

  
#   if return_chain:
#     data_fields = ['actions', 'energies', 'grad_norms']
#     ChainData = collections.namedtuple('ChainData', data_fields)
#     chain_data = ChainData(full_chain_actions, full_chain_energies,
#                            full_chain_grad_norms)
#     return actions, chain_data
#   else:
#     return actions


# def get_probabilities(energy_network,
#                       batch_size,
#                       num_action_samples,
#                       observations,
#                       actions,
#                       training,
#                       # temperature=1.0
#                       temperature=0.1
#                       ):
#   """Get probabilities to post-process Langevin results."""
#   # net_logits, _ = energy_network(
#   #     (observations, actions), training=training)
#   net_logits = energy_network(
#       (observations, actions), training=training)
#   net_logits = tf.reshape(net_logits, (batch_size, num_action_samples))
#   probs = tf.nn.softmax(net_logits / temperature, axis=1)
#   probs = tf.reshape(probs, (-1,))
#   return probs



import collections
from typing import Tuple, Optional

import torch
import torch.nn.functional as F

# Global range alias for debugging
my_range = range


def gradient_wrt_act(
    energy_network,
    observations,
    actions,
    training,
    network_state,
    tfa_step_type,
    apply_exp,
    obs_encoding=None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute dE(obs,act)/dact, also return energy."""
    # Ensure actions require gradient
    actions = actions.clone().detach().requires_grad_(True)
    # observations_ = observations.clone().detach()
    observations_ = observations
    # Forward pass
    if obs_encoding is not None:
        energies, _ = energy_network(
            (observations_, actions),
            training=training,
            network_state=network_state,
            step_type=tfa_step_type,
            observation_encoding=obs_encoding
        )
    else:
        if training:
            energy_network.train()
        else:
            energy_network.eval()
        energies = energy_network(observations_, actions)

    if apply_exp:
        energies = torch.exp(energies)

    # Gradients
    denergies_dactions = torch.autograd.grad(
        outputs=energies.sum(),
        inputs=actions,
        create_graph=True,  # set to True to penalize the norm of gradients
        retain_graph=True
        # create_graph=False,
        # retain_graph=False
    )[0] * -1.0

    return denergies_dactions, energies.detach()


def compute_grad_norm(grad_norm_type, de_dact):
    """Given de_dact and the type, compute the norm."""
    if grad_norm_type is not None:
        grad_type = {'1': 1, '2': 2, 'inf': float('inf')}[grad_norm_type]
        grad_norms = torch.norm(de_dact, p=grad_type, dim=1)
    else:
        grad_norms = torch.zeros(de_dact.size(0), device=de_dact.device)
    return grad_norms


def langevin_step(
    energy_network,
    observations,
    actions,
    training,
    policy_state,
    tfa_step_type,
    noise_scale,
    grad_clip,
    delta_action_clip,
    stepsize,
    apply_exp,
    min_actions,
    max_actions,
    grad_norm_type,
    obs_encoding
):
    """Single step of Langevin update."""
    l_lambda = 1.0
    de_dact, energies = gradient_wrt_act(
        energy_network,
        observations,
        actions,
        training,
        policy_state,
        tfa_step_type,
        apply_exp,
        obs_encoding
    )

    # Scale delta clip
    delta_clip_scaled = delta_action_clip * 0.5 * (max_actions - min_actions)

    unclipped = de_dact.clone()
    grad_norms = compute_grad_norm(grad_norm_type, unclipped)

    if grad_clip is not None:
        de_dact = torch.clamp(de_dact, -grad_clip, grad_clip)

    de_dact = 0.5 * l_lambda * de_dact + torch.randn_like(actions) * l_lambda * noise_scale
    delta_actions = stepsize * de_dact

    # Clip and update
    delta_actions = torch.clamp(delta_actions, -delta_clip_scaled, delta_clip_scaled)
    actions = actions - delta_actions
    actions = actions.clamp(min=min_actions, max=max_actions)

    return actions.detach(), energies, grad_norms


class ExponentialSchedule:
    def __init__(self, init, decay):
        self._decay = decay
        self._latest = init

    def get_rate(self, index):
        self._latest *= self._decay
        return self._latest


class PolynomialSchedule:
    def __init__(self, init, final, power, num_steps):
        self._init = init
        self._final = final
        self._power = power
        self._num_steps = num_steps

    def get_rate(self, index):
        frac = 1.0 - (index / float(self._num_steps - 1))
        return ((self._init - self._final) * (frac ** self._power)) + self._final


def update_chain_data(
    num_iterations,
    step_index,
    actions,
    energies,
    grad_norms,
    full_chain_actions,
    full_chain_energies,
    full_chain_grad_norms
):
    """Helper to record data during MCMC."""
    mask = F.one_hot(torch.full_like(energies, step_index, dtype=torch.long), num_iterations).float()
    full_chain_energies += energies * mask
    full_chain_grad_norms += grad_norms * mask

    mask_actions = mask.unsqueeze(-1).expand(-1, -1, actions.size(-1))
    actions_exp = actions.unsqueeze(0).expand_as(mask_actions)
    full_chain_actions += actions_exp * mask_actions

    return full_chain_actions, full_chain_energies, full_chain_grad_norms


def langevin_actions_given_obs(
    energy_network,
    observations,
    action_samples,
    policy_state,
    min_actions,
    max_actions,
    num_action_samples,
    num_iterations=25,
    training=False,
    tfa_step_type=(),
    sampler_stepsize_init=1e-1,
    sampler_stepsize_decay=0.8,
    noise_scale=1.0,
    grad_clip=None,
    delta_action_clip=0.1,
    stop_chain_grad=True,
    apply_exp=False,
    use_polynomial_rate=True,
    sampler_stepsize_final=1e-5,
    sampler_stepsize_power=2.0,
    return_chain=False,
    grad_norm_type='inf',
    late_fusion=False
):
    """Perform Langevin MCMC sampling of actions given observations."""
    stepsize = sampler_stepsize_init
    if use_polynomial_rate:
        schedule = PolynomialSchedule(sampler_stepsize_init, sampler_stepsize_final,
                                      sampler_stepsize_power, num_iterations)
    else:
        schedule = ExponentialSchedule(sampler_stepsize_init, sampler_stepsize_decay)

    actions = action_samples.clone()
    b_times_n = actions.size(0)
    act_dim = actions.size(-1)

    full_chain_actions = torch.zeros((num_iterations, b_times_n, act_dim), device=actions.device)
    full_chain_energies = torch.zeros((num_iterations, b_times_n), device=actions.device)
    full_chain_grad_norms = torch.zeros((num_iterations, b_times_n), device=actions.device)

    for step_index in my_range(num_iterations):
        actions, energies, grad_norms = langevin_step(
            energy_network,
            observations,
            actions,
            training,
            policy_state,
            tfa_step_type,
            noise_scale,
            grad_clip,
            delta_action_clip,
            stepsize,
            apply_exp,
            min_actions,
            max_actions,
            grad_norm_type,
            None
        )
        if stop_chain_grad:
            actions = actions.detach()
        stepsize = schedule.get_rate(step_index + 1)

        if return_chain:
            (full_chain_actions, full_chain_energies,
             full_chain_grad_norms) = update_chain_data(
                num_iterations, step_index,
                actions, energies, grad_norms,
                full_chain_actions, full_chain_energies,
                full_chain_grad_norms
            )

    if return_chain:
        ChainData = collections.namedtuple('ChainData', ['actions', 'energies', 'grad_norms'])
        chain_data = ChainData(full_chain_actions, full_chain_energies, full_chain_grad_norms)
        return actions, chain_data
    return actions


def get_probabilities(
    energy_network,
    batch_size,
    num_action_samples,
    observations,
    actions,
    training,
    temperature=0.1
):
    """Get probabilities to post-process Langevin results."""
    # net_logits = energy_network([observations, actions], training=training)
    if training:
        energy_network.train()
    else:
        energy_network.eval()
    net_logits = energy_network(observations, actions)
    net_logits = net_logits.view(batch_size, num_action_samples)
    probs = F.softmax(net_logits / temperature, dim=1)
    return probs.view(-1)
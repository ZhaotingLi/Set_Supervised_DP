from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
logger = logging.getLogger(__name__)

from agents.DP_model.diffusion.conditional_unet1d_original import ConditionalResidualBlock1D

from agents.DP_model.diffusion.conv1d_components import (
    Downsample1d, Upsample1d, Conv1dBlock)

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionValueFunctionModel(nn.Module):
    """
    PyTorch implementation of the TF Keras action-value function model.
    Takes state and action inputs, concatenates them, and passes through
    successive Linear -> ReLU -> LayerNorm layers, ending in a scalar Q-value.
    """
    def __init__(self, dim_o: int, dim_a: int, hidden_dims=None):
        super(ActionValueFunctionModel, self).__init__()
        self.dim_o = dim_o
        self.dim_a = dim_a

        # Define network dimensions
        input_dim = dim_o + dim_a
        # default to original architecture if not provided
        if hidden_dims is None:
            hidden_dims = [512, 512, 512, 256]
        self.hidden_dims = hidden_dims

        # Build Linear + LayerNorm stacks
        self.fcs = nn.ModuleList()
        self.lns = nn.ModuleList()

        prev_dim = input_dim
        for h_dim in hidden_dims:
            self.fcs.append(nn.Linear(prev_dim, h_dim))
            self.lns.append(nn.LayerNorm(h_dim))
            prev_dim = h_dim

        # Output layer: scalar Q-value
        self.fc_out = nn.Linear(prev_dim, 1)

        # Initialize output layer weights and bias to zero for stable start
        nn.init.zeros_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        :param state: Tensor of shape (batch_size, dim_o)
        :param action: Tensor of shape (batch_size, dim_a)
        :return: Q-value tensor of shape (batch_size, 1)
        """
        # Concatenate state and action along last dim
        x = torch.cat((state, action), dim=1)

        # Hidden layers with ReLU then LayerNorm
        for fc, ln in zip(self.fcs, self.lns):
            x = F.relu(fc(x))
            x = ln(x)

        # Output Q-value
        q_value = self.fc_out(x)
        return q_value


# class ActionValueFunctionModel(nn.Module):
#     """
#     PyTorch implementation of the TF Keras action-value function model.
#     Takes state and action inputs, concatenates them, and passes through
#     successive Linear -> ReLU -> LayerNorm layers, ending in a scalar Q-value.
#     """
#     def __init__(
#         self,
#         dim_o: int,
#         dim_a: int,
#         hidden_dims=(512, 512, 512, 256, 256, 256),
#         use_layer_norm: bool = True,
#         activation: nn.Module = nn.ReLU,
#     ):
#         super().__init__()
#         self.dim_o = dim_o
#         self.dim_a = dim_a

#         input_dim = dim_o + dim_a

#         layers = []
#         prev_dim = input_dim
#         for h in hidden_dims:
#             fc = nn.Linear(prev_dim, h)
#             # Orthogonal init with gain ~sqrt(2) (good for ReLU-family)
#             nn.init.orthogonal_(fc.weight, gain=math.sqrt(2))
#             nn.init.zeros_(fc.bias)
#             layers.append(fc)

#             if use_layer_norm:
#                 layers.append(nn.LayerNorm(h))

#             layers.append(activation())
#             prev_dim = h

#         self.backbone = nn.Sequential(*layers)

#         # Output layer: small random init around 0
#         self.fc_out = nn.Linear(prev_dim, 1)
#         nn.init.uniform_(self.fc_out.weight, -1e-3, 1e-3)
#         nn.init.zeros_(self.fc_out.bias)

#     def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
#         x = torch.cat([state, action], dim=-1)
#         # print("x shape: ", x.shape)
#         h = self.backbone(x)
#         q_value = self.fc_out(h)
#         return q_value  # energy = -q_value if you want EBM
    

class UnetEncoder_QModel(nn.Module):
    """
    Q-network that reuses the encoder (down + mid blocks) of a UNet
    to process an H-step action chunk, then collapses temporally
    via purely CNN-based pooling, fuses with a state embedding,
    and outputs a scalar Q-value.
    """
    def __init__(self, 
        input_dim:          int,
        # if you want to pass per-time local conditioning (else leave None)
        local_cond_dim:     int    = None,
        # UNet encoder parameters
        down_dims:          list   = [256, 512, 1024],
        kernel_size:        int    = 3,
        n_groups:           int    = 8,
        cond_predict_scale: bool   = False,
        # intermediate channel dim for collapse
        attn_dim:           int    = 256,
        # state embedding + MLP
        dim_o:              int    = 100,
        state_emb_dim:      int    = 256,
        mlp_dims:           list   = [512, 256],
    ):
        super().__init__()
        self.input_dim = input_dim
        all_dims = [input_dim] + down_dims
        cond_dim = state_emb_dim  # use state embedding as “global cond”

        # ——— local-condition encoder (optional) ———
        self.local_cond_encoder = None
        if local_cond_dim is not None:
            in_dim, out_dim = all_dims[0], all_dims[1]
            self.local_cond_encoder = nn.ModuleList([
                ConditionalResidualBlock1D(
                    local_cond_dim, out_dim, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale
                ),
                ConditionalResidualBlock1D(
                    local_cond_dim, out_dim, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale
                ),
            ])

        # ——— build the down-path ———
        self.down_modules = nn.ModuleList()
        for i, (c_in, c_out) in enumerate(zip(all_dims[:-1], all_dims[1:])):
            is_last = (i == len(all_dims)-2)
            self.down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    c_in, c_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale
                ),
                ConditionalResidualBlock1D(
                    c_out, c_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale
                ),
                Downsample1d(c_out) if not is_last else nn.Identity()
            ]))

        # ——— mid-blocks ———
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            )
            for _ in range(2)
        ])

        # ——— reduce channels for temporal‐CNN pooling ———
        C = down_dims[-1]
        self.dim_reducer   = Conv1dBlock(C, attn_dim, kernel_size=1)
        # self.collapse_conv = Conv1dBlock(attn_dim, attn_dim, kernel_size=3)

        # ——— state embedding + final MLP ———
        self.state_fc  = nn.Linear(dim_o, state_emb_dim)
        self.ln_state  = nn.LayerNorm(state_emb_dim)

        joint_dim     = state_emb_dim + attn_dim
        self.fc1      = nn.Linear(joint_dim, mlp_dims[0])
        self.ln1      = nn.LayerNorm(mlp_dims[0])
        self.fc2      = nn.Linear(mlp_dims[0], mlp_dims[1])
        self.ln2      = nn.LayerNorm(mlp_dims[1])
        self.fc_out   = nn.Linear(mlp_dims[1], 1)
        nn.init.zeros_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

        
        # self._modules

        # print(f"UnetEncoder_QModel parameters: {sum(p.numel() for p in self.parameters()):,}")
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )
        # import pdb; pdb.set_trace()


    def forward(self, 
        state:        torch.Tensor,  # (B, dim_o)
        action_chunk: torch.Tensor,  # (B, H, input_dim)
        local_cond:   torch.Tensor = None
    ) -> torch.Tensor:
        B, H, D = action_chunk.shape
        assert D == self.input_dim, f"Expected action_dim={self.input_dim}, got {D}"

        # ——— 1) global cond from state ———
        s = F.relu(self.state_fc(state))
        s = self.ln_state(s)             # (B, state_emb_dim)
        global_feature = s               # passed into each ResBlock

        # ——— 2) optional local-condition encoding ———
        local_feats = None
        if self.local_cond_encoder and local_cond is not None:
            lc = local_cond.transpose(1, 2)  # (B, local_cond_dim, H)
            down_loc, up_loc = self.local_cond_encoder
            local_feats = [
                down_loc(lc, global_feature),
                up_loc(lc, global_feature)
            ]

        # ——— 3) UNet down-path ———
        x = action_chunk.transpose(1, 2)  # (B, input_dim, H)
        for idx, (res1, res2, downsample) in enumerate(self.down_modules):
            x = res1(x, global_feature)
            if idx == 0 and local_feats is not None:
                x = x + local_feats[0]
            x = res2(x, global_feature)
            x = downsample(x)

        # ——— 4) mid-blocks ———
        for mid in self.mid_modules:
            x = mid(x, global_feature)

        
        # import pdb; pdb.set_trace()
        # x is now (B, C, T′). Reduce channels, then pooling via CNN+max:
        x = self.dim_reducer(x)              # (B, attn_dim, T′)
        # x = F.relu(self.collapse_conv(x))    # (B, attn_dim, T′)
        chunk_emb = F.adaptive_max_pool1d(x, 1).squeeze(-1)  # (B, attn_dim)
        # import pdb; pdb.set_trace()
        # ——— 5) fuse with state and predict Q ———
        j = torch.cat([s, chunk_emb], dim=1)
        j = F.relu(self.fc1(j)); j = self.ln1(j)
        j = F.relu(self.fc2(j)); j = self.ln2(j)
        q = self.fc_out(j)                  # (B, 1)
        return q
    

    """
Energy-Based Transformer (EBT) inspired Q‑function.

This module defines a PyTorch neural network that implements an
energy‑based transformer style architecture for approximating an action‑value
function.  The architecture draws inspiration from the Energy‑Based
Transformer (EBT) proposed by Gladstone et al. while keeping the same
signature as a standard Q‑function: it takes a batch of observations and
actions and outputs a scalar Q‑value for each pair.

Key features:

* Inputs are first projected to a shared hidden dimensionality and treated
  as a short sequence (state token and action token).  A learned positional
  embedding distinguishes the two tokens.
* A stack of EBT‑style blocks processes the sequence.  Each block uses
  multi‑head self‑attention and a feed‑forward network, modulated by
  adaptive layer‑norm (AdaLN) parameters derived from a global context
  vector.  This mirrors the conditioning and gating used in EBTs for
  diffusion models.
* The output Q‑value is predicted from a pooled representation of the
  processed tokens.  Layer normalisation and a linear head map the
  hidden representation to a single scalar per example.

This module is self contained and does not depend on any of the original
EBT training code.  It can be dropped into an existing RL/IL codebase as a
replacement for a simple MLP critic.  To use it as an energy model,
remember that the energy of a state‑action pair is the negative of the
Q‑value returned by this network.
"""



import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply scale and shift modulation.

    The Energy‑Based Transformer uses adaptive layer normalisation where
    the normalised features are scaled and shifted before passing through
    attention or MLP blocks.  This helper performs the modulation in a
    broadcast‑safe way.

    Args:
        x: Normalised features of shape (batch, seq_len, hidden_dim).
        shift: Per‑example shift, shape (batch, hidden_dim).
        scale: Per‑example scale, shape (batch, hidden_dim).

    Returns:
        Modulated features of shape (batch, seq_len, hidden_dim).
    """
    # The shift/scale are of shape (B, H).  Unsqueeze to broadcast over
    # the sequence dimension.  Scale is additive around 1.0 to allow zero
    # initialisation of AdaLN parameters to recover vanilla transformer
    # behaviour at initialisation.
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class EBTBlock(nn.Module):
    """A single Energy‑Based Transformer block.

    Each block consists of layer normalisation, multi‑head self attention,
    and a feed‑forward MLP.  Adaptive layer normalisation parameters
    (shift, scale, and gating factors) are generated from a context vector
    using a small neural network.  See Gladstone et al. (2025) for
    details.
    """

    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.mlp_hidden_dim = int(hidden_dim * mlp_ratio)

        # Normalisation layers.
        self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-5)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-5)

        # Self‑attention.  batch_first=True makes the input shape (B, S, H).
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

        # Feed‑forward network.
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, self.mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.mlp_hidden_dim, hidden_dim),
        )

        # Adaptive layer normalisation modulation.  For each block we
        # produce six hidden_dim‑dimensional vectors: shift_msa,
        # scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp.  The gates
        # control the strength of the residual contribution for attention and
        # MLP separately.
        self.adaln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim, bias=True),
        )

        # Initialise the second linear in adaln to zero so that at
        # initialisation the gates are zero and the block reduces to a
        # standard Transformer block (attention and MLP residuals are
        # initially off).  We follow the practice from the DiT codebase.
        nn.init.constant_(self.adaln[-1].weight, 0.0)
        nn.init.constant_(self.adaln[-1].bias, 0.0)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Apply the block to a sequence.

        Args:
            x: Input sequence of shape (batch, seq_len, hidden_dim).
            context: Context vector of shape (batch, hidden_dim) used
                to compute AdaLN modulation parameters.

        Returns:
            Updated sequence of the same shape as `x`.
        """
        # Generate modulation parameters from context.  We split the
        # 6*hidden_dim output into six pieces along the last dimension.
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaln(context).chunk(6, dim=-1)
        )

        # --- Multi‑head self‑attention ---
        # Apply layer norm to the input.
        x_norm = self.norm1(x)
        # Modulate by shift/scale for the attention branch.
        x_mod = _modulate(x_norm, shift_msa, scale_msa)
        # Self‑attention expects (B, S, H).  We pass query, key and value as x_mod.
        attn_out, _ = self.attn(x_mod, x_mod, x_mod, need_weights=False)
        # Residual connection with gating.  gate_msa has shape (B, H), so we
        # unsqueeze it over the sequence dimension to broadcast.  A SiLU
        # activation on gate_msa could be added, but leaving it linear
        # follows the DiT formulation.
        x = x + gate_msa.unsqueeze(1) * attn_out

        # --- Feed‑forward ---
        x_norm2 = self.norm2(x)
        # Modulate by shift/scale for the MLP branch.
        x_mod2 = _modulate(x_norm2, shift_mlp, scale_mlp)
        mlp_out = self.mlp(x_mod2)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        return x

# import torch.backends.cuda as cuda_back

# # print(">>> PyTorch version:", torch.__version__)

# # Force math SDPA, disable flash/mem-efficient/cudnn
# cuda_back.enable_math_sdp(True)
# cuda_back.enable_flash_sdp(False)
# cuda_back.enable_mem_efficient_sdp(False)
# cuda_back.enable_cudnn_sdp(False)

class EBTQFunctionModel(nn.Module):
    """Energy‑based transformer Q‑function.

    This class implements a Q‑function using a stack of EBT blocks.  The
    network projects the state and action inputs into a shared hidden
    dimension, processes the resulting two‑token sequence with attention
    blocks, and then predicts a scalar Q‑value from the pooled hidden
    representation.

    Args:
        dim_o: Dimensionality of the observation/state vector.
        dim_a: Dimensionality of the action vector.
        hidden_dim: Size of the token embeddings and internal hidden state.
        num_layers: Number of EBT blocks to use.
        num_heads: Number of attention heads in each block.
        mlp_ratio: Expansion factor for the MLP inside each block.
    """

    def __init__(
        self,
        dim_o: int,
        dim_a: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.dim_o = dim_o
        self.dim_a = dim_a
        self.hidden_dim = hidden_dim

        # Linear projections for state and action into a common hidden space.
        self.state_proj = nn.Linear(dim_o, hidden_dim)
        self.action_proj = nn.Linear(dim_a, hidden_dim)

        # Positional embedding to distinguish state and action tokens.  We
        # initialise with zeros so the model learns the relative positions
        # during training.
        self.pos_embed = nn.Parameter(torch.zeros(1, 2, hidden_dim))

        # Stack of EBT blocks.
        self.blocks: List[EBTBlock] = nn.ModuleList(
            [EBTBlock(hidden_dim, num_heads, mlp_ratio) for _ in range(num_layers)]
        )

        # Final layer normalisation and linear head for Q‑value prediction.
        self.norm_final = nn.LayerNorm(hidden_dim, eps=1e-5)
        self.q_head = nn.Linear(hidden_dim, 1)

        # Initialise the linear head weights to small values for stable start.
        nn.init.uniform_(self.q_head.weight, -1e-3, 1e-3)
        nn.init.zeros_(self.q_head.bias)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute Q‑values for a batch of state/action pairs.

        Args:
            state: Tensor of shape (batch_size, dim_o).
            action: Tensor of shape (batch_size, dim_a).

        Returns:
            Tensor of shape (batch_size, 1) containing the Q‑values.
        """
        # Project inputs into hidden dimension.
        s = self.state_proj(state)  # (B, H)
        a = self.action_proj(action)  # (B, H)

        # Form a two‑token sequence (state token followed by action token) and
        # add positional embeddings.
        x = torch.stack([s, a], dim=1)  # (B, 2, H)
        x = x + self.pos_embed

        # Compute a context vector as the mean of the token embeddings.  This
        # simple pooling provides a global representation from which AdaLN
        # modulation parameters are derived in each block.
        context = x.mean(dim=1)  # (B, H)

        # Process through the stack of EBT blocks.
        for block in self.blocks:
            x = block(x, context)

        # Pool the final hidden representations.  We take the mean over the
        # sequence dimension (state/action tokens) but other schemes such as
        # selecting the state token could also be used.
        pooled = x.mean(dim=1)  # (B, H)
        # Apply final normalisation and linear head.
        pooled = self.norm_final(pooled)
        q_value = self.q_head(pooled)  # (B, 1)
        return q_value
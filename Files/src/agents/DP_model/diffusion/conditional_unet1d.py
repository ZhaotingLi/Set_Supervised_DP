from typing import Union
import logging
import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
import torch.nn.functional as F
# from diffusion_policy.model.diffusion.conv1d_components import (
#     Downsample1d, Upsample1d, Conv1dBlock)
# from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb

from agents.DP_model.diffusion.conv1d_components import (
    Downsample1d, Upsample1d, Conv1dBlock)
from agents.DP_model.diffusion.positional_embedding import SinusoidalPosEmb

logger = logging.getLogger(__name__)

class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, 
            in_channels, 
            out_channels, 
            cond_dim,
            kernel_size=3,
            n_groups=8,
            cond_predict_scale=False):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])


        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange('batch t -> batch t 1'),
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        # print("conv1d: x0: ", x.shape)

        out = self.blocks[0](x)   

        # print("conv1d: x1: ",out.shape) # x:(B, in_channels, 1), out: (B, out_channels, 1)

        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.reshape(
                embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:,0,...]
            bias = embed[:,1,...]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out
    
# # ## 2 times faster than conv1d, only works for Ta = 1
# class ConditionalResidualBlock1D(nn.Module):
#     def __init__(self, 
#                  in_channels, 
#                  out_channels, 
#                  cond_dim,
#                  seq_len = 1,
#                  kernel_size = None,
#                  n_groups=8,
#                  cond_predict_scale=False):
#         super().__init__()
        
#         self.seq_len = seq_len
#         self.in_dim = in_channels * seq_len
#         self.out_dim = out_channels * seq_len

#         self.blocks = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(self.in_dim, self.out_dim),
#                 # nn.GroupNorm(n_groups, self.out_dim),
#                 nn.LayerNorm(self.out_dim),
#                 nn.Mish(),
#             ),
#             nn.Sequential(
#                 nn.Linear(self.out_dim, self.out_dim),
#                 # nn.GroupNorm(n_groups, self.out_dim),
#                 nn.LayerNorm(self.out_dim),
#                 nn.Mish(),
#             ),
#         ])

#         cond_channels = out_channels
#         if cond_predict_scale:
#             cond_channels = out_channels * 2

#         self.cond_predict_scale = cond_predict_scale
#         self.out_channels = out_channels

#         # self.cond_encoder = nn.Sequential(
#         #     nn.Linear(cond_dim, cond_channels * seq_len),
#         #     nn.Mish(),
#         # )

#         self.cond_encoder = nn.Sequential(
#             nn.Mish(),
#             nn.Linear(cond_dim, cond_channels * seq_len),
#         )

#         self.residual_linear = nn.Linear(self.in_dim, self.out_dim) \
#             if self.in_dim != self.out_dim else nn.Identity()


#     def forward(self, x, cond):
#         '''
#         x: [batch_size, in_channels, seq_len]
#         cond: [batch_size, cond_dim]

#         returns:
#         out: [batch_size, out_channels, seq_len]
#         '''
#         batch_size = x.size(0)

#         # Flatten temporal and channel dimension
#         x_flat = x.view(batch_size, -1)  # [batch_size, in_channels * seq_len]

#         out = self.blocks[0](x_flat)  # [batch_size, out_channels * seq_len]

#         embed = self.cond_encoder(cond)  # [batch_size, cond_channels * seq_len]

#         if self.cond_predict_scale:
#             embed = embed.view(batch_size, 2, self.out_channels * self.seq_len)
#             scale = embed[:, 0, :]
#             bias = embed[:, 1, :]
#             out = scale * out + bias
#         else:
#             out = out + embed

#         out = self.blocks[1](out)
#         out = out + self.residual_linear(x_flat)

#         # reshape back to [batch_size, out_channels, seq_len]
#         out = out.view(batch_size, self.out_channels, self.seq_len)

#         return out




class ConditionalUnet1D(nn.Module):
    def __init__(self, 
        input_dim,
        local_cond_dim=None,
        global_cond_dim=None,
        # diffusion_step_embed_dim=256,
        diffusion_step_embed_dim=32,
        # down_dims=[256,512,1024],
        down_dims=[128,256,512],
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=False
        ):
        super().__init__()
        self.input_dim = input_dim
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        # diffusion_step_encoder = nn.Sequential(
        #     SinusoidalPosEmb(dsed),
        #     nn.Linear(dsed, dsed * 2),
        #     nn.Mish(),
        #     nn.Linear(dsed * 2, dsed),
        # )

        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 2),
            nn.Mish(),
            nn.Linear(dsed * 2, dsed),
        )
        cond_dim = dsed
        if global_cond_dim is not None:
            cond_dim += global_cond_dim

        
        # self.diffusion_global_cond_encoder = nn.Sequential(
        #     nn.Linear(global_cond_dim, global_cond_dim),
        #     nn.LayerNorm(global_cond_dim),
        #     nn.Mish(),
        #     nn.Linear(global_cond_dim, 512),
        #     # nn.LayerNorm(512),
        #     # nn.Mish(),
        #     # nn.Linear(512, 256),
        #     # nn.Mish(),
        #     # nn.Linear(512, 256),
        # )
        # cond_dim = dsed + 512

        in_out = list(zip(all_dims[:-1], all_dims[1:]))  # [(7, 256), (256, 512), (512, 1024)]

        local_cond_encoder = None
        if local_cond_dim is not None:
            _, dim_out = in_out[0]
            dim_in = local_cond_dim
            local_cond_encoder = nn.ModuleList([
                # down encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                # up encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale)
            ])

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))
        
        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )
        
        # seq_len = 1
        # final_conv = nn.Sequential(
        #     nn.Linear(start_dim * seq_len, start_dim * seq_len),
        #     # nn.GroupNorm(n_groups, start_dim* seq_len),
        #     nn.LayerNorm(start_dim * seq_len),
        #     nn.Mish(),
        #     nn.Linear(start_dim * seq_len, input_dim * seq_len),
        # )


        self.diffusion_step_encoder = diffusion_step_encoder
        self.local_cond_encoder = local_cond_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def forward(self, 
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int], 
            local_cond=None, global_cond=None, **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        local_cond: (B,T,local_cond_dim)
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        sample = einops.rearrange(sample, 'b h t -> b t h')

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)
        # global_cond = self.diffusion_global_cond_encoder(global_cond)
        # if global_cond is not None:
        global_feature = torch.cat([
            global_feature, global_cond
        ], axis=-1)
        
        # encode local features
        h_local = list()
        # if local_cond is not None:
        #     local_cond = einops.rearrange(local_cond, 'b h t -> b t h')
        #     resnet, resnet2 = self.local_cond_encoder
        #     x = resnet(local_cond, global_feature)
        #     h_local.append(x)
        #     x = resnet2(local_cond, global_feature)
        #     h_local.append(x)
        
        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
        # for idx, (resnet, resnet2) in enumerate(self.down_modules):
            # print(idx, " x: ", x.shape)
            x = resnet(x, global_feature)
            # if idx == 0 and len(h_local) > 0:
            #     x = x + h_local[0]
            # print("resnet x: ", x.shape)
            x = resnet2(x, global_feature)
            # print("resnet2 x: ", x.shape)
            h.append(x)
            # x = downsample(x)
            # print("downsample x: ", x.shape)
        # print(" ----------------------------------------")
        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
        # for idx, (resnet, resnet2) in enumerate(self.up_modules):
            h_pop = h.pop()
            # print("h_pop: ", h_pop.shape)
            # print("x: ", x.shape)
            x = torch.cat((x, h_pop), dim=1)
            # print("x after 1: ", x.shape)
            x = resnet(x, global_feature)
            # The correct condition should be:
            # print("len h_local", len(h_local))
            # if idx == (len(self.up_modules)-1) and len(h_local) > 0:
            # # However this change will break compatibility with published checkpoints.
            # # Therefore it is left as a comment.
            # # if idx == len(self.up_modules) and len(h_local) > 0:
            #     x = x + h_local[1]
            x = resnet2(x, global_feature)
            # print("x after resnet2: ", x.shape)
            # x = upsample(x)  # should have this line commented if horizon = 1

            # print("x after upsample: ", x.shape)

        # batch_size = x.size(0)
        # x_flat = x.view(batch_size, -1)
        # x = self.final_conv(x_flat)
        # x = x.view(batch_size, self.input_dim, -1)
        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')
        return x


class HumanFunctionModel(nn.Module):
    def __init__(self, dim_o, dim_a, diffusion_step_embed_dim=32):
        super(HumanFunctionModel, self).__init__()
        self.dim_o = dim_o
        self.dim_a = dim_a

        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 2),
            nn.Mish(),
            nn.Linear(dsed * 2, dsed),
        )

        input_dim = dim_o + dim_a + dsed

        # self.model = nn.Sequential(
        #     nn.Linear(input_dim, 512),
        #     nn.LayerNorm(512),
        #     nn.ReLU(),

        #     nn.Linear(512, 512),
        #     nn.LayerNorm(512),
        #     nn.ReLU(),

        #     nn.Linear(512, 512),  # Additional layer specific to robosuite task
        #     nn.LayerNorm(512),
        #     nn.ReLU(),

        #     nn.Linear(512, 256),
        #     nn.LayerNorm(256),
        #     nn.ReLU(),

        #     nn.Linear(256, dim_a),
        # )

        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.Mish(),

            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.Mish(),

            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.Mish(),

            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.Mish(),

            nn.Linear(512, 512),  # Additional layer specific to robosuite task
            nn.LayerNorm(512),
            nn.Mish(),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.Mish(),

            nn.Linear(256, dim_a),
        )

        # Initialize last layer weights and biases to zero
        # nn.init.zeros_(self.model[-1].weight)
        # nn.init.zeros_(self.model[-1].bias)

    def forward(self, action_input, timestep, local_cond=None, global_cond=None):
        state_input = global_cond
        if state_input is None:
            raise ValueError("HumanFunctionModel requires global_cond as state input.")

        action_shape = action_input.shape
        batch_size = action_shape[0]
        action_input_flat = action_input.reshape(batch_size, -1)
        if action_input_flat.shape[1] != self.dim_a:
            raise ValueError(
                f"HumanFunctionModel expected flattened action dim {self.dim_a}, "
                f"but got action_input shape {tuple(action_shape)} "
                f"with flattened dim {action_input_flat.shape[1]}."
            )

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=action_input.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(action_input.device)
        else:
            timesteps = timesteps.to(action_input.device)
        timesteps = timesteps.expand(batch_size)

        global_feature = self.diffusion_step_encoder(timesteps)

        concat_input = torch.cat([state_input, action_input_flat, global_feature], dim=1)
        output = self.model(concat_input)
        output = output.reshape(action_shape)

        return output
    
import math
class ResidualBlock(nn.Module):
    """A simple fully‑connected residual block with LayerNorm + ReLU."""

    def __init__(self, dim: int):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.lin2 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)

        # Kaiming initialization (fan_in, ReLU) is a good default for residual MLPs
        nn.init.kaiming_uniform_(self.lin1.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lin2.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lin1.bias)
        nn.init.zeros_(self.lin2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.norm1(self.lin1(x)))
        x = self.norm2(self.lin2(x))
        return F.relu(x + residual)  # post‑activation after addition for stability


class HumanFunctionModel_Resnet(nn.Module):
    """
    Re‑implementation of the original HumanFunctionModel where the core MLP has
    been replaced by a stack of residual blocks (a fully‑connected ResNet).
    """

    def __init__(
        self,
        dim_o: int,
        dim_a: int,
        diffusion_step_embed_dim: int = 32,
        hidden_dim: int = 512,
        num_blocks: int = 5,
    ) -> None:
        super().__init__()
        self.dim_o = dim_o
        self.dim_a = dim_a

        # ---------------------------------------------------------------------
        # Diffusion‑step (timestep) encoder ‑ unchanged from original version
        # ---------------------------------------------------------------------
        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 2),
            nn.Mish(),
            nn.Linear(dsed * 2, dsed),
        )

        # ------------------------------
        # Core ResNet architecture
        # ------------------------------
        input_dim = dim_o + dim_a + dsed
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.Mish(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        
        self.res_blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(num_blocks)])
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim + dim_a, 128),
            nn.Mish(),
            nn.Linear(128, dim_a),
        )

        # # Initialize the output layer to zeros to mirror original behaviour
        # nn.init.zeros_(self.output_head[-1].weight)
        # nn.init.zeros_(self.output_head[-1].bias)

    # ---------------------------------------------------------------------
    # Forward pass
    # ---------------------------------------------------------------------
    def forward(
        self,
        action_input,
        timestep,
        local_cond=None, global_cond=None,
    ) -> torch.Tensor:
        # Original implementation collapses local/global conditioning to just
        # `state_input = global_cond`. Keeping the same behaviour for parity.
        state_input = global_cond

        # `action_input` arrives as (B, 1, dim_a); flatten last two dims.
        action_input = action_input.squeeze(1)  # (B, dim_a)

        # Expand t so it matches the batch size expected by the sinusoidal emb.
        t = timestep.expand(action_input.shape[0])
        global_feature = self.diffusion_step_encoder(t)

        # Concatenate [state | action | t‑embedding] then feed through ResNet
        x = torch.cat([state_input, action_input, global_feature], dim=1)
        x = F.relu(self.input_proj(x))
        for block in self.res_blocks:
            x = block(x)

        state_action_emb = x
        state_action_emb_x = torch.cat([action_input, state_action_emb], dim=1)


        out = self.output_head(state_action_emb_x).unsqueeze(1)  # restore (B, 1, dim_a) shape
        return out

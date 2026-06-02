import logging
from typing import Dict, Tuple, Union, List, Optional
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from agents.DP_model.vision.crop_randomizer import CropRandomizer
from agents.DP_model.common.module_attr_mixin import ModuleAttrMixin
from agents.DP_model.common.pytorch_util import replace_submodules
from agents.DP_model.vision.model_getter import get_resnet
from agents.DP_model.vision.multi_image_obs_encoder import ResNet18Conv, SpatialSoftmax

logger = logging.getLogger(__name__)


class MultiImageObsEncoderWithDecoder(ModuleAttrMixin):
    """
    Multi-camera observation encoder with an attached decoder per camera.

    This version enforces a *global bottleneck* z ∈ R^{256} (i.e., [256,1]).
    - Encoder trunk: conv features (B, 512, H', W').
    - Pool + Linear(512→256) to get z (B, 256).
    - z -> Linear(256→C·H'·W') -> reshape to (B, C, H', W') -> existing deconv decoder.
    - Policy can consume z (concat across cameras), controlled by flag.

    Flags:
      - use_global_bottleneck_for_policy: return z (B,256) per camera to the policy.
      - decode_from_global_z: reconstruct by decoding from z (hard bottleneck). If False,
        fall back to decoding from conv feature maps (legacy behavior).
      - add_coord_channels_to_seed: optionally append (x,y) coord channels once at seed.

    All resizing/cropping/normalization behavior mirrors your original.
    """

    def __init__(self,
                 shape_meta: dict,
                 resize_shape: Union[Tuple[int, int], Dict[str, tuple], None] = None,
                 crop_shape: Union[Tuple[int, int], Dict[str, tuple], None] = None,
                 random_crop: bool = True,
                 use_group_norm: bool = False,
                 share_rgb_model: bool = False,
                 imagenet_norm: bool = False,
                 use_spatial_softmax: bool = False,
                 # ---- decoder / AE options ----
                 share_rgb_decoder: bool = False,
                 decoder_channels: Tuple[int, int, int, int, int] = (256, 128, 64, 32, 16),
                 recon_loss: str = 'l2',
                 recon_loss_weight: float = 1.0,
                 perceptual_loss_weight: float = 0.0,
                 detach_encoder_in_ae: bool = False,
                 # ---- bottleneck flags ----
                 use_global_bottleneck_for_policy: bool = True,
                 decode_from_global_z: bool = True,
                 add_coord_channels_to_seed: bool = False,
                 # ---- bottleneck size ----
                 bottleneck_dim: int = 256,   # enforce [256,1],
                 lowdim_embed_dim: int = 32,
                 ):
        super().__init__()

        assert recon_loss in ['l1', 'l2']
        self.use_spatial_softmax = use_spatial_softmax
        self.share_rgb_model = share_rgb_model
        self.imagenet_norm = imagenet_norm
        self.random_crop = random_crop
        self.recon_loss = recon_loss
        self.recon_loss_weight = recon_loss_weight
        self.perceptual_loss_weight = perceptual_loss_weight
        self.detach_encoder_in_ae = detach_encoder_in_ae

        self.use_global_bottleneck_for_policy = use_global_bottleneck_for_policy
        self.decode_from_global_z = decode_from_global_z
        self.add_coord_channels_to_seed = add_coord_channels_to_seed
        self.bottleneck_dim = bottleneck_dim

        # ------------------- Build encoder(s) -------------------
        rgb_model_template = ResNet18Conv(input_channel=3, input_coord_conv=False)

        rgb_keys: List[str] = []
        low_dim_keys: List[str] = []
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map: Dict[str, Tuple[int, ...]] = {}
        key_conv_out_shape: Dict[str, Tuple[int, int, int]] = {}  # (C, H', W')
        key_target_hw: Dict[str, Tuple[int, int]] = {}            # (H, W)

        if share_rgb_model:
            key_model_map['rgb'] = rgb_model_template

        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            typ = attr.get('type', 'low_dim')
            key_shape_map[key] = shape
            if typ == 'rgb':
                rgb_keys.append(key)

                this_model = None
                if not share_rgb_model:
                    if isinstance(rgb_model_template, dict):
                        this_model = rgb_model_template[key]
                    else:
                        this_model = copy.deepcopy(rgb_model_template)

                if this_model is not None:
                    if use_group_norm:
                        this_model = replace_submodules(
                            root_module=this_model,
                            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                            func=lambda x: nn.GroupNorm(
                                num_groups=max(1, x.num_features // 16),
                                num_channels=x.num_features)
                        )
                    key_model_map[key] = this_model

                # resize
                input_shape = shape
                this_resizer = nn.Identity()
                if resize_shape is not None:
                    if isinstance(resize_shape, dict):
                        h, w = resize_shape[key]
                    else:
                        h, w = resize_shape
                    this_resizer = torchvision.transforms.Resize(size=(h, w))
                    input_shape = (shape[0], h, w)

                # crop
                this_randomizer = nn.Identity()
                if crop_shape is not None:
                    if isinstance(crop_shape, dict):
                        ch, cw = crop_shape[key]
                    else:
                        ch, cw = crop_shape
                    if random_crop:
                        this_randomizer = CropRandomizer(
                            input_shape=input_shape,
                            crop_height=ch,
                            crop_width=cw,
                            num_crops=1,
                            pos_enc=False
                        )
                    else:
                        this_randomizer = torchvision.transforms.CenterCrop(size=(ch, cw))
                    key_target_hw[key] = (ch, cw)
                else:
                    key_target_hw[key] = (input_shape[1], input_shape[2])

                # normalize
                this_normalizer = nn.Identity()
                if imagenet_norm:
                    this_normalizer = torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

                key_transform_map[key] = nn.Sequential(this_resizer, this_randomizer, this_normalizer)

                # conv output shape from ResNet18Conv
                c_out, h_out, w_out = ResNet18Conv().output_shape((3, *key_target_hw[key]))
                key_conv_out_shape[key] = (c_out, h_out, w_out)

            elif typ == 'low_dim':
                low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {typ}")

        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)

        # optional spatial softmax head (not used when bottlenecking to z for policy)
        if self.use_spatial_softmax:
            self.spatial_softmax_map = nn.ModuleDict()
            for key in rgb_keys:
                c, h, w = key_conv_out_shape[key]
                self.spatial_softmax_map[key] = SpatialSoftmax(
                    input_shape=(c, h, w),
                    num_kp=32,
                    temperature=1.0,
                    learnable_temperature=False,
                    output_variance=False,
                    noise_std=0.0,
                )
        else:
            self.spatial_softmax_map = None

        self.lowdim_embed_dim = lowdim_embed_dim
        # ---- low-dim projection heads ----
        self.lowdim_proj_map = nn.ModuleDict()
        for key in low_dim_keys:
            in_dim = int(math.prod(key_shape_map[key]))
            self.lowdim_proj_map[key] = nn.Linear(in_dim, self.lowdim_embed_dim)

        # ------------------- Decoders -------------------
        key_decoder_map = nn.ModuleDict()
        shared_decoder = None
        for key in rgb_keys:
            c, h, w = key_conv_out_shape[key]
            target_h, target_w = key_target_hw[key]
            if share_rgb_decoder and shared_decoder is not None:
                key_decoder_map[key] = shared_decoder
                continue
            decoder = self._make_decoder(c, h, w, target_h, target_w, decoder_channels)
            if share_rgb_decoder:
                shared_decoder = decoder
            key_decoder_map[key] = decoder

        # ------------------- Bottleneck heads -------------------
        # 512 (pooled) -> 256 bottleneck
        self.encoder_to_bottleneck = nn.Linear(512, self.bottleneck_dim)

        # For each RGB key, z(256) -> seed map (C*H'*W')
        z2seed_map = nn.ModuleDict()
        for key in rgb_keys:
            c, h, w = key_conv_out_shape[key]
            flat_hw = c * h * w
            z2seed_map[key] = nn.Sequential(
                nn.Linear(self.bottleneck_dim, flat_hw),
                nn.ReLU(inplace=True),
            )

        # ------------------- Optional perceptual head -------------------
        if self.perceptual_loss_weight > 0.0:
            vgg = torchvision.models.vgg11(weights=None)
            self.perc_feat_extractor = nn.Sequential(*list(vgg.features.children())[:9]).eval()
            for p in self.perc_feat_extractor.parameters():
                p.requires_grad = False
        else:
            self.perc_feat_extractor = None

        # keep refs
        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.key_shape_map = key_shape_map
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.key_conv_out_shape = key_conv_out_shape
        self.key_target_hw = key_target_hw
        self.key_decoder_map = key_decoder_map
        self.z2seed_map = z2seed_map

    # ------------------------- Public API -------------------------
    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = None
        features = []

        if self.share_rgb_model:
            imgs = []
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert img.shape[1:] == self.key_shape_map[key]
                enc_in = self.key_transform_map[key](img)
                imgs.append(enc_in)

            if len(imgs) > 0:
                imgs_cat = torch.cat(imgs, dim=0)
                conv = self.key_model_map['rgb'](imgs_cat)  # (N*B,512,H',W')
                conv_per_key = torch.chunk(conv, chunks=len(self.rgb_keys), dim=0)
                for key, conv_feat in zip(self.rgb_keys, conv_per_key):
                    if self.use_spatial_softmax:
                        kps = self.spatial_softmax_map[key](conv_feat)
                        features.append(kps.reshape(batch_size, -1))
                    else:
                        if self.use_global_bottleneck_for_policy:
                            z = self._pool_and_project(conv_feat)  # (B,256)
                            features.append(z)
                        else:
                            features.append(conv_feat.reshape(batch_size, -1))
        else:
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert img.shape[1:] == self.key_shape_map[key]
                enc_in = self.key_transform_map[key](img)
                conv_feat = self.key_model_map[key](enc_in)  # (B,512,H',W')
                if self.use_spatial_softmax:
                    kps = self.spatial_softmax_map[key](conv_feat)
                    features.append(kps.reshape(batch_size, -1))
                else:
                    if self.use_global_bottleneck_for_policy:
                        z = self._pool_and_project(conv_feat)  # (B,256)
                        features.append(z)
                    else:
                        features.append(conv_feat.reshape(batch_size, -1))

        # for key in self.low_dim_keys:
        #     data = obs_dict[key]
        #     if batch_size is None:
        #         batch_size = data.shape[0]
        #     else:
        #         assert batch_size == data.shape[0]
        #     assert data.shape[1:] == self.key_shape_map[key]
        #     features.append(data)

        for key in self.low_dim_keys:
            data = obs_dict[key]
            if batch_size is None:
                batch_size = data.shape[0]
            else:
                assert batch_size == data.shape[0]
            assert data.shape[1:] == self.key_shape_map[key]
            flat = data.view(batch_size, -1)
            embed = self.lowdim_proj_map[key](flat)
            features.append(embed)

        return torch.cat(features, dim=-1) if len(features) > 0 else torch.empty(batch_size, 0, device=self.device)

    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = {}
        batch_size = 1
        for key, attr in self.shape_meta['obs'].items():
            shape = tuple(attr['shape'])
            example_obs_dict[key] = torch.zeros((batch_size,) + shape, dtype=self.dtype, device=self.device)
        example_output = self.forward(example_obs_dict)
        return example_output.shape[1:]

    # ----------------- Autoencoder utilities / training -----------------
    def compute_autoencoder_loss(self,
                                 obs_dict: Dict[str, torch.Tensor],
                                 reduction: str = 'mean',
                                 return_per_key: bool = True,
                                 ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        assert reduction in ['mean', 'sum']
        device = self.device
        total = torch.zeros((), device=device)
        per_key_losses: Dict[str, torch.Tensor] = {}

        if len(self.rgb_keys) == 0:
            return total if not return_per_key else (total, per_key_losses)

        if self.share_rgb_model:
            imgs = []
            enc_inputs = []
            for key in self.rgb_keys:
                x = obs_dict[key]
                x_t = self.key_transform_map[key](x)
                imgs.append(x_t)
                enc_inputs.append(x_t)
            imgs_cat = torch.cat(imgs, dim=0)
            conv = self.key_model_map['rgb'](imgs_cat)
            conv_per_key = torch.chunk(conv, chunks=len(self.rgb_keys), dim=0)
            for key, conv_feat, x_t in zip(self.rgb_keys, conv_per_key, enc_inputs):
                loss_k = self._recon_loss_single_key(key, conv_feat, x_t, reduction)
                per_key_losses[key] = loss_k.detach()
                total = total + loss_k
        else:
            for key in self.rgb_keys:
                x = obs_dict[key]
                x_t = self.key_transform_map[key](x)
                conv_feat = self.key_model_map[key](x_t)
                loss_k = self._recon_loss_single_key(key, conv_feat, x_t, reduction)
                per_key_losses[key] = loss_k.detach()
                total = total + loss_k

        return total if not return_per_key else (total, per_key_losses)

    @torch.no_grad()
    def reconstruct(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        recons: Dict[str, torch.Tensor] = {}
        if len(self.rgb_keys) == 0:
            return recons

        if self.share_rgb_model:
            imgs = []
            for key in self.rgb_keys:
                imgs.append(self.key_transform_map[key](obs_dict[key]))
            imgs_cat = torch.cat(imgs, dim=0)
            conv = self.key_model_map['rgb'](imgs_cat)
            conv_per_key = torch.chunk(conv, chunks=len(self.rgb_keys), dim=0)
            for key, conv_feat in zip(self.rgb_keys, conv_per_key):
                recons[key] = self._decode_key(key, conv_feat)
        else:
            for key in self.rgb_keys:
                x_t = self.key_transform_map[key](obs_dict[key])
                conv_feat = self.key_model_map[key](x_t)
                recons[key] = self._decode_key(key, conv_feat)
        return recons

    # ---------------------------- Internals ----------------------------
    def _make_decoder(self,
                      in_c: int,
                      in_h: int,
                      in_w: int,
                      target_h: int,
                      target_w: int,
                      channels: Tuple[int, int, int, int, int]) -> nn.Module:
        layers: List[nn.Module] = []
        c = in_c
        h, w = in_h, in_w
        for ch in channels:
            layers += [
                nn.ConvTranspose2d(c, ch, kernel_size=4, stride=2, padding=1, bias=True),
                nn.ReLU(inplace=True),
            ]
            c = ch
            h *= 2
            w *= 2
        layers.append(nn.Conv2d(c, 3, kernel_size=3, padding=1, bias=True))
        decoder = nn.Sequential(*layers)

        class _Decoder(nn.Module):
            def __init__(self, dec: nn.Module, tgt_hw: Tuple[int, int]):
                super().__init__()
                self.dec = dec
                self.tgt_hw = tgt_hw
            def forward(self, zmap):
                y = self.dec(zmap)
                if y.shape[-2:] != self.tgt_hw:
                    y = F.interpolate(y, size=self.tgt_hw, mode='bilinear', align_corners=False)
                return y
        return _Decoder(decoder, (target_h, target_w))

    def _maybe_add_coords(self, seed: torch.Tensor) -> torch.Tensor:
        if not self.add_coord_channels_to_seed:
            return seed
        B, _, H, W = seed.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=seed.device, dtype=seed.dtype),
            torch.linspace(-1, 1, W, device=seed.device, dtype=seed.dtype),
            indexing='ij'
        )
        coords = torch.stack([yy, xx], dim=0).unsqueeze(0).expand(B, -1, -1, -1)  # (B,2,H,W)
        return torch.cat([seed, coords], dim=1)

    def _pool_and_project(self, conv_feat: torch.Tensor) -> torch.Tensor:
        """Pool conv_feat -> [B,512], then project -> [B,256]."""
        pooled = F.adaptive_avg_pool2d(conv_feat, output_size=(1, 1)).flatten(1)  # [B,512]
        z = self.encoder_to_bottleneck(pooled)  # [B,256]
        return z

    def _decode_key(self, key: str, conv_feat: torch.Tensor) -> torch.Tensor:
        """Decode to encoder-input space.
        If decode_from_global_z=True, use z (B,256) -> seed -> decoder.
        Else, decode from conv feature map directly (legacy).
        """
        if self.detach_encoder_in_ae:
            conv_feat = conv_feat.detach()

        if self.decode_from_global_z and not self.use_spatial_softmax:
            z = self._pool_and_project(conv_feat)  # (B,256)
            c, h, w = self.key_conv_out_shape[key]
            seed = self.z2seed_map[key](z).view(-1, c, h, w)
            seed = self._maybe_add_coords(seed)
            recon = self.key_decoder_map[key](seed)
            return recon
        else:
            return self.key_decoder_map[key](conv_feat)

    def _recon_loss_single_key(self,
                               key: str,
                               conv_feat: torch.Tensor,
                               target_enc_input: torch.Tensor,
                               reduction: str) -> torch.Tensor:
        recon = self._decode_key(key, conv_feat)
        if self.recon_loss == 'l1':
            pix = F.l1_loss(recon, target_enc_input, reduction=reduction)
        else:
            pix = F.mse_loss(recon, target_enc_input, reduction=reduction)

        if self.perceptual_loss_weight > 0.0 and self.perc_feat_extractor is not None:
            x = target_enc_input
            y = recon
            if self.imagenet_norm:
                mean = torch.as_tensor([0.485, 0.456, 0.406], device=x.device)[None, :, None, None]
                std = torch.as_tensor([0.229, 0.224, 0.225], device=x.device)[None, :, None, None]
                x = x * std + mean
                y = y * std + mean
            with torch.no_grad():
                self.perc_feat_extractor.eval()
            fx = self.perc_feat_extractor(x)
            fy = self.perc_feat_extractor(y)
            perc = F.l1_loss(fy, fx, reduction=reduction)
            return pix * self.recon_loss_weight + perc * self.perceptual_loss_weight
        else:
            return pix * self.recon_loss_weight


# ------------------------------
# Smoke test harness
# ------------------------------
if __name__ == '__main__':
    torch.manual_seed(0)

    def make_dummy_obs(shape_meta, batch_size=2, device='cpu', dtype=torch.float32):
        obs = {}
        for key, attr in shape_meta['obs'].items():
            shape = tuple(attr['shape'])
            if attr.get('type', 'low_dim') == 'rgb':
                obs[key] = torch.rand((batch_size,) + shape, device=device, dtype=dtype)
            else:
                obs[key] = torch.randn((batch_size,) + shape, device=device, dtype=dtype)
        return obs

    def run_smoke_tests(encoder, shape_meta, name='encoder', batch_size=32, lr=1e-3, do_backward=True):
        logger.info(f'\n===== Running tests: {name} =====')
        device = next(encoder.parameters()).device
        dtype = next(encoder.parameters()).dtype

        obs = make_dummy_obs(shape_meta, batch_size=batch_size, device=device, dtype=dtype)

        feats = encoder(obs)
        logger.debug(f'[forward] features shape: {tuple(feats.shape)}')

        out_shape = encoder.output_shape()
        logger.debug(f'[output_shape] reported: {tuple(out_shape)} | actual: {tuple(feats.shape[1:])}')

        recons = encoder.reconstruct(obs)
        if len(recons) == 0:
            logger.info('[reconstruct] no RGB keys detected, nothing to reconstruct.')
        else:
            for k, r in recons.items():
                mn, mx = r.min().item(), r.max().item()
                logger.debug(f'[reconstruct] {k}: shape={tuple(r.shape)} value_range=({mn:.3f},{mx:.3f})')

        ae = encoder.compute_autoencoder_loss(obs, reduction='mean', return_per_key=True)
        total_loss, per_key = ae
        logger.debug(f'[ae loss] total={total_loss.item():.6f}')
        for k, v in per_key.items():
            logger.info(f'           {k}={v.item():.6f}')

        if do_backward and total_loss.requires_grad and any(p.requires_grad for p in encoder.parameters()):
            opt = torch.optim.Adam(encoder.parameters(), lr=lr)
            opt.zero_grad(set_to_none=True)
            total_loss.backward()
            total_norm = 0.0
            for p in encoder.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2).item()
                    total_norm += param_norm ** 2
            total_norm = math.sqrt(total_norm)
            logger.debug(f'[backward] total grad L2 norm: {total_norm:.6f}')
            opt.step()

        logger.info('===== Done =====\n')

    shape_meta = {
        "obs": {
            "agentview_image": {"shape": [3, 240, 320], "type": "rgb"},
            "robot0_eye_in_hand_image": {"shape": [3, 240, 320], "type": "rgb"},
            "robot0_eef_pos": {"shape": [3]},
            "robot0_eef_quat": {"shape": [4]},
            "robot0_gripper_qpos": {"shape": [2]}
        },
        "action": {"shape": [7]}
    }
    obs_encoder_crop_shape = [216, 288]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32

    # Enforce 256-d bottleneck end-to-end
    obs_encoder = MultiImageObsEncoderWithDecoder(
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
    ).to(device=device, dtype=dtype)

    run_smoke_tests(obs_encoder, shape_meta, name='z256→seed→decoder (policy uses z256)')
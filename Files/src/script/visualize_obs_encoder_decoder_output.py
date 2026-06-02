# === Visualize decoder output for MultiImageObsEncoderWithDecoder =================
import argparse
import os, sys
import math
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
)
import torch
import torch.nn.functional as F

# --- your project imports ---
from tools.buffer import HDF5Buffer, Buffer_uniform_sampling
from agents.DP_model.vision.multi_image_obs_encoder_with_decoder import MultiImageObsEncoderWithDecoder
from agents.DP_model.vision.multi_image_obs_encoder_with_decoder_no_transformer import MultiImageObsEncoderWithDecoder as MultiImageObsEncoderWithDecoder_no_transformer
from agents.Set_Supervised_diffusion_policy_image import collate_obs_dict
from agents.DP_model.common.pytorch_util import dict_apply



# ------------------ helpers ------------------
def _ensure_chw_last_frame(x_np, device="cuda:0"):
    """
    Accepts image arrays with shape:
      - (stack, 3, H, W) or (3, H, W) or (H, W, 3)
    Returns torch.Tensor of shape (1, 3, H, W), preserving the original numeric range.
    """
    img = x_np
    # stacked frames -> take last
    if img.ndim == 4:
        img = img[-1]
    # (H,W,3) -> (3,H,W)
    if img.ndim == 3 and img.shape[-1] == 3 and img.shape[0] != 3:
        img = np.transpose(img, (2, 0, 1))
    assert img.ndim == 3 and img.shape[0] in (1, 3), f"Unexpected image shape: {img.shape}"
    t = torch.from_numpy(img).float().unsqueeze(0).to(device)  # (1,3,H,W)
    return t

def _prep_for_display(x_3chw, imagenet_norm=False, device="cuda:0"):
    """
    x_3chw: torch (3,H,W) in encoder-input space. Returns HWC float in [0,1] for plotting.
    """
    x = x_3chw.detach()
    if imagenet_norm:
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device)[:, None, None]
        std  = torch.tensor([0.229, 0.224, 0.225], device=x.device)[:, None, None]
        x = x * std + mean
    # If data were stored in [-1,1], bring to [0,1] only for display
    if torch.min(x) < 0:
        x = (x + 1.0) / 2.0
    x = x.clamp(0, 1)
    x = x[[2, 1, 0], :, :] # Rgb -> bgr
    # return x.permute(1, 2, 0).cpu().numpy()  # HWC
    # print("x shape: ", x.shape)
    return x.permute(1, 2, 0).cpu().numpy()

def _last_frame_to_hwc_uint8(x_np):
    """
    Return last frame as HWC uint8 for plotting the raw image and drawing a crop box.
    Accepts (stack,3,H,W) or (3,H,W) or (H,W,3). If values are float in [-1,1] or [0,1],
    it scales to [0,255]. If already uint8, returns as-is.
    """
    img = x_np
    if img.ndim == 4:
        img = img[-1]
    if img.ndim == 3 and img.shape[-1] == 3 and img.shape[0] != 3:
        hwc = img
    else:
        # (3,H,W) -> (H,W,3)
        hwc = np.transpose(img, (1, 2, 0))
    # to uint8 for consistent display
    if hwc.dtype != np.uint8:
        arr = hwc.astype(np.float32)
        # try to infer range
        if arr.min() >= -1.0 and arr.max() <= 1.0:
            arr = (arr + 1.0) * 127.5
        elif arr.min() >= 0.0 and arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255)
        hwc = arr.astype(np.uint8)
    return hwc

def _random_crop_coords(h, w, crop_h, crop_w):
    """
    Return (y0, x0, y1, x1) for a random crop box.
    Ensures the box stays inside the image.
    """
    if h < crop_h or w < crop_w:
        # fallback: clamp to image if crop is bigger (shouldn't happen with your shapes)
        y0, x0 = 0, 0
    else:
        y0 = np.random.randint(0, h - crop_h + 1)
        x0 = np.random.randint(0, w - crop_w + 1)
    y1 = min(h, y0 + crop_h)
    x1 = min(w, x0 + crop_w)
    return y0, x0, y1, x1

def visualize_reconstruction_for_indices(state_batch, indices, obs_encoder, rows=2, device="cuda:0", n_obs_steps = 2, save_dir=None):
    import matplotlib.patches as patches
    """
    Now shows 3 columns per camera:
      1) Raw original (last frame) with the crop box overlayed
      2) Cropped patch (pre-normalization)
      3) Reconstruction (decoder output)
    """
    rgb_keys = list(obs_encoder.rgb_keys)
    assert len(rgb_keys) >= 1, "No RGB keys found in encoder."

    # Pull target crop shape from the encoder config
    crop_h, crop_w = obs_encoder.crop_shape if hasattr(obs_encoder, "crop_shape") else (228, 304)

    for idx in indices:
        obs_np = state_batch[idx]  # dict of numpy arrays
        obs_1 = {}

        # RGB inputs -> tensors for encode/recon
        for k in rgb_keys:
            if k not in obs_np:
                raise KeyError(f"Missing key '{k}' in observation at index {idx}")
            obs_1[k] = _ensure_chw_last_frame(obs_np[k])

        # Low-dim keys
        for k in obs_encoder.low_dim_keys:
            shape = obs_encoder.key_shape_map[k]
            if k in obs_np:
                arr = obs_np[k]
                if arr.ndim >= 2 and arr.shape[0] == n_obs_steps:
                    arr = arr[-1]
                obs_1[k] = torch.from_numpy(arr).float().unsqueeze(0).to(device)
            else:
                obs_1[k] = torch.zeros((1, *shape), device=device)

        # Reconstructions
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            recons = obs_encoder.reconstruct(obs_1)  # dict: key -> (1,3,h,w)

        # Plot per camera
        n_rows = len(rgb_keys)
        fig, axes = plt.subplots(n_rows, 3, figsize=(12, 3.8 * n_rows))
        if n_rows == 1:
            axes = np.array([axes])

        for r, k in enumerate(rgb_keys):
            # --- (1) Raw original with crop box ---
            raw_hwc = _last_frame_to_hwc_uint8(obs_np[k])
            H, W = raw_hwc.shape[:2]
            y0, x0, y1, x1 = _random_crop_coords(H, W, crop_h, crop_w)

            raw_hwc = raw_hwc[:, :, [2, 1, 0]] # Rgb -> bgr
            axes[r, 0].imshow(raw_hwc)
            rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                     linewidth=2, edgecolor='r', facecolor='none')
            axes[r, 0].add_patch(rect)
            axes[r, 0].set_title(f"{k} — raw (crop box)")
            axes[r, 0].axis("off")

            # --- (2) Cropped patch (pre-normalization) ---
            cropped_patch = raw_hwc[y0:y1, x0:x1]
            axes[r, 1].imshow(cropped_patch)
            axes[r, 1].set_title(f"{k} — cropped {crop_h}×{crop_w}")
            axes[r, 1].axis("off")

            # --- (3) Reconstruction ---
            rec_hwc = _prep_for_display(recons[k][0], imagenet_norm=obs_encoder.imagenet_norm)
            axes[r, 2].imshow(rec_hwc)
            axes[r, 2].set_title(f"{k} — reconstruction")
            axes[r, 2].axis("off")

        plt.suptitle(f"Sample index {idx}")
        plt.tight_layout()

        if save_dir is not None:
            save_path = os.path.join(save_dir, f"reconstruction_{idx}.png")
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            print(f"Saved reconstruction figure to: {save_path}")
            plt.close(fig)
        else:
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize decoder reconstructions from an HDF5 replay buffer."
    )
    parser.add_argument("buffer_path", help="Path to the HDF5 replay buffer.")
    parser.add_argument(
        "--obs-enc-dir",
        default=None,
        help="Optional directory containing obs_encoder.pth.",
    )
    args = parser.parse_args()

    # ------------------ config / paths ------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Buffer file (contains observations to visualize)
    
    # buffer_path = "${BD_COACH_SRC_ROOT}/outputs_docker/data_buffer_demo30_interventionNov20_franka.h5"
    # buffer_path = "${BD_COACH_SRC_ROOT}/outputs_docker/data_buffer_Nov19_demo30_franka_shuffled.h5"
    
    # buffer_path = "${HOME}/outputs/trajectory_buffer_0929_InsertBlueT_GreenU_demonstration_73trajs_0930_Intervention_1002_Inter_trajs60_Uint8_shuffled.h5"
    # buffer_path = "${HOME}/outputs/1002/trajectory_buffer_1002_intervention_eps60.h5"
    # buffer_path = "${BD_COACH_SRC_ROOT}/outputs_docker/trajectory_buffer_0929_InsertBlueT_GreenU_demonstration_73trajs_0930_Intervention_1002_Inter_trajs60_Uint8_shuffled.h5"
    # buffer_path = "${BD_COACH_SRC_ROOT}/outputs_docker/trajectory_buffer_0929_InsertBlueT_GreenU_demonstration_73trajs_0930_Intervention.h5"
    # buffer_path = '${BD_COACH_SRC_ROOT}/outputs_docker/data_buffer_Nov10_demo_noisy.h5'
    # buffer_path_hdf5 = '${HOME}/outputs/1006/trajectory_buffer_test_1006_CLIC_DP_SA32_eps194.hdf5'
    # buffer_path = '${HOME}/outputs/1006/trajectory_buffer_test_1006_CLIC_DP_SA32_eps194.h5'
    # buffer_path = '${HOME}/outputs/1006/trajectory_buffer_1006_intervention01_trajs15.h5'
    # Optional trained obs-encoder checkpoint
    # obs_enc_dir = "${BD_COACH_SRC_ROOT}/outputs_docker/20251003_100154_Diffusion_CLIC_intervention_Circular_kuka-pushT-img_Ta16_offlineTrue_Scale0.05/network_params/"
    # obs_enc_dir = '${BD_COACH_SRC_ROOT}/outputs/2025-11-20/Nov20_franka_22-32-58/saved_data/network_params/'
    # obs_enc_dir = '${BD_COACH_SRC_ROOT}/outputs/2025-11-24/23-05-13_franka/saved_data/network_params/'
    
    buffer_path = os.path.expanduser(os.path.expandvars(args.buffer_path))
    obs_enc_dir = (
        os.path.expanduser(os.path.expandvars(args.obs_enc_dir))
        if args.obs_enc_dir is not None
        else None
    )
    # obs_enc_dir = "${BD_COACH_SRC_ROOT}/outputs_docker/2025_1111_CLIC_DP/network_params/"

    obs_enc_path = os.path.join(obs_enc_dir, "obs_encoder.pth") if obs_enc_dir else None

    # Observation schema used by the model
    n_obs_steps = 2
    shape_meta = {
        "obs": {
            "image1": {"shape": [3, 240, 320], "type": "rgb"},
            "image2": {"shape": [3, 240, 320], "type": "rgb"},
            "robot0_eef_pos_vel": {"shape": [4]},
        },
        "action": {"shape": [2]},
    }

    field_shapes = {
        "image1": (n_obs_steps, 3, 240, 320),
        "image2": (n_obs_steps, 3, 240, 320),
        "robot0_eef_pos_vel": (n_obs_steps, 4),
        "teacher_action": (2,),
        "robot_action": (2,),
    }

    # shape_meta = {
    #     "obs": {
    #         "image1": {"shape": [3, 240, 320], "type": "rgb"},
    #         "image2": {"shape": [3, 240, 320], "type": "rgb"},
    #         "robot0_eef_pos_vel": {"shape": [8]},
    #     },
    #     "action": {"shape": [10]},
    # }

    
    # field_shapes = {
    #     "image1": (n_obs_steps, 3, 240, 320),
    #     "image2": (n_obs_steps, 3, 240, 320),
    #     "robot0_eef_pos_vel": (n_obs_steps, 8),
    #     "teacher_action": (10,),
    #     "robot_action": (10,),
    # }

    # ------------------ build the obs encoder ------------------
    obs_encoder = MultiImageObsEncoderWithDecoder_no_transformer(
        shape_meta=shape_meta,
        resize_shape=None,
        crop_shape=[228, 304],
        random_crop=True,                  # eval() will make crop deterministic in typical implementations
        use_group_norm=True,
        share_rgb_model=False,
        imagenet_norm=False,
        use_spatial_softmax=False,         # bottleneck z is used for policy; decoder decodes from z
        use_global_bottleneck_for_policy=True,
        decode_from_global_z=True,
        add_coord_channels_to_seed=False,
        bottleneck_dim=256,
    ).to(device)
    # obs_encoder = MultiImageObsEncoderWithDecoder(
    #                 shape_meta=shape_meta,
    #                 resize_shape=None,
    #                 # crop_shape=[216, 288],
    #                 crop_shape=[228, 304],
    #                 random_crop=True,
    #                 use_group_norm=True,
    #                 share_rgb_model=False,
    #                 imagenet_norm=False,
    #                 use_spatial_softmax=False,
    #                 use_global_bottleneck_for_policy=True,   # policy tokens are z: (B,256) per camera
    #                 decode_from_global_z=True,               # recon from z -> seed -> decoder
    #                 add_coord_channels_to_seed=False,
    #                 bottleneck_dim=256,
    #                 # Transformer fusion settings
    #                 use_transformer_fusion=True,
    #                 # use_transformer_fusion=False,
    #                 # d_model=256,
    #                 d_model=512,
    #                 nhead=8,
    #                 num_transformer_layers=6,
    #                 dim_feedforward=512,
    #                 transformer_dropout=0.1,
    #                 add_cls_token=True,
    #                 fusion_output_mode="cls",
    #             ).to(device)

    # Optional: load trained weights
    if obs_enc_path and os.path.isfile(obs_enc_path):
        ckpt = torch.load(obs_enc_path, map_location=device)
        if "obs_encoder_state_dict" in ckpt:
            obs_encoder.load_state_dict(ckpt["obs_encoder_state_dict"], strict=True)
            print(f"[OK] Loaded encoder weights from {obs_enc_path}")
        else:
            print(f"[WARN] 'obs_encoder_state_dict' not found in {obs_enc_path}; skipping.")
    elif obs_enc_path:
        print(f"[INFO] No checkpoint at {obs_enc_path}; using fresh weights.")
    else:
        print("[INFO] No --obs-enc-dir provided; using fresh weights.")

    obs_encoder.eval()  # disable dropout / randomization where applicable

    # ------------------ data: sample a batch from buffer ------------------
   
    # buffer = HDF5Buffer(
    #     filename='buffer.h5',
    #     field_shapes=field_shapes,
    #     min_size=32,
    #     max_size=50000,
    #     dtype_map={},
    # )
    buffer = HDF5Buffer(
        filename='buffer.h5',
        field_shapes=field_shapes,
        min_size=32,
        max_size=50000,
        dtype_map={"image1": "uint8", "image2": "uint8",},
        image_saved_in_Uint8=True
    )
    buffer.load_from_file(buffer_path, read_only=True)  # load from h5
    # buffer.ingest_trajectory_hdf5(buffer_path_hdf5, skip_no_teacher_action = False)  # load from traj buffer hdf5

    # buffer = Buffer_uniform_sampling(min_size=32, max_size=50000 )  # selfplay data 
    # buffer.load_from_h5_buffer_file(buffer_path)

    batch_size = 80
    # sampled = buffer.sample_randomly(batch_size)  # list of (obs_dict, human_action, robot_action)
    sampled = buffer.sample(batch_size)  # list of (obs_dict, human_action, robot_action)
    sampled = buffer.sample(batch_size)  # list of (obs_dict, human_action, robot_action)
    state_batch = [pair[0] for pair in sampled]
    teacher_action_batch = [pair[1] for pair in sampled]
    robot_action_batch = [pair[2] for pair in sampled]

    import pdb; pdb.set_trace()
    # ------------------ run the visualization ------------------
    # Visualize the first few samples in the batch (change indices as you like)
    # indices_to_show = list(range(min(8, len(state_batch))))
    indices_to_show = list(range(0, len(state_batch), 5))
    visualize_reconstruction_for_indices(state_batch, indices_to_show, obs_encoder, save_dir='outputs/decoder_image')

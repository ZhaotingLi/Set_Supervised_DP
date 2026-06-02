import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys, os
import time
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
)
from buffer_trajectory import TrajectoryBuffer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize buffer samples and optional observation encoder features."
    )
    parser.add_argument("buffer_path", help="Path to the replay buffer file.")
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Optional directory containing network_params/obs_encoder.pth or obs_encoder.pth.",
    )
    return parser.parse_args()


args = parse_args()



def visualize_sampled_batch(
        sampled_batch,
        img_keys=("agentview_image", "robot0_eye_in_hand_image"),
        cols=8,
        normalize_if_needed=True,
):
    """
    Show the first N samples of `sampled_batch` in a grid for each image key.

    Parameters
    ----------
    sampled_batch : list[tuple]
        Output of HDF5Buffer.sample(); each element is (obs_dict, act_r, act_t).
    img_keys : str | list[str]
        Which observation keys to visualize.  Provide one or several.  If you
        give a single str it's turned into a 1-elem list automatically.
    cols : int
        Number of columns in the grid.  Rows are computed automatically.
    normalize_if_needed : bool
        If True, images stored in [-1,1] range are linearly mapped to [0,1].
    """
    if isinstance(img_keys, str):
        img_keys = [img_keys]

    n = len(sampled_batch)
    import math
    rows = math.ceil(n / cols)

    for k in img_keys:
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.0))
        axes = np.atleast_1d(axes).reshape(rows, cols)

        for ax in axes.flat:
            ax.axis("off")     # blank them first

        for i, (obs, _, _) in enumerate(sampled_batch):
            if k not in obs:
                raise KeyError(f"Key '{k}' not found in obs[{i}]")

            img = obs[k]

            # ─── handle stacked frames / channel-first ─────────────────────────
            if img.ndim == 4:          # (stack, C, H, W) or (stack, H, W, C)
                img = img[1]           # take the last frame
            if img.ndim == 3 and img.shape[0] in (1, 3):  # CHW → HWC
                img = np.transpose(img, (1, 2, 0))

            # ─── normalise if stored in [-1,1] floats ─────────────────────────
            if normalize_if_needed and img.dtype.kind == "f" and img.min() < 0:
                img = (img + 1.0) / 2.0
            img = np.clip(img, 0.0, 1.0)

            r, c = divmod(i, cols)
            axes[r, c].imshow(img)
            axes[r, c].set_title(f"{k}\nidx {i}", fontsize=8)
            axes[r, c].axis("off")

        plt.suptitle(f"{k} — {n} samples")
        plt.tight_layout()
        plt.show()

def visualize_sampled_batch_with_obs_feature(
        sampled_batch,
        obs_feature, # (batch, obs_feature_dim)
        img_keys=("agentview_image", "robot0_eye_in_hand_image"), crop_shape = [72, 72],
        cols=8,
        normalize_if_needed=True,
):
    """
    Show the first N samples of `sampled_batch` in a grid for each image key.

    Parameters
    ----------
    sampled_batch : list[tuple]
        Output of HDF5Buffer.sample(); each element is (obs_dict, act_r, act_t).
    img_keys : str | list[str]
        Which observation keys to visualize.  Provide one or several.  If you
        give a single str it's turned into a 1-elem list automatically.
    cols : int
        Number of columns in the grid.  Rows are computed automatically.
    normalize_if_needed : bool
        If True, images stored in [-1,1] range are linearly mapped to [0,1].
    """
    if isinstance(img_keys, str):
        img_keys = [img_keys]

    n = len(sampled_batch)
    import math
    rows = math.ceil(n / cols)

    for k in img_keys:
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.0))
        axes = np.atleast_1d(axes).reshape(rows, cols)

        for ax in axes.flat:
            ax.axis("off")     # blank them first

        for i, (obs, _, _) in enumerate(sampled_batch):
            if k not in obs:
                raise KeyError(f"Key '{k}' not found in obs[{i}]")

            img = obs[k]

            # ─── handle stacked frames / channel-first ─────────────────────────
            if img.ndim == 4:          # (stack, C, H, W) or (stack, H, W, C)
                img = img[1]           # take the last frame
            if img.ndim == 3 and img.shape[0] in (1, 3):  # CHW → HWC
                img = np.transpose(img, (1, 2, 0))

            # ─── normalise if stored in [-1,1] floats ─────────────────────────
            if normalize_if_needed and img.dtype.kind == "f" and img.min() < 0:
                img = (img + 1.0) / 2.0
            img = np.clip(img, 0.0, 1.0)

            # kp_img1_flat = obs_feature[i, 128+4:192+4]             # next 64 dims are img2
            # kp_img2_flat = obs_feature[i, 192+4:256+4]             # next 64 dims are img2
            # kp_img1_flat = obs_feature[i, 128+9:192+9]             # next 64 dims are img2  # for robosuite tasks as the lowdim is 9
            # kp_img2_flat = obs_feature[i, 192+9:256+9]             # next 64 dims are img2
            kp_img1_flat = obs_feature[ i, :64]                # first 64 dims are img1 (32 kp×2)
            kp_img2_flat = obs_feature[i, 64:128]             # next 64 dims are img2

            # ─── convert to pixel coords & visualise ───────────────────────────────────────
            # import pdb; pdb.set_trace()
            kp1_xy = _flat_to_xy(kp_img1_flat, crop_shape).cpu().numpy()
            kp2_xy = _flat_to_xy(kp_img2_flat, crop_shape).cpu().numpy()

            # import pdb; pdb.set_trace()
            # bring the two crops back to [0,1] HWC so matplotlib can show them
            def prep(img):
                if img.ndim == 3 and img.shape[0] in (1,3):  # CHW → HWC
                    img = np.transpose(img, (1,2,0))
                if img.min() < 0:                            # [-1,1] → [0,1]
                    img = (img + 1.) / 2.
                return np.clip(img,0,1)

            if k == 'image1':
                kp_xy = kp1_xy
            else:
                kp_xy = kp2_xy
            


            r, c = divmod(i, cols)
            axes[r, c].imshow(img)
            axes[r, c].scatter(kp_xy[:,0], kp_xy[:,1], s=35, marker='o', edgecolors='white',
                facecolors='red', linewidths=0.6)
            axes[r, c].set_title(f"{k}\nidx {i}", fontsize=8)
            axes[r, c].axis("off")

        plt.suptitle(f"{k} — {n} samples")
        plt.tight_layout()
        plt.show()
        
# # buffer_path = '${HOME}/outputs/trajectory_buffer_0.pkl'
# # buffer_path = 'outputs/trajectory_buffer_0_0702.pkl'
# buffer_path = 'trajectory_buffer_0_0702.pkl'
# visualize_traj_path(buffer_path)

buffer_path = os.path.expanduser(os.path.expandvars(args.buffer_path))
n_obs_steps =2 
from tools.buffer import Buffer

buffer = Buffer(min_size=30, max_size = 50000)
buffer.load_from_file(buffer_path)


# from tools.buffer import HDF5Buffer
# # field_shapes = {
# #     'agentview_image':           (n_obs_steps, 3, 84, 84),
# #     'robot0_eye_in_hand_image':  (n_obs_steps,3, 84, 84),
# #     'robot0_eef_pos':            (n_obs_steps,3),
# #     'robot0_eef_quat':           (n_obs_steps,4),
# #     'robot0_gripper_qpos':       (n_obs_steps,2),
# #     'teacher_action':                    (10,),
# #     'robot_action':                    (10,),
# # }
# field_shapes = {
#     'image1':           (n_obs_steps, 3, 240, 320),
#     'image2':  (n_obs_steps, 3,  240, 320),
#     'robot0_eef_pos_vel':            (n_obs_steps, 4),
#     'teacher_action':                    (2,),
#     'robot_action':                    (2,),
# }

# model_dir = '${BD_COACH_SRC_ROOT}/outputs/20250712_111252_Implicit_BC_kuka-pushT-img_Ta1_offlineTrue_copy/'
# buffer = HDF5Buffer(filename =model_dir+'buffer.h5', field_shapes=field_shapes, min_size=32,
#                                         max_size=50000, dtype_map={})
# sampled_data = buffer.sample(32)
# visualize_sampled_batch(sampled_data, img_keys=('image1', 'image2'))

# shape_meta = {  # should be defined outside this class, as this depends on the env
#             "obs": {
#                 "image1": {"shape": [3, 240, 320], "type": "rgb"},
#                 "image2": {"shape": [3, 240, 320], "type": "rgb"},
#                 "robot0_eef_pos_vel": {"shape": [4]},
#             },
#             "action": {"shape": [2]}
#         }


shape_meta = {  
            "obs": {
                "agentview_image": {"shape": [3, 84, 84], "type": "rgb"},
                "robot0_eye_in_hand_image": {"shape": [3, 84, 84], "type": "rgb"},
                "robot0_eef_pos": {"shape": [3]},
                "robot0_eef_quat": {"shape": [4]},
                # "robot0_eef_quat": {"shape": [9]},  # use rotation matrix instead of quat
                "robot0_gripper_qpos": {"shape": [2]}
            },
            "action": {"shape": [7]}
        }
model_dir = (
    os.path.expanduser(os.path.expandvars(args.model_dir))
    if args.model_dir is not None
    else None
)
crop_shape=[72, 72]

sampled_data = buffer.sample(32)
visualize_sampled_batch(sampled_data, img_keys=('agentview_image', 'robot0_eye_in_hand_image'))
### visualize obsencoder
import torch
from agents.DP_model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from agents.Set_Supervised_diffusion_policy_image import collate_obs_dict
from agents.DP_model.common.pytorch_util import dict_apply
device = torch.device("cuda:0")  
obs_encoder = MultiImageObsEncoder(
            shape_meta=shape_meta,
            resize_shape=None,
            crop_shape=crop_shape,
            random_crop=True,
            use_group_norm=True,
            share_rgb_model=False,
            imagenet_norm=False,
            use_spatial_softmax = True,
        ).to(device)

if model_dir is not None:
    obs_enc_path = os.path.join(model_dir, 'obs_encoder.pth')
    if os.path.isfile(obs_enc_path):
        checkpoint = torch.load(obs_enc_path, map_location=device)
        obs_encoder.load_state_dict(checkpoint['obs_encoder_state_dict'])
        print(f"Obs encoder loaded from {obs_enc_path}")
    else:
        print(f"Obs encoder file not found at {obs_enc_path}, skipping.")
else:
    print("No --model-dir provided; skipping obs encoder checkpoint load.")

state_batch = [pair[0] for pair in sampled_data]
action_batch = [np.array(pair[2]) for pair in sampled_data]  # robot action
h_human_batch = [np.array(pair[1]) for pair in sampled_data]  # human action
batch_size = len(sampled_data)

# import pdb; pdb.set_trace()
nobs = collate_obs_dict(state_batch)
nobs = dict_apply(nobs,lambda x: torch.from_numpy(x).to(
                        device=device,  dtype=torch.float32))
this_nobs = dict_apply(nobs, 
        lambda x: x[:,:n_obs_steps,...].reshape(-1,*x.shape[2:]))
# with torch.no_grad():
#     nobs_feature = obs_encoder(this_nobs)
#     obs_feature = nobs_feature.reshape(batch_size, -1)
#     obs_feature_img1 = obs_feature[:, :32] 
# TODO draw obs_feature (key points on the image)


# ─── helper ────────────────────────────────────────────────────────────────────
def _flat_to_xy(kp_flat, crop_hw):
    """
    kp_flat : (B, 32*2) tensor in [-1,1]  →  (B, 32, 2) pix coords (x,y)
    crop_hw : (H,W) of the random crop given to the encoder
    """
    H, W = crop_hw
    kp = kp_flat.view( 32, 2)                        # (B,32,2)  [-1,1]
    kp = (-kp + 1.0) *0.5                                  # → [0,1]
    kp[:, 0] = kp[:, 0] * (W - 1)                   # x
    kp[: ,1] = kp[:, 1] * (H - 1)                   # y
    return kp                                           # (B,32,2) in pixels

def show_kp_on_img(img, kp_xy, title=None):
    """
    img     : H×W×3 array in [0,1] (HWC, RGB)
    kp_xy   : (32,2) array (x,y) in pixels, same H,W as img
    """
    plt.figure(figsize=(4,4))
    plt.imshow(img)
    plt.scatter(kp_xy[:,0], kp_xy[:,1], s=35, marker='o', edgecolors='white',
                facecolors='red', linewidths=0.6)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

# ─── grab one sample from your already-prepared batch ──────────────────────────
device = torch.device("cuda:0")
obs_encoder.eval()                            # turn off dropout/random_crop noise
with torch.no_grad():
    # forward through encoder
    feat = obs_encoder(this_nobs)             # (B, 256)     2 images × 32 kp ×2 =128
    feat = feat.reshape(batch_size, -1)


# import pdb; pdb.set_trace()
# visualize_sampled_batch_with_obs_feature(sampled_data, feat, img_keys=('image1', 'image2'))
visualize_sampled_batch_with_obs_feature(sampled_data, feat, img_keys=('agentview_image', 'robot0_eye_in_hand_image'), crop_shape = crop_shape)

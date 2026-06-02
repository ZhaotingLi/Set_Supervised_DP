import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys, os
import time
import os, time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor


sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
)
from tools.buffer_trajectory import TrajectoryBuffer


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize trajectory-buffer data.")
    parser.add_argument("buffer_path", help="Path to the trajectory HDF5 buffer.")
    parser.add_argument(
        "--model-hdf5-path",
        default=None,
        help="Optional HDF5 replay buffer used by the later sampled-batch section.",
    )
    parser.add_argument(
        "--obs-enc-dir",
        default=None,
        help="Optional directory containing obs_encoder.pth.",
    )
    return parser.parse_args()


ARGS = parse_args()
      
def visualize_loaded_trajectory(traj, img_key="img"):
    """
    Visualizes all frames in a loaded trajectory.
    
    Parameters:
        traj (list of dict): The trajectory (list of transitions).
        img_key (str): The key in 'obs' where the image is stored.
    """
    print(f"Trajectory contains {len(traj)} steps.")

    for step_id, step in enumerate(traj):
        img = step["obs"][img_key][-1]

        # If (C, H, W), transpose to (H, W, C)
        if img.ndim == 3 and img.shape[0] in [1, 3]:
            img = np.transpose(img, (1, 2, 0))

        # If in [-1,1], map to [0,1]
        if img.min() < 0:
            img = (img + 1.0) / 2.0

        img = np.clip(img, 0.0, 1.0)

        plt.imshow(img)
        plt.title(f"Step {step_id}")
        plt.axis("off")
        plt.show()


import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def replay_loaded_trajectory_interactive(
    traj,
    img_key="img",
    save_img=False,
    save_path="./saved_frames",
    pad=6,
    max_visual_length=None,
    traj_id=0,
    bgr=True,
    min_seq_len=16 # New parameter for your logic
):
    """
    Interactive player that highlights valid extraction points.
    Condition: Valid if no_teacher_action is False for [t : t + min_seq_len]
    """
    
    # 1. Setup Data
    total_steps = len(traj)
    if max_visual_length is not None:
        total_steps = min(total_steps, max_visual_length)
    
    timesteps   = np.array([t['timestep'] for t in traj[:total_steps]])
    no_robot    = np.array([int(t['no_robot_action']) for t in traj[:total_steps]])
    no_teacher  = np.array([int(t['no_teacher_action']) for t in traj[:total_steps]])

    # --- NEW: Calculate Valid Extraction Indices ---
    valid_indices = []
    for t in range(total_steps - min_seq_len + 1):
        # Slice the next 16 steps
        window = no_teacher[t : t + min_seq_len]
        # Check if ALL are 0 (False)
        if np.sum(window) == 0:
            valid_indices.append(t)
            
    print(f"Total Steps: {total_steps}")
    print(f"Found {len(valid_indices)} valid start points (continuous teacher action >= {min_seq_len} steps).")

    # 2. Setup Plot
    fig = plt.figure(figsize=(8, 10))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 3, 0.2])
    ax_plot = fig.add_subplot(gs[0])
    ax_img = fig.add_subplot(gs[1])
    ax_slider = fig.add_subplot(gs[2])

    # 3. Draw the Flags Graph
    # Plot original flags
    # ax_plot.step(timesteps, no_robot, where='post', label='no_robot_actions', alpha=0.5)
    ax_plot.step(timesteps, no_teacher, where='post', label='no_teacher_actions', color='orange')
    
    # --- NEW: Plot Valid Extraction Points ---
    if valid_indices:
        # We plot these as green triangles slightly below the 0 line
        y_valid = [-0.15] * len(valid_indices)
        ax_plot.scatter(valid_indices, y_valid, color='green', marker='^', s=15, label='Valid Extraction Start')

    vline = ax_plot.axvline(timesteps[0], color='k', linestyle='--')
    
    ax_plot.set_xlim(timesteps[0], timesteps[-1])
    ax_plot.set_ylim(-0.3, 1.1) # Expanded Y-limit to show the green markers
    ax_plot.set_ylabel('Flag')
    ax_plot.set_yticks([0, 1])
    ax_plot.set_yticklabels(['False', 'True'])
    ax_plot.legend(loc='upper right', fontsize='small', ncol=3)
    ax_plot.set_title(f"Data & Extraction Logic (Window={min_seq_len})")

    # 4. Initialize Image Container
    initial_img = np.zeros((100, 100, 3))
    img_display = ax_img.imshow(initial_img)
    ax_img.axis("off")
    title_text = ax_img.set_title("Initializing...")

    # 5. Define Update Logic
    def get_processed_image(idx):
        step = traj[idx]
        img = step["obs"][img_key]
        
        if img.ndim == 4 and img.shape[0] >= 1: img = img[0]
        if img.ndim == 3 and img.shape[0] in [1, 3]: img = np.transpose(img, (1, 2, 0))
        if bgr and (img.ndim == 3 and img.shape[-1] == 3): img = img[..., ::-1]
        
        if img.min() < 0: img = (img + 1.0) / 2.0
        img = np.clip(img, 0.0, 1.0)
        return img, step

    def update(val):
        idx = int(slider.val)
        
        # Update Image
        img, step = get_processed_image(idx)
        img_display.set_data(img)
        
        # Check if current index is a valid start point for the title info
        is_valid_start = idx in valid_indices
        status_str = " [VALID START]" if is_valid_start else ""
        
        info = (f"Step: {idx}{status_str} | No_Teacher: {no_teacher[idx]}")
        title_text.set_text(info)
        
        # Update Graph Cursor
        curr_time = timesteps[idx]
        vline.set_xdata([curr_time, curr_time])

        if save_img:
            fname = f"traj_{traj_id}_{img_key}_{str(idx).zfill(pad)}.png"
            fpath = os.path.join(save_path, fname)
            if not os.path.exists(fpath):
                os.makedirs(save_path, exist_ok=True)
                plt.imsave(fpath, img, format="png")

        fig.canvas.draw_idle()

    # 6. Configure Slider
    slider = Slider(ax=ax_slider, label='Timestep', valmin=0, valmax=total_steps - 1, valinit=0, valstep=1)
    slider.on_changed(update)

    # 7. Keyboard Controls
    def on_key(event):
        curr = slider.val
        if event.key == 'right': slider.set_val(min(curr + 1, total_steps - 1))
        elif event.key == 'left': slider.set_val(max(curr - 1, 0))

    fig.canvas.mpl_connect('key_press_event', on_key)
    update(0)
    plt.show()

def replay_loaded_trajectory_old(
    traj,
    img_key="img",
    interval_sec=0.00002,
    save_img=False,
    save_path="./saved_frames",
    pad=6,
    max_visual_length = None,
    visualize_image = True,
    traj_id = 0,
    bgr=True,               # <- set True if your source is OpenCV/BGR
):
    """
    Replays frames in a loaded trajectory, and optionally saves them with matplotlib.

    Parameters:
        traj (list of dict): The trajectory.
        img_key (str): Key in obs containing the image.
        interval_sec (float): Delay between frames in seconds.
        save_img (bool): Whether to save images.
        save_path (str): Directory where images will be saved.
        pad (int): Zero-padding for step number in filenames.
    """
    if save_img:
        os.makedirs(save_path, exist_ok=True)

    # Pre-extract time-series
    timesteps   = [t['timestep']               for t in traj]
    no_robot    = [int(t['no_robot_action'])   for t in traj]
    no_teacher  = [int(t['no_teacher_action']) for t in traj]

    print(f"Trajectory contains {len(traj)} steps.")
    plt.ion()
    fig, (ax_plot, ax_img) = plt.subplots(
        2, 1,
        figsize=(6, 8),
        gridspec_kw={'height_ratios': [1, 3]}
    )

    # Plot the two flags once
    ax_plot.step(timesteps, no_robot,   where='post', label='no_robot_actions')
    ax_plot.step(timesteps, no_teacher, where='post', label='no_teacher_actions')
    vline = ax_plot.axvline(timesteps[0], color='k')
    ax_plot.set_xlim(timesteps[0], timesteps[-1])
    ax_plot.set_ylim(-0.1, 1.1)
    ax_plot.set_ylabel('Flag (1=True, 0=False)')
    ax_plot.set_xlabel('Timestep')
    ax_plot.legend(loc='upper right')

    img_display = None
    title = None

    # Loop through frames
    for i, step in enumerate(traj):
        if max_visual_length is not None and i > max_visual_length:
            break
        
        print('no_teacher: ', no_teacher[i], " teacher action: ", step['teacher_action'], " robot action: ", step['robot_action'])
        img = step["obs"][img_key]

        # Handle stacked frames and CHW→HWC transpose
        if img.ndim == 4 and img.shape[0] >= 1:
            img = img[0]
        if img.ndim == 3 and img.shape[0] in [1, 3]:
            img = np.transpose(img, (1, 2, 0))

        # Convert BGR -> RGB if requested and image is color
        if bgr and (img.ndim == 3 and img.shape[-1] == 3):
            img = img[..., ::-1]

        # Normalize if in [-1,1]
        if img.min() < 0:
            img = (img + 1.0) / 2.0
        img = np.clip(img, 0.0, 1.0)  # plt.imsave expects [0,1] float or uint8

        # Save image to disk if requested (works for HxW or HxWx3)
        if save_img:
            fname = f"traj_{traj_id}_{img_key}_{str(i).zfill(pad)}.png"
            fpath = os.path.join(save_path, fname)
            # For grayscale 2D arrays, plt.imsave will default to a colormap; force no colormap
            if img.ndim == 2:
                plt.imsave(fpath, img, format="png", cmap="gray", vmin=0.0, vmax=1.0)
            else:
                plt.imsave(fpath, img, format="png")

        # Initialize/update the display
        if img_display is None:
            img_display = ax_img.imshow(img)
            title = ax_img.set_title(f"Step {i}")
            ax_img.axis("off")
        else:
            img_display.set_data(img)
            title.set_text(f"Step {i}")

        # Move the vertical line
        vline.set_xdata([timesteps[i], timesteps[i]])

        fig.canvas.draw()
        fig.canvas.flush_events()
        # time.sleep(interval_sec)
    if visualize_image:
        plt.ioff()
        plt.show()
        plt.close()


def replay_loaded_trajectory(
    traj,
    img_key="img",
    visualize_image=True,
    show_plot=True,
    plot_update_stride=10,
    bgr=True,                 # True if source images are in BGR (OpenCV) order
    save_video=False,
    video_path="./traj.mp4",
    video_fps=30,
    save_img=False,        # if you truly need individual frames
    save_path="./saved_frames_fast",
    img_pad=6,
    print_stride=50,
    max_visual_length=None,
    jpeg_images=False,
):
    n = len(traj)
    if max_visual_length is not None:
        n = min(n, max_visual_length + 1)

    # ---- optional plot (unchanged) ----
    if show_plot:
        timesteps   = [t['timestep']               for t in traj[:n]]
        no_robot    = [int(t['no_robot_action'])   for t in traj[:n]]
        no_teacher  = [int(t['no_teacher_action']) for t in traj[:n]]

        plt.ion()
        fig, ax_plot = plt.subplots(figsize=(6, 2.5))
        ax_plot.step(timesteps, no_robot,   where='post', label='no_robot_actions')
        ax_plot.step(timesteps, no_teacher, where='post', label='no_teacher_actions')
        vline = ax_plot.axvline(timesteps[0], color='k')
        ax_plot.set_xlim(timesteps[0], timesteps[-1])
        ax_plot.set_ylim(-0.1, 1.1)
        ax_plot.set_ylabel('Flag')
        ax_plot.set_xlabel('Timestep')
        ax_plot.legend(loc='upper right')
        fig.canvas.draw()

    if visualize_image:
        cv2.namedWindow("replay", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("replay", 640, 480)

    # ---- frame prep helper (unchanged) ----
    def _prep_frame(frame):
        img = frame
        if img.ndim == 4 and img.shape[0] >= 1:
            img = img[0]
        if img.ndim == 3 and img.shape[0] in [1, 3]:
            img = np.transpose(img, (1, 2, 0))
        if img.dtype != np.uint8:
            if img.min() < 0:
                img = (img + 1.0) * 127.5
            else:
                img = img * 255.0
            img = np.clip(img, 0, 255).astype(np.uint8)
        # Ensure BGR for OpenCV
        if not bgr:
            if img.ndim == 3 and img.shape[-1] == 3:
                img = img[..., ::-1]
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    # ---- Optional video writer (robust) ----
    writer = None
    writer_out_path = None
    if save_video:
        # Make sure directory exists
        out_dir = os.path.dirname(video_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        # Probe size using the SAME pipeline used for all frames
        sample_raw = traj[0]["obs"][img_key]
        sample_bgr = _prep_frame(sample_raw)
        h, w = sample_bgr.shape[:2]

        # Try MP4 first
        fourcc_str = 'mp4v'  # try 'avc1' if you have H.264
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(video_path, fourcc, float(video_fps), (w, h))
        writer_out_path = video_path

        # If that failed, fall back to AVI/MJPG (very compatible)
        if not writer.isOpened():
            writer.release()
            alt_path = os.path.splitext(video_path)[0] + ".avi"
            fourcc_alt = cv2.VideoWriter_fourcc(*'MJPG')
            writer = cv2.VideoWriter(alt_path, fourcc_alt, float(video_fps), (w, h))
            writer_out_path = alt_path

        # If still not opened, raise with a clear message
        if not writer.isOpened():
            raise RuntimeError(
                f"Failed to open video writer for:\n"
                f"  MP4: {video_path} (codec {fourcc_str}) and\n"
                f"  AVI: {alt_path} (codec MJPG).\n"
                "Check that the path exists and your OpenCV build has the required codecs."
            )
        print(f"[Video] Writing to: {writer_out_path} at {video_fps} FPS, size {w}x{h}")

    # ---- Optional threaded still-image saving ----
    if save_img:
        os.makedirs(save_path, exist_ok=True)
        pool = ThreadPoolExecutor(max_workers=4)
        ext = ".jpg" if jpeg_images else ".png"

        def _save_still(idx, bgr_img):
            fname = f"traj_{str(idx).zfill(img_pad)}{ext}"
            fpath = os.path.join(save_path, fname)
            if jpeg_images:
                cv2.imwrite(fpath, bgr_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            else:
                cv2.imwrite(fpath, bgr_img)
    else:
        pool = None  # keep symbol defined

    # ---- Main loop ----
    for i, step in enumerate(traj[:n]):
        if (i % print_stride) == 0:
            print(f"[{i}/{n}] no_teacher={int(step['no_teacher_action'])}  "
                  f"teacher={step.get('teacher_action')}  robot={step.get('robot_action')}")

        frame_bgr = _prep_frame(step["obs"][img_key])

        # Ensure writer/frame sizes match exactly
        if save_video:
            if frame_bgr.shape[0] != h or frame_bgr.shape[1] != w:
                frame_bgr = cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_AREA)
            writer.write(frame_bgr)

        if visualize_image:
            cv2.imshow("replay", frame_bgr)
            cv2.waitKey(1)

        if save_img:
            pool.submit(_save_still, i, frame_bgr)

        if show_plot and ((i % plot_update_stride) == 0 or i == n - 1):
            vline.set_xdata([step['timestep'], step['timestep']])
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

    # ---- Cleanup ----
    if visualize_image:
        cv2.destroyWindow("replay")
    if save_video and writer is not None:
        writer.release()
        print(f"[Video] Saved: {writer_out_path}")
    if save_img and pool is not None:
        pool.shutdown(wait=True)
    if show_plot:
        plt.ioff()
        plt.show()




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

            kp_img1_flat = obs_feature[ i, :64]                # first 64 dims are img1 (32 kp×2)
            kp_img2_flat = obs_feature[i, 64:128]             # next 64 dims are img2
            # kp_img1_flat = obs_feature[ i, 128+4:192+4]
            # kp_img2_flat = obs_feature[i, 192+4:256+4]             # next 64 dims are img2

            # ─── convert to pixel coords & visualise ───────────────────────────────────────
            crop_shape = (216, 288)                         # (H,W) given in the encoder cfg
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

# buffer_path = 'outputs/20250815_120309_Diffusion_CLIC_intervention_Circular_square_image_abs_Ta16_offlineFalse/trajectory_buffer_0.hdf5'
# buffer_path = '${HOME}/outputs/trajectory_buffer_0711_PushT_abs_50hz.hdf5'
# buffer_path = 'outputs_docker/trajectory_buffer_0716_insertT_abs_86eps.hdf5'
# buffer_path = '${HOME}/outputs/0925/traj_trajectory_buffer_0925_combined_Insert_demonstration_70.hdf5'
# buffer_path ='${HOME}/outputs/1002/trajectory_buffer_0929_InsertBlueT_GreenU_demonstration_73trajs_0930_Intervention_1002_Inter_trajs60.hdf5'
# buffer_path = '${HOME}/NEXTGENCoR/CLIC_DP_real_exp_data/1006/trajectory_buffer_1006_intervention01_trajs15.hdf5'
# buffer_path = '${HOME}/outputs/1008/trajectory_buffer_CLIC_DP_AELoss_13hours_batch64.hdf5'
# buffer_path = '${HOME}/outputs/1009/trajectory_buffer_CLIC_DP_AELoss_20hours_batch64.hdf5'
# buffer_path = '${HOME}/outputs/Nov_evaluation/trajectory_buffer_evaluation_Nov12_CLID_DP_delftblue_0to20.hdf5'
buffer_path = os.path.expanduser(os.path.expandvars(ARGS.buffer_path))
# buffer_path = '${HOME}/NEXTGENCoR/CLIC_DP_real_exp_data/trajectory_buffer_Nov11_noisy_intervention.hdf5'
# buffer_path = '${HOME}/outputs_franka/outputs/20251118_124920_Diffusion_CLIC_intervention_Circular_Franka-2d-img_Ta16_offlineFalse_Scale0.05/trajectory_buffer_0.hdf5'
# buffer_path = '${HOME}/outputs_franka/trajectory_buffer_0_test.hdf5'
# buffer_path = '${HOME}/outputs_franka/outputs/20251119_151345_Diffusion_CLIC_intervention_Circular_Franka-2d-img_Ta16_offlineFalse_Scale0.05/trajectory_buffer_0.hdf5'
# buffer_path = '${HOME}/outputs_franka/trajectory_buffer_intervention_combined.hdf5'



# buffer_path = '${BD_COACH_SRC_ROOT}/outputs_docker/20250817_172838_Diffusion_policy_square_image_abs_Ta16_offlineFalse_NN2048/trajectory_buffer_0.hdf5'
# with teacher action: 70186, no teacher action: 22684

traj_buffer = TrajectoryBuffer()
traj_number = traj_buffer.count_trajectories_in_hdf5(buffer_path)
print("traj_number: ", traj_number)
# no_teacher   = [int(traj['no_teacher_action']) for t in traj]

'''save the initial image for every traj'''
# for traj_id in range(traj_number):
#     traj_ = traj_buffer.load_from_file(buffer_path, traj_id=traj_id)
#     replay_loaded_trajectory_old(traj_, img_key='image1',save_img= True, save_path= 'outputs/img_blue_green_insertT_CLIC_20hours_Inits/', max_visual_length=1, visualize_image=False, traj_id= traj_id)
#     plt.close()

''' visualize the traj data'''
for traj_id in range(0, traj_number):
    print("traj_id: ", traj_id)
    traj = traj_buffer.load_from_file(buffer_path, traj_id = traj_id)
    # img_key = 'image2'
    img_key = 'image1'
    # replay_loaded_trajectory_interactive(traj, img_key=img_key,save_img= False, save_path= 'outputs/Nov19_demo30/', max_visual_length=None, traj_id= traj_id)
    # replay_loaded_trajectory_old(traj, img_key=img_key,save_img= False, save_path= 'outputs/img_blue_green_insertT_CLIC/', max_visual_length=None, visualize_image=False, traj_id= traj_id)
    replay_loaded_trajectory(traj, img_key=img_key,save_img= False, save_path= 'outputs/Nov25_interventions/', video_fps=20, show_plot=False, save_video=True, video_path=f'outputs_docker/INsertT_videos_0to20_DP/{img_key}_{traj_id}.mp4')  # for InsertT
# # replay_loaded_trajectory(traj, img_key='image2')  # for robosuite simulated env

import pdb; pdb.set_trace()
'''Check how many teacher actions are inside the trajectory buffer'''
# total_no_teacher = 0
# total_steps = 0

# for traj_id in range(traj_number):
#     traj = traj_buffer.load_from_file(buffer_path, traj_id=traj_id)
#     # import pdb; pdb.set_trace()
#     no_teacher_arr = np.array([int(t['no_teacher_action']) for t in traj])  # boolean array
#     total_no_teacher += np.sum(no_teacher_arr)
#     total_steps += len(no_teacher_arr)

# print(f"Total steps: {total_steps}")
# print(f"Steps with no teacher action: {total_no_teacher}")
# print(f"Steps with teacher action: {total_steps - total_no_teacher}")

'''load  data buffer from traj_buffer'''
from tools.buffer import HDF5Buffer
# n_obs_steps = 2
# field_shapes = {
#     'agentview_image':           (n_obs_steps, 3, 84, 84),
#     'robot0_eye_in_hand_image':  (n_obs_steps,3, 84, 84),
#     'robot0_eef_pos':            (n_obs_steps,3),
#     'robot0_eef_quat':           (n_obs_steps,4),
#     'robot0_gripper_qpos':       (n_obs_steps,2),
#     # 'teacher_action':                    (10,),
#     # 'robot_action':                    (10,),
#     'teacher_action':                    (16, 10),
#     'robot_action':                    (16, 10),
# }
# buffer = HDF5Buffer(filename ='outputs/buffer.h5', field_shapes=field_shapes, min_size=32,
#                                         max_size=50000, dtype_map={})
# # buffer.ingest_trajectory_hdf5(traj_filename =buffer_path )
# buffer.ingest_trajectory_hdf5_Ta(traj_filename =buffer_path, Ta=16)

# # import pdb; pdb.set_trace()
# # # visualize_loaded_trajectory(traj, img_key='image')
# # # visualize_loaded_trajectory(traj, img_key = "agentview_image")


''' check the statistics of the traj data'''
global_min = None
global_max = None

# action_key = "teacher_action"
action_key = "teacher_action"
for tid in range(traj_number):
    traj = traj_buffer.load_from_file(buffer_path, traj_id=tid)
    # stack into shape (T, action_dim)
    # actions = np.stack([step[action_key] for step in traj], axis=0)
    actions = np.stack([step["obs"]['robot0_eef_pos_vel'] for step in traj], axis=0)
    

    # per-traj min/max
    tmin = actions.min(axis=0)
    tmax = actions.max(axis=0)

    if global_min is None:
        global_min = tmin.copy()
        global_max = tmax.copy()
    else:
        global_min = np.minimum(global_min, tmin)
        global_max = np.maximum(global_max, tmax)

print("global_min: ", global_min, " global_max: ", global_max)

import pdb; pdb.set_trace()
n_obs_steps =2 
field_shapes = {
    'image1':           (n_obs_steps, 3, 240, 320),
    'image2':  (n_obs_steps, 3,  240, 320),
    'robot0_eef_pos_vel':            (n_obs_steps, 4),
    'teacher_action':                    (2,),
    'robot_action':                    (2,),
}


# buffer = HDF5Buffer(filename ='outputs/buffer.h5', field_shapes=field_shapes, min_size=32,
#                                         max_size=50000, dtype_map={})
# buffer.ingest_trajectory_hdf5(traj_filename =buffer_path )
# # import pdb; pdb.set_trace()

if ARGS.model_hdf5_path is None:
    raise ValueError("--model-hdf5-path is required for the sampled HDF5-buffer section.")
model_dir = os.path.expanduser(os.path.expandvars(ARGS.model_hdf5_path))
# model_dir = '${BD_COACH_SRC_ROOT}/outputs_docker/data_buffer_Nov10_demo_noisy.h5'
buffer = HDF5Buffer(filename =model_dir, field_shapes=field_shapes, min_size=32,
                                        max_size=50000, dtype_map={"image1": "uint8", "image2": "uint8",}, image_saved_in_Uint8=True)
buffer.load_from_file(model_dir)
sampled_data = buffer.sample(32)
visualize_sampled_batch(sampled_data, img_keys=('image1', 'image2'))

shape_meta = {  # should be defined outside this class, as this depends on the env
            "obs": {
                "image1": {"shape": [3, 240, 320], "type": "rgb"},
                "image2": {"shape": [3, 240, 320], "type": "rgb"},
                "robot0_eef_pos_vel": {"shape": [4]},
            },
            "action": {"shape": [2]}
        }
### visualize obsencoder
import torch
from agents.DP_model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from agents.DP_model.vision.multi_image_obs_encoder_with_decoder import MultiImageObsEncoderWithDecoder
from agents.Set_Supervised_diffusion_policy_image import collate_obs_dict
from agents.DP_model.common.pytorch_util import dict_apply
device = torch.device("cuda:0") 

obs_encoder = MultiImageObsEncoderWithDecoder(
        shape_meta=shape_meta,
        resize_shape=None,
        crop_shape=[216, 288],
        random_crop=True,
        use_group_norm=True,
        share_rgb_model=False,
        imagenet_norm=False,
        use_spatial_softmax=False,
        use_global_bottleneck_for_policy=True,   # policy uses z: (B,256) per camera
        decode_from_global_z=True,               # recon from z -> seed -> decoder
        add_coord_channels_to_seed=False,        # set True to append (x,y) to seed
        bottleneck_dim=256,
    ).to(device)

# obs_encoder = MultiImageObsEncoder(
#             shape_meta=shape_meta,
#             resize_shape=None,
#             crop_shape=[216, 288],
#             random_crop=True,
#             use_group_norm=True,
#             share_rgb_model=False,
#             imagenet_norm=False,
#             use_spatial_softmax = True,
#         ).to(device)


'''DP obs encoder load'''
# from robomimic.algo import algo_factory
# from robomimic.algo.algo import PolicyAlgo
# import robomimic.utils.obs_utils as ObsUtils
# # import robomimic.models.base_nets as rmbn
# import robomimic.models.obs_core as rmbn
# import agents.DP_model.vision.crop_randomizer as dmvc
# from agents.DP_model.common.pytorch_util import dict_apply, replace_submodules


# from robomimic.config import config_factory
# import robomimic.scripts.generate_paper_configs as gpc
# from robomimic.scripts.generate_paper_configs import (
#     modify_config_for_default_image_exp,
#     modify_config_for_default_low_dim_exp,
#     modify_config_for_dataset,
# )
# from agents.diffusion_policy_image_original import get_robomimic_config
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# obs_shape_meta = shape_meta['obs']
# obs_config = {
#     'low_dim': [],
#     'rgb': [],
#     'depth': [],
#     'scan': []
# }
# obs_key_shapes = dict()
# for key, attr in obs_shape_meta.items():
#     shape = attr['shape']
#     obs_key_shapes[key] = list(shape)

#     type = attr.get('type', 'low_dim')
#     if type == 'rgb':
#         obs_config['rgb'].append(key)
#     elif type == 'low_dim':
#         obs_config['low_dim'].append(key)
#     else:
#         raise RuntimeError(f"Unsupported obs type: {type}")
# config = get_robomimic_config(
#     algo_name='bc_rnn',
#     hdf5_type='image',
#     task_name='square',
#     dataset_type='ph')
# crop_shape = (216, 288)  
# action_dim = 2
# obs_encoder_group_norm= True
# eval_fixed_crop = True
# with config.unlocked():
#     # set config with shape_meta
#     config.observation.modalities.obs = obs_config

#     if crop_shape is None:
#         for key, modality in config.observation.encoder.items():
#             if modality.obs_randomizer_class == 'CropRandomizer':
#                 modality['obs_randomizer_class'] = None
#     else:
#         # set random crop parameter
#         ch, cw = crop_shape
#         for key, modality in config.observation.encoder.items():
#             if modality.obs_randomizer_class == 'CropRandomizer':
#                 modality.obs_randomizer_kwargs.crop_height = ch
#                 modality.obs_randomizer_kwargs.crop_width = cw
# # init global state
# ObsUtils.initialize_obs_utils_with_config(config)
# # load model
# policy: PolicyAlgo = algo_factory(
#         algo_name=config.algo_name,
#         config=config,
#         obs_key_shapes=obs_key_shapes,
#         ac_dim=action_dim,
#         device='cpu',
#     )
# obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']

# if obs_encoder_group_norm:
#     # replace batch norm with group norm
#     replace_submodules(
#         root_module=obs_encoder,
#         predicate=lambda x: isinstance(x, nn.BatchNorm2d),
#         func=lambda x: nn.GroupNorm(
#             num_groups=x.num_features//16, 
#             num_channels=x.num_features)
#     )
#     # obs_encoder.obs_nets['agentview_image'].nets[0].nets

# # obs_encoder.obs_randomizers['agentview_image']
# if eval_fixed_crop:
#     replace_submodules(
#         root_module=obs_encoder,
#         predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
#         func=lambda x: dmvc.CropRandomizer(
#             input_shape=x.input_shape,
#             crop_height=x.crop_height,
#             crop_width=x.crop_width,
#             num_crops=x.num_crops,
#             pos_enc=x.pos_enc
#         )
#     )

obs_enc_dir = (
    os.path.expanduser(os.path.expandvars(ARGS.obs_enc_dir))
    if ARGS.obs_enc_dir is not None
    else None
)
obs_enc_path = os.path.join(obs_enc_dir, 'obs_encoder.pth') if obs_enc_dir else None
print('obs_enc_path: ', obs_enc_path)
if obs_enc_path and os.path.isfile(obs_enc_path):
    checkpoint = torch.load(obs_enc_path, map_location=device)
    obs_encoder.load_state_dict(checkpoint['obs_encoder_state_dict'])
    print(f"Obs encoder loaded from {obs_enc_path}")
elif obs_enc_path:
    print(f"Obs encoder file not found at {obs_enc_path}, skipping.")
else:
    print("No --obs-enc-dir provided; skipping obs encoder checkpoint load.")

obs_encoder.to(device)
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
    kp = (-1*kp + 1.0) *0.5                                  # → [0,1]
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
    # pick first example in the batch
    img1 = state_batch[0]['image1'][0, :]           # (2,3,240,320) or (3,240,320)
    img2 = state_batch[0]['image2'][0, :]

    # keep only the last frame & make CHW
    if img1.ndim == 4: img1 = img1[-1]
    if img2.ndim == 4: img2 = img2[-1]

    # forward through encoder
    feat = obs_encoder(this_nobs)             # (B, 256)     2 images × 32 kp ×2 =128
    feat = feat.reshape(batch_size, -1)
    kp_img1_flat = feat[0, :64]                # first 64 dims are img1 (32 kp×2)
    kp_img2_flat = feat[0, 128:192]             # next 64 dims are img2
    print("feat: ", feat.shape)

    ## test decoder


# import pdb; pdb.set_trace()
visualize_sampled_batch_with_obs_feature(sampled_data, feat, img_keys=('image1', 'image2'))


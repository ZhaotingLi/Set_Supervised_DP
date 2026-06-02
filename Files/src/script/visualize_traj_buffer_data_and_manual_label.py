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


from matplotlib.widgets import Slider


def manual_label_trajectory(
    traj,
    img_key="img",
    save_path="./saved_frames",
    pad=6,
    max_visual_length=None,
    traj_id=0,
    bgr=True,
    min_seq_len=16
):
    """
    Interactive tool to manually label regions for modification.
    
    Visual Aids:
    - Orange Line: Current no_teacher_action flag.
    - Green Triangles: Algorithm suggested valid start points (min_seq_len consecutive False).
    - Red Shading: Your manually selected regions.
    - Extra line plot: (teacher_action * (not no_teacher_action)) - robot_action.
    
    Returns:
        list of tuples: [(start_time_1, end_time_1), (start_time_2, end_time_2), ...]
    """
    
    # --- 1. Data Setup ---
    total_steps = len(traj)
    if max_visual_length is not None:
        total_steps = min(total_steps, max_visual_length)
    
    timesteps      = np.array([t['timestep']        for t in traj[:total_steps]])
    no_robot       = np.array([int(t['no_robot_action'])   for t in traj[:total_steps]])
    no_teacher     = np.array([int(t['no_teacher_action']) for t in traj[:total_steps]])
    teacher_action = np.array([t['teacher_action']  for t in traj[:total_steps]])
    robot_action   = np.array([t['robot_action']    for t in traj[:total_steps]])

    # --- Calculate Valid Extraction Indices (The Logic you wanted to keep) ---
    valid_indices = []
    for t in range(total_steps - min_seq_len + 1):
        window = no_teacher[t : t + min_seq_len]
        if np.sum(window) == 0:
            valid_indices.append(t)
            
    print("-" * 50)
    print("CONTROLS:")
    print(" [Left/Right Arrow] : Navigate frames")
    print(" [A]                : Set START of modification region")
    print(" [E]                : Set END of modification region (saves it)")
    print(" [U]                : UNDO last saved region")
    print(" [Close Window]     : Finish and return list")
    print(f"INFO: Found {len(valid_indices)} algorithmic valid start points (Green Triangles).")
    print("-" * 50)

    # --- 2. State Management ---
    state = {
        'start_idx': None,          # currently selected start index
        'regions': [],              # list of (start, end) tuples
        'start_line_artist': None,  # vertical line for start
        'span_artists': []          # list of red spans for regions
    }

    # --- 3. Plot Setup ---
    fig = plt.figure(figsize=(10, 10))
    # Now 4 rows: flags, diff, image, slider
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1., 3.3, 0.2])
    ax_plot = fig.add_subplot(gs[0])   # flags (no_teacher)
    ax_diff = fig.add_subplot(gs[1])   # NEW: difference plot
    ax_img = fig.add_subplot(gs[2])    # image
    ax_slider = fig.add_subplot(gs[3]) # slider

    # Plot Flags
    ax_plot.step(timesteps, no_teacher, where='post',
                 label='no_teacher_actions', color='orange')
    
    # Plot Valid Extraction Points
    if valid_indices:
        y_valid = [-0.15] * len(valid_indices)
        ax_plot.scatter(valid_indices, y_valid, color='green',
                        marker='^', s=15,
                        label=f'Valid Start (len>={min_seq_len})')

    # Configure Plot Axis
    ax_plot.set_xlim(timesteps[0], timesteps[-1])
    ax_plot.set_ylim(-0.3, 1.1)  # Extra space at bottom for triangles
    ax_plot.set_ylabel('Flag')
    ax_plot.set_yticks([0, 1])
    ax_plot.set_yticklabels(['False', 'True'])
    ax_plot.legend(loc='upper right', fontsize='small', ncol=2)
    ax_plot.set_title("Press 'S' to start region, 'E' to end region")

    # Cursor line (Timeline position) on flag plot
    cursor_line = ax_plot.axvline(timesteps[0], color='k', linestyle='-', alpha=0.8)

    # --- 3b. Difference Plot Setup ---
    # teacher_effective = teacher_action * (not no_teacher_action)
    # no_teacher is 0/1 -> (1 - no_teacher) is 1 when teacher is present
    if teacher_action.ndim == 1:
        mask = (1 - no_teacher)          # shape [T]
        teacher_eff = teacher_action * mask
        robot_action_eff = robot_action * mask
    else:
        mask = (1 - no_teacher)[:, None] # shape [T, 1]
        teacher_eff = teacher_action * mask
        robot_action_eff = robot_action * mask

    action_diff = teacher_eff - robot_action_eff  # (teacher * ~no_teacher) - robot
    # action_diff = np.sum(action_diff**2, axis=-1)
    action_diff = np.linalg.norm(action_diff, axis= - 1)

    # import pdb;pdb.set_trace()

    if action_diff.ndim == 1:
        action_diff = action_diff[:, None]

    num_dims = action_diff.shape[1]
    x_range = np.arange(total_steps)

    diff_lines = []
    for d in range(num_dims):
        line, = ax_diff.plot(x_range, action_diff[:, d], label=f'dim {d}')
        diff_lines.append(line)

    diff_cursor_line = ax_diff.axvline(0, color='k', linestyle='--', alpha=0.7)

    ax_diff.set_title("Action diff: (teacher * (not no_teacher)) - robot")
    ax_diff.set_xlim(timesteps[0], timesteps[-1])
    ax_diff.set_xlabel("Timestep")
    ax_diff.set_ylabel("Difference")
    ax_diff.legend(loc='upper right', fontsize='small')
    ax_diff.grid(True)

    # Image container
    initial_img = np.zeros((60, 80, 3))
    img_display = ax_img.imshow(initial_img)
    ax_img.axis("off")
    title_text = ax_img.set_title("Step 0")

    # Text box for actions (teacher & robot) under the image
    actions_text = ax_img.text(
        0.5, -0.01, "",
        transform=ax_img.transAxes,
        ha='center', va='top',
        fontsize=14
    )

    # --- 4. Helper Functions ---
    def get_processed_image(idx):
        step = traj[idx]
        img = step["obs"][img_key]
        if img.ndim == 4 and img.shape[0] >= 1:
            img = img[0]
        if img.ndim == 3 and img.shape[0] in [1, 3]:
            img = np.transpose(img, (1, 2, 0))
        if bgr and (img.ndim == 3 and img.shape[-1] == 3):
            img = img[..., ::-1]
        if img.min() < 0:
            img = (img + 1.0) / 2.0
        return np.clip(img, 0.0, 1.0)

    def redraw_regions():
        """Refreshes the red shaded regions"""
        for span in state['span_artists']:
            span.remove()
        state['span_artists'] = []

        for (s, e) in state['regions']:
            span = ax_plot.axvspan(s, e, color='red', alpha=0.2)
            state['span_artists'].append(span)
        
        fig.canvas.draw_idle()

    def update(val):
        idx = int(slider.val)
        
        # Update Image
        img = get_processed_image(idx)
        img_display.set_data(img)
        
        # Update Title with Context
        mode_str = " [RECORDING...]" if state['start_idx'] is not None else ""
        valid_str = " [VALID START]" if idx in valid_indices else ""
        
        info = (f"Step: {idx} | No_Teacher: {no_teacher[idx]}{mode_str}{valid_str}")
        title_text.set_text(info)

        # Update teacher & robot actions text
        ta = teacher_action[idx]
        ra = robot_action[idx]
        ta_str = np.array2string(ta, precision=3, separator=', ')
        ra_str = np.array2string(ra, precision=3, separator=', ')
        actions_text.set_text(f"Teacher action: {ta_str}\nRobot action:   {ra_str}")
        
        # Update Cursor on flag plot
        cursor_line.set_xdata([idx, idx])

        # Update cursor on diff plot
        diff_cursor_line.set_xdata([idx, idx])
        
        fig.canvas.draw_idle()

    # --- 5. Slider ---
    slider = Slider(
        ax=ax_slider,
        label='Timestep',
        valmin=0,
        valmax=total_steps - 1,
        valinit=0,
        valstep=1
    )
    slider.on_changed(update)

    # --- 6. Keyboard Logic ---
    def on_key(event):
        curr_idx = int(slider.val)

        # START Selection  (use 's' as documented)
        if event.key == 'a':
            state['start_idx'] = curr_idx
            if state['start_line_artist']:
                state['start_line_artist'].remove()
            state['start_line_artist'] = ax_plot.axvline(
                curr_idx, color='green', linestyle='--', linewidth=2
            )
            print(f"-> Start marked at step {curr_idx}")
            update(curr_idx)

        # END Selection
        elif event.key == 'e':
            if state['start_idx'] is None:
                print("XX Cannot End: No Start point set! Press 'S' first.")
                return
            start = state['start_idx']
            end = curr_idx
            if end <= start:
                print("XX Invalid Region: End time must be after Start time.")
                return
            
            state['regions'].append((start, end))
            print(f"-> Region Saved: [{start}, {end}]")

            # Reset Start Logic
            state['start_idx'] = None
            if state['start_line_artist']:
                state['start_line_artist'].remove()
                state['start_line_artist'] = None

            redraw_regions()
            update(curr_idx)

        # UNDO Selection
        elif event.key == 'u':
            if len(state['regions']) > 0:
                removed = state['regions'].pop()
                print(f"-> Undid Region: {removed}")
                redraw_regions()
            else:
                print("Nothing to undo.")

        # Navigation
        elif event.key == 'right':
            slider.set_val(min(curr_idx + 1, total_steps - 1))
        elif event.key == 'left':
            slider.set_val(max(curr_idx - 1, 0))

    fig.canvas.mpl_connect('key_press_event', on_key)

    # Initialize first frame
    update(0)
    plt.show()
    
    return state['regions']



buffer_path = 'outputs/square_dataset_CDP/trajectory_buffer_0.hdf5'


# buffer_path = '${HOME}/outputs_franka/trajectory_buffer_intervention_Nov25.hdf5'
# buffer_path = '${HOME}/outputs_franka/trajectory_buffer_intervention_combined.hdf5'
# buffer_path = '${HOME}/outputs_franka/trajectory_buffer_intervention_nov21.hdf5'





# buffer_path = '${BD_COACH_SRC_ROOT}/outputs_docker/20250817_172838_Diffusion_policy_square_image_abs_Ta16_offlineFalse_NN2048/trajectory_buffer_0.hdf5'
# with teacher action: 70186, no teacher action: 22684

traj_buffer = TrajectoryBuffer()
traj_number = traj_buffer.count_trajectories_in_hdf5(buffer_path)
print("traj_number: ", traj_number)
# no_teacher   = [int(traj['no_teacher_action']) for t in traj]

import json


# Extract the base name without extension (e.g., 'trajectory_buffer_intervention_combined')
# --- 1. Dynamic Path Generation ---
# Get directory: '${HOME}/outputs_franka'
parent_dir = os.path.dirname(buffer_path)

# Get filename without extension: 'trajectory_buffer_intervention_combined'
filename_no_ext = os.path.splitext(os.path.basename(buffer_path))[0]

# Construct new path: '.../labeled_regions_trajectory_buffer_intervention_combined.json'
json_filename = f"labeled_regions_{filename_no_ext}.json"
json_output_path = os.path.join(parent_dir, json_filename)

print(f"Labeling output will be saved to:\n{json_output_path}\n")

# Initialize the dictionary to store results
# Format: { "0": [[start, end], ...], "1": ... }
all_labeled_regions = {}

# Check if file exists to potentially resume (Optional, currently starts fresh)
if os.path.exists(json_output_path):
    print(f"Warning: Overwriting existing file at {json_output_path}")

for traj_id in range(0, traj_number):
# for traj_id in range(227, traj_number):
    print(f"\n=== Processing Trajectory {traj_id} / {traj_number-1} ===")
    
    # 1. Load Trajectory
    traj = traj_buffer.load_from_file(buffer_path, traj_id=traj_id)
    # img_key = 'image2'
    img_key = 'agentview_image'
    
    # 2. Run the Interactive Tool
    # This pauses the loop until you close the matplotlib window
    state_regions = manual_label_trajectory(
        traj, 
        img_key=img_key, 
        save_path='outputs/save_frames/', 
        max_visual_length=None, 
        traj_id=traj_id
    )
    
    # 3. Store in Dictionary (using string key is standard for JSON)
    all_labeled_regions[str(traj_id)] = state_regions
    
    print(f" -> Traj {traj_id} finished. Selected regions: {state_regions}")

    # 4. Save to Disk Immediately
    # We save inside the loop so you don't lose progress if you quit early
    with open(json_output_path, 'w') as f:
        json.dump(all_labeled_regions, f, indent=4)
        print(f" -> Progress saved to {json_output_path}")

print("\nAll trajectories processed.")
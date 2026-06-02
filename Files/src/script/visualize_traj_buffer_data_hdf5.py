import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider


sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from tools.buffer_trajectory import TrajectoryBuffer


def visualize_trajectory(
    traj,
    img_key="agentview_image",
    max_visual_length=None,
    bgr=True,
    min_seq_len=16,
):
    """
    Interactive viewer for one trajectory buffer trajectory.

    Keeps the same visual layout as the manual-label tool:
    - no_teacher_action timeline
    - action-difference plot
    - current image frame
    - slider + arrow-key navigation
    """
    total_steps = len(traj)
    if max_visual_length is not None:
        total_steps = min(total_steps, max_visual_length)

    timesteps = np.array([t["timestep"] for t in traj[:total_steps]])
    no_teacher = np.array([int(t["no_teacher_action"]) for t in traj[:total_steps]])
    teacher_action = np.array([t["teacher_action"] for t in traj[:total_steps]])
    robot_action = np.array([t["robot_action"] for t in traj[:total_steps]])

    valid_indices = []
    for idx in range(total_steps - min_seq_len + 1):
        if np.sum(no_teacher[idx: idx + min_seq_len]) == 0:
            valid_indices.append(idx)

    print("-" * 50)
    print("CONTROLS:")
    print(" [Left/Right Arrow] : Navigate frames")
    print(" [Close Window]     : Finish trajectory")
    print(f"INFO: Found {len(valid_indices)} valid start points (Green Triangles).")
    print("-" * 50)

    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1.0, 3.3, 0.2])
    ax_plot = fig.add_subplot(gs[0])
    ax_diff = fig.add_subplot(gs[1])
    ax_img = fig.add_subplot(gs[2])
    ax_slider = fig.add_subplot(gs[3])

    ax_plot.step(timesteps, no_teacher, where="post", label="no_teacher_actions", color="orange")
    if valid_indices:
        ax_plot.scatter(
            timesteps[valid_indices],
            [-0.15] * len(valid_indices),
            color="green",
            marker="^",
            s=15,
            label=f"Valid Start (len>={min_seq_len})",
        )
    cursor_line = ax_plot.axvline(timesteps[0], color="k", linestyle="-", alpha=0.8)
    ax_plot.set_xlim(timesteps[0], timesteps[-1])
    ax_plot.set_ylim(-0.3, 1.1)
    ax_plot.set_ylabel("Flag")
    ax_plot.set_yticks([0, 1])
    ax_plot.set_yticklabels(["False", "True"])
    ax_plot.legend(loc="upper right", fontsize="small", ncol=2)
    ax_plot.set_title("Trajectory Visualization")

    if teacher_action.ndim == 1:
        mask = 1 - no_teacher
        teacher_eff = teacher_action * mask
        robot_action_eff = robot_action * mask
    else:
        mask = (1 - no_teacher)[:, None]
        teacher_eff = teacher_action * mask
        robot_action_eff = robot_action * mask

    action_delta = teacher_eff - robot_action_eff
    if action_delta.ndim == 1:
        action_diff = np.abs(action_delta)[:, None]
    else:
        action_diff = np.linalg.norm(action_delta, axis=-1)[:, None]

    x_range = timesteps
    for dim_idx in range(action_diff.shape[1]):
        ax_diff.plot(x_range, action_diff[:, dim_idx], label=f"dim {dim_idx}")

    diff_cursor_line = ax_diff.axvline(0, color="k", linestyle="--", alpha=0.7)
    ax_diff.set_title("Action diff: (teacher * (not no_teacher)) - robot")
    ax_diff.set_xlim(timesteps[0], timesteps[-1])
    ax_diff.set_xlabel("Timestep")
    ax_diff.set_ylabel("Difference")
    ax_diff.legend(loc="upper right", fontsize="small")
    ax_diff.grid(True)

    img_display = ax_img.imshow(np.zeros((60, 80, 3)))
    ax_img.axis("off")
    title_text = ax_img.set_title("Step 0")
    actions_text = ax_img.text(
        0.5,
        -0.01,
        "",
        transform=ax_img.transAxes,
        ha="center",
        va="top",
        fontsize=14,
    )

    def get_processed_image(idx):
        img = traj[idx]["obs"][img_key]
        if img.ndim == 4 and img.shape[0] >= 1:
            img = img[0]
        if img.ndim == 3 and img.shape[0] in [1, 3]:
            img = np.transpose(img, (1, 2, 0))
        if bgr and img.ndim == 3 and img.shape[-1] == 3:
            img = img[..., ::-1]
        if img.min() < 0:
            img = (img + 1.0) / 2.0
        return np.clip(img, 0.0, 1.0)

    def update(_):
        idx = int(slider.val)

        img_display.set_data(get_processed_image(idx))
        valid_str = " [VALID START]" if idx in valid_indices else ""
        title_text.set_text(f"Step: {idx} | No_Teacher: {no_teacher[idx]}{valid_str}")

        teacher_str = np.array2string(teacher_action[idx], precision=3, separator=", ")
        robot_str = np.array2string(robot_action[idx], precision=3, separator=", ")
        actions_text.set_text(
            f"Teacher action: {teacher_str}\nRobot action:   {robot_str}"
        )

        cursor_line.set_xdata([timesteps[idx], timesteps[idx]])
        diff_cursor_line.set_xdata([timesteps[idx], timesteps[idx]])
        fig.canvas.draw_idle()

    slider = Slider(
        ax=ax_slider,
        label="Timestep",
        valmin=0,
        valmax=total_steps - 1,
        valinit=0,
        valstep=1,
    )
    slider.on_changed(update)

    def on_key(event):
        curr_idx = int(slider.val)
        if event.key == "right":
            slider.set_val(min(curr_idx + 1, total_steps - 1))
        elif event.key == "left":
            slider.set_val(max(curr_idx - 1, 0))

    fig.canvas.mpl_connect("key_press_event", on_key)
    update(0)
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize trajectory buffer data without labeling.")
    parser.add_argument(
        "--buffer-path",
        type=str,
        default="outputs/square_dataset_CDP/trajectory_buffer_0.hdf5",
        help="Path to the trajectory buffer hdf5 file.",
    )
    parser.add_argument(
        "--traj-id",
        type=int,
        default=0,
        help="Trajectory index to visualize.",
    )
    parser.add_argument(
        "--img-key",
        type=str,
        default="agentview_image",
        help="Image key inside step['obs'].",
    )
    parser.add_argument(
        "--max-visual-length",
        type=int,
        default=None,
        help="Optional maximum number of steps to display.",
    )
    parser.add_argument(
        "--min-seq-len",
        type=int,
        default=16,
        help="Window length used to mark valid start points.",
    )
    parser.add_argument(
        "--rgb",
        action="store_true",
        help="Treat stored images as RGB instead of BGR.",
    )
    parser.add_argument(
        "--all-trajectories",
        action="store_true",
        help="Visualize every trajectory in sequence.",
    )
    parser.add_argument(
        "--single-trajectory",
        action="store_true",
        help="Visualize only the selected traj_id and then exit.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    traj_buffer = TrajectoryBuffer()
    traj_number = traj_buffer.count_trajectories_in_hdf5(args.buffer_path)
    print("traj_number:", traj_number)

    if not 0 <= args.traj_id < traj_number:
        raise ValueError(f"traj_id {args.traj_id} is out of range [0, {traj_number - 1}]")

    if args.single_trajectory:
        traj_ids = [args.traj_id]
    elif args.all_trajectories:
        traj_ids = range(traj_number)
    else:
        traj_ids = range(args.traj_id, traj_number)

    for traj_id in traj_ids:
        print(f"\n=== Visualizing Trajectory {traj_id} / {traj_number - 1} ===")
        traj = traj_buffer.load_from_file(args.buffer_path, traj_id=traj_id)
        visualize_trajectory(
            traj=traj,
            img_key=args.img_key,
            max_visual_length=args.max_visual_length,
            bgr=not args.rgb,
            min_seq_len=args.min_seq_len,
        )
        if traj_id + 1 in traj_ids:
            print(f"Closed current window. Next trajectory: traj_id={traj_id + 1}")
        else:
            print("Closed current window. No more trajectories.")


if __name__ == "__main__":
    main()
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys, os
import time
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
)
from buffer_trajectory import TrajectoryBuffer




def replay_loaded_trajectory(traj, img_key="img", interval_sec=0.00002):
    """
    Replays frames in a loaded trajectory as an animation, showing:
      - Top: binary flags no_robot_actions / no_teacher_actions over time,
             with a vertical line marking the current step.
      - Bottom: the camera image at each step.

    Parameters:
        traj (list of dict): The trajectory.
        img_key (str): Key in obs containing the image.
        interval_sec (float): Delay between frames in seconds.
    """
    # Pre-extract time-series
    timesteps    = [t['timestep']            for t in traj]
    no_robot     = [int(t['no_robot_action'])   for t in traj]
    no_teacher   = [int(t['no_teacher_action']) for t in traj]
    teacher_action = [t['teacher_action'] for t in traj]
    obs_tactile_avg = [t['obs']['low_dim'][-1] for t in traj]

    # print("teacher_action: ", teacher_action)
    '''save the action list so that we can reply it in main kuka.py'''
    # import pickle
    # with open('teacher_action.pkl', 'wb') as f:
    #     # protocol=pickle.HIGHEST_PROTOCOL is recommended but optional
    #     pickle.dump(teacher_action, f, protocol=pickle.HIGHEST_PROTOCOL)

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

  
    # Loop through frames
    for i, step in enumerate(traj):
        print("teacher action: ", step['teacher_action'])


        # Move the vertical line (use a 2‐element list here!)
        x = timesteps[i]
        vline.set_xdata([x, x])

        fig.canvas.draw()
        fig.canvas.flush_events()
        # time.sleep(interval_sec)

    plt.ioff()
    plt.show()
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize tactile low-dimensional trajectory buffer data."
    )
    parser.add_argument("buffer_path", help="Path to the trajectory HDF5 buffer.")
    parser.add_argument("--traj-id", type=int, default=4, help="Trajectory id to load.")
    parser.add_argument(
        "--csv-path",
        default="trajectory_6.csv",
        help="Where to save the extracted low-dimensional CSV.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    buffer_path = os.path.expanduser(os.path.expandvars(args.buffer_path))

    traj_buffer = TrajectoryBuffer()
    traj_number = traj_buffer.count_trajectories_in_hdf5(buffer_path)
    print("traj_number: ", traj_number)
    traj = traj_buffer.load_from_file(buffer_path, traj_id=args.traj_id)
    obs_tactile_avg = [50 * t['obs']['low_dim'][:, -1] for t in traj]

    robot_z_vals = [t['obs']['low_dim'][:, 0] for t in traj]

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

    # tactile on top
    ax1.plot(obs_tactile_avg,   marker='o', linestyle='-')
    ax1.set_ylabel('Tactile (scaled)')
    ax1.set_title('Tactile Obs & Robot Z over Time')
    ax1.grid(True)

    # robot z below
    ax2.plot(robot_z_vals,      marker='s', linestyle='--')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Robot Z')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    import pandas as pd
    df = pd.DataFrame({
        'robot_x': [t['obs']['low_dim'][:, 0][0] for t in traj] ,
        'robot_y': [t['obs']['low_dim'][:, 1][0] for t in traj] ,
        'robot_z': [t['obs']['low_dim'][:, 2][0] for t in traj] ,
        'sensor_1': [t['obs']['low_dim'][:, -5][0] for t in traj] ,
        'sensor_2': [t['obs']['low_dim'][:, -4][0] for t in traj] ,
        'sensor_3': [t['obs']['low_dim'][:, -3][0] for t in traj] ,
        'sensor_4': [t['obs']['low_dim'][:, -2][0] for t in traj] ,
        'sensor_avg': [t['obs']['low_dim'][:, -1][0] for t in traj] ,
    })

    df.to_csv(args.csv_path, index=False)
    print(f"Saved data to {args.csv_path}")


if __name__ == "__main__":
    main()


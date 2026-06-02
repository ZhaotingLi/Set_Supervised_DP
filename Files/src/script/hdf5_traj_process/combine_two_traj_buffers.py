"""
Combine two HDF5 trajectory buffers into one buffer with renumbered episodes.

Example:
    python Files/src/script/hdf5_traj_process/combine_two_traj_buffers.py \
        --buffer-path1 /path/to/demo_buffer.hdf5 \
        --buffer-path2 /path/to/correction_buffer.hdf5 \
        --combined-buffer-path /path/to/combined_buffer.hdf5
"""

import argparse
import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC_DIR))

def main():
    parser = argparse.ArgumentParser(
        description="Combine two trajectory buffers into a new HDF5 file."
    )
    parser.add_argument(
        "--buffer-path1",
        required=True,
        help="Path to the first source HDF5 trajectory buffer.",
    )
    parser.add_argument(
        "--buffer-path2",
        required=True,
        help="Path to the second source HDF5 trajectory buffer.",
    )
    parser.add_argument(
        "--combined-buffer-path",
        required=True,
        help="Path for the combined output HDF5 trajectory buffer.",
    )
    args = parser.parse_args()

    from tools.buffer_trajectory import (
        combine_two_traj_buffers as _combine_two_traj_buffers,
    )

    episode_count = _combine_two_traj_buffers(
        args.buffer_path1,
        args.buffer_path2,
        args.combined_buffer_path,
    )
    print(f"Combined trajectories: {episode_count}")
    print(f"Combined buffer path: {args.combined_buffer_path}")


if __name__ == "__main__":
    main()

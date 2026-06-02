"""
Remove selected trajectories from an HDF5 trajectory buffer.

Example:
    python Files/src/script/hdf5_traj_process/remove_trajectories.py \
        --buffer-path /path/to/trajectory_buffer.hdf5 \
        --traj-ids 21 \
        --suffix _remove_traj_id
"""

import argparse
import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC_DIR))

def parse_ids(values):
    ids = []
    for value in values:
        if ":" in value:
            parts = value.split(":")
            if len(parts) != 2:
                raise argparse.ArgumentTypeError(
                    f"Invalid range '{value}'. Use START:STOP, for example 0:34."
                )
            start, stop = (int(part) for part in parts)
            ids.extend(range(start, stop))
        else:
            ids.append(int(value))
    return ids


def main():
    parser = argparse.ArgumentParser(
        description="Create a new HDF5 buffer without selected trajectories."
    )
    parser.add_argument(
        "--buffer-path",
        required=True,
        help="Path to the source HDF5 trajectory buffer.",
    )
    parser.add_argument(
        "--traj-ids",
        nargs="+",
        required=True,
        help="Trajectory IDs or Python-style ranges to remove, e.g. 21 or 0:3.",
    )
    parser.add_argument(
        "--suffix",
        default="_remove_traj_id",
        help="Suffix appended before .hdf5 for the output file.",
    )
    args = parser.parse_args()

    from tools.buffer_trajectory import remove_trajectories as _remove_trajectories

    traj_ids = parse_ids(args.traj_ids)
    output_path, episode_count = _remove_trajectories(
        args.buffer_path,
        traj_ids,
        suffix=args.suffix,
    )
    print(f"New buffer path: {output_path}")
    print(f"Remaining trajectories: {episode_count}")


if __name__ == "__main__":
    main()

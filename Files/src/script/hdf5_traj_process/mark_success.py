"""
Mark selected trajectories as success or failure in an HDF5 trajectory buffer.

Example:
    python Files/src/script/hdf5_traj_process/mark_success.py \
        --buffer-path /path/to/trajectory_buffer.hdf5 \
        --episode-ids all \
        --success
"""

import argparse
import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC_DIR))


def resolve_hdf5_path(buffer_path):
    if buffer_path.endswith(".hdf5"):
        return buffer_path
    return f"{buffer_path}.hdf5"


def count_episodes(buffer_path):
    import h5py

    with h5py.File(resolve_hdf5_path(buffer_path), "r") as f:
        return len(sorted(f.keys()))


def parse_ids(values, buffer_path):
    if len(values) == 1 and values[0].lower() == "all":
        return range(count_episodes(buffer_path))

    ids = []
    for value in values:
        if ":" in value:
            parts = value.split(":")
            if len(parts) != 2:
                raise argparse.ArgumentTypeError(
                    f"Invalid range '{value}'. Use START:STOP, for example 0:34."
                )
            start, stop = (int(part) for part in parts)
            if stop == -1:
                stop = count_episodes(buffer_path)
            ids.extend(range(start, stop))
        else:
            ids.append(int(value))
    return ids


def main():
    parser = argparse.ArgumentParser(
        description="Set the final if_success flag for selected trajectories."
    )
    parser.add_argument(
        "--buffer-path",
        required=True,
        help="Path to the source HDF5 trajectory buffer.",
    )
    parser.add_argument(
        "--episode-ids",
        nargs="+",
        required=True,
        help=(
            "Episode IDs, Python-style ranges, all, or ranges ending at -1, "
            "e.g. 0 1 2, 0:34, all, or 0:-1."
        ),
    )
    success_group = parser.add_mutually_exclusive_group()
    success_group.add_argument(
        "--success",
        dest="success",
        action="store_true",
        default=True,
        help="Mark selected trajectories as successful. This is the default.",
    )
    success_group.add_argument(
        "--failure",
        dest="success",
        action="store_false",
        help="Mark selected trajectories as failed.",
    )
    args = parser.parse_args()

    from tools.buffer_trajectory import mark_success as _mark_success

    episode_ids = parse_ids(args.episode_ids, args.buffer_path)
    _mark_success(args.buffer_path, episode_ids, success=args.success)


if __name__ == "__main__":
    main()

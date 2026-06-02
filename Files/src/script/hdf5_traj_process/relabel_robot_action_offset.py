"""
Relabel robot_actions as teacher_actions plus a fixed L2 offset.

This is intended for demonstration buffers. Do not apply it to correction
datasets unless you specifically want to overwrite robot_actions there.

Example:
    python Files/src/script/hdf5_traj_process/relabel_robot_action_offset.py \
        --buffer-path /path/to/demo_buffer.hdf5 \
        --episode-ids all \
        --scale 0.1
"""

import argparse
import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC_DIR))

def parse_ids(values):
    if len(values) == 1 and values[0].lower() == "all":
        return "all"

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


def parse_direction(value):
    if value is None:
        return None
    return [float(part.strip()) for part in value.split(",")]


def parse_clip(value):
    if value is None:
        return None
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"Invalid clip '{value}'. Use MIN,MAX, for example -1.0,1.0."
        )
    return tuple(parts)


def main():
    parser = argparse.ArgumentParser(
        description="Overwrite robot_actions with teacher_actions plus an offset."
    )
    parser.add_argument(
        "--buffer-path",
        required=True,
        help="Path to the source HDF5 trajectory buffer.",
    )
    parser.add_argument(
        "--episode-ids",
        nargs="+",
        default=["all"],
        help="Episode IDs, Python-style ranges, or all. Default: all.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.1,
        help="Target L2 distance between robot and teacher actions.",
    )
    parser.add_argument(
        "--direction",
        default=None,
        help="Comma-separated fixed direction vector, e.g. 1,0,0,0.",
    )
    parser.add_argument(
        "--randomize",
        action="store_true",
        help="Use random unit directions when --direction is not set.",
    )
    parser.add_argument(
        "--per-timestep",
        action="store_true",
        help="With --randomize, sample a direction per timestep.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed used with --randomize.",
    )
    parser.add_argument(
        "--clip",
        default=None,
        help="Optional comma-separated min,max clipping range, e.g. -1.0,1.0.",
    )
    args = parser.parse_args()

    from tools.buffer_trajectory import (
        relabel_robot_action_offset as _relabel_robot_action_offset,
    )

    _relabel_robot_action_offset(
        args.buffer_path,
        episode_ids=parse_ids(args.episode_ids),
        scale=args.scale,
        direction=parse_direction(args.direction),
        randomize=args.randomize,
        per_timestep=args.per_timestep,
        seed=args.seed,
        clip=parse_clip(args.clip),
    )


if __name__ == "__main__":
    main()

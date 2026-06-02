"""
Relabel correction/intervention trajectory segments from a manual-label JSON.

The JSON should be produced by:
    Files/src/script/visualize_traj_buffer_data_and_manual_label.py

For selected ranges where no_teacher_actions[t] is True, this creates a new
buffer with:
    teacher_actions[t] = old robot_actions[t]
    robot_actions[t] = teacher_actions[t] + unit_direction * scale
    no_teacher_actions[t] = False

Example:
    python3 Files/src/script/hdf5_traj_process/relabel_teacher_robot_segments_from_json.py \
        --buffer-path /path/to/correction_buffer.hdf5 \
        --json-path /path/to/labeled_regions_correction_buffer.json \
        --scale 0.3 \
        --randomize \
        --seed 42
"""

import argparse
import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC_DIR))


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
        description=(
            "Create a relabeled correction buffer from manually labeled "
            "trajectory segment ranges."
        )
    )
    parser.add_argument(
        "--buffer-path",
        required=True,
        help="Path to the source correction/intervention HDF5 trajectory buffer.",
    )
    parser.add_argument(
        "--json-path",
        required=True,
        help=(
            "Path to labeled_regions_*.json produced by "
            "visualize_traj_buffer_data_and_manual_label.py."
        ),
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.1,
        help="Target L2 distance between relabeled robot and teacher actions.",
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
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the default *_relabeled.hdf5 output if it exists.",
    )
    args = parser.parse_args()

    from tools.buffer_trajectory import (
        relabel_teacher_robot_segments_from_json as _relabel_segments,
    )

    _relabel_segments(
        src_buffer_path=args.buffer_path,
        json_path=args.json_path,
        scale=args.scale,
        direction=parse_direction(args.direction),
        randomize=args.randomize,
        per_timestep=args.per_timestep,
        seed=args.seed,
        clip=parse_clip(args.clip),
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

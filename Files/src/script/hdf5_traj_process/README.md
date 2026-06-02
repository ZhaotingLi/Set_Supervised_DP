# HDF5 Trajectory Processing

This folder contains small command-line scripts for processing trajectory HDF5
buffers. The scripts wrap the helper functions in `Files/src/tools/buffer_trajectory.py`.

Recommended workflow:

For demonstration datasets:

1. [Visualize and inspect the recorded buffer](#1-visualize-and-inspect).
2. [Mark successful trajectories](#2-mark-success).
3. [Remove low-quality trajectories](#3-remove-trajectories).
4. [Relabel demonstration `robot_actions`](#4-relabel-robot-action-offset-for-demonstration-data) if needed.

For correction/intervention datasets:

1. [Visualize and inspect the recorded buffer](#1-visualize-and-inspect).
2. [Mark successful trajectories](#2-mark-success).
3. [Remove low-quality trajectories](#3-remove-trajectories).
4. [Create manual-label JSON and relabel teacher/robot action segments](#5-relabel-teacherrobot-segments-from-json-for-correction-data) if needed.
5. [Combine demonstration and correction buffers](#6-combine-two-trajectory-buffers) for training.

## 1. Visualize and Inspect

Before editing a trajectory buffer, inspect the recorded data and identify:

- which trajectories are successful,
- which trajectories should be removed,
- for correction/intervention data, which timestep windows should be relabeled.

For general visualization, use:

```bash
python3 Files/src/script/visualize_traj_buffer_data_and_manual_label.py
```

Before running it, edit `buffer_path` inside
this file so it points to the HDF5 file
you want to inspect. This script is useful for checking trajectory quality,
image observations, actions, and trajectory IDs before running the processing
steps below.

For correction/intervention data, this script is also used to collect manual timestep labels, described in detail in
[Section 5](#5-relabel-teacherrobot-segments-from-json-for-correction-data).

## 2. Mark Success

Use `mark_success.py` to set the final `if_success` value for selected
episodes. This edits the input HDF5 file in place.

To mark all episodes as successful:

```bash
python3 Files/src/script/hdf5_traj_process/mark_success.py \
  --buffer-path /path/to/trajectory_buffer.hdf5 \
  --episode-ids all \
  --success
```

Use a normal Python-style range for a subset:

```bash
python3 Files/src/script/hdf5_traj_process/mark_success.py \
  --buffer-path /path/to/trajectory_buffer.hdf5 \
  --episode-ids 0:34 \
  --success
```

To mark specific episodes:

```bash
python3 Files/src/script/hdf5_traj_process/mark_success.py \
  --buffer-path /path/to/trajectory_buffer.hdf5 \
  --episode-ids 0 1 2 5 \
  --success
```

To mark selected episodes as failures:

```bash
python3 Files/src/script/hdf5_traj_process/mark_success.py \
  --buffer-path /path/to/trajectory_buffer.hdf5 \
  --episode-ids 3 7 \
  --failure
```

Notes:

- `0:34` means episode IDs `0` through `33`, following Python range behavior.
- `0:-1` is a special convenience form in this script. It marks through the
  final episode.
- The script modifies the original file directly.

## 3. Remove Trajectories

Use `remove_trajectories.py` to create a new HDF5 buffer without selected
trajectories. The original file is not modified.

```bash
python3 Files/src/script/hdf5_traj_process/remove_trajectories.py \
  --buffer-path /path/to/trajectory_buffer.hdf5 \
  --traj-ids 21
```

Remove multiple trajectories:

```bash
python3 Files/src/script/hdf5_traj_process/remove_trajectories.py \
  --buffer-path /path/to/trajectory_buffer.hdf5 \
  --traj-ids 3 9 14 17
```

Remove a range:

```bash
python3 Files/src/script/hdf5_traj_process/remove_trajectories.py \
  --buffer-path /path/to/trajectory_buffer.hdf5 \
  --traj-ids 0:3
```

By default, the output file name is:

```text
trajectory_buffer_remove_traj_id.hdf5
```

You can choose a different suffix:

```bash
python3 Files/src/script/hdf5_traj_process/remove_trajectories.py \
  --buffer-path /path/to/trajectory_buffer.hdf5 \
  --traj-ids 21 \
  --suffix _cleaned
```

This creates:

```text
trajectory_buffer_cleaned.hdf5
```

## 4. Relabel Robot Action Offset for Demonstration Data

Use `relabel_robot_action_offset.py` to overwrite `robot_actions` with:

```text
robot_actions = teacher_actions + unit_direction * scale
```

This is mainly intended for demonstration buffers. **Do not apply this to
correction or intervention datasets** unless you intentionally want to overwrite
their `robot_actions`.

Relabel all episodes with the default scale `0.1`:

```bash
python3 Files/src/script/hdf5_traj_process/relabel_robot_action_offset.py \
  --buffer-path /path/to/demo_buffer.hdf5 \
  --episode-ids all \
  --scale 0.1
```

Notes:

- This script modifies the input HDF5 file in place.
- The input buffer must contain both `teacher_actions` and `robot_actions`.

## 5. Relabel Teacher/Robot Segments From JSON for Correction Data

Use `relabel_teacher_robot_segments_from_json.py` to process a correction or
intervention dataset using the JSON file produced by
`Files/src/script/visualize_traj_buffer_data_and_manual_label.py`.

First create the JSON labels with the manual labeling script:

Before running it, edit `buffer_path` inside
`Files/src/script/visualize_traj_buffer_data_and_manual_label.py` so it points
to the correction HDF5 file you want to label.

```bash
python3 Files/src/script/visualize_traj_buffer_data_and_manual_label.py
```

In the labeling window:

1. Drag the slider to choose the timestep.
2. Press `A` to add the selected timestep as the start timestep.
3. Drag the slider to the corresponding end timestep.
4. Press `E` to add the end timestep.
5. The selected time window is marked in red.
6. Press `U` to undo the latest selected time window if needed.

The manual labeling script saves a JSON file next to the buffer named like:

```text
labeled_regions_<buffer_name>.json
```

The expected JSON format maps trajectory IDs to labeled timestep ranges:

```json
{
  "0": [[10, 25], [40, 55]],
  "1": [[5, 30]]
}
```

Here, `"0"` and `"1"` are trajectory IDs (`traj_id`). Each inner pair is a
selected time window in the format `[start_t, end_t]`, inclusive.

After the JSON file is created, `relabel_teacher_robot_segments_from_json.py`
reads it and creates a new relabeled HDF5 buffer. For each labeled timestep
where `no_teacher_actions[t]` is `True`, it applies:

```text
teacher_actions[t] = old robot_actions[t]
robot_actions[t] = teacher_actions[t] + unit_direction * scale
no_teacher_actions[t] = False
```

Then run this command to relabel the correction dataset with the JSON file:

```bash
python3 Files/src/script/hdf5_traj_process/relabel_teacher_robot_segments_from_json.py \
  --buffer-path /path/to/correction_buffer.hdf5 \
  --json-path /path/to/labeled_regions_correction_buffer.json \
  --scale 0.3 \
  --seed 42
```

Notes:

- This script does not modify the source file.
- The output is written next to the source as `<buffer_name>_relabeled.hdf5`.
- The source buffer must contain `teacher_actions`, `robot_actions`, and
  `no_teacher_actions`.
- Only timesteps with `no_teacher_actions[t] == True` are changed.

## 6. Combine Two Trajectory Buffers

Use `combine_two_traj_buffers.py` to combine two HDF5 buffers into one output
file. Episodes are copied from both files and renumbered consecutively starting
from `episode_0000`.

```bash
python3 Files/src/script/hdf5_traj_process/combine_two_traj_buffers.py \
  --buffer-path1 /path/to/demo_buffer_relabeled.hdf5 \
  --buffer-path2 /path/to/correction_buffer_relabeled.hdf5 \
  --combined-buffer-path /path/to/combined_buffer.hdf5
```

Notes:

- The combined output file is overwritten if it already exists.
- The two input paths must point to different files.

## Typical Full Example

```bash
# 1. Inspect the buffer first. Edit buffer_path inside the script before running.
python3 Files/src/script/visualize_traj_buffer_data.py

# 2. Mark all good demonstration trajectories as successful.
python3 Files/src/script/hdf5_traj_process/mark_success.py \
  --buffer-path /path/to/demo_buffer.hdf5 \
  --episode-ids all \
  --success

# 3. Remove bad trajectories.
python3 Files/src/script/hdf5_traj_process/remove_trajectories.py \
  --buffer-path /path/to/demo_buffer.hdf5 \
  --traj-ids 21 \
  --suffix _cleaned

# 4. Relabel robot actions in the cleaned demonstration buffer.
python3 Files/src/script/hdf5_traj_process/relabel_robot_action_offset.py \
  --buffer-path /path/to/demo_buffer_cleaned.hdf5 \
  --episode-ids all \
  --scale 0.1

# 5. Create correction labels. Edit buffer_path inside the script before running.
python3 Files/src/script/visualize_traj_buffer_data_and_manual_label.py

# 6. Relabel correction segments using the manual-label JSON.
python3 Files/src/script/hdf5_traj_process/relabel_teacher_robot_segments_from_json.py \
  --buffer-path /path/to/correction_buffer.hdf5 \
  --json-path /path/to/labeled_regions_correction_buffer.json \
  --scale 0.3 \
  --randomize \
  --seed 42

# 7. Combine demonstration and relabeled correction buffers.
python3 Files/src/script/hdf5_traj_process/combine_two_traj_buffers.py \
  --buffer-path1 /path/to/demo_buffer_cleaned.hdf5 \
  --buffer-path2 /path/to/correction_buffer_relabeled.hdf5 \
  --combined-buffer-path /path/to/combined_buffer.hdf5
```

After processing, inspect the combined dataset before using it for training.

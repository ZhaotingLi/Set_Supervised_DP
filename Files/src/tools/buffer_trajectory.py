import logging
# trajectory_buffer.py
import pickle

import argparse
import os
import pickle

import imageio
import cv2
import pdb
import h5py
import numpy as np

import zarr

logger = logging.getLogger(__name__)


def relabel_robot_action_offset(
    buffer_path,
    episode_ids="all",
    scale=0.1,
    direction=None,          # np.ndarray of shape (action_dim,), or None
    randomize=False,         # if True and direction is None, sample random unit directions
    per_timestep=False,      # when randomize=True, choose new dir each timestep instead of per-episode
    seed=None,
    clip=None,               # (min, max) or None
):
    """
    Overwrite robot_actions with teacher_actions + u * scale, ensuring
    ||robot - teacher||_2 == scale.

    Parameters
    ----------
    buffer_path : str
        Path to the HDF5 buffer (.hdf5). '.hdf5' is appended if missing.
    episode_ids : 'all' | list[int]
        Episodes to relabel (0-based). Default 'all'.
    scale : float
        Target L2 distance between robot and teacher actions.
    direction : np.ndarray | None
        If provided, a fixed direction vector of shape (action_dim,). Will be normalized.
        If None and randomize=False, a fixed deterministic direction of all-ones is used.
    randomize : bool
        If True (and direction is None), sample random unit directions.
        - per_timestep=False: one random direction per episode (applied to all timesteps).
        - per_timestep=True: new random direction for each timestep.
    per_timestep : bool
        See 'randomize'.
    seed : int | None
        RNG seed for reproducibility when randomize=True.
    clip : tuple[float, float] | None
        (min, max) to clip resulting robot actions. No clipping if None.
    """
    def _unit(v):
        v = np.asarray(v, dtype=np.float32)
        n = np.linalg.norm(v)
        if n == 0:
            # fallback to e1
            v = np.zeros_like(v)
            v[0] = 1.0
            return v
        return v / n

    def _random_unit(shape, rng):
        v = rng.normal(size=shape).astype(np.float32)
        n = np.linalg.norm(v, axis=-1, keepdims=True)
        # handle rare zeros
        n[n == 0] = 1.0
        return v / n

    if not buffer_path.endswith(".hdf5"):
        buffer_path = buffer_path + ".hdf5"
    if not os.path.exists(buffer_path):
        raise FileNotFoundError(f"Buffer not found: {buffer_path}")

    rng = np.random.default_rng(seed)

    with h5py.File(buffer_path, "r+") as f:
        episode_keys = sorted(k for k in f.keys() if k.startswith("episode_"))
        if episode_ids == "all":
            target_ids = list(range(len(episode_keys)))
        else:
            max_id = len(episode_keys) - 1
            for eid in episode_ids:
                if eid < 0 or eid > max_id:
                    raise IndexError(f"Episode index {eid} out of range (0–{max_id})")
            target_ids = list(episode_ids)

        for eid in target_ids:
            ep_key = episode_keys[eid]
            grp = f[ep_key]
            if "teacher_actions" not in grp or "robot_actions" not in grp:
                logger.warning(f"Skipping {ep_key}: required datasets missing.")
                continue

            teacher = grp["teacher_actions"][()]            # shape (T, A)
            T, A = teacher.shape[-2], teacher.shape[-1]

            # Build unit direction(s)
            if direction is not None:
                u = _unit(direction).reshape(1, A)          # fixed, broadcast across T
                u = np.repeat(u, T, axis=0)
            elif randomize:
                if per_timestep:
                    u = _random_unit((T, A), rng)           # different per timestep
                else:
                    u = _random_unit((1, A), rng)           # one per episode
                    u = np.repeat(u, T, axis=0)
            else:
                # deterministic fixed direction of ones
                u = _unit(np.ones(A, dtype=np.float32)).reshape(1, A)
                u = np.repeat(u, T, axis=0)

            new_robot = teacher + scale * u

            if clip is not None:
                lo, hi = clip
                new_robot = np.clip(new_robot, lo, hi)

            # preserve dtype/shape of original dataset
            # cast if needed to match existing dtype
            dtype = grp["robot_actions"].dtype
            grp["robot_actions"][...] = new_robot.astype(dtype, copy=False)

            mode = (
                "fixed direction"
                if direction is not None else
                ("random per timestep" if (randomize and per_timestep) else
                 ("random per episode" if randomize else "deterministic fixed"))
            )
            logger.info(f"Relabeled {ep_key}: ||Δ||_2={scale} ({mode})")


def mark_success(buffer_path, episode_ids, success=True):
    """
    Manually change the last `if_success` flag of selected trajectories.

    Parameters
    ----------
    buffer_path : str
        Path to the HDF5 buffer file (.hdf5).
    episode_ids : list[int]
        List of episode indices to modify (0-based, matching episode_0000, ...).
    success : bool, optional
        Value to set for the last step (default True).
    """
    # Ensure extension
    if not buffer_path.endswith(".hdf5"):
        buffer_path = buffer_path + ".hdf5"

    with h5py.File(buffer_path, "r+") as f:  # r+ allows modification
        episode_keys = sorted(f.keys())  # ['episode_0000', 'episode_0001', ...]

        for eid in episode_ids:
            if eid < 0 or eid >= len(episode_keys):
                raise IndexError(
                    f"Episode index {eid} out of range (0–{len(episode_keys)-1})"
                )
            ep_key = episode_keys[eid]
            dset = f[ep_key]["if_success"]

            # Ensure it's writable
            arr = dset[()]
            arr[-1] = success  # modify last timestep
            dset[...] = arr    # overwrite dataset in-place

            logger.info(f"Updated {ep_key}: last if_success set to {success}")
            

def remove_trajectories(buffer_path, traj_ids, suffix="_remove_traj_id"):
    """
    Create a new HDF5 buffer without the given trajectory IDs.

    Parameters
    ----------
    buffer_path : str
        Path to the source HDF5 buffer (.hdf5).
    traj_ids : list[int]
        List of episode indices (0-based) to remove.
    suffix : str, optional
        Extra name suffix for the output file (default "_remove_traj_id").

    Returns
    -------
    str
        Path to the new file containing filtered trajectories.
    int
        Number of episodes in the new file.
    """
    if not buffer_path.endswith(".hdf5"):
        buffer_path = buffer_path + ".hdf5"

    if not os.path.exists(buffer_path):
        raise FileNotFoundError(f"Buffer not found: {buffer_path}")

    output_path = buffer_path.replace(".hdf5", f"{suffix}.hdf5")

    with h5py.File(buffer_path, "r") as f_in, h5py.File(output_path, "w") as f_out:
        episode_keys = sorted(f_in.keys())  # ['episode_0000', ...]

        # Validate IDs
        max_id = len(episode_keys) - 1
        for tid in traj_ids:
            if tid < 0 or tid > max_id:
                raise IndexError(f"Trajectory index {tid} out of range (0–{max_id})")

        keep_ids = [i for i in range(len(episode_keys)) if i not in traj_ids]

        # Copy episodes, renumber consecutively
        for new_idx, old_idx in enumerate(keep_ids):
            old_key = episode_keys[old_idx]
            new_key = f"episode_{new_idx:04d}"
            f_in.copy(f_in[old_key], f_out, name=new_key)

    return output_path, len(keep_ids)

def combine_two_traj_buffers(buffer_path1, buffer_path2, combined_buffer_path):
    """
    Read two HDF5 trajectory buffers (with groups named 'episode_XXXX') and
    write a new HDF5 file that contains all episodes from both, renumbered
    consecutively starting at episode_0000.

    Parameters
    ----------
    buffer_path1 : str
        Path to the first buffer file. If it doesn't end with '.hdf5', the
        function will also try '<path>.hdf5'.
    buffer_path2 : str
        Path to the second buffer file. Same resolution behavior as buffer_path1.
    combined_buffer_path : str
        Path for the output combined buffer file. If it doesn't end with
        '.hdf5', '.hdf5' will be appended. The file will be overwritten if it
        exists.

    Returns
    -------
    int
        Total number of episodes written to the combined file.
    """
    import os
    import h5py

    def _resolve_hdf5_path(p, must_exist=True):
        if p.endswith(".hdf5"):
            cand = p
        else:
            cand = p + ".hdf5"
            if os.path.exists(p) and not os.path.isdir(p):
                # User gave an existing file without .hdf5 extension
                cand = p
        if must_exist and not os.path.exists(cand):
            raise FileNotFoundError(f"Buffer not found: {p} (tried '{cand}')")
        return cand

    def _episode_keys(h5file):
        # Sort numerically by the XXXX part, but accept any order if parse fails
        keys = [k for k in h5file.keys() if k.startswith("episode_")]
        def _keynum(k):
            try:
                return int(k.split("_")[-1])
            except Exception:
                return float("inf")
        return sorted(keys, key=_keynum)

    # Resolve paths
    src1 = _resolve_hdf5_path(buffer_path1, must_exist=True)
    src2 = _resolve_hdf5_path(buffer_path2, must_exist=True)
    if os.path.abspath(src1) == os.path.abspath(src2):
        raise ValueError("buffer_path1 and buffer_path2 refer to the same file.")
    dst  = _resolve_hdf5_path(combined_buffer_path, must_exist=False)

    # Create / overwrite destination
    total_written = 0
    with h5py.File(src1, "r") as f1, h5py.File(src2, "r") as f2, h5py.File(dst, "w") as fout:
        # Helper: copy all episodes from a source file into fout, renaming/renumbering
        def _copy_from(src_file, start_index):
            idx = start_index
            for ep_key in _episode_keys(src_file):
                new_name = f"episode_{idx:04d}"
                # h5py high-level copy preserves datasets, compression, and attributes.
                # Call copy from the *source* file, with destination group and new name.
                src_file.copy(src_file[ep_key], fout, name=new_name)
                idx += 1
            return idx

        next_idx = _copy_from(f1, 0)
        next_idx = _copy_from(f2, next_idx)
        total_written = next_idx

    return total_written






import json



def relabel_teacher_robot_segments_from_json(
    src_buffer_path,
    json_path,
    scale=0.1,
    direction=None,          # same semantics as relabel_robot_action_offset
    randomize=False,
    per_timestep=False,
    seed=None,
    clip=None,               # (min, max) or None
    overwrite=False,
):
    """
    Create a NEW HDF5 buffer with relabeled trajectories, leaving the
    original buffer untouched.

    Reads trajectories from `src_buffer_path`, copies everything into
    `dst_buffer_path`, and then, according to `json_path`, for selected
    timesteps where no_teacher_actions[t] is True, performs:

        teacher_actions[t]   = robot_actions[t]          (old robot)
        robot_actions[t]     = teacher_actions[t] + scale * u[t]
        no_teacher_actions[t] = False

    where u[t] is a unit direction (same logic as relabel_robot_action_offset).

    Parameters
    ----------
    src_buffer_path : str
        Path to the source HDF5 buffer (.hdf5). '.hdf5' is appended if missing.
    dst_buffer_path : str
        Path to the NEW HDF5 buffer to be created. '.hdf5' is appended if missing.
    json_path : str
        Path to JSON mapping "traj_id" -> list of [start_idx, end_idx].
    scale, direction, randomize, per_timestep, seed, clip, overwrite :
        Same semantics as discussed before.
    """
    # --- helpers (same as in relabel_robot_action_offset) ------------------
    def _unit(v):
        v = np.asarray(v, dtype=np.float32)
        n = np.linalg.norm(v)
        if n == 0:
            v = np.zeros_like(v)
            v[0] = 1.0
            return v
        return v / n

    def _random_unit(shape, rng):
        v = rng.normal(size=shape).astype(np.float32)
        n = np.linalg.norm(v, axis=-1, keepdims=True)
        n[n == 0] = 1.0
        return v / n
    # ----------------------------------------------------------------------

    if not src_buffer_path.endswith(".hdf5"):
        src_buffer_path = src_buffer_path + ".hdf5"
    dst_buffer_path = src_buffer_path.replace(".hdf5", "_relabeled.hdf5")

    if not os.path.exists(src_buffer_path):
        raise FileNotFoundError(f"Source buffer not found: {src_buffer_path}")

    if os.path.exists(dst_buffer_path) and not overwrite:
        raise FileExistsError(
            f"Destination buffer already exists: {dst_buffer_path} "
            f"(set overwrite=True if you want to replace it)."
        )

    # Load edit specification
    with open(json_path, "r") as f:
        edit_spec = json.load(f)

    rng = np.random.default_rng(seed)

    # 1) Copy source file -> destination file
    with h5py.File(src_buffer_path, "r") as src_f, h5py.File(dst_buffer_path, "w") as dst_f:
        # Copy groups/datasets
        for k in src_f.keys():
            src_f.copy(k, dst_f, name=k)
        # Copy root attributes if any
        for k, v in src_f.attrs.items():
            dst_f.attrs[k] = v

        # 2) Now modify ONLY the destination file
        episode_keys = sorted(k for k in dst_f.keys() if k.startswith("episode_"))

        for traj_id_str, ranges in edit_spec.items():
            logger.debug('traj_id_str:  %s', traj_id_str)
            traj_id = int(traj_id_str)
            if traj_id < 0 or traj_id >= len(episode_keys):
                logger.warning(f"[WARN] traj_id {traj_id} out of range (0–{len(episode_keys)-1}), skipping.")
                continue

            ep_key = episode_keys[traj_id]
            grp = dst_f[ep_key]

            # required datasets
            required = ["teacher_actions", "robot_actions", "no_teacher_actions"]
            if any(r not in grp for r in required):
                logger.warning(f"[WARN] Skipping {ep_key}: required datasets missing.")
                continue

            teacher = grp["teacher_actions"][()]          # (T, A)
            robot   = grp["robot_actions"][()]            # (T, A)
            no_teacher = grp["no_teacher_actions"][()]    # (T,)

            T, A = teacher.shape[-2], teacher.shape[-1]

            # --- build unit direction(s) u[t] ------------------------------
            if direction is not None:
                u = _unit(direction).reshape(1, A)
                u = np.repeat(u, T, axis=0)               # (T, A)
                mode = "fixed direction"
            elif randomize:
                if per_timestep:
                    u = _random_unit((T, A), rng)
                    mode = "random per timestep"
                else:
                    u_ep = _random_unit((1, A), rng)
                    u = np.repeat(u_ep, T, axis=0)
                    mode = "random per episode"
            else:
                u = _unit(np.ones(A, dtype=np.float32)).reshape(1, A)
                u = np.repeat(u, T, axis=0)
                mode = "deterministic fixed direction"
            # ----------------------------------------------------------------

            logger.info(f"Processing trajectory {traj_id} ({ep_key}), T={T}, mode={mode}")

            for start_idx, end_idx in ranges:
                start = max(0, start_idx)
                end   = min(T - 1, end_idx)

                if start > end:
                    logger.warning(f"[WARN] Range [{start_idx}, {end_idx}] "
                          f"-> invalid after clamping [{start}, {end}], skipping.")
                    continue

                for t in range(start, end + 1):
                    if not no_teacher[t]:
                        continue  # only change where no_teacher_actions[t] is True

                    old_robot = robot[t].astype(np.float32, copy=True)

                    # 1) teacher_actions[t] = old robot
                    teacher[t] = old_robot

                    # 2) robot_actions[t] = teacher_actions[t] + scale * u[t]
                    new_robot = teacher[t] + scale * u[t]

                    if clip is not None:
                        lo, hi = clip
                        new_robot = np.clip(new_robot, lo, hi)

                    robot[t] = new_robot.astype(robot.dtype, copy=False)

                    # 3) no_teacher_actions[t] = False
                    no_teacher[t] = False

            # write back
            grp["teacher_actions"][...] = teacher.astype(grp["teacher_actions"].dtype, copy=False)
            grp["robot_actions"][...]   = robot.astype(grp["robot_actions"].dtype, copy=False)
            grp["no_teacher_actions"][...] = no_teacher

            logger.info(f"Finished relabeling segments in trajectory {traj_id} ({ep_key}).")

    logger.info(f"Done. New buffer written to: {dst_buffer_path}")


def remove_teacher_actions_segments_from_json(
    src_buffer_path,
    json_path,
    overwrite=False,
):
    """
    Read from src_buffer_path, write a *new* buffer where specified teacher
    actions are marked as removed.

    For each trajectory id in `json_path` and for each [start_idx, end_idx]:
        for t in [start_idx, end_idx]:
            no_teacher_actions[t] = True

    The original file is not modified. The new file is saved as
        src_buffer_path (normalized to .hdf5) with suffix '_rmteacher.hdf5'.

    Parameters
    ----------
    src_buffer_path : str
        Path to the source HDF5 buffer. '.hdf5' is appended if missing.
    json_path : str
        Path to JSON:
            {
              "0": [[start0, end0], [start1, end1], ...],
              "1": [[...], ...],
              ...
            }
    overwrite : bool
        If False (default) and destination already exists, raise FileExistsError.
    """
    # Normalize source path
    if not src_buffer_path.endswith(".hdf5"):
        src_buffer_path = src_buffer_path + ".hdf5"

    if not os.path.exists(src_buffer_path):
        raise FileNotFoundError(f"Source buffer does not exist: {src_buffer_path}")

    # Destination path: src + '_rmteacher'
    dst_buffer_path = src_buffer_path.replace(".hdf5", "_rmteacher.hdf5")

    if os.path.exists(dst_buffer_path) and not overwrite:
        raise FileExistsError(
            f"Destination buffer already exists: {dst_buffer_path}\n"
            f"Use overwrite=True if you want to regenerate it."
        )

    # Load JSON spec
    with open(json_path, "r") as f:
        edit_spec = json.load(f)

    # Copy + modify
    with h5py.File(src_buffer_path, "r") as src_f, h5py.File(dst_buffer_path, "w") as dst_f:
        # Copy all groups/datasets
        for key in src_f.keys():
            src_f.copy(key, dst_f)

        # Copy root attributes if any
        for k, v in src_f.attrs.items():
            dst_f.attrs[k] = v

        # Work on destination only
        episode_keys = sorted(k for k in dst_f.keys() if k.startswith("episode_"))

        for traj_id_str, ranges in edit_spec.items():
            traj_id = int(traj_id_str)
            if traj_id < 0 or traj_id >= len(episode_keys):
                logger.warning(f"[WARN] traj_id {traj_id} out of range (0–{len(episode_keys)-1}), skipping.")
                continue

            ep_key = episode_keys[traj_id]
            grp = dst_f[ep_key]

            if "no_teacher_actions" not in grp:
                logger.warning(f"[WARN] Skipping {ep_key}: 'no_teacher_actions' dataset missing.")
                continue

            no_teacher = grp["no_teacher_actions"][()]
            T = no_teacher.shape[0]

            logger.info(f"Processing trajectory {traj_id} ({ep_key}), T={T}")

            for start_idx, end_idx in ranges:
                # Clamp to valid range
                start = max(0, start_idx)
                end   = min(T - 1, end_idx)

                if start > end:
                    logger.warning(f"[WARN] Invalid range [{start_idx}, {end_idx}] "
                          f"after clamping -> [{start}, {end}], skipping.")
                    continue

                # Mark teacher as "removed" in this segment
                no_teacher[start:end+1] = True

            # Write back
            grp["no_teacher_actions"][...] = no_teacher

    logger.debug(f"\n✔ Done.\nNew buffer with teacher actions removed in segments saved to:\n  {dst_buffer_path}\n")
    return dst_buffer_path


import numpy as np
import h5py

class TrajectoryBuffer:
    def __init__(self):
        self.trajectories = []
        self.current_trajectory = []
        self.file_name = None

    # -----------------------------
    # Image helpers
    # -----------------------------
    @staticmethod
    def _looks_like_image(key, arr) -> bool:
        """
        Heuristic: treat as image if:
          - key name suggests image/rgb/cam, OR
          - array shape looks like (C,H,W) or (H,W,C) with C in {1,3,4}
        """
        if not isinstance(arr, np.ndarray):
            return False

        key_l = str(key).lower()
        key_hint = any(s in key_l for s in ["rgb", "image", "img", "camera", "cam"])

        return key_hint 

    @staticmethod
    def _float_img_to_uint8(x: np.ndarray) -> np.ndarray:
        """
        Assumes float image is in [-1, 1]. Converts to uint8 [0, 255].
        """
        x = np.clip(x, -1.0, 1.0)
        x01 = (x + 1.0) * 0.5  # [-1,1] -> [0,1]
        return np.clip(x01 * 255.0, 0.0, 255.0).astype(np.uint8)

    @staticmethod
    def _uint8_img_to_float(x: np.ndarray) -> np.ndarray:
        """
        Converts uint8 [0,255] to float32 [-1,1].
        """
        x = x.astype(np.float32)
        x01 = x / 255.0
        return (2.0 * x01 - 1.0).astype(np.float32)

    @staticmethod
    def _is_float_dtype(arr: np.ndarray) -> bool:
        return np.issubdtype(arr.dtype, np.floating)

    @staticmethod
    def _serialize_state(state):
        return np.frombuffer(
            pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL),
            dtype=np.uint8,
        )

    @staticmethod
    def _deserialize_state(state_bytes):
        return pickle.loads(np.asarray(state_bytes, dtype=np.uint8).tobytes())

    # -----------------------------
    # Buffer methods
    # -----------------------------
    def add_transition(
        self, obs, teacher_action, robot_action, done, timestep,
        no_robot_action, no_teacher_action, if_success=False,
        episode_id=None, state=None
    ):
        transition = {
            "obs": obs,
            "robot_action": robot_action,
            "teacher_action": teacher_action,
            "done": done,
            "timestep": timestep,
            "no_teacher_action": no_teacher_action,
            "no_robot_action": no_robot_action,
            "episode_id": episode_id,
            "if_success": if_success,
            "state": state,
        }
        self.current_trajectory.append(transition)

    def finish_trajectory(self):
        if self.current_trajectory:
            self.trajectories.append(self.current_trajectory)
            self.current_trajectory = []

    def save_to_file(self, filename):
        if not self.trajectories:
            return

        filename = filename + ".hdf5"
        self.file_name = filename

        with h5py.File(filename, "a") as f:
            current_num_episodes = len(f.keys())

            for idx, traj in enumerate(self.trajectories):
                group = f.create_group(f"episode_{current_num_episodes + idx:04d}")

                obs_dict = {}
                robot_actions, teacher_actions, dones, timesteps = [], [], [], []
                no_robot_actions, no_teacher_actions = [], []
                if_success = []
                states = []

                for t in traj:
                    for k, v in t["obs"].items():
                        obs_dict.setdefault(k, []).append(v)

                    robot_actions.append(t["robot_action"])
                    teacher_actions.append(t["teacher_action"])
                    dones.append(t["done"])
                    timesteps.append(t["timestep"])
                    no_robot_actions.append(t["no_robot_action"])
                    no_teacher_actions.append(t["no_teacher_action"])
                    if_success.append(t["if_success"])
                    states.append(self._serialize_state(t.get("state", None)))

                # Save observation
                obs_group = group.create_group("observation")
                for k, v_list in obs_dict.items():
                    arr = np.stack(v_list, axis=0)  # (T, ...)
                    # ---- NEW: image float[-1,1] -> uint8 ----
                    if self._looks_like_image(k, arr) and self._is_float_dtype(arr):
                        # Only convert if it actually looks like normalizeda [-1,1]
                        # shape of arr for image key: [traj_length, obs_T, C, H, W]
                        arr_min = float(np.min(arr))
                        arr_max = float(np.max(arr))
                        if arr_min >= -1.01 and arr_max <= 1.01:
                            arr = self._float_img_to_uint8(arr)
                        # else: keep as-is (could be other float obs)
            
                    obs_group.create_dataset(k, data=arr, compression="gzip")

                group.create_dataset("robot_actions", data=np.stack(robot_actions), compression="gzip")
                group.create_dataset("teacher_actions", data=np.stack(teacher_actions), compression="gzip")
                group.create_dataset("dones", data=np.array(dones))
                group.create_dataset("timesteps", data=np.array(timesteps))
                group.create_dataset("no_robot_actions", data=np.array(no_robot_actions))
                group.create_dataset("no_teacher_actions", data=np.array(no_teacher_actions))
                group.create_dataset("if_success", data=np.array(if_success))
                state_dtype = h5py.vlen_dtype(np.dtype("uint8"))
                states_ds = group.create_dataset(
                    "states", shape=(len(states),), dtype=state_dtype
                )
                for state_idx, state_bytes in enumerate(states):
                    states_ds[state_idx] = state_bytes

        self.clear()

    def load_traj_i(self, traj_id):
        if self.file_name is None:
            return []
        return self.load_from_file(self.file_name, traj_id)

    def load_from_file(self, filename, traj_id):
        self.trajectories = []

        with h5py.File(filename, "r") as f:
            episode_keys = sorted(f.keys())

            if traj_id < 0 or traj_id >= len(episode_keys):
                raise IndexError(
                    f"Trajectory index {traj_id} out of range (0 to {len(episode_keys)-1})"
                )

            ep_key = episode_keys[traj_id]
            group = f[ep_key]

            obs_group = group["observation"]
            obs_keys = list(obs_group.keys())
            obs_data = {}

            # ---- NEW: uint8 image -> float32[-1,1] ----
            for k in obs_keys:
                arr = obs_group[k][()]  # (T, ...)
                if self._looks_like_image(k, arr) and arr.dtype == np.uint8:
                    arr = self._uint8_img_to_float(arr)
                obs_data[k] = arr

            robot_actions = group["robot_actions"][()]
            teacher_actions = group["teacher_actions"][()]
            dones = group["dones"][()]
            timesteps = group["timesteps"][()]
            no_robot_actions = group["no_robot_actions"][()]
            no_teacher_actions = group["no_teacher_actions"][()]
            if_success = group["if_success"][()]
            states = group["states"][()] if "states" in group else None

            T = len(timesteps)

            trajectory = []
            for t in range(T):
                obs_t = {k: obs_data[k][t] for k in obs_keys}
                transition = {
                    "obs": obs_t,
                    "robot_action": robot_actions[t],
                    "teacher_action": teacher_actions[t],
                    "done": dones[t],
                    "timestep": timesteps[t],
                    "no_robot_action": no_robot_actions[t],
                    "no_teacher_action": no_teacher_actions[t],
                    "episode_id": ep_key,
                    "if_success": if_success[t],
                    "state": (
                        self._deserialize_state(states[t]) if states is not None else None
                    ),
                }
                trajectory.append(transition)

        return trajectory

    def count_trajectories_in_hdf5(self, filename):
        with h5py.File(filename, "r") as f:
            count = sum(1 for key in f.keys() if key.startswith("episode_"))
        return count

    def clear(self):
        self.trajectories = []
        self.current_trajectory = []



def convert_hdf5_to_zarr(hdf5_path, zarr_path, compressor=None):
    """
    Converts an HDF5 file to a Zarr directory store.

    Parameters:
        hdf5_path: str, path to the input HDF5 file
        zarr_path: str, path to output Zarr directory
        compressor: zarr compressor (e.g., zarr.Blosc()), or None for no compression
    """
    # Open HDF5 for reading
    with h5py.File(hdf5_path, 'r') as f_in:
        # Create (or overwrite) Zarr store
        if os.path.exists(zarr_path):
            logger.info(f"Removing existing Zarr store at {zarr_path}")
            import shutil
            shutil.rmtree(zarr_path)

        root = zarr.open(zarr_path, mode='w')

        # Recursive function to copy
        def copy_group(h5_group, zarr_group):
            # Copy attributes (if needed)
            for k, v in h5_group.attrs.items():
                zarr_group.attrs[k] = v

            for name, item in h5_group.items():
                if isinstance(item, h5py.Dataset):
                    logger.info(f"Copying dataset: {item.name} -> {zarr_group.name}/{name}")
                    data = item[()]  # Load dataset into memory
                    zarr_group.array(
                        name,
                        data=data,
                        chunks=True,
                        compressor=compressor
                    )
                elif isinstance(item, h5py.Group):
                    logger.info(f"Creating group: {item.name}")
                    sub_group = zarr_group.create_group(name)
                    copy_group(item, sub_group)

        # Start recursive copy
        copy_group(f_in, root)

    logger.info(f"Conversion complete: {hdf5_path} -> {zarr_path}")


def create_preference_batch(traj1, traj2, segment_length):
    # Determine the maximum valid starting index in each trajectory.
    max_idx1 = len(traj1) - segment_length + 1
    max_idx2 = len(traj2) - segment_length + 1

    if max_idx1 < 1 or max_idx2 < 1:
        raise ValueError("Trajectories are too short for the given segment_length.")


    min_traj_length = min(len(traj1), len(traj2))
    segments1 = [traj1[i:i + segment_length] for i in range(0, min_traj_length, segment_length)]
    segments2 = [traj2[i:i + segment_length] for i in range(0, min_traj_length, segment_length)]


    # Split each trajectory into segments.
    segments1 = [traj1[i:i + segment_length] for i in range(0, len(traj1), segment_length)]
    segments2 = [traj2[i:i + segment_length] for i in range(0, len(traj2), segment_length)]

    # Use the minimum number of segments available from both trajectories.
    num_segments = min(len(segments1), len(segments2))
    preference_data = []

    for i in range(num_segments):
        seg1 = segments1[i]
        seg2 = segments2[i]

        preference_label =  1 # seg2 is prefred

        # Append the tuple (segment from traj1, segment from traj2, preference label)
        preference_data.append((seg1, seg2, preference_label))

    return preference_data




class TrajectoryBuffer_pickle:
    """
    A buffer for storing trajectories. Each trajectory is a list of transitions,
    where each transition is a dictionary containing:
        - 'obs': The observation at the current timestep.
        - 'action': The action taken by the agent.
        - 'done': A boolean flag indicating if the episode is done.
        - 'state': Additional state information (if any).
        - 'timestep': The timestep within the episode.
        - 'h': The feedback or correction signal (if any).
    """

    def __init__(self):
        # List of trajectories (each trajectory is a list of transitions)
        self.trajectories = []
        # Current trajectory (list of transitions)
        self.current_trajectory = []
        self.file_name = None

    def add_transition(self, obs, teacher_action, robot_action, done, timestep,
        no_robot_action, no_teacher_action, robot_position= None,
     episode_id = None, if_success=None, relabeled_action= None, state = None):
        """
        Add a single transition to the current trajectory.
        
        Parameters:
            obs: Observation data.
            action: Action taken by the agent.
            done: Boolean indicating if the episode is finished.
            state: Additional state information.
            timestep: The current timestep in the episode.
            no_robot_action: if true, the robot_action is set to zeros by default. 
        """
        transition = {
            "obs": obs,
            "robot_action": robot_action,
            'teacher_action': teacher_action,
            "done": done,
            # "state": state,
            "timestep": timestep,
            "no_teacher_action": no_teacher_action,
            "no_robot_action": no_robot_action,
            "episode_id": episode_id, 
            'robot_position': robot_position, 
            'if_success': if_success,
            'relabeled_action': relabeled_action,
            "state": state,
        }
        self.current_trajectory.append(transition)

    def load_traj_i(self, traj_id):
        if traj_id < self.count_trajectories_in_hdf5():
            return self.trajectories[traj_id]
        else:
            return None


    def finish_trajectory(self):
        """
        Ends the current trajectory and appends it to the list of saved trajectories.
        Resets the current trajectory for the next episode.
        """
        if self.current_trajectory:
            self.trajectories.append(self.current_trajectory)
            self.current_trajectory = []

    def save_to_file(self, filename):
        """
        Saves the entire buffer (all trajectories) to a file using pickle.

        Parameters:
            filename (str): The file path where the buffer will be saved.
        """
        filename = filename + ".pkl"
        self.file_name = filename
        with open(filename, "wb") as f:
            pickle.dump(self.trajectories, f)

    

    def load_from_file(self, filename):
        """
        Loads trajectories from a file into the buffer.

        Parameters:
            filename (str): The file path from where the buffer will be loaded.
        """
        with open(filename, "rb") as f:
            self.trajectories = pickle.load(f)

    def count_trajectories_in_hdf5(self, filename=None):
        return len(self.trajectories)

    def clear(self):
        """
        Clears all trajectories from the buffer.
        """
        self.trajectories = []
        self.current_trajectory = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=64,
                        help="Resolution for rendering")
    parser.add_argument("--output_dir", type=str, default="gifs",
                        help="Directory to save GIFs")
    parser.add_argument("--fps", type=int, default=10,
                        help="Frames per second for the GIF")
    args = parser.parse_args()

    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)


    buffer = TrajectoryBuffer()

    # buffer.load_from_file("trajectory_buffer_tested.pkl")
    buffer.load_from_file('trajectory_buffer.pkl')

    # from main_init import env_eval
    from env.metaworld_env.metaworld import MetaWorldSawyerEnv
    # task = "hammer-v2-goal-observable"
    task = "drawer-open-v2"
    task = task.strip('"')
    env_eval = MetaWorldSawyerEnv(task)

    trajectories = buffer.trajectories

    '''create the preference data'''
    preference_data_all = []
    for traj_idx in range(0, len(trajectories)-1, 2):
        traj1 = trajectories[traj_idx]
        traj2 = trajectories[traj_idx + 1]
        try:
            pref_batch = create_preference_batch(traj1, traj2, segment_length=64)
            # You could further process pref_batch (e.g., convert segments to torch tensors)
            preference_data_all.extend(pref_batch)
            logger.info(f"Processed preference batch for trajectory pair ({traj_idx}, { traj_idx + 1}).")
        except ValueError as e:
            logger.warning(f"Skipping trajectory pair ({ traj_idx}, {traj_idx + 1}): {e}")

    # save the preference_data_all
    save_path = "preference_data_all.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(preference_data_all, f)

    '''Visualize the preference data'''
    # For each preference batch, we render both segments and then combine the frames side by side.
    for idx, (seg1, seg2, pref_label) in enumerate(preference_data_all):
        seg1_frames = []
        seg2_frames = []
        env_eval.reset()
        # Render frames for the first segment
        for transition in seg1:
            # state_q, state_v = transition["state"]
            # pdb.set_trace()
            env_eval.set_state(transition["state"])
            env_eval.render_mode = 'rgb_array'
            env_eval.camera_name = "corner2"
            img1 = env_eval.render()
            # pdb.set_trace()
            img1 = cv2.resize(img1, (0, 0), fx=0.6, fy=0.6)
            # img1 = cv2.rotate(img1, cv2.ROTATE_180)
            seg1_frames.append(img1)
        
        # Render frames for the second segment
        for transition in seg2:
            # state_q, state_v = transition["state"]
            env_eval.set_state(transition["state"])
            env_eval.render_mode = 'rgb_array'
            env_eval.camera_name = "corner2"
            img2 = env_eval.render()
            img2 = cv2.resize(img2, (0, 0), fx=0.6, fy=0.6)
            # img2 = cv2.rotate(img2, cv2.ROTATE_180)
            seg2_frames.append(img2)
        
        # Ensure both segments have the same number of frames by taking the minimum length
        n_frames = min(len(seg1_frames), len(seg2_frames))
        combined_frames = []
        for i in range(n_frames):
            # Convert the frames to BGR if needed (OpenCV expects BGR)
            frame_left = cv2.cvtColor(seg1_frames[i], cv2.COLOR_RGB2BGR)
            frame_right = cv2.cvtColor(seg2_frames[i], cv2.COLOR_RGB2BGR)
            # Concatenate the two frames horizontally
            combined_frame = cv2.hconcat([frame_left, frame_right])
            combined_frames.append(combined_frame)
        
        if not combined_frames:
            logger.info(f"No frames to process for preference segment {idx}.")
            continue

        # Determine the dimensions of the combined frame
        height, width, _ = combined_frames[0].shape
        video_path = os.path.join(args.output_dir, f"preference_video_{idx}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(video_path, fourcc, args.fps, (width, height))
        for frame in combined_frames:
            video_writer.write(frame)
        video_writer.release()
        logger.info(f"Saved preference video for segment {idx} at: {video_path}")

    logger.info("All preference videos have been processed and saved.")

    '''visualize the trajectory data'''
    # Iterate over each trajectory
    for traj_idx in range(len(trajectories)):
        env_eval.reset()
        # pdb.set_trace()
        trajectory = trajectories[traj_idx]
        frames = []
        logger.info(f"Processing trajectory {traj_idx} with {len(trajectory)} transitions...")
        for transition in trajectory:
            # Use the 'state' field stored in the transition to set the environment's state.
            # Make sure that your environment's `set_state` method is compatible with the saved state.
            # env.set_state(transition["state"])
            state_q, state_v = transition["state"]
            # print("state: ", state_q)
            env_eval.set_env_state((state_q, state_v))
            # Render the image from the environment; returns an (H, W, 3) NumPy array.
            env_eval.render_mode = 'rgb_array'
            env_eval.camera_name = "corner2"
            img = env_eval.render()
            # img = cv2.resize(img, (0, 0), fx=0.267, fy=0.267)
            img = cv2.resize(img, (0, 0), fx=0.6, fy=0.6)
            img = cv2.rotate(img, cv2.ROTATE_180)
            frames.append(img)
            
        # Determine the frame dimensions from the first frame
        height, width, _ = frames[0].shape
        # Define the video file name
        video_path = os.path.join(args.output_dir, f"trajectory_{traj_idx}.mp4")
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(video_path, fourcc, args.fps, (width, height))
        # Write each frame to the video file
        for frame in frames:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame)
        
        # Release the video writer
        video_writer.release()
        logger.info(f"Saved video for trajectory {traj_idx} at: {video_path}")

    logger.info("All trajectories have been processed and GIFs are saved.")

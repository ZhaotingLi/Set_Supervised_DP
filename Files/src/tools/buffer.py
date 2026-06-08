import logging
import random
import pickle
import numpy as np
# import tensorflow as tf
import os
import time

# from env.metaworld_env.sawyer_hammer_v2_policy import SawyerHammerV2Policy


from collections import deque

import h5py


import shutil
from tqdm import tqdm 
from typing import Optional


from typing import Optional, Literal, List, Tuple

logger = logging.getLogger(__name__)


_H5PY_SUPPORTS_LOCKING_KW = None


def _h5py_supports_locking_kw() -> bool:
    global _H5PY_SUPPORTS_LOCKING_KW
    if _H5PY_SUPPORTS_LOCKING_KW is None:
        try:
            import inspect

            _H5PY_SUPPORTS_LOCKING_KW = (
                "locking" in inspect.signature(h5py.File).parameters
            )
        except Exception:
            _H5PY_SUPPORTS_LOCKING_KW = False
    return bool(_H5PY_SUPPORTS_LOCKING_KW)


def _open_h5_read_only(
    filename: str,
    *,
    mode: str = "r",
    rdcc_nbytes: int,
    rdcc_nslots: int,
    rdcc_w0: float,
    retry_attempts: int = 3,
    retry_delay_sec: float = 0.25,
):
    if mode != "r":
        raise ValueError(f"_open_h5_read_only only supports mode='r', got {mode!r}.")

    open_kwargs = {
        "mode": mode,
        "rdcc_nbytes": int(rdcc_nbytes),
        "rdcc_nslots": int(rdcc_nslots),
        "rdcc_w0": float(rdcc_w0),
    }

    if _h5py_supports_locking_kw():
        open_kwargs["locking"] = False
    else:
        os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

    last_error = None
    for attempt_idx in range(max(int(retry_attempts), 1)):
        try:
            return h5py.File(filename, **open_kwargs)
        except TypeError:
            if "locking" not in open_kwargs:
                raise
            open_kwargs.pop("locking", None)
            os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
            continue
        except OSError as exc:
            last_error = exc
            if attempt_idx + 1 >= max(int(retry_attempts), 1):
                break
            time.sleep(float(retry_delay_sec) * float(attempt_idx + 1))

    if last_error is None:
        return h5py.File(filename, **open_kwargs)
    raise last_error



def combine_data_buffers(
    src_a: str,
    src_b: str,
    dst_filename: Optional[str] = None,
    capacity: Optional[int] = None,
    order: Literal["A_then_B", "B_then_A"] = "A_then_B",
    keep: Literal["tail", "head"] = "tail",
    chunk_rows: int = 8192,
    overwrite: bool = False,
) -> str:
    """
    Combine two HDF5 replay buffers (matching schema) into a new file.

    Differences from the original:
      - Instead of concatenating A then B in chronological order, this version
        *mixes* samples from A and B by alternately copying blocks from each.
        No global random shuffle is performed, but the resulting buffer
        interleaves A/B so that sequential iteration during training sees a mix.

      - 'order' now controls which buffer is taken first in the alternation:
          * 'A_then_B' -> A, B, A, B, ...
          * 'B_then_A' -> B, A, B, A, ...

    Assumptions:
      - Each source has datasets for fields with leading dim = max_size,
        plus scalar 'ptr' and 'count'.
      - The *valid* region is reconstructed with wrap-awareness:
          if count < max_size   -> [0:count]
          if count == max_size  -> [ptr:max_size] + [0:ptr]
      - Destination capacity defaults to (count_a + count_b). If 'capacity' is smaller,
        'keep' controls whether to keep the head or tail of the mixed sequence.

    Args:
        src_a, src_b: Paths to the two source .h5 files.
        dst_filename: Path to the output .h5; if None, creates '<base>_combined.h5' next to src_a.
        capacity: Optional destination capacity. Default is n_a + n_b.
        order: Which buffer starts the alternation: "A_then_B" or "B_then_A".
        keep: If capacity < n_a + n_b, whether to keep the 'head' (earliest in the
              mixed stream) or 'tail' (latest) of the mixed sequence.
        chunk_rows: Copy block size for I/O efficiency and for mixing granularity.
        overwrite: Overwrite dst if it already exists.

    Returns:
        Path to the newly written combined .h5.
    """
    if dst_filename is None:
        root, ext = os.path.splitext(src_a)
        dst_filename = f"{root}_combined{ext}"

    if os.path.exists(dst_filename):
        if overwrite:
            os.remove(dst_filename)
        else:
            raise FileExistsError(f"{dst_filename} exists. Set overwrite=True to replace.")

    def _field_names(f: h5py.File) -> List[str]:
        return [k for k in f.keys() if k not in ("ptr", "count")]

    def _chronology_slices(ptr: int, count: int, max_size: int) -> List[Tuple[int, int]]:
        """Return list of [start, end) slices in chronological order."""
        if count == 0:
            return []
        if count < max_size:
            return [(0, count)]
        # full ring: oldest starts at ptr
        return [(ptr, max_size), (0, ptr)]

    def _validate_schemas(a: h5py.File, b: h5py.File) -> Tuple[List[str], int, int, dict]:
        fa = _field_names(a)
        fb = _field_names(b)
        if set(fa) != set(fb):
            missing_a = set(fb) - set(fa)
            missing_b = set(fa) - set(fb)
            raise RuntimeError(f"Field mismatch.\nMissing in A: {missing_a}\nMissing in B: {missing_b}")
        fields = sorted(fa)
        # Check shapes (beyond leading dim) and dtype consistency
        meta = {}
        for name in fields:
            da, db = a[name], b[name]
            if da.shape[1:] != db.shape[1:]:
                raise RuntimeError(f"Shape mismatch for '{name}': {da.shape[1:]} vs {db.shape[1:]}")
            if da.dtype != db.dtype:
                raise RuntimeError(f"Dtype mismatch for '{name}': {da.dtype} vs {db.dtype}")
            meta[name] = dict(
                shape_tail=da.shape[1:],
                dtype=da.dtype,
                chunks=da.chunks,
                compression=da.compression,
                attrs=dict(da.attrs.items()),
            )
        return fields, a[fields[0]].shape[0], b[fields[0]].shape[0], meta

    # Open sources with big read caches for speed
    with h5py.File(src_a, 'r', rdcc_nbytes=1_500_000_000, rdcc_nslots=1_000_003, rdcc_w0=0.75) as A, \
         h5py.File(src_b, 'r', rdcc_nbytes=1_500_000_000, rdcc_nslots=1_000_003, rdcc_w0=0.75) as B:

        fields, max_a, max_b, meta = _validate_schemas(A, B)

        n_a = int(A["count"][()]) if "count" in A else 0
        n_b = int(B["count"][()]) if "count" in B else 0
        if n_a > max_a or n_b > max_b:
            raise RuntimeError(f"Invalid count(s): A={n_a}/{max_a}, B={n_b}/{max_b}")

        ptr_a = int(A["ptr"][()]) if "ptr" in A else (n_a % max_a)
        ptr_b = int(B["ptr"][()]) if "ptr" in B else (n_b % max_b)
        slices_a = _chronology_slices(ptr_a, n_a, max_a)
        slices_b = _chronology_slices(ptr_b, n_b, max_b)

        total = n_a + n_b
        cap = total if capacity is None else int(capacity)
        if cap <= 0:
            raise ValueError("capacity must be positive.")
        write_count = min(cap, total)

        # Decide how many to skip from the mixed stream
        if keep == "tail":
            skip_from_front = total - write_count
        elif keep == "head":
            skip_from_front = 0
        else:
            raise ValueError("keep must be 'tail' or 'head'.")

        # Create destination with chosen capacity
        with h5py.File(dst_filename, 'w') as D:
            for name in fields:
                tail_shape = meta[name]["shape_tail"]
                D_ds = D.create_dataset(
                    name,
                    shape=(cap, *tail_shape),
                    dtype=meta[name]["dtype"],
                    chunks=meta[name]["chunks"],
                    compression=meta[name]["compression"],
                )
                for ak, av in meta[name]["attrs"].items():
                    D_ds.attrs[ak] = av

            # Prepare ptr/count (ptr = next write index; count = number of valid rows)
            D.create_dataset("ptr", data=(write_count % cap))
            D.create_dataset("count", data=write_count)

            if write_count == 0:
                return dst_filename

            # Per-source streaming state
            state = {
                "A": {
                    "slices": slices_a,
                    "slice_idx": 0,
                    "pos": slices_a[0][0] if slices_a else 0,
                    "remaining": n_a,
                    "file": A,
                },
                "B": {
                    "slices": slices_b,
                    "slice_idx": 0,
                    "pos": slices_b[0][0] if slices_b else 0,
                    "remaining": n_b,
                    "file": B,
                },
            }

            def _advance_in_src(tag: str, advance_len: int, do_write: bool,
                                dst_pos: int, remaining_to_write: int) -> Tuple[int, int, int]:
                """
                Advance within src 'tag' by up to advance_len samples.
                If do_write is True, copy into D starting at dst_pos.
                Returns (actually_advanced, new_dst_pos, new_remaining_to_write).
                """
                st = state[tag]
                slices = st["slices"]
                file = st["file"]
                advanced = 0

                while advance_len > 0 and st["remaining"] > 0 and slices:
                    si = st["slice_idx"]
                    if si >= len(slices):
                        break
                    s, e = slices[si]
                    pos = st["pos"]

                    # Ensure pos is within current slice
                    if pos < s or pos >= e:
                        pos = s

                    avail = e - pos
                    if avail <= 0:
                        st["slice_idx"] += 1
                        if st["slice_idx"] >= len(slices):
                            break
                        st["pos"] = slices[st["slice_idx"]][0]
                        continue

                    step = min(avail, advance_len)
                    if do_write and remaining_to_write > 0:
                        # Don't write beyond remaining_to_write
                        write_step = min(step, remaining_to_write)
                        if write_step > 0:
                            src_slice = slice(pos, pos + write_step)
                            dst_slice = slice(dst_pos, dst_pos + write_step)
                            for name in fields:
                                D[name][dst_slice] = file[name][src_slice]
                            dst_pos += write_step
                            remaining_to_write -= write_step
                            pos += write_step
                            step = write_step  # we consider only what we advanced
                    else:
                        # Just skip ahead without writing
                        pos += step

                    advance_len -= step
                    advanced += step
                    st["remaining"] -= step
                    st["pos"] = pos

                    if st["pos"] >= e:
                        st["slice_idx"] += 1
                        if st["slice_idx"] < len(slices):
                            st["pos"] = slices[st["slice_idx"]][0]

                    if remaining_to_write <= 0 and not do_write:
                        # only skipping; we can still continue skipping if needed
                        continue

                    if remaining_to_write <= 0 and do_write:
                        break

                return advanced, dst_pos, remaining_to_write

            # Main mixing loop: alternate between A and B in blocks
            turn = "A" if order == "A_then_B" else "B"
            other = {"A": "B", "B": "A"}

            skip = skip_from_front
            dst_pos = 0
            remaining_to_write = write_count

            while remaining_to_write > 0 and (state["A"]["remaining"] > 0 or state["B"]["remaining"] > 0):
                # Pick which source to use this step
                if state[turn]["remaining"] == 0 and state[other[turn]]["remaining"] > 0:
                    turn = other[turn]
                if state[turn]["remaining"] == 0:
                    # Both empty
                    break

                block_len = min(chunk_rows, state[turn]["remaining"])

                # First, skip from front if needed (logical skip)
                if skip > 0:
                    # We may skip part or all of this block
                    to_skip = min(skip, block_len)
                    advanced, _, _ = _advance_in_src(
                        turn, to_skip, do_write=False,
                        dst_pos=dst_pos, remaining_to_write=remaining_to_write
                    )
                    skip -= advanced
                    block_len -= advanced

                    # If we used the whole block just for skipping, flip turn and continue
                    if block_len == 0:
                        turn = other[turn]
                        continue

                # Now actually write from this source
                if block_len > 0 and remaining_to_write > 0:
                    advanced, dst_pos, remaining_to_write = _advance_in_src(
                        turn, block_len, do_write=True,
                        dst_pos=dst_pos, remaining_to_write=remaining_to_write
                    )

                # Alternate source for next block
                turn = other[turn]

            # If capacity > total, the unused tail remains uninitialized—leave as zeros.

    return dst_filename





def shuffle_data_buffer(
    src_filename: str,
    dst_filename: Optional[str] = None,
    seed: Optional[int] = None,
    chunk_rows: int = 8192,
    overwrite: bool = False,
    show_progress: bool = True,    target_chunk_bytes: int = 512 * 1024 * 1024,  # ~512MB per chunk across all fields
) -> str:
    """
    Shuffle an existing HDF5 replay buffer on disk and write the result to a new file.

    Assumptions:
      - File schema matches your HDF5Buffer: datasets for fields (all with leading
        dimension 'max_size'), plus scalar datasets 'ptr' and 'count'.
      - Only the first 'count' rows are considered valid and will be shuffled.

    Args:
        src_filename: Path to the source .h5 file.
        dst_filename: Path for the shuffled .h5. If None, appends '_shuffled' before the extension.
        seed: Optional RNG seed for reproducibility.
        chunk_rows: Upper bound on number of rows to process per write (controls memory / I/O).
        overwrite: If True and dst file exists, it will be overwritten.
        show_progress: If True, prints a progress bar (uses tqdm if installed).
        target_chunk_bytes: Approximate maximum memory per chunk across all fields.

    Returns:
        The path to the newly written shuffled .h5 file.
    """
    if dst_filename is None:
        root, ext = os.path.splitext(src_filename)
        dst_filename = f"{root}_shuffled{ext}"

    if os.path.exists(dst_filename):
        if overwrite:
            os.remove(dst_filename)
        else:
            raise FileExistsError(
                f"Destination exists: {dst_filename}. Set overwrite=True to replace."
            )

    rng = np.random.default_rng(seed)

    # Large read cache for speed; SWMR not needed for offline shuffle
    with h5py.File(
        src_filename,
        "r",
        rdcc_nbytes=1_500_000_000,
        rdcc_nslots=1_000_003,
        rdcc_w0=0.75,
    ) as src, h5py.File(dst_filename, "w") as dst:
        # Identify data fields (everything except the control scalars)
        field_names = [k for k in src.keys() if k not in ("ptr", "count")]
        if not field_names:
            raise RuntimeError("No data fields found in source file.")

        # Infer max_size from first field
        any_ds = src[field_names[0]]
        if any_ds.ndim < 1:
            raise RuntimeError(
                "Field datasets must be at least 1D with leading dimension = max_size."
            )
        max_size = any_ds.shape[0]

        # Read how many rows are valid
        if "count" not in src:
            raise RuntimeError("Source file missing 'count' dataset.")
        n = int(src["count"][()])

        if n < 0 or n > max_size:
            raise RuntimeError(f"Invalid count {n} for max_size {max_size}.")

        # Prepare destination datasets mirroring source metadata
        for name in field_names:
            sds = src[name]
            compression = sds.compression
            chunks = sds.chunks
            dds = dst.create_dataset(
                name,
                shape=sds.shape,
                dtype=sds.dtype,
                chunks=chunks,
                compression=compression,
            )
            # Copy attrs (if any)
            for ak, av in sds.attrs.items():
                dds.attrs[ak] = av

        # Create ptr/count in destination: start at 0, keep the same 'count'
        dst.create_dataset("ptr", data=0)
        dst.create_dataset("count", data=n)

        if n == 0:
            # Nothing to shuffle
            return dst_filename

        # --- Estimate per-row memory and adjust chunk_rows for better performance ---
        bytes_per_row = 0
        for name in field_names:
            sds = src[name]
            # product of remaining dimensions * itemsize
            row_elems = int(np.prod(sds.shape[1:], dtype=np.int64))
            bytes_per_row += row_elems * sds.dtype.itemsize

        if bytes_per_row <= 0:
            # Degenerate case, just fall back
            effective_chunk_rows = min(chunk_rows, n)
        else:
            max_rows_by_mem = max(1, target_chunk_bytes // bytes_per_row)
            effective_chunk_rows = int(max(1, min(chunk_rows, max_rows_by_mem, n)))

        # Generate a permutation for the valid region [0, n)
        perm = rng.permutation(n)

        # Setup progress tracking
        if show_progress and tqdm is not None:
            pbar = tqdm(total=n, desc="Shuffling buffer", unit="rows")
            use_simple_progress = False
        elif show_progress:
            pbar = None
            use_simple_progress = True
            logger.info(f"Shuffling {n} rows with chunk size {effective_chunk_rows} "
                f"(~{bytes_per_row * effective_chunk_rows / 1e6:.1f} MB per chunk)...")
        else:
            pbar = None
            use_simple_progress = False

        # --- Main shuffle loop: write shuffled rows in chunks ---
        start = 0
        last_simple_report = 0

        while start < n:
            end = min(start + effective_chunk_rows, n)
            src_idx = perm[start:end]  # indices in source to read

            # For h5py efficiency, read sorted, then permute back to requested order
            order = np.argsort(src_idx)
            sorted_src_idx = src_idx[order]
            inv_order = np.empty_like(order)
            inv_order[order] = np.arange(order.size)

            for name in field_names:
                sds = src[name]
                dds = dst[name]
                # Read rows in sorted order (h5py advanced indexing)
                block_sorted = sds[sorted_src_idx]
                # Reorder to match destination [start:end]
                block = block_sorted[inv_order]
                # Write contiguously
                dds[start:end] = block

            # Update progress
            if pbar is not None:
                pbar.update(end - start)
            elif use_simple_progress:
                processed = end
                # print every ~5% or at the end
                if processed == n or processed - last_simple_report >= max(n // 20, 1):
                    pct = 100.0 * processed / n
                    logger.info(f"  progress: {processed}/{n} rows ({pct:5.1f}%)")
                    last_simple_report = processed

            start = end

        if pbar is not None:
            pbar.close()

        # --- Copy any unused tail rows [n:max_size) verbatim so capacity is preserved ---
        if n < max_size:
            tail_start = n
            tail_end = max_size
            # Tail copy is contiguous; do it in reasonably large chunks as well
            ts = tail_start
            while ts < tail_end:
                te = min(ts + effective_chunk_rows, tail_end)
                sl = slice(ts, te)
                for name in field_names:
                    dst[name][sl] = src[name][sl]
                ts = te

    return dst_filename



def downsample_hdf5buffer_uniform(
    src_h5: str,
    dst_h5: str,
    target_size: int,
    compression: str = "lzf",
):
    """
    Uniformly downsample an existing HDF5Buffer file to `target_size` items
    WITHOUT repetition, writing to `dst_h5`.

    - Preserves dataset names/shapes/dtypes.
    - Copies ONLY selected indices (evenly spaced across [0, n-1]).
    - Sets ptr=target_size % target_size == 0, count=target_size in output.

    Args:
        src_h5: path to input .h5 created by HDF5Buffer
        dst_h5: path to output .h5
        target_size: number of items to keep (<= original count)
        compression: compression filter for output datasets (e.g., "lzf", "gzip", None)
    """
    if target_size <= 0:
        raise ValueError("target_size must be > 0")

    with h5py.File(src_h5, "r") as fsrc:
        if "count" not in fsrc:
            raise RuntimeError("Source file missing 'count' dataset")
        n = int(fsrc["count"][()])
        if target_size > n:
            raise ValueError(f"target_size ({target_size}) > source count ({n})")

        # pick evenly spaced indices, then force uniqueness + increasing
        idx = np.linspace(0, n - 1, num=target_size)
        idx = np.rint(idx).astype(np.int64)
        idx = np.unique(idx)
        # if rounding caused us to lose some (rare), fill by adding nearest missing indices
        if idx.size != target_size:
            keep = np.zeros(n, dtype=bool)
            keep[idx] = True
            missing = target_size - idx.size
            # add missing indices uniformly from the remaining ones
            remaining = np.nonzero(~keep)[0]
            add = np.rint(np.linspace(0, remaining.size - 1, num=missing)).astype(np.int64)
            idx = np.sort(np.concatenate([idx, remaining[add]]))
        else:
            idx = np.sort(idx)

        # Create output (overwrite if exists)
        if os.path.exists(dst_h5):
            os.remove(dst_h5)

        with h5py.File(dst_h5, "w") as fdst:
            # copy each dataset except ptr/count by indexing
            for name, dsrc in fsrc.items():
                if name in ("ptr", "count"):
                    continue

                # HDF5Buffer datasets are (max_size, *shape). We will write (target_size, *shape).
                # Use same dtype, same per-item shape.
                out_shape = (target_size,) + dsrc.shape[1:]
                chunks = (1,) + dsrc.shape[1:] if len(dsrc.shape) > 1 else (1,)

                ddst = fdst.create_dataset(
                    name,
                    shape=out_shape,
                    dtype=dsrc.dtype,
                    chunks=chunks,
                    compression=compression,
                )

                # copy in reasonably sized blocks to avoid big RAM spikes
                # (tweak block_size if you want)
                block_size = 4096
                for start in range(0, target_size, block_size):
                    end = min(start + block_size, target_size)
                    sel = idx[start:end]
                    ddst[start:end] = dsrc[sel]

            # write new ptr/count
            fdst.create_dataset("ptr", data=0, dtype=np.int64)
            fdst.create_dataset("count", data=target_size, dtype=np.int64)

    return idx  # useful for debugging / reproducibility


class HDF5Buffer:
    def __init__(
        self,
        filename: str,
        field_shapes: dict,
        min_size: int,
        max_size: int,
        dtype_map: dict = None,
        compression: str = "lzf",
        unique_per_run = True,
        image_saved_in_Uint8 = False,
        contiguous_sampling = True,  # TODO, during online training, should be False
    ):
        """
        Args:
            filename: path to .h5 file (will be created/opened).
            field_shapes: dict mapping field_name -> tuple(shape,),
                e.g. {
                    'agentview_image': (3,84,84),
                    'robot0_eye_in_hand_image': (3,84,84),
                    'robot0_eef_pos': (3,),
                    'robot0_eef_quat': (4,),
                    'robot0_gripper_qpos': (2,),
                    'robot_action': (10,),
                    'teacher_action': (10,)
                }
            min_size, max_size: same semantics as before.
            dtype_map: optional dict mapping field_name -> numpy dtype.
            compression: HDF5 compression filter.
        """
        # self.filename    = filename
        self.filename    = (f"{os.path.splitext(filename)[0]}_{os.getpid()}.h5"
                            if unique_per_run else filename)
        self.min_size    = min_size
        self.max_size    = max_size
        self.field_shapes = field_shapes
        self.dtype_map   = dtype_map or {}
        self.compression = compression
        self.image_saved_in_Uint8 = image_saved_in_Uint8

        # open/create HDF5 file
        self.f = h5py.File(self.filename , 'a')
        self.datasets = {}
        for name, shape in field_shapes.items():
            dtype = self.dtype_map.get(name, 'float32')
            if name not in self.f:
                self.datasets[name] = self.f.create_dataset(
                    name,
                    shape=(max_size, *shape),
                    dtype=dtype,
                    chunks=(1, *shape),
                    compression=self.compression
                )
            else:
                self.datasets[name] = self.f[name]

        # pointer & count trackers
        if 'ptr' not in self.f:
            self.ptr_ds   = self.f.create_dataset('ptr',   data=0)
        else:
            self.ptr_ds   = self.f['ptr']
        if 'count' not in self.f:
            self.count_ds = self.f.create_dataset('count', data=0)
        else:
            self.count_ds = self.f['count']

    def full(self) -> bool:
        return int(self.count_ds[()]) >= self.max_size

    def initialized(self) -> bool:
        return int(self.count_ds[()]) >= self.min_size

    def length(self) -> int:
        return int(self.count_ds[()])

    def add(self, step: tuple): #TODO, consider transfer the input image to Uint8 format
        """
        step: a tuple (obs_dict, action_robot, action_teacher)
            - obs_dict: dict mapping all non-action field names to arrays/scalars
            - action_robot: array or list matching field_shapes['robot_action']
            - action_teacher: array or list matching field_shapes['teacher_action']
        """
        # obs_dict, action_robot, action_teacher = step
        obs_dict, action_teacher, action_robot = step
        i = int(self.ptr_ds[()])

        # write observation fields
        for name in self.field_shapes:
            if name in ('robot_action', 'teacher_action'):
                continue
            
            if self.image_saved_in_Uint8: # image data has been processed into [-1, 1]
                if name in self.dtype_map and self.dtype_map[name] == "uint8":
                    obs_dict[name] = np.clip((obs_dict[name] + 1.0) * 127.5, 0, 255).astype(np.uint8)
            self.datasets[name][i] = obs_dict[name]

        # write actions
        # print("action_robot: ", action_robot, " action_teacher: ", action_teacher)
        self.datasets['robot_action'][i]  = np.array(action_robot, dtype=self.datasets['robot_action'].dtype)
        self.datasets['teacher_action'][i] = np.array(action_teacher, dtype=self.datasets['teacher_action'].dtype)

        # advance pointer & count
        self.ptr_ds[...]   = (i + 1) % self.max_size
        self.count_ds[...] = min(self.count_ds[()] + 1, self.max_size)

    # def sample(self, batch_size: int):
    #     """
    #     Returns a list of tuples (obs_dict, action_robot, action_teacher)
    #     matching the structure given to `add`.
    #     """
    #     n = self.length()
    #     idxs = np.random.randint(0, n, size=batch_size)
    #     batch = []
    #     for idx in idxs:
    #         # reconstruct observation dict
    #         obs_dict = {
    #             name: self.datasets[name][idx]
    #             for name in self.field_shapes
    #             if name not in ('robot_action', 'teacher_action')
    #         }
    #         action_robot  = self.datasets['robot_action'][idx]
    #         action_teacher = self.datasets['teacher_action'][idx]
    #         # batch.append((obs_dict, action_robot, action_teacher))
    #         batch.append((obs_dict, action_teacher, action_robot))
    #     return batch

    def sample_randomly(self, batch_size: int): 
        # this one is only used for visualization and testing, not for training as it's too slow in HPC
        n = self.length()
        idxs = np.random.randint(0, n, size=batch_size)

        # 1) Make indices strictly increasing & unique for h5py
        uniq_sorted, inv = np.unique(idxs, return_inverse=True)  # uniq_sorted is strictly increasing

        # 2) One read per dataset using uniq_sorted
        obs_batch_unique = {}
        for name in self.field_shapes:
            if name in ('robot_action', 'teacher_action'):
                continue
            obs_batch_unique[name] = self.datasets[name][uniq_sorted]
            if self.image_saved_in_Uint8 and name in self.dtype_map and self.dtype_map[name] == "uint8":
                obs_batch_unique[name] = obs_batch_unique[name].astype(np.float32) / 255. 
                obs_batch_unique[name] = 2.0 * obs_batch_unique[name] - 1.0

        robot_unique   = self.datasets['robot_action'][uniq_sorted]
        teacher_unique = self.datasets['teacher_action'][uniq_sorted]

        # 3) Expand back to the original (with duplicates) via inverse map
        batch = []
        for j in range(batch_size):
            u = inv[j]  # which unique row to use
            obs_dict = {k: v[u] for k, v in obs_batch_unique.items()}
            batch.append((obs_dict, teacher_unique[u], robot_unique[u]))  # (obs, teacher, robot)
        return batch

    def _ensure_seq_ptr(self):
        if not hasattr(self, "_seq_ptr"):
            self._seq_ptr = 0  # start of sequential stream

    def sample(self, batch_size: int):
        """
        Sequential, contiguous read: rows [i : i+batch_size], wrapping at buffer end.
        Returns a list of (obs_dict, teacher_action, robot_action).
        """
        n = self.length()
        if n == 0:
            raise RuntimeError("Buffer is empty")
        self._ensure_seq_ptr()

        i = self._seq_ptr
        j = i + batch_size

        # --- Read contiguous slices (handle wrap) ---
        def _read_field(name):
            if j <= n:  # no wrap
                return self.datasets[name][i:j]
            else:
                first = self.datasets[name][i:n]
                second = self.datasets[name][0:j - n]
                return np.concatenate([first, second], axis=0)

        # Observations
        obs_batch = {}
        for name in self.field_shapes:
            if name in ("robot_action", "teacher_action"):
                continue
            obs_batch[name] = _read_field(name)
            if self.image_saved_in_Uint8 and name in self.dtype_map and self.dtype_map[name] == "uint8":
                obs_batch[name] = obs_batch[name].astype(np.float32) / 255. 
                obs_batch[name] = 2.0 * obs_batch[name] - 1.0

        # Actions
        robot   = _read_field("robot_action")
        teacher = _read_field("teacher_action")

        # Advance pointer (wrap within current length)
        self._seq_ptr = (j % n)

        # Rebuild original structure: list of tuples
        B = robot.shape[0]
        batch = []
        for k in range(B):
            obs_k = {fname: arr[k] for fname, arr in obs_batch.items()}
            batch.append((obs_k, teacher[k], robot[k]))  # (obs, teacher, robot)
        return batch

    
    def save_to_file(self, filename: str):
        """
        Copy the underlying HDF5 file to `filename`.
        """
        self.f.flush()
        self.f.close()
        shutil.copy(self.filename, filename)
        self.f = h5py.File(self.filename, 'a')
        self._rebind_handles()

    def load_from_file(self, filename: str, read_only: bool = False):
        """
        Attach the buffer to an existing HDF5 file.

        Args:
            filename: path to .h5 file to open.
            read_only: 
                - True → open in read-only mode, you can sample but NOT add.
                - False → open in read/write mode, you can still add.
        """
        try:
            self.f.close()
        except Exception:
            pass

        self.filename = filename
        mode = 'r' if read_only else 'a'   # 'r' = read-only, 'a' = read/write
        # self.f = h5py.File(self.filename, mode)
        if read_only: 
            self.f = h5py.File(self.filename, 'r',
                       rdcc_nbytes=1_500_000_000,
                       rdcc_nslots=1_000_003,
                       rdcc_w0=0.75)  # no swmr unless you truly need it
        else:
            self.f = h5py.File(self.filename, mode, rdcc_nbytes=64*1024*1024, rdcc_nslots=100_003, rdcc_w0=0.75)
        self._rebind_handles()

        logger.debug('length:  %s', self.length())

    def _rebind_handles(self):
        self.datasets = {name: self.f[name] for name in self.field_shapes}
        self.ptr_ds   = self.f['ptr']
        self.count_ds = self.f['count']

    def close(self):
        self.f.close()


    #  ingest previously-saved TrajectoryBuffer HDF5 files (defined in buffer_trajectory.py)
    def ingest_trajectory_hdf5(
        self,
        traj_filename: str,
        *,
        chunk_size: int = 2048,
        skip_no_robot_action: bool = False,
        # skip_no_teacher_action: bool = False,
        skip_no_teacher_action: bool = True,
        show_progress: bool = True,
    ) -> int:
        """
        Stream transitions from <traj_filename>.hdf5 into *this* buffer's file
        (`self.f`) in chunks, without keeping the whole data in RAM.

        Parameters
        ----------
        traj_filename : str
            Base name used in TrajectoryBuffer.save_to_file ('.hdf5' added
            automatically).
        chunk_size : int
            How many transitions to copy in one I/O block.  Tune for your SSD /
            network FS; 2-8 k is usually a good balance between speed & memory.
        skip_no_robot_action / skip_no_teacher_action : bool
            Drop transitions whose flags are True.
        show_progress : bool
            If True, show a tqdm bar while copying.

        Returns
        -------
        n_written : int
            Number of transitions actually written into the replay buffer.
        """
        from hydra.utils import get_original_cwd
        from hydra.core.hydra_config import HydraConfig
        # if use hydra, define path as follows
        if HydraConfig and getattr(HydraConfig, "is_initialized", None) and HydraConfig.is_initialized():
            project_root = get_original_cwd()
            # import pdb; pdb.set_trace()
            full_path = os.path.join(project_root, traj_filename)
            src_path = full_path 
        else: # if not use hydra, simply use the traj_filename
            src_path = traj_filename
        logger.info('load hdf5 buffer from trajectory dataset:  %s', src_path)
        n_written = 0

        with h5py.File(src_path, "r") as src:
            episode_keys = sorted(k for k in src if k.startswith("episode_"))

            # Pre-extract dest handles so attribute lookups stay cheap
            dst_obs_dsets = {
                name: self.datasets[name]
                for name in self.field_shapes
                if name not in ("robot_action", "teacher_action")
            }
            dst_robot = self.datasets["robot_action"]
            dst_teacher = self.datasets["teacher_action"]

            cur_ptr = int(self.ptr_ds[()])
            cur_cnt = int(self.count_ds[()])

            pbar = tqdm(total=len(episode_keys), disable=not show_progress,
                        desc="Episodes") if show_progress else None

            for ep in episode_keys:
                g = src[ep]

                # Source handles
                # s_obs  = {k: g["observation"][k] for k in g["observation"]}
                s_obs = {}
                for k in g["observation"]:
                    ds_np = g["observation"][k][()]          # load the whole array for key k
                    if self.image_saved_in_Uint8:
                        # orginal image is [-1, 1]
                        if k in self.dtype_map and self.dtype_map[k] == "uint8":
                            # print(k)
                            ds_np = np.clip((ds_np + 1.0) * 127.5, 0, 255).astype(np.uint8)
                    
                    exp_shape = self.field_shapes.get(k)
                    # Guard: if we don’t know the expected shape, just keep it
                    if exp_shape is None:
                        s_obs[k] = ds_np
                        continue

                    per_sample_shape = ds_np.shape[1:]
                    need_aug = (
                        per_sample_shape == tuple(exp_shape[1:])
                        and (len(per_sample_shape) == len(exp_shape) - 1)
                    )


                    if need_aug:                      # (T, C, H, W)  →  make (T, 2, C, H, W)
                        T = ds_np.shape[0]
                        stacked = np.empty((T, 2, *ds_np.shape[1:]), dtype=ds_np.dtype)
                        stacked[:, 1, ...] = ds_np           # current frame in slot 1
                        stacked[0, 0, ...] = ds_np[0]        # t==0: duplicate itself
                        if T > 1:
                            stacked[1:, 0, ...] = ds_np[:-1] # slot 0 gets previous frame
                        s_obs[k] = stacked
                    else:
                        # Already has leading stack dimension (T, 2, C, H, W) or non-image
                        s_obs[k] = ds_np

                s_r    = g["robot_actions"]
                s_t    = g["teacher_actions"]
                s_no_r = g["no_robot_actions"]
                s_no_t = g["no_teacher_actions"]
                # import pdb; pdb.set_trace()

                T = s_r.shape[0]
                start = 0
                while start < T:
                    end = min(start + chunk_size, T)

                    # Apply skip masks if needed --------------------------------
                    if skip_no_robot_action or skip_no_teacher_action:
                        # build a boolean mask for the slice
                        skip_mask = np.zeros(end - start, dtype=bool)
                        if skip_no_robot_action:
                            skip_mask |= s_no_r[start:end]
                        if skip_no_teacher_action:
                            skip_mask |= s_no_t[start:end]
                            # import pdb; pdb.set_trace()

                        keep_idx = np.nonzero(~skip_mask)[0]
                        if keep_idx.size == 0:
                            start = end
                            continue  # nothing worth copying in this slice

                        # Fancy-index the arrays into an in-RAM chunk
                        # (h5py must materialise them for selection anyway)
                        obs_chunk = {
                            k: s_obs[k][start:end][keep_idx] for k in s_obs
                        }
                        r_chunk = s_r[start:end][keep_idx]
                        t_chunk = s_t[start:end][keep_idx]
                    else:
                        # Straight, contiguous slice; pull once per field
                        obs_chunk = {k: s_obs[k][start:end] for k in s_obs}
                        r_chunk   = s_r[start:end]
                        t_chunk   = s_t[start:end]

                    n_chunk = r_chunk.shape[0]
                    if n_chunk == 0:
                        start = end
                        continue

                    # Where to write in destination --------------------------------
                    dst_end = cur_ptr + n_chunk
                    wrap = dst_end > self.max_size
                    if wrap:
                        first_part = self.max_size - cur_ptr
                        second_part = n_chunk - first_part
                    else:
                        first_part, second_part = n_chunk, 0

                    # -------- write FIRST part (no wrap or pre-wrap region) -------
                    sl = slice(cur_ptr, cur_ptr + first_part)
                    for k, arr in obs_chunk.items():
                        dst_obs_dsets[k][sl] = arr[:first_part]
                    dst_robot[sl]   = r_chunk[:first_part]
                    dst_teacher[sl] = t_chunk[:first_part]

                    # -------- optional wrap-around write --------------------------
                    if wrap and second_part:
                        sl2 = slice(0, second_part)
                        for k, arr in obs_chunk.items():
                            dst_obs_dsets[k][sl2] = arr[first_part:]
                        dst_robot[sl2]   = r_chunk[first_part:]
                        dst_teacher[sl2] = t_chunk[first_part:]

                    # Update pointers ---------------------------------------------
                    cur_ptr = (cur_ptr + n_chunk) % self.max_size
                    cur_cnt = min(cur_cnt + n_chunk, self.max_size)
                    n_written += n_chunk

                    if self.full():            # stop once buffer saturated
                        break

                    start = end

                if pbar:
                    pbar.update(1)

                if self.full():
                    break

            if pbar:
                pbar.close()

            # persist updated ptr / count to the file
            self.ptr_ds[...]   = cur_ptr
            self.count_ds[...] = cur_cnt
            self.f.flush()
        # import pdb; pdb.set_trace()
        return n_written


    def ingest_trajectory_hdf5_Ta(
        self,
        traj_filename: str,
        *,
        Ta: int = 16,
        chunk_size: int = 2048,
        require_all_teacher_present: bool = True,
        pad_tail: bool = True,
        show_progress: bool = True,
        selfplay_data = False,
        end_episode: int | None = None,   # <-- ADD THIS
    ) -> int:
        """
        Stream length-Ta windows from <traj_filename>.hdf5 into this buffer.

        For each episode with length T:
        - Build windows of actions [t : t+Ta) for both robot & teacher.
        - Save the observation at index t+1 (mimics train_interactive_learning_repetition's
            data_id=1 logic, where obs aligns with the 2nd element of the window).
        - A window is valid if teacher actions are present over the whole window
            (ALL no_teacher_actions==False) when require_all_teacher_present=True;
            otherwise at least one step present (ANY).
        - If pad_tail=True and t+Ta > T, pad with the last available action.

        NOTE:
        Your buffer must have action shapes (Ta, act_dim).

        Returns
        -------
        n_written : int
            Number of Ta-windows written.
        """
        from hydra.utils import get_original_cwd
        from hydra.core.hydra_config import HydraConfig

        if HydraConfig and getattr(HydraConfig, "is_initialized", None) and HydraConfig.is_initialized():
            project_root = get_original_cwd()
            src_path = os.path.join(project_root, traj_filename)
        else:
            src_path = traj_filename

        logger.info(f"[Ta={Ta}] load trajectory dataset from: {src_path}")
        n_written = 0

        # Pre-bind destination handles
        dst_obs_dsets = {
            name: self.datasets[name]
            for name in self.field_shapes
            if name not in ("robot_action", "teacher_action")
        }
        dst_robot = self.datasets["robot_action"]
        dst_teacher = self.datasets["teacher_action"]

        cur_ptr = int(self.ptr_ds[()])
        cur_cnt = int(self.count_ds[()])

        sum_of_feedback = 0
        sum_of_padding = 0

        with h5py.File(src_path, "r") as src:
            episode_keys_all = sorted(k for k in src if k.startswith("episode_"))
            if end_episode is None:
                episode_keys = episode_keys_all
            else:
                episode_keys = episode_keys_all[: end_episode + 1]
                
            pbar = tqdm(total=len(episode_keys), disable=not show_progress,
                        desc=f"Episodes (Ta={Ta})") if show_progress else None

            for ep in episode_keys:
                g = src[ep]

                # ---- Load & shape observations (reuse your frame-stack augmentation) ----
                s_obs = {}
                for k in g["observation"]:
                    ds_np = g["observation"][k][()]  # shape (T, ...)
                    if self.image_saved_in_Uint8:
                        # orginal image is [-1, 1]
                        if k in self.dtype_map and self.dtype_map[k] == "uint8":
                            # print(k)
                            ds_np = np.clip((ds_np + 1.0) * 127.5, 0, 255).astype(np.uint8)
                    exp_shape = self.field_shapes.get(k)

                    if exp_shape is None:
                        s_obs[k] = ds_np
                        continue

                    per_sample_shape = ds_np.shape[1:]
                    need_aug = (
                        per_sample_shape == tuple(exp_shape[1:])
                        and (len(per_sample_shape) == len(exp_shape) - 1)
                    )
                    if need_aug:
                        T = ds_np.shape[0]
                        stacked = np.empty((T, 2, *ds_np.shape[1:]), dtype=ds_np.dtype)
                        stacked[:, 1, ...] = ds_np
                        stacked[0, 0, ...] = ds_np[0]
                        if T > 1:
                            stacked[1:, 0, ...] = ds_np[:-1]
                        s_obs[k] = stacked
                    else:
                        s_obs[k] = ds_np

                # ---- Load actions & masks ----
                s_r    = g["robot_actions"][()]          # (T, A)
                s_t    = g["teacher_actions"][()]        # (T, A)
                s_no_r = g["no_robot_actions"][()]       # (T,) bool
                s_no_t = g["no_teacher_actions"][()]     # (T,) bool
                if_success_list = g['if_success'][()]

                T, act_dim = s_r.shape
                if T == 0:
                    if pbar: pbar.update(1)
                    continue

                # ---- Build list of valid window start indices ----
                starts = []
                # observation index is t+1; require it be within [0, T-1]
                # so t must be in [ -1, T-2 ], but we start from 0 and clamp obs index.
                for t in range(T):  # consider all starts; we'll clamp/pad as needed
                    end = t + Ta
                    window_mask = s_no_t[t:min(end, T)]
                    if require_all_teacher_present:
                        # TODO the traj_data should also save if_success iterm so that we can do padding for the success expisodes
                        if_success = if_success_list[-1] ## to do, load from data
                        ok_teacher = (window_mask.size == Ta and not window_mask.any()) \
                                    or (pad_tail and not window_mask.any() and if_success) \
                                    or (not pad_tail and end <= T and not window_mask.any())
                        
                        # ok_teacher = (window_mask.size == Ta and not window_mask[1:].any()) \
                        #             or (not pad_tail and end <= T and not window_mask[1:].any())
                    else:
                        if_success = None
                        ok_teacher = not window_mask.all()  # at least one present
                        if selfplay_data: ok_teacher = True
                    
                    # if sum_of_feedback > 20695 and t >= T -3:
                    #     import pdb; pdb.set_trace()

                    if end <= T:
                        # exact window in-range
                        if ok_teacher:
                            starts.append(t)
                    else:
                        # tail window
                        if pad_tail and ok_teacher:
                            starts.append(t)
                            sum_of_padding = sum_of_padding + 1
                        # else drop

                sum_of_feedback += len(starts)
                logger.info('sum_of_feedback:  %s  ep:  %s  sum_of_padding:  %s  if_success:  %s', sum_of_feedback, ep, sum_of_padding, if_success)
                if not starts:
                    if pbar: pbar.update(1)
                    continue

                # ---- Stream in blocks of starts to limit RAM ----
                for block_beg in range(0, len(starts), chunk_size):
                    block_starts = starts[block_beg:block_beg + chunk_size]
                    n_chunk = len(block_starts)

                    # Prepare destination slices (handle wrap)
                    dst_end = cur_ptr + n_chunk
                    wrap = dst_end > self.max_size
                    first_part = (self.max_size - cur_ptr) if wrap else n_chunk
                    second_part = n_chunk - first_part if wrap else 0

                    # Allocate temp arrays for actions
                    robot_chunk   = np.empty((n_chunk, Ta, act_dim), dtype=dst_robot.dtype)
                    teacher_chunk = np.empty((n_chunk, Ta, act_dim), dtype=dst_teacher.dtype)

                    # Build windows + choose observation index = t+1 (clamped to T-1)
                    # We also build per-field obs selection for this block.
                    obs_idx_list = []
                    for i, t in enumerate(block_starts):
                        end = t + Ta
                        in_range = min(end, T)
                        win_len = in_range - t
                        # fill actions
                        robot_chunk[i, :win_len]   = s_r[t:in_range]
                        teacher_chunk[i, :win_len] = s_t[t:in_range]
                        if win_len < Ta:
                            # pad with the last available action
                            robot_chunk[i, win_len:]   = s_r[in_range - 1]
                            teacher_chunk[i, win_len:] = s_t[in_range - 1]
                        # obs index: t+1 (data_id=1 in your training), clamp to [0, T-1]
                        obs_idx_list.append(min(t + 1, T - 1))

                    obs_idx_arr = np.asarray(obs_idx_list, dtype=np.int64)

                    # Write FIRST part
                    sl = slice(cur_ptr, cur_ptr + first_part)
                    # observations: fancy-index each obs field
                    for k, arr in s_obs.items():
                        dst_obs_dsets[k][sl] = arr[obs_idx_arr[:first_part]]
                    dst_robot[sl]   = robot_chunk[:first_part]
                    dst_teacher[sl] = teacher_chunk[:first_part]

                    # Optional wrap-around write
                    if wrap and second_part:
                        sl2 = slice(0, second_part)
                        for k, arr in s_obs.items():
                            dst_obs_dsets[k][sl2] = arr[obs_idx_arr[first_part:]]
                        dst_robot[sl2]   = robot_chunk[first_part:]
                        dst_teacher[sl2] = teacher_chunk[first_part:]

                    # Update pointers
                    cur_ptr = (cur_ptr + n_chunk) % self.max_size
                    cur_cnt = min(cur_cnt + n_chunk, self.max_size)
                    n_written += n_chunk

                    if self.full():
                        break

                if pbar: pbar.update(1)
                if self.full():
                    break

            if pbar:
                pbar.close()

        # persist updated ptr / count
        self.ptr_ds[...]   = cur_ptr
        self.count_ds[...] = cur_cnt
        self.f.flush()
        return n_written



class hdf5buffer_fromList:
    """
    Read-only wrapper around multiple HDF5Buffer files.
    Input: list of .h5 paths
    Internally: creates HDF5Buffer objects in read-only mode.
    """

    def __init__(
        self,
        list_of_paths,
        field_shapes,
        min_size=0,
        sampling="proportional",
        dtype_map=None,
        compression="lzf",
        image_saved_in_Uint8 = True,
    ):
        """
        Args:
            list_of_paths: list[str], paths to .h5 files (each created by HDF5Buffer).
            field_shapes: same dict used to construct each HDF5Buffer.
            min_size: logical threshold for initialized().
            sampling:
                - "proportional": weight ∝ buffer size
                - "uniform": each buffer equally likely
        """
        assert len(list_of_paths) > 0, "list_of_paths must not be empty."
        assert sampling in ("proportional", "uniform")

        self.field_shapes = field_shapes
        self.min_size = int(min_size)
        self.sampling = sampling
        self.sample_ratio = None  # <-  optional manual per-buffer weights
        self.dtype_map = dtype_map
        self.image_saved_in_Uint8 = image_saved_in_Uint8

        # create + load HDF5Buffer objects
        self.buffers = []
        # for path in list_of_paths:
        #     buf = HDF5Buffer(
        #         filename=path,
        #         field_shapes=field_shapes,
        #         min_size=0,
        #         max_size=1_000_000_000,  # ignored for read-only
        #         unique_per_run=False,
        #         dtype_map=dtype_map,
        #         compression=compression,
        #     )
        #     # buf.load_from_file(path, read_only=True)
        #     self.buffers.append(buf)

        # # init sampling statistics
        # self._refresh_lengths()
        # self._seq_buf_idx = 0

    def load_from_file(self, paths, read_only=True, sample_ratio=None):
        """
        Load one or many HDF5 files into this combined buffer.

        Args:
            paths: str or list[str]
            read_only: passed to each underlying HDF5Buffer
            sample_ratio: None or list/array of floats, one per buffer.
                        They do NOT need to sum to 1 (we normalize).
        """
        # Normalize paths to list
        if isinstance(paths, str):
            paths = [paths]
        
        logger.info("Using %s buffer files:", len(paths))
        for p in paths:
            logger.info('  - %s', p)

        # Normalize / store sample_ratio if provided
        if sample_ratio is not None:
            sample_ratio = np.array(sample_ratio, dtype=np.float64)
            if sample_ratio.ndim != 1 or sample_ratio.shape[0] != len(paths):
                raise ValueError(
                    f"sample_ratio must be a 1D array of length {len(paths)}, "
                    f"got shape {sample_ratio.shape}"
                )
            # we keep raw ratios; normalization happens in _refresh_lengths
            self.sample_ratio = sample_ratio
        else:
            # If you want to clear manual ratios when not provided, uncomment:
            # self.sample_ratio = None
            pass

        # Close previous buffers if any
        for buf in getattr(self, "buffers", []):
            try:
                buf.close()
            except Exception:
                pass

        # Create new HDF5Buffer objects
        new_buffers = []
        for path in paths:
            buf = HDF5Buffer(
                filename=path,
                field_shapes=self.field_shapes,
                min_size=0,
                max_size=1_000_000_000,
                dtype_map=getattr(self, "dtype_map", {}),
                compression=getattr(self, "compression", "lzf"),
                unique_per_run=False,
                image_saved_in_Uint8= self.image_saved_in_Uint8
            )
            buf.load_from_file(path, read_only=read_only)
            new_buffers.append(buf)

        self.buffers = new_buffers
        self._seq_buf_idx = 0
        self._refresh_lengths()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _refresh_lengths(self):
        """Recompute per-buffer lengths and sampling weights."""
        if not self.buffers:
            self._lengths = np.array([], dtype=np.int64)
            self._total_len = 0
            self._weights = None
            return

        self._lengths = np.array([buf.length() for buf in self.buffers], dtype=np.int64)
        self._total_len = int(self._lengths.sum())

        # ---- NEW: manual ratios override sampling mode ----
        if self.sample_ratio is not None:
            w = np.array(self.sample_ratio, dtype=np.float64)
            if w.shape[0] != self._lengths.shape[0]:
                raise ValueError(
                    f"sample_ratio has length {w.shape[0]} but there are "
                    f"{self._lengths.shape[0]} buffers"
                )
            # zero out empty buffers
            w[self._lengths == 0] = 0.0
            s = w.sum()
            if s == 0:
                # fall back to uniform over non-empty buffers
                nonempty = (self._lengths > 0)
                cnt = nonempty.sum()
                if cnt == 0:
                    self._weights = None
                else:
                    u = np.zeros_like(self._lengths, dtype=np.float64)
                    u[nonempty] = 1.0 / cnt
                    self._weights = u
            else:
                self._weights = w / s
            return

        # ---- original behavior when no manual ratio is set ----
        if self.sampling == "proportional":
            if self._total_len == 0:
                self._weights = None
            else:
                self._weights = self._lengths / self._total_len
        else:  # "uniform"
            n = len(self.buffers)
            self._weights = np.full(n, 1.0 / n, dtype=np.float64)

    def _check_non_empty(self):
        if self._total_len == 0:
            raise RuntimeError("All underlying buffers are empty")

    # ------------------------------------------------------------------
    # Public API (similar to HDF5Buffer)
    # ------------------------------------------------------------------
    def length(self) -> int:
        """Total number of transitions across all underlying buffers."""
        # Make sure we reflect any growth in underlying buffers
        self._refresh_lengths()
        return self._total_len

    def initialized(self) -> bool:
        """True if global length >= min_size."""
        return self.length() >= self.min_size

    def full(self) -> bool:
        """
        Semantics are a bit ambiguous for a combined buffer.
        Here we just mirror 'initialized': if you rely on 'full' for algo logic,
        adapt as needed.
        """
        return self.initialized()

    def add_buffer(self, buf):
        """
        Dynamically add a new HDF5Buffer to the list (e.g., a newly recorded file).
        """
        if buf.field_shapes != self.field_shapes:
            raise ValueError("New buffer must share the same field_shapes")
        self.buffers.append(buf)
        self._refresh_lengths()

    def _allocate_counts(self, batch_size: int):
        """
        Decide how many samples to draw from each buffer for a batch.
        Returns an int array 'counts' of length len(buffers) that sums to batch_size.
        """
        self._refresh_lengths()
        self._check_non_empty()

        n_buf = len(self.buffers)

        effective_weights = self._weights.copy()
        effective_weights[self._lengths == 0] = 0.0
        total_w = effective_weights.sum()
        if total_w == 0:
            raise RuntimeError("All underlying buffers have zero length or zero weight")
        effective_weights /= total_w

        counts = np.random.multinomial(batch_size, effective_weights)

        # Clip by each buffer length (optional safety)
        for i in range(n_buf):
            if counts[i] > self._lengths[i] and self._lengths[i] > 0:
                counts[i] = self._lengths[i]

        deficit = batch_size - int(counts.sum())
        if deficit > 0:
            cap = self._lengths - counts
            cap[cap < 0] = 0
            cap_total = cap.sum()
            if cap_total > 0:
                extra = np.random.multinomial(deficit, cap / cap_total)
                counts += extra

        return counts

    def sample_randomly(self, batch_size: int):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        counts = self._allocate_counts(batch_size)
        batch = []

        for buf, n in zip(self.buffers, counts):
            if n <= 0 or buf.length() == 0:
                continue
            sub_batch = buf.sample_randomly(n)
            batch.extend(sub_batch)

        if len(batch) > batch_size:
            batch = batch[:batch_size]

        np.random.shuffle(batch)
        return batch

    # ------------------------------------------------------------------
    # "Sequential" sampling – delegate to each buffer's sequential stream
    # ------------------------------------------------------------------
    def sample(self, batch_size: int):
        """
        Sequential-ish sampling with per-buffer ratios:
        We allocate how many samples to draw from each buffer using the same
        weights as random sampling (including sample_ratio, if set),
        then draw each portion using that buffer's sequential sampler.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self._refresh_lengths()
        self._check_non_empty()

        counts = self._allocate_counts(batch_size)
        n_buf = len(self.buffers)

        batch = []
        start = self._seq_buf_idx

        # Iterate buffers in a rotating order for some fairness
        for offset in range(n_buf):
            i = (start + offset) % n_buf
            buf = self.buffers[i]
            take = counts[i]

            if take <= 0 or buf.length() == 0:
                continue

            # safety: don't exceed its length (even though _allocate_counts already tries not to)
            take = min(take, buf.length())
            if take <= 0:
                continue

            sub_batch = buf.sample(take)  # each uses its own sequential stream
            batch.extend(sub_batch)

        self._seq_buf_idx = (self._seq_buf_idx + 1) % n_buf

        if len(batch) > batch_size:
            batch = batch[:batch_size]
        elif len(batch) < batch_size and len(batch) > 0:
            extra_idxs = np.random.randint(0, len(batch), size=batch_size - len(batch))
            for ei in extra_idxs:
                batch.append(batch[ei])

        np.random.shuffle(batch)
        return batch

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------
    def close(self):
        """Close all underlying HDF5 files."""
        for buf in self.buffers:
            try:
                buf.close()
            except Exception:
                pass


class Buffer:
    def __init__(self, min_size, max_size):
        self.buffer = []
        self.min_size, self.max_size = min_size, max_size

    def full(self):
        return len(self.buffer) >= self.max_size

    def initialized(self):
        return len(self.buffer) >= self.min_size

    def add(self, step):
        if self.full():
            self.buffer.pop(0)
        self.buffer.append(step)

    def sample(self, batch_size):
        return [random.choice(self.buffer) for _ in range(batch_size)]

    def length(self):
        return len(self.buffer)

    def save_to_file(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.buffer, file)

    def load_from_file(self, filename):
        with open(filename, 'rb') as file:
            self.buffer = pickle.load(file)



class Buffer_uniform_sampling:
    """
    A replay-buffer that lets you draw batches without replacement,
    while keeping memory overhead tiny.

    ── Behaviour ──────────────────────────────────────────────
    • add(step)              – append a new step (drops oldest if full)
    • sample(batch_size)     – uniform batch; every element is visited
                               once before the buffer is reshuffled
    • full(), initialized()  – same semantics as before
    • save_to_file / load_from_file – unchanged
    """
    def __init__(self, min_size: int, max_size: int) -> None:
        self.buffer: list = []                # or deque(maxlen=max_size) if you prefer
        self.min_size, self.max_size = min_size, max_size

        # internal bookkeeping for sampling
        self._indices: list[int] = []         # current permutation of indices
        self._ptr: int = 0                    # cursor into _indices
        self._need_shuffle: bool = True       # flag to reshuffle before next sample

    # ── helpers ──────────────────────────────────────────────
    def _reshuffle(self) -> None:
        """Create a new random permutation of indices."""
        self._indices = list(range(len(self.buffer)))
        random.shuffle(self._indices)
        self._ptr = 0
        self._need_shuffle = False

    # ── public API ───────────────────────────────────────────
    def full(self) -> bool:
        return len(self.buffer) >= self.max_size

    def initialized(self) -> bool:
        return len(self.buffer) >= self.min_size

    def add(self, step) -> None:
        """Add a transition; if full, drop the oldest one."""
        if self.full():
            # drop-oldest: pop(0) is O(n); switch to deque if this is a bottleneck
            self.buffer.pop(0)
        self.buffer.append(step)

        # the buffer changed → reshuffle before the next sample epoch starts
        self._need_shuffle = True

    def sample(self, batch_size: int):
        if len(self.buffer) < batch_size:
            raise ValueError(
                f"Buffer currently holds {len(self.buffer)} elements, "
                f"batch_size={batch_size} is too large."
            )

        # prepare permutation if needed, or if current epoch is almost exhausted
        if self._need_shuffle or self._ptr + batch_size > len(self._indices):
            self._reshuffle()

        # slice the next chunk of indices and advance the cursor
        start, end = self._ptr, self._ptr + batch_size
        self._ptr = end
        idx_batch = self._indices[start:end]

        return [self.buffer[i] for i in idx_batch]

    def length(self) -> int:
        return len(self.buffer)

    # ── persistence ─────────────────────────────────────────
    def save_to_file(self, filename: str) -> None:
        with open(filename, "wb") as f:
            pickle.dump(self.buffer, f)

    def load_from_file(self, filename: str) -> None:

        # used to obtain the src path instead of hydra's temporary run directory
        from hydra.utils import get_original_cwd
        project_root = get_original_cwd()
        # import pdb; pdb.set_trace()
        full_path = os.path.join(project_root, filename)

        # sanity‐check (optional):
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Cannot find pickle file: {full_path!r}")
        with open(full_path, "rb") as f:
            self.buffer = pickle.load(f)

        # force a reshuffle so the first sample after loading behaves correctly
        self._need_shuffle = True

    def ingest_trajectory_hdf5_to_Intervention_buffer_Ta(
        self,
        traj_filename: str,
        *,
        Ta: int = 16,
        show_progress: bool = True,
        end_episode: int | None = None,
        reset_buffer: bool = True,
        copy_hdf5_to_hydra: bool = True,
        require_all_teacher_present: bool = True,
        pad_tail: bool = True,
        selfplay_data: bool = False,
    ) -> int:
        """
        Rebuild the in-memory intervention buffer from a trajectory HDF5 file
        while matching `HDF5Buffer.ingest_trajectory_hdf5_Ta` for:
        - teacher-availability filtering
        - padded tail-window handling

        Each loaded sample matches the online intervention format:
            [obs_t, preferred_action_chunk, robot_action_chunk]

        `obs_t` is loaded at `min(t + 1, T - 1)` for a chunk that starts at
        action index `t`, matching the HDF5 buffer path.
        """
        from hydra.utils import get_original_cwd
        from hydra.core.hydra_config import HydraConfig

        if Ta <= 0:
            raise ValueError(f"Ta must be positive, got {Ta}.")

        if HydraConfig and getattr(HydraConfig, "is_initialized", None) and HydraConfig.is_initialized():
            project_root = get_original_cwd()
            src_path = os.path.join(project_root, traj_filename)
        else:
            src_path = traj_filename

        logger.info(f"[intervention Ta={Ta} updated] load trajectory dataset from: {src_path}")

        if reset_buffer:
            self.reset_to_empty()

        n_written = 0

        with h5py.File(src_path, "r") as src:
            episode_keys_all = sorted(k for k in src if k.startswith("episode_"))
            if end_episode is None:
                episode_keys = episode_keys_all
            else:
                episode_keys = episode_keys_all[: end_episode + 1]

            pbar = tqdm(
                total=len(episode_keys),
                disable=not show_progress,
                desc=f"Intervention episodes updated (Ta={Ta})",
            ) if show_progress else None

            for ep in episode_keys:
                g = src[ep]
                obs_group = g["observation"]

                teacher_actions = np.asarray(g["teacher_actions"][()], dtype=np.float32)
                robot_actions = np.asarray(g["robot_actions"][()], dtype=np.float32)
                no_teacher_actions = np.asarray(g["no_teacher_actions"][()], dtype=np.bool_)
                if_success_list = g["if_success"][()] if "if_success" in g else np.asarray([False], dtype=np.bool_)

                T = int(robot_actions.shape[0])
                if T == 0:
                    if pbar:
                        pbar.update(1)
                    continue

                starts = []
                for t in range(T):
                    end = t + Ta
                    window_mask = no_teacher_actions[t:min(end, T)]

                    if require_all_teacher_present:
                        if_success = bool(if_success_list[-1]) if np.size(if_success_list) > 0 else False
                        ok_teacher = (
                            (window_mask.size == Ta and not window_mask.any())
                            or (pad_tail and not window_mask.any() and if_success)
                            or (not pad_tail and end <= T and not window_mask.any())
                        )
                    else:
                        ok_teacher = not window_mask.all()
                        if selfplay_data:
                            ok_teacher = True

                    if end <= T:
                        if ok_teacher:
                            starts.append(t)
                    else:
                        if pad_tail and ok_teacher:
                            starts.append(t)

                if not starts:
                    if pbar:
                        pbar.update(1)
                    continue

                for t in starts:
                    end = t + Ta
                    in_range = min(end, T)
                    win_len = in_range - t
                    if win_len <= 0:
                        continue

                    robot_chunk = np.empty((Ta, robot_actions.shape[1]), dtype=np.float32)
                    teacher_chunk = np.empty_like(robot_chunk, dtype=np.float32)

                    robot_chunk[:win_len] = robot_actions[t:in_range]
                    teacher_chunk[:win_len] = teacher_actions[t:in_range]
                    if win_len < Ta:
                        robot_chunk[win_len:] = robot_actions[in_range - 1]
                        teacher_chunk[win_len:] = teacher_actions[in_range - 1]

                    obs_t_index = min(t + 1, T - 1)
                    obs_t = self._load_obs_from_traj_group(obs_group, obs_t_index)
                    self.add([obs_t, teacher_chunk, robot_chunk])
                    n_written += 1

                if pbar:
                    pbar.update(1)

            if pbar:
                pbar.close()

        return n_written

    
    def load_from_h5_buffer_file(self, filename: str, action_horizon: int = 0) -> None:
        """
        Load transitions from an HDF5Buffer-formatted .h5 file into self.buffer.

        Each loaded item is a tuple: (obs_dict, action_teacher, action_robot).

        If the file contains more transitions than this buffer's max_size,
        only the most recent self.max_size transitions are kept (based on the
        circular buffer's ptr/count metadata).

        Parameters
        ----------
        filename : str
            Path to the HDF5 buffer file.
        action_horizon : int, optional (default=0)
            If the stored actions are chunks with shape (T, A), this selects
            which time-step to use. For example, if `action_horizon == 1`,
            the second action in the chunk (index 1) is used.
            If actions are 1D (shape (A,)), this is ignored.
        """
        full_path = filename
        with h5py.File(full_path, "r") as f:
            # Basic validation
            required_keys = {"robot_action", "teacher_action", "ptr", "count"}
            if not required_keys.issubset(set(f.keys())):
                missing = required_keys.difference(set(f.keys()))
                raise ValueError(
                    f"HDF5 file {full_path!r} is missing required datasets: {sorted(missing)}"
                )

            d_robot = f["robot_action"]
            d_teacher = f["teacher_action"]
            file_max_size = int(d_robot.shape[0])  # same length for all per-design
            count = int(f["count"][()])
            ptr = int(f["ptr"][()])

            # Nothing to load
            n_available = min(count, file_max_size)
            if n_available <= 0:
                self.buffer = []
                self._need_shuffle = True
                return

            # Determine chronological indices (oldest → newest) in the ring buffer
            if n_available < file_max_size:
                # Not full yet: valid data are at [0, n_available)
                chronological = list(range(n_available))
            else:
                # Full: oldest starts at ptr, then wraps
                chronological = list(range(ptr, file_max_size)) + list(range(0, ptr))

            # Keep only the most recent up to this buffer's capacity
            n_keep = min(n_available, self.max_size)
            indices = chronological[-n_keep:]

            # Pre-bind observation datasets (everything except actions and metadata)
            obs_names = [
                k for k in f.keys()
                if k not in ("robot_action", "teacher_action", "ptr", "count")
            ]
            obs_dsets = {k: f[k] for k in obs_names}

            loaded = []
            for i in indices:
                # Build obs dict lazily per index
                obs_dict = {k: obs_dsets[k][i] for k in obs_names}

                # Read actions
                action_robot_arr = d_robot[i][()]      # shape: (A,) or (T, A)
                action_teacher_arr = d_teacher[i][()]  # shape: (A,) or (T, A)

                # If actions are chunks over time, select the requested horizon index
                # e.g., if action_horizon == 1, take the second action in the chunk.
                if hasattr(action_robot_arr, "ndim") and action_robot_arr.ndim > 1:
                    if not (0 <= action_horizon <= action_robot_arr.shape[0]):
                        raise ValueError(
                            f"action_horizon={action_horizon} is out of bounds for "
                            f"action chunk with length {action_robot_arr.shape[0]}"
                        )
                    # action_robot = action_robot_arr[action_horizon]
                    if action_horizon == 1:
                        action_robot = action_robot_arr[1]
                    else:
                        action_robot = action_robot_arr[:action_horizon]
                else:
                    action_robot = action_robot_arr

                if hasattr(action_teacher_arr, "ndim") and action_teacher_arr.ndim > 1:
                    if not (0 <= action_horizon <= action_teacher_arr.shape[0]):
                        raise ValueError(
                            f"action_horizon={action_horizon} is out of bounds for "
                            f"action chunk with length {action_teacher_arr.shape[0]}"
                        )
                    if action_horizon == 1:
                        action_teacher = action_teacher_arr[1]
                    else:
                        action_teacher = action_teacher_arr[:action_horizon]
                else:
                    action_teacher = action_teacher_arr

                # Match your tuple convention used elsewhere
                loaded.append((obs_dict, action_teacher, action_robot))

            self.buffer = loaded
            # Ensure next sample() reshuffles
            self._need_shuffle = True

            
class Buffer_uniform_refer_Traj_hdf5:
    """
    Stores (traj_id, t_index, a_pos, a_neg[, should_save_mask]) and dereferences obs from traj HDF5 on sample.
    Uniform sampling without replacement per-epoch (same behavior as your Buffer_uniform_sampling).
    """

    def __init__(self, min_size: int, max_size: int, traj_hdf5_path: str,
                 image_key_hints=("rgb", "image", "img", "camera", "cam"),
                 traj_obs_combine_previous: bool = False):
        self.buffer = []
        self.min_size, self.max_size = min_size, max_size
        self.traj_hdf5_path = self._resolve_traj_hdf5_path(traj_hdf5_path)
        self.image_key_hints = tuple(s.lower() for s in image_key_hints)
        self.traj_obs_combine_previous = bool(traj_obs_combine_previous)

        self._indices = []
        self._ptr = 0
        self._need_shuffle = True

    @staticmethod
    def _resolve_traj_hdf5_path(path: str) -> str:
        if os.path.isabs(path):
            return path

        candidates = [path]
        try:
            from hydra.utils import get_original_cwd
            from hydra.core.hydra_config import HydraConfig

            if HydraConfig and getattr(HydraConfig, "is_initialized", None) and HydraConfig.is_initialized():
                candidates.append(os.path.join(get_original_cwd(), path))
        except Exception:
            pass

        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate

        return candidates[-1]

    # ---------- sampling bookkeeping ----------
    def _reshuffle(self):
        self._indices = list(range(len(self.buffer)))
        random.shuffle(self._indices)
        self._ptr = 0
        self._need_shuffle = False

    def full(self) -> bool:
        return len(self.buffer) >= self.max_size

    def initialized(self) -> bool:
        return len(self.buffer) >= self.min_size

    def add(self, ref_step) -> None:
        # ref_step: (traj_id, t_index, a_pos, a_neg[, should_save_mask])
        if self.full():
            self.buffer.pop(0)
        self.buffer.append(ref_step)
        self._need_shuffle = True

    def length(self) -> int:
        return len(self.buffer)
    
    def reset_to_empty(self) -> None:
        """Completely clear the buffer and reset sampling state."""
        self.buffer.clear()
        self._indices.clear()
        self._ptr = 0
        self._need_shuffle = True

    def save_to_file(self, filename: str, episode_id: Optional[int] = None) -> str:
        """
        Save a compact snapshot of the current non-intervention reference buffer.
        To reduce storage, a_neg is only saved when traj_id == episode_id
        (if episode_id is provided).
        """
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)

        entries = []
        for sample in self.buffer:
            if len(sample) == 5:
                traj_id, t_index, a_pos, a_neg, should_save_mask = sample
            elif len(sample) == 4:
                traj_id, t_index, a_pos, a_neg = sample
                should_save_mask = None
            else:
                continue

            entry = {
                "traj_id": int(traj_id),
                "t_index": int(t_index),
                "a_pos": np.asarray(a_pos, dtype=np.float32),
            }
            if episode_id is None or int(traj_id) == int(episode_id):
                entry["a_neg"] = np.asarray(a_neg, dtype=np.float32)
            if should_save_mask is not None:
                entry["should_save"] = np.asarray(should_save_mask, dtype=np.bool_)
            entries.append(entry)

        payload = {
            "episode_id": int(episode_id) if episode_id is not None else None,
            "buffer_length": int(len(entries)),
            "traj_hdf5_path": self.traj_hdf5_path,
            "entries": entries,
        }
        with open(filename, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        return filename

    def ingest_trajectory_hdf5_to_Intervention_buffer_Ta(
        self,
        traj_filename: str,
        *,
        Ta: int = 16,
        show_progress: bool = True,
        end_episode: int | None = None,
        reset_buffer: bool = True,
        copy_hdf5_to_hydra: bool = False,
        require_all_teacher_present: bool = True,
        pad_tail: bool = True,
        selfplay_data: bool = False,
    ) -> int:
        """
        Load Ta chunks from a trajectory HDF5 into self.buffer while matching
        `HDF5Buffer.ingest_trajectory_hdf5_Ta` for:
        - teacher-availability filtering
        - padded tail-window handling

        Each saved entry follows this buffer's reference format:
        [traj_id, t_index, h_chunk, r_chunk, should_save]

        `t_index` stores the observation index used by the HDF5 buffer path,
        i.e. `min(t + 1, T - 1)` for a chunk that starts at action index `t`.
        """
        if Ta <= 0:
            raise ValueError(f"Ta must be positive, got {Ta}.")

        src_path = self._resolve_traj_hdf5_path(traj_filename)
        if not copy_hdf5_to_hydra:
            self.traj_hdf5_path = src_path

        logger.info(f"[ref Ta={Ta} updated] load trajectory dataset from: {src_path}")

        if reset_buffer:
            self.reset_to_empty()

        n_written = 0

        with h5py.File(src_path, "r") as src:
            episode_keys_all = sorted(k for k in src if k.startswith("episode_"))
            if end_episode is None:
                episode_keys = episode_keys_all
            else:
                episode_keys = episode_keys_all[: end_episode + 1]

            pbar = tqdm(
                total=len(episode_keys),
                disable=not show_progress,
                desc=f"Ref episodes updated (Ta={Ta})",
            ) if show_progress else None

            for traj_id, ep in enumerate(episode_keys):
                g = src[ep]

                teacher_actions = np.asarray(g["teacher_actions"][()], dtype=np.float32)
                robot_actions = np.asarray(g["robot_actions"][()], dtype=np.float32)
                no_teacher_actions = np.asarray(g["no_teacher_actions"][()], dtype=np.bool_)
                if_success_list = g["if_success"][()] if "if_success" in g else np.asarray([False], dtype=np.bool_)

                T = int(robot_actions.shape[0])
                if T == 0:
                    if pbar:
                        pbar.update(1)
                    continue

                starts = []
                for t in range(T):
                    end = t + Ta
                    window_mask = no_teacher_actions[t:min(end, T)]

                    if require_all_teacher_present:
                        if_success = bool(if_success_list[-1]) if np.size(if_success_list) > 0 else False
                        ok_teacher = (
                            (window_mask.size == Ta and not window_mask.any())
                            or (pad_tail and not window_mask.any() and if_success)
                            or (not pad_tail and end <= T and not window_mask.any())
                        )
                    else:
                        ok_teacher = not window_mask.all()
                        if selfplay_data:
                            ok_teacher = True

                    if end <= T:
                        if ok_teacher:
                            starts.append(t)
                    else:
                        if pad_tail and ok_teacher:
                            starts.append(t)

                if not starts:
                    if pbar:
                        pbar.update(1)
                    continue

                for t in starts:
                    end = t + Ta
                    in_range = min(end, T)
                    win_len = in_range - t
                    if win_len <= 0:
                        continue

                    r_chunk = np.empty((Ta, robot_actions.shape[1]), dtype=np.float32)
                    h_chunk = np.empty_like(r_chunk, dtype=np.float32)

                    r_chunk[:win_len] = robot_actions[t:in_range]
                    h_chunk[:win_len] = teacher_actions[t:in_range]
                    if win_len < Ta:
                        r_chunk[win_len:] = robot_actions[in_range - 1]
                        h_chunk[win_len:] = teacher_actions[in_range - 1]

                    should_save = np.ones((max(Ta - 1, 0),), dtype=np.bool_)
                    obs_t_index = min(t + 1, T - 1)
                    self.add([traj_id, obs_t_index, h_chunk, r_chunk, should_save])
                    n_written += 1

                if pbar:
                    pbar.update(1)

            if pbar:
                pbar.close()

        return n_written

    # ---------- obs helpers ----------
    def _looks_like_image(self, key: str, arr: np.ndarray) -> bool:
        k = str(key).lower()
        if any(h in k for h in self.image_key_hints):
            return True
        # # fallback shape heuristic
        # if isinstance(arr, np.ndarray) and arr.ndim == 3:
        #     if arr.shape[0] in (1, 3, 4) or arr.shape[-1] in (1, 3, 4):
        #         return True
        return False

    def _uint8_to_float_img(self, x: np.ndarray) -> np.ndarray:
        # uint8 [0,255] -> float32 [-1,1]
        x = x.astype(np.float32) / 255.0
        return (2.0 * x - 1.0).astype(np.float32)

    def _get_traj_hdf5_open_kwargs(self) -> dict:
        return {
            "mode": "r",
            "rdcc_nbytes": 256 * 1024 * 1024,
            "rdcc_nslots": 100_003,
            "rdcc_w0": 0.75,
        }

    def _open_traj_store(self):
        traj_hdf5_path = self._resolve_traj_hdf5_path(self.traj_hdf5_path)
        if not os.path.exists(traj_hdf5_path):
            raise FileNotFoundError(
                f"Trajectory HDF5 not found: {traj_hdf5_path!r}. "
                "Load the trajectory dataset first or point the buffer to the correct file."
            )

        self.traj_hdf5_path = traj_hdf5_path
        return _open_h5_read_only(
            traj_hdf5_path,
            **self._get_traj_hdf5_open_kwargs(),
        )

    def _close_traj_store(self, traj_store) -> None:
        if traj_store is None:
            return
        if hasattr(traj_store, "close"):
            try:
                traj_store.close()
            except Exception:
                pass

    def _list_traj_episode_keys(self, traj_store) -> list[str]:
        episode_keys = sorted(
            k for k in traj_store.keys() if str(k).startswith("episode_")
        )
        if episode_keys:
            return episode_keys
        return sorted(traj_store.keys())

    def _extract_traj_request(self, sample) -> tuple[int, int]:
        if isinstance(sample, dict):
            return int(sample["traj_id"]), int(sample["t_index"])
        if len(sample) < 2:
            raise ValueError(
                "Trajectory-reference samples must provide traj_id and t_index."
            )
        return int(sample[0]), int(sample[1])

    def _load_obs_batch(
        self,
        batch_requests,
        *,
        h5_file=None,
        episode_keys: list[str] | None = None,
    ) -> dict:
        """
        Load a batch of observations while minimizing tiny random trajectory
        HDF5 reads.

        Requests are sorted by (traj_id, t_index), grouped by trajectory, and
        each trajectory is fetched with one contiguous [min_t : max_t + 1]
        slice per observation key.
        """
        if not batch_requests:
            raise ValueError("Cannot load an empty batch of trajectory observations.")

        owns_handle = h5_file is None
        if owns_handle:
            h5_file = self._open_traj_store()

        try:
            if episode_keys is None:
                episode_keys = self._list_traj_episode_keys(h5_file)

            parsed_requests = []
            for original_index, sample in enumerate(batch_requests):
                traj_id, t_index = self._extract_traj_request(sample)
                parsed_requests.append((traj_id, t_index, original_index))
            parsed_requests.sort(key=lambda item: (item[0], item[1], item[2]))

            collated_obs = None
            expected_keys = None
            request_ptr = 0
            while request_ptr < len(parsed_requests):
                traj_id = parsed_requests[request_ptr][0]
                group_end = request_ptr + 1
                while (
                    group_end < len(parsed_requests)
                    and parsed_requests[group_end][0] == traj_id
                ):
                    group_end += 1

                if traj_id < 0 or traj_id >= len(episode_keys):
                    raise IndexError(
                        f"traj_id={traj_id} out of range (0..{len(episode_keys) - 1})"
                    )

                group_requests = parsed_requests[request_ptr:group_end]
                min_t = min(item[1] for item in group_requests)
                max_t = max(item[1] for item in group_requests)
                fetch_min_t = min_t
                if self.traj_obs_combine_previous and min_t > 0:
                    fetch_min_t = min_t - 1

                obs_group = h5_file[episode_keys[traj_id]]["observation"]
                obs_keys = tuple(sorted(obs_group.keys()))
                if expected_keys is None:
                    expected_keys = obs_keys
                    collated_obs = {
                        key: [None] * len(batch_requests) for key in expected_keys
                    }
                elif obs_keys != expected_keys:
                    raise ValueError(
                        "Observation schema mismatch across trajectories in "
                        f"{self.traj_hdf5_path!r}: expected keys {expected_keys}, "
                        f"got {obs_keys} for traj_id={traj_id}."
                    )

                for key in expected_keys:
                    dataset = obs_group[key]
                    traj_length = int(dataset.shape[0])
                    if min_t < 0 or max_t >= traj_length:
                        raise IndexError(
                            f"t_index range [{min_t}, {max_t}] is out of bounds for "
                            f"traj_id={traj_id} with length {traj_length}."
                        )

                    obs_block = dataset[fetch_min_t : max_t + 1]
                    if (
                        isinstance(obs_block, np.ndarray)
                        and obs_block.dtype == np.uint8
                        and self._looks_like_image(key, obs_block)
                    ):
                        obs_block = self._uint8_to_float_img(obs_block)

                    for _, t_index, original_index in group_requests:
                        current_offset = t_index - fetch_min_t
                        obs_current = np.array(obs_block[current_offset], copy=True)
                        if self.traj_obs_combine_previous:
                            past_t_index = t_index - 1
                            if past_t_index < 0:
                                past_offset = current_offset
                            else:
                                past_offset = past_t_index - fetch_min_t
                            obs_past = np.array(obs_block[past_offset], copy=True)
                            collated_obs[key][original_index] = np.stack(
                                [obs_past, obs_current],
                                axis=0,
                            )
                        else:
                            collated_obs[key][original_index] = obs_current

                request_ptr = group_end

            return {
                key: np.stack(values, axis=0) for key, values in collated_obs.items()
            }
        finally:
            if owns_handle and h5_file is not None:
                self._close_traj_store(h5_file)

    def _load_single_obs_at(self, obs_group, t_index: int) -> dict:
        obs = {}
        for k in obs_group.keys():
            # Read single timestep slice
            arr = obs_group[k][t_index]   # (...), not (T,...) ## TODO check if the order is correct
            # Convert images if stored uint8
            if isinstance(arr, np.ndarray) and arr.dtype == np.uint8 and self._looks_like_image(k, arr):
                arr = self._uint8_to_float_img(arr)
            obs[k] = arr
        return obs

    def _load_obs_at(self, traj_id: int, t_index: int) -> dict:
        """
        Read obs_dict for one timestep from trajectory HDF5.
        Converts uint8 images to float32 [-1,1] on load (matching your env preprocessing).
        If traj_obs_combine_previous=True, return n_obs=2 observations by stacking
        [previous, current] for each key, matching franka_env_img._get_obs_dict().
        """
        f = self._open_traj_store()
        try:
            episode_keys = self._list_traj_episode_keys(f)
            if traj_id < 0 or traj_id >= len(episode_keys):
                raise IndexError(f"traj_id={traj_id} out of range (0..{len(episode_keys)-1})")

            ep = f[episode_keys[traj_id]]
            obs_group = ep["observation"]

            obs_current = self._load_single_obs_at(obs_group, t_index)
            if not self.traj_obs_combine_previous:
                return obs_current

            obs_past = self._load_single_obs_at(obs_group, max(t_index - 1, 0))
            return {
                key: np.stack([obs_past[key], obs_current[key]], axis=0)
                for key in obs_current
            }
        finally:
            self._close_traj_store(f)

    # ---------- main sampling ----------
    def sample(self, batch_size: int):
        if len(self.buffer) < batch_size:
            raise ValueError(f"Buffer holds {len(self.buffer)} elements, batch_size={batch_size} too large.")

        if self._need_shuffle or self._ptr + batch_size > len(self._indices):
            self._reshuffle()

        start, end = self._ptr, self._ptr + batch_size
        self._ptr = end
        idx_batch = self._indices[start:end]

        metadata_batch = []
        batch_requests = []
        for i in idx_batch:
            sample = self.buffer[i]
            if len(sample) == 5:
                traj_id, t_index, a_pos, a_neg, should_save_mask = sample
            elif len(sample) == 4:
                traj_id, t_index, a_pos, a_neg = sample
                should_save_mask = None
            else:
                raise ValueError(
                    f"Unexpected ref_step length={len(sample)}. "
                    "Expected 4 or 5 elements: (traj_id, t_index, a_pos, a_neg[, should_save_mask])."
                )
            batch_requests.append((traj_id, t_index))
            metadata_batch.append((a_pos, a_neg, should_save_mask))

        obs_batch = self._load_obs_batch(batch_requests)

        out = []
        for batch_index, (a_pos, a_neg, should_save_mask) in enumerate(metadata_batch):
            obs = {
                key: values[batch_index]
                for key, values in obs_batch.items()
            }
            if should_save_mask is None:
                out.append([obs, a_pos, a_neg])
            else:
                out.append([obs, a_pos, a_neg, should_save_mask])
        return out

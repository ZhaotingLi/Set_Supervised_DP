import numpy as np
import torch
from typing import Optional
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from agents.DP_model.common.pytorch_util import dict_apply
from tools.buffer import Buffer_uniform_refer_Traj_hdf5


def collate_obs_dict(obs_list):
    if not obs_list:
        raise ValueError("The list of observation dictionaries is empty.")

    keys = list(obs_list[0].keys())
    collated = {key: [] for key in keys}

    for obs in obs_list:
        for key in keys:
            collated[key].append(obs[key])

    for key in keys:
        collated[key] = np.stack(collated[key])

    return collated


def _collate_replay_training_batch(batch, collated_obs):
    collated = {
        "nobs": dict_apply(
            collated_obs,
            lambda x: torch.from_numpy(x).to(dtype=torch.float32),
        ),
        "action_preferred": torch.from_numpy(
            np.stack([item["action_preferred"] for item in batch], axis=0)
        ).to(dtype=torch.float32),
        "action_negative": torch.from_numpy(
            np.stack([item["action_negative"] for item in batch], axis=0)
        ).to(dtype=torch.float32),
    }

    if any("should_save" in item for item in batch):
        collated["should_save"] = torch.from_numpy(
            np.stack(
                [
                    np.asarray(item.get("should_save"), dtype=np.bool_)
                    for item in batch
                ],
                axis=0,
            )
        )
    else:
        collated["should_save"] = None

    if any("sample_id" in item for item in batch):
        collated["sample_ids"] = torch.tensor(
            [int(item["sample_id"]) for item in batch],
            dtype=torch.long,
        )
    else:
        collated["sample_ids"] = None

    if any("radius_ratio" in item for item in batch):
        collated["radius_ratio"] = torch.from_numpy(
            np.stack(
                [
                    np.asarray(
                        item.get(
                            "radius_ratio",
                            np.ones(
                                np.asarray(item["action_preferred"]).shape[:-1],
                                dtype=np.float32,
                            ),
                        ),
                        dtype=np.float32,
                    )
                    for item in batch
                ],
                axis=0,
            )
        ).to(dtype=torch.float32)
    else:
        collated["radius_ratio"] = None

    return collated


def replay_training_collate_fn(batch):
    if not batch:
        raise ValueError("Cannot collate an empty replay training batch.")

    return _collate_replay_training_batch(
        batch,
        collate_obs_dict([item["obs"] for item in batch]),
    )


class _ReplayBufferDataset(Dataset):
    """
    Training-side adapter that exposes existing replay buffers through the
    Dataset interface without modifying the buffer implementations.
    """

    def __init__(self, buffer, buffer_type: str, include_sample_ids: bool = False):
        self.buffer = buffer
        self.buffer_type = buffer_type
        self.include_sample_ids = include_sample_ids

    def __len__(self):
        return self.buffer.length()

    def get_sample_weights(self):
        if not hasattr(self.buffer, "get_sample_weights"):
            return None
        weights = self.buffer.get_sample_weights()
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (len(self),):
            return None
        invalid = (~np.isfinite(weights)) | (weights <= 0.0)
        if np.any(invalid):
            weights = weights.copy()
            weights[invalid] = 1.0
        return weights

    def __getitem__(self, index):
        if self.buffer_type == "obs_buffer":
            buffer_sample = self.buffer.buffer[index]
            obs, action_preferred, action_negative = buffer_sample[:3]
            sample = {
                "obs": obs,
                "action_preferred": np.asarray(action_preferred, dtype=np.float32),
                "action_negative": np.asarray(action_negative, dtype=np.float32),
            }
            if len(buffer_sample) >= 4:
                sample["should_save"] = np.asarray(buffer_sample[3], dtype=np.bool_)
            if len(buffer_sample) >= 5:
                sample["radius_ratio"] = np.asarray(buffer_sample[4], dtype=np.float32)
            if self.include_sample_ids and hasattr(self.buffer, "_sample_ids"):
                sample["sample_id"] = int(self.buffer._sample_ids[index])
            return sample

        if self.buffer_type == "traj_ref_buffer":
            ref_sample = self.buffer.buffer[index]
            radius_ratio = None
            if len(ref_sample) == 6:
                traj_id, t_index, action_preferred, action_negative, should_save, radius_ratio = ref_sample
            elif len(ref_sample) == 5:
                traj_id, t_index, action_preferred, action_negative, should_save = ref_sample
            elif len(ref_sample) == 4:
                traj_id, t_index, action_preferred, action_negative = ref_sample
                should_save = None
            else:
                raise ValueError(
                    f"Unexpected non-intervention sample length={len(ref_sample)}."
                )

            sample = {
                "traj_id": int(traj_id),
                "t_index": int(t_index),
                "action_preferred": np.asarray(action_preferred, dtype=np.float32),
                "action_negative": np.asarray(action_negative, dtype=np.float32),
            }
            if should_save is not None:
                sample["should_save"] = np.asarray(should_save, dtype=np.bool_)
            if radius_ratio is not None:
                sample["radius_ratio"] = np.asarray(radius_ratio, dtype=np.float32)
            return sample

        raise ValueError(f"Unsupported buffer_type={self.buffer_type!r}.")


class _TrajRefBatchCollator:
    """
    Batch-level trajectory-store dereferencing for trajectory-reference samples.

    The collator runs inside the DataLoader worker, keeps one trajectory-store
    handle open per worker process, and batches reads by trajectory to avoid
    many tiny open/read/close operations on network filesystems.
    """

    def __init__(self, buffer: Buffer_uniform_refer_Traj_hdf5):
        self.buffer = buffer
        self._traj_store = None
        self._episode_keys = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_traj_store"] = None
        state["_episode_keys"] = None
        return state

    def _close_store(self):
        if self._traj_store is not None:
            self.buffer._close_traj_store(self._traj_store)
        self._traj_store = None
        self._episode_keys = None

    def __del__(self):
        self._close_store()

    def _open_store(self):
        if self._traj_store is None:
            self._traj_store = self.buffer._open_traj_store()
            self._episode_keys = self.buffer._list_traj_episode_keys(self._traj_store)
        return self._traj_store, self._episode_keys

    def __call__(self, batch):
        if not batch:
            raise ValueError("Cannot collate an empty trajectory-reference batch.")

        try:
            traj_store, episode_keys = self._open_store()
            collated_obs = self.buffer._load_obs_batch(
                batch,
                h5_file=traj_store,
                episode_keys=episode_keys,
            )
        except (OSError, KeyError, RuntimeError):
            self._close_store()
            traj_store, episode_keys = self._open_store()
            collated_obs = self.buffer._load_obs_batch(
                batch,
                h5_file=traj_store,
                episode_keys=episode_keys,
            )
        return _collate_replay_training_batch(batch, collated_obs)


class ReplayBatchLoader:
    def __init__(
        self,
        *,
        online_buffer=None,
        reference_buffer=None,
        use_hdf5_dataset: bool = False,
        intervention_replay_buffer_type: Optional[str] = None,
        use_action_buffer: bool = False,
        training_dataloader_num_workers: int = 4,
        training_dataloader_traj_ref_num_workers: int = 0,
        training_dataloader_pin_memory: bool = False,
        training_dataloader_prefetch_factor: int = 2,
        persistent_workers: bool = False,
        collate_fn=replay_training_collate_fn,
        collate_hdf5_intervention_samples: bool = True,
    ):
        self.online_buffer = online_buffer
        self.reference_buffer = reference_buffer
        if intervention_replay_buffer_type is None:
            intervention_replay_buffer_type = (
                "hdf5_obs_action_buffer"
                if use_hdf5_dataset
                else "pickle_obs_action_buffer"
            )
        self.intervention_replay_buffer_type = str(intervention_replay_buffer_type)
        self.use_hdf5_dataset = (
            self.intervention_replay_buffer_type == "hdf5_obs_action_buffer"
        )
        self.use_action_buffer = bool(use_action_buffer)
        self.training_dataloader_num_workers = int(training_dataloader_num_workers)
        self.training_dataloader_traj_ref_num_workers = int(
            training_dataloader_traj_ref_num_workers
        )
        self.training_dataloader_pin_memory = bool(training_dataloader_pin_memory)
        self.training_dataloader_prefetch_factor = int(
            training_dataloader_prefetch_factor
        )
        self.persistent_workers = bool(persistent_workers)
        self.collate_fn = collate_fn
        self.collate_hdf5_intervention_samples = bool(
            collate_hdf5_intervention_samples
        )
        self._training_batch_loader_state = {}

    def set_buffers(self, *, online_buffer=None, reference_buffer=None):
        if online_buffer is not None:
            self.online_buffer = online_buffer
        if reference_buffer is not None:
            self.reference_buffer = reference_buffer

    def _get_training_buffer(self, buffer_name):
        if buffer_name == "online_buffer":
            if self.online_buffer is None:
                raise ValueError("online_buffer is not configured.")
            return self.online_buffer
        if buffer_name == "reference_buffer":
            if self.reference_buffer is None:
                raise ValueError("reference_buffer is not configured.")
            return self.reference_buffer
        raise ValueError(f"Unsupported buffer_name={buffer_name!r}.")

    def infer_replay_buffer_type(self, buffer):
        if isinstance(buffer, Buffer_uniform_refer_Traj_hdf5):
            return "traj_ref_buffer"
        return "obs_buffer"

    def _normalize_training_sample(self, sample, buffer, buffer_type, sample_id=None):
        if buffer_type == "traj_ref_buffer":
            if len(sample) >= 3 and isinstance(sample[0], dict):
                item = {
                    "obs": sample[0],
                    "action_preferred": np.asarray(sample[1], dtype=np.float32),
                    "action_negative": np.asarray(sample[2], dtype=np.float32),
                }
                if len(sample) >= 4:
                    item["should_save"] = np.asarray(sample[3], dtype=np.bool_)
                if len(sample) >= 5:
                    item["radius_ratio"] = np.asarray(sample[4], dtype=np.float32)
            else:
                radius_ratio = None
                if len(sample) == 6:
                    traj_id, t_index, action_preferred, action_negative, should_save, radius_ratio = sample
                elif len(sample) == 5:
                    traj_id, t_index, action_preferred, action_negative, should_save = sample
                elif len(sample) == 4:
                    traj_id, t_index, action_preferred, action_negative = sample
                    should_save = None
                else:
                    raise ValueError(
                        "Expected traj-ref sample to contain 4, 5, or 6 elements."
                    )
                item = {
                    "obs": buffer._load_obs_at(int(traj_id), int(t_index)),
                    "action_preferred": np.asarray(action_preferred, dtype=np.float32),
                    "action_negative": np.asarray(action_negative, dtype=np.float32),
                }
                if should_save is not None:
                    item["should_save"] = np.asarray(should_save, dtype=np.bool_)
                if radius_ratio is not None:
                    item["radius_ratio"] = np.asarray(radius_ratio, dtype=np.float32)
        elif buffer_type == "obs_buffer":
            if len(sample) < 3:
                raise ValueError(
                    "Expected obs-buffer sample to contain [obs, action_preferred, action_negative]."
                )
            item = {
                "obs": sample[0],
                "action_preferred": np.asarray(sample[1], dtype=np.float32),
                "action_negative": np.asarray(sample[2], dtype=np.float32),
            }
            if len(sample) >= 4:
                item["should_save"] = np.asarray(sample[3], dtype=np.bool_)
            if len(sample) >= 5:
                item["radius_ratio"] = np.asarray(sample[4], dtype=np.float32)
        else:
            raise ValueError(f"Unsupported buffer_type={buffer_type!r}.")

        if sample_id is not None:
            item["sample_id"] = int(sample_id)
        return item

    def _build_weighted_sampler(self, dataset):
        weights = dataset.get_sample_weights()
        if weights is None or weights.size <= 0:
            return None
        if np.allclose(weights, weights[0]):
            return None
        return WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(dataset),
            replacement=True,
        )

    def _build_training_batch_loader(self, buffer_name, batch_size):
        target_buffer = self._get_training_buffer(buffer_name)
        buffer_type = self.infer_replay_buffer_type(target_buffer)

        if buffer_name == "online_buffer" and buffer_type == "obs_buffer":
            num_workers = 0
        elif buffer_type == "traj_ref_buffer":
            num_workers = self.training_dataloader_traj_ref_num_workers
        else:
            num_workers = self.training_dataloader_num_workers

        dataset = _ReplayBufferDataset(
            target_buffer,
            buffer_type=buffer_type,
            include_sample_ids=(
                buffer_name == "online_buffer"
                and self.use_action_buffer
                and not self.use_hdf5_dataset
            ),
        )
        collate_fn = self.collate_fn
        if buffer_type == "traj_ref_buffer":
            collate_fn = _TrajRefBatchCollator(target_buffer)

        weighted_sampler = self._build_weighted_sampler(dataset)
        loader_kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "drop_last": True,
            "num_workers": num_workers,
            "pin_memory": self.training_dataloader_pin_memory,
            "collate_fn": collate_fn,
        }
        if weighted_sampler is None:
            loader_kwargs["shuffle"] = True
        else:
            loader_kwargs["sampler"] = weighted_sampler
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = self.persistent_workers
            loader_kwargs["prefetch_factor"] = self.training_dataloader_prefetch_factor
        return DataLoader(**loader_kwargs)

    def _get_training_buffer_version(self, buffer):
        if hasattr(buffer, "get_version"):
            try:
                return int(buffer.get_version())
            except Exception:
                return None
        version = getattr(buffer, "_version", None)
        if version is None:
            return None
        try:
            return int(version)
        except Exception:
            return None

    def _close_training_loader_state(self, state):
        if state is None:
            return

        try:
            iterator = state.get("iterator", None)
            if iterator is not None:
                del iterator
        except Exception:
            pass

        try:
            loader = state.get("loader", None)
            if loader is not None and hasattr(loader, "close"):
                loader.close()
        except Exception:
            pass

        state["iterator"] = None
        state["loader"] = None

    def close_training_batch_loaders(self):
        for _, state in list(self._training_batch_loader_state.items()):
            self._close_training_loader_state(state)
        self._training_batch_loader_state.clear()

    def close(self):
        self.close_training_batch_loaders()

    def next_training_batch_from_loader(self, buffer_name, batch_size):
        target_buffer = self._get_training_buffer(buffer_name)

        current_length = target_buffer.length()
        current_version = self._get_training_buffer_version(target_buffer)
        if current_length < batch_size:
            raise ValueError(
                f"{buffer_name} holds {current_length} elements, "
                f"but batch_size={batch_size} was requested."
            )

        loader_key = (buffer_name, batch_size)
        state = self._training_batch_loader_state.get(loader_key)
        if state is None:
            loader = self._build_training_batch_loader(
                buffer_name=buffer_name,
                batch_size=batch_size,
            )
            state = {
                "loader": loader,
                "iterator": iter(loader),
                "buffer_length": current_length,
                "buffer_version": current_version,
            }
            self._training_batch_loader_state[loader_key] = state
        elif (
            state["buffer_length"] != current_length
            or state.get("buffer_version") != current_version
        ):
            self._close_training_loader_state(state)
            loader = self._build_training_batch_loader(
                buffer_name=buffer_name,
                batch_size=batch_size,
            )
            state = {
                "loader": loader,
                "iterator": iter(loader),
                "buffer_length": current_length,
                "buffer_version": current_version,
            }
            self._training_batch_loader_state[loader_key] = state

        try:
            return next(state["iterator"])
        except StopIteration:
            state["iterator"] = iter(state["loader"])
            state["buffer_length"] = current_length
            state["buffer_version"] = current_version
            return next(state["iterator"])

    def collate_training_samples(self, samples, sample_ids=None, buffer=None):
        target_buffer = self.online_buffer if buffer is None else buffer
        buffer_type = self.infer_replay_buffer_type(target_buffer)
        normalized_samples = []
        for idx, sample in enumerate(samples):
            current_sample_id = None if sample_ids is None else sample_ids[idx]
            normalized_samples.append(
                self._normalize_training_sample(
                    sample,
                    buffer=target_buffer,
                    buffer_type=buffer_type,
                    sample_id=current_sample_id,
                )
            )
        return self.collate_fn(normalized_samples)

    def inject_sample_into_collated_batch(
        self,
        batch,
        sample,
        sample_id=None,
        index=-1,
        buffer=None,
    ):
        if not isinstance(batch, dict):
            raise TypeError("Batch injection requires a collated DataLoader batch.")

        replacement = self.collate_training_samples(
            [sample],
            sample_ids=[sample_id] if sample_id is not None else None,
            buffer=self.online_buffer if buffer is None else buffer,
        )
        target_index = index if index >= 0 else batch["action_preferred"].shape[0] + index

        for key in batch["nobs"].keys():
            batch["nobs"][key][target_index] = replacement["nobs"][key][0]
        batch["action_preferred"][target_index] = replacement["action_preferred"][0]
        batch["action_negative"][target_index] = replacement["action_negative"][0]

        if batch.get("should_save") is not None and replacement.get("should_save") is not None:
            batch["should_save"][target_index] = replacement["should_save"][0]
        if batch.get("sample_ids") is not None and replacement.get("sample_ids") is not None:
            batch["sample_ids"][target_index] = replacement["sample_ids"][0]
        return batch

    def sample_intervention_batch(self, batch_size):
        if self.use_hdf5_dataset:
            batch = self.online_buffer.sample(batch_size=batch_size)
            if self.collate_hdf5_intervention_samples:
                batch = self.collate_training_samples(
                    batch,
                    buffer=self.online_buffer,
                )
            return batch, None

        if self.intervention_replay_buffer_type == "pickle_obs_action_buffer":
            if self.use_action_buffer:
                return self.online_buffer.sample(
                    batch_size=batch_size,
                    return_sample_ids=True,
                )
            return self.online_buffer.sample(batch_size=batch_size), None

        batch = self.next_training_batch_from_loader(
            buffer_name="online_buffer",
            batch_size=batch_size,
        )
        sample_ids = None
        if isinstance(batch, dict) and batch.get("sample_ids") is not None:
            sample_ids = batch["sample_ids"].tolist()
        return batch, sample_ids

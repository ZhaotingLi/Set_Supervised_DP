import logging
from dataclasses import dataclass
import inspect
from typing import Optional

from agents.replay_batch_loader import ReplayBatchLoader, replay_training_collate_fn
from tools.buffer import Buffer_uniform_refer_Traj_hdf5, Buffer_uniform_sampling

logger = logging.getLogger(__name__)


@dataclass
class ReplayBufferSetup:
    buffer: object
    buffer_non_intervention: Optional[object] = None
    replay_batch_loader: Optional[ReplayBatchLoader] = None
    buffer_selfplay: Optional[object] = None
    buffer_demo: Optional[object] = None
    intervention_replay_buffer_type: str = "pickle_obs_action_buffer"
    non_intervention_replay_buffer_type: str = "pickle_obs_action_buffer"
    use_hdf5_dataset: bool = False
    use_traj_ref_buffer_for_intervention_buffer: bool = False
    use_traj_ref_buffer: bool = False
    use_action_buffer: bool = False


def _read_replay_buffer_type(config_agent, field_name, default_value):
    value = getattr(config_agent, field_name, None)
    if value is None:
        value = default_value
    value = str(value).strip().lower()
    valid_values = (
        "hdf5_obs_action_buffer",
        "traj_ref_buffer",
        "pickle_obs_action_buffer",
    )
    if value not in valid_values:
        raise ValueError(
            f"{field_name} must be one of {list(valid_values)}, got {value!r}."
        )
    return value


def _hdf5_field_spec(shape_meta, n_obs_steps, horizon, action_dim):
    field_shapes = {}
    dtype_map = {}
    for name, meta in shape_meta.get("obs", {}).items():
        dims = meta.get("shape", [])
        field_shapes[name] = (n_obs_steps, *tuple(dims))
        if len(dims) >= 3 and ("image" in name.lower()):
            dtype_map[name] = "uint8"
        else:
            dtype_map[name] = "float32"

    if horizon == 1:
        field_shapes["robot_action"] = (action_dim,)
        field_shapes["teacher_action"] = (action_dim,)
    else:
        field_shapes["robot_action"] = (horizon, action_dim)
        field_shapes["teacher_action"] = (horizon, action_dim)
    return field_shapes, dtype_map


def _build_hdf5_obs_action_buffer(
    *,
    config_agent,
    shape_meta,
    n_obs_steps,
    horizon,
    action_dim,
    buffer_min_size,
    buffer_max_size,
):
    from tools.buffer import HDF5Buffer

    field_shapes, dtype_map = _hdf5_field_spec(
        shape_meta,
        n_obs_steps,
        horizon,
        action_dim,
    )
    logger.debug('field_shapes:  %s  dtype_map:  %s', field_shapes, dtype_map)

    buffer_dataset_path = getattr(config_agent, "buffer_dataset_path", None)
    if buffer_dataset_path is None or isinstance(buffer_dataset_path, str):
        return (
            HDF5Buffer(
                filename="buffer.h5",
                field_shapes=field_shapes,
                min_size=buffer_min_size,
                max_size=buffer_max_size,
                dtype_map=dtype_map,
                image_saved_in_Uint8=True,
            ),
            field_shapes,
            dtype_map,
        )

    from tools.buffer import hdf5buffer_fromList

    return (
        hdf5buffer_fromList(
            list_of_paths=buffer_dataset_path,
            field_shapes=field_shapes,
            min_size=32,
            sampling="proportional",
            dtype_map=dtype_map,
            compression="lzf",
            image_saved_in_Uint8=True,
        ),
        field_shapes,
        dtype_map,
    )


def _build_traj_ref_buffer(config_agent, buffer_min_size, buffer_max_size):
    kwargs = {
        "min_size": buffer_min_size,
        "max_size": buffer_max_size,
        "traj_hdf5_path": config_agent.traj_buffer_file_name + ".hdf5",
        "recency_bias_alpha": getattr(config_agent, "traj_ref_recency_bias_alpha", 0.0),
        "sampling_strategy": getattr(config_agent, "traj_ref_sampling_strategy", None),
        "traj_storage_backend": getattr(config_agent, "traj_ref_storage_backend", "hdf5"),
        "traj_zarr_path": getattr(config_agent, "traj_ref_zarr_path", None),
        "traj_zarr_force_rebuild": getattr(config_agent, "traj_ref_zarr_force_rebuild", False),
        "traj_obs_combine_previous": getattr(config_agent, "traj_obs_combine_previous", False),
    }
    supported_kwargs = inspect.signature(Buffer_uniform_refer_Traj_hdf5).parameters
    kwargs = {key: value for key, value in kwargs.items() if key in supported_kwargs}
    return Buffer_uniform_refer_Traj_hdf5(**kwargs)


def _build_selfplay_buffer(
    *,
    selfplay_replay_buffer_type,
    field_shapes,
    dtype_map,
    buffer_min_size,
    buffer_max_size,
):
    if selfplay_replay_buffer_type == "pickle_obs_action_buffer":
        return Buffer_uniform_sampling(
            min_size=buffer_min_size,
            max_size=buffer_max_size,
        )
    if selfplay_replay_buffer_type == "hdf5_obs_action_buffer":
        from tools.buffer import HDF5Buffer

        return HDF5Buffer(
            filename="buffer_selfplay.h5",
            field_shapes=field_shapes,
            min_size=buffer_min_size,
            max_size=buffer_max_size,
            dtype_map=dtype_map,
            image_saved_in_Uint8=True,
        )
    raise ValueError(
        "selfplay_replay_buffer_type must be 'hdf5_obs_action_buffer' or "
        f"'pickle_obs_action_buffer', got {selfplay_replay_buffer_type!r}."
    )


def _build_demo_buffer(field_shapes, dtype_map, buffer_min_size, buffer_max_size):
    from tools.buffer import HDF5Buffer

    return HDF5Buffer(
        filename="buffer_demo.h5",
        field_shapes=field_shapes,
        min_size=buffer_min_size,
        max_size=buffer_max_size,
        dtype_map=dtype_map,
        image_saved_in_Uint8=True,
    )


def build_replay_buffer_setup(
    *,
    config_agent,
    shape_meta,
    horizon,
    action_dim,
    n_obs_steps,
    buffer_min_size,
    buffer_max_size,
    use_hdf5_dataset=False,
    use_action_buffer=False,
    use_autoencoder_loss=False,
    create_non_intervention_buffer=False,
    default_intervention_replay_buffer_type="pickle_obs_action_buffer",
    default_non_intervention_replay_buffer_type="pickle_obs_action_buffer",
    allow_traj_ref_intervention=False,
    selfplay_replay_buffer_type="hdf5_obs_action_buffer",
    create_demo_buffer=False,
    create_replay_batch_loader=False,
    replay_batch_loader_kwargs=None,
):
    intervention_default = (
        "hdf5_obs_action_buffer"
        if use_hdf5_dataset
        else default_intervention_replay_buffer_type
    )
    if (
        not use_hdf5_dataset
        and bool(
            getattr(
                config_agent,
                "use_traj_ref_buffer_for_intervention_buffer",
                getattr(config_agent, "use_traj_ref_buffer_intervention_data", False),
            )
        )
    ):
        intervention_default = "traj_ref_buffer"

    non_intervention_default = default_non_intervention_replay_buffer_type
    if bool(getattr(config_agent, "use_traj_ref_buffer", False)):
        non_intervention_default = "traj_ref_buffer"

    intervention_replay_buffer_type = _read_replay_buffer_type(
        config_agent,
        "intervention_replay_buffer_type",
        intervention_default,
    )
    non_intervention_replay_buffer_type = _read_replay_buffer_type(
        config_agent,
        "non_intervention_replay_buffer_type",
        non_intervention_default,
    )

    if (
        intervention_replay_buffer_type == "traj_ref_buffer"
        and not allow_traj_ref_intervention
    ):
        raise NotImplementedError(
            "intervention_replay_buffer_type='traj_ref_buffer' requires a "
            "training loop that can dereference trajectory observations."
        )
    if (
        intervention_replay_buffer_type == "traj_ref_buffer"
        and not bool(getattr(config_agent, "offline_training", False))
    ):
        raise ValueError(
            "intervention_replay_buffer_type='traj_ref_buffer' is only supported "
            "for offline training. For online learning, use "
            "'pickle_obs_action_buffer' for intervention data and reserve "
            "'traj_ref_buffer' for non-intervention data."
        )
    if (
        create_non_intervention_buffer
        and non_intervention_replay_buffer_type == "hdf5_obs_action_buffer"
    ):
        raise NotImplementedError(
            "non_intervention_replay_buffer_type='hdf5_obs_action_buffer' is not "
            "implemented yet because non-intervention replay samples carry "
            "should_save/radius_ratio metadata used by the DataLoader."
        )

    field_shapes = None
    dtype_map = None
    if intervention_replay_buffer_type == "hdf5_obs_action_buffer":
        buffer, field_shapes, dtype_map = _build_hdf5_obs_action_buffer(
            config_agent=config_agent,
            shape_meta=shape_meta,
            n_obs_steps=n_obs_steps,
            horizon=horizon,
            action_dim=action_dim,
            buffer_min_size=buffer_min_size,
            buffer_max_size=buffer_max_size,
        )
    elif intervention_replay_buffer_type == "traj_ref_buffer":
        buffer = _build_traj_ref_buffer(
            config_agent,
            buffer_min_size,
            buffer_max_size,
        )
    else:
        buffer = Buffer_uniform_sampling(
            min_size=buffer_min_size,
            max_size=buffer_max_size,
        )

    buffer_non_intervention = None
    if create_non_intervention_buffer:
        if non_intervention_replay_buffer_type == "traj_ref_buffer":
            buffer_non_intervention = _build_traj_ref_buffer(
                config_agent,
                buffer_min_size,
                buffer_max_size,
            )
        else:
            buffer_non_intervention = Buffer_uniform_sampling(
                min_size=buffer_min_size,
                max_size=buffer_max_size,
            )

    buffer_selfplay = None
    if use_autoencoder_loss:
        effective_selfplay_replay_buffer_type = selfplay_replay_buffer_type
        if intervention_replay_buffer_type != "hdf5_obs_action_buffer":
            effective_selfplay_replay_buffer_type = "pickle_obs_action_buffer"
        buffer_selfplay = _build_selfplay_buffer(
            selfplay_replay_buffer_type=effective_selfplay_replay_buffer_type,
            field_shapes=field_shapes,
            dtype_map=dtype_map,
            buffer_min_size=buffer_min_size,
            buffer_max_size=buffer_max_size,
        )

    buffer_demo = None
    if create_demo_buffer and intervention_replay_buffer_type == "hdf5_obs_action_buffer":
        buffer_demo = _build_demo_buffer(
            field_shapes,
            dtype_map,
            buffer_min_size,
            buffer_max_size,
        )

    if use_action_buffer and intervention_replay_buffer_type != "pickle_obs_action_buffer":
        logger.info("use_action_buffer only supports "
            "intervention_replay_buffer_type='pickle_obs_action_buffer'. "
            "Disable it for the configured intervention replay buffer.")
        use_action_buffer = False

    replay_batch_loader = None
    if create_replay_batch_loader:
        replay_batch_loader_kwargs = replay_batch_loader_kwargs or {}
        replay_batch_loader = ReplayBatchLoader(
            online_buffer=buffer,
            reference_buffer=buffer_non_intervention,
            use_hdf5_dataset=intervention_replay_buffer_type == "hdf5_obs_action_buffer",
            intervention_replay_buffer_type=intervention_replay_buffer_type,
            use_action_buffer=use_action_buffer,
            collate_fn=replay_training_collate_fn,
            **replay_batch_loader_kwargs,
        )

    return ReplayBufferSetup(
        buffer=buffer,
        buffer_non_intervention=buffer_non_intervention,
        replay_batch_loader=replay_batch_loader,
        buffer_selfplay=buffer_selfplay,
        buffer_demo=buffer_demo,
        intervention_replay_buffer_type=intervention_replay_buffer_type,
        non_intervention_replay_buffer_type=non_intervention_replay_buffer_type,
        use_hdf5_dataset=intervention_replay_buffer_type == "hdf5_obs_action_buffer",
        use_traj_ref_buffer_for_intervention_buffer=intervention_replay_buffer_type == "traj_ref_buffer",
        use_traj_ref_buffer=non_intervention_replay_buffer_type == "traj_ref_buffer",
        use_action_buffer=use_action_buffer,
    )

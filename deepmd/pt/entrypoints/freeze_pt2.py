# SPDX-License-Identifier: LGPL-3.0-or-later
"""DPA4 / SeZM → AOTInductor ``.pt2`` freeze path for the pt backend.

SeZM relies on a nested ``autograd.grad(create_graph=True)`` inside
``fit_output_to_model_output``; TorchScript cannot represent that
graph, so DPA4 / SeZM checkpoints are routed through AOTInductor instead.
The output archive layout follows the ``pt_expt`` convention, including the
metadata consumed by ``DeepPotPTExpt.cc`` and ``DeepSpinPTExpt.cc``.

Tracing runs on CPU (``make_fx`` with ``_allow_non_fake_inputs=True``
is brittle on CUDA because the proxy-tensor dispatcher does not set
up CUDA streams for the captured parameters).  The compiled package
is moved to the target device via ``move_to_device_pass`` before
``aoti_compile_and_package``.

``.pt2`` I/O is always float64, matching the C++ contract in
``DeepPotPTExpt::compute`` where LAMMPS coordinates are unconditionally
cast to ``torch::kFloat64``.  SeZM's own ``_input_type_cast`` bridges
fp64 inputs to whatever internal compute dtype the checkpoint uses.
"""

from __future__ import (
    annotations,
)

import ctypes
import json
import logging
import os
import tempfile
import zipfile
from copy import (
    deepcopy,
)
from typing import (
    Any,
)

import numpy as np
import torch

from deepmd.dpmodel.utils.nlist import (
    build_neighbor_list,
    extend_coord_with_ghosts,
)
from deepmd.dpmodel.utils.region import (
    normalize_coord,
)
from deepmd.kernels.utils import (
    cuda_compute_capability,
    gpu_capability_description,
    triton_infer_level,
)
from deepmd.pt.model.descriptor.sezm_nn.so2 import (
    SO2Convolution,
    SO2Linear,
)
from deepmd.pt.model.model import (
    get_model,
)
from deepmd.pt.train.wrapper import (
    ModelWrapper,
)
from deepmd.pt.utils.compile_compat import (
    build_inductor_compile_options,
)
from deepmd.pt.utils.env import (
    DEVICE,
)
from deepmd.pt_expt.utils.edge_schema import (
    edge_schema_from_extended,
)
from deepmd.utils.model_branch_dict import (
    get_model_dict,
)

log = logging.getLogger(__name__)

# Fixed nloc used for sample inputs during .pth freeze tracing.
_PTH_SAMPLE_NLOC = 7


def _model_has_spin(model: torch.nn.Module) -> bool:
    """Return whether ``model`` uses the spin lower interface."""
    has_spin = getattr(model, "has_spin", False)
    return bool(has_spin() if callable(has_spin) else has_spin)


def _get_model_ntypes(model: torch.nn.Module) -> int:
    """Return atom type count even when the exported type map is empty."""
    type_map = list(model.get_type_map())
    if type_map:
        return len(type_map)
    descriptor = model.get_descriptor()
    return int(descriptor.get_ntypes())


def _model_has_message_passing(model: torch.nn.Module) -> bool:
    """Return whether the regular .pt2 graph requires a real atom mapping."""
    for obj in (
        model,
        getattr(model, "atomic_model", None),
        model.get_descriptor() if hasattr(model, "get_descriptor") else None,
    ):
        if obj is None or not hasattr(obj, "has_message_passing"):
            continue
        try:
            return bool(obj.has_message_passing())
        except (AttributeError, NotImplementedError):
            continue
    return False


def _strip_shape_assertions(graph_module: torch.nn.Module) -> None:
    """Remove deferred shape assertions from SeZM export graphs.

    SeZM lower inputs intentionally keep extended-atom and local-atom axes
    independent: regular exports pass ghost coordinates through ``coord`` while
    ``atype`` remains local-only, and spin exports slice both ``nall`` and
    ``nloc`` after virtual atom expansion. ``torch.export`` may turn these valid
    dynamic cases into deferred ``Ne(nall, nloc)`` assertions.
    """
    graph = graph_module.graph
    for node in list(graph.nodes):
        if (
            node.op == "call_function"
            and node.target is torch.ops.aten._assert_scalar.default
        ):
            graph.erase_node(node)
    graph.eliminate_dead_code()
    graph_module.recompile()


def _log_dpa4_citation() -> None:
    """Log the DPA-4 paper citation BibTeX."""
    log.info(
        "Thank you for using the DPA4/SeZM model! If it benefits your "
        "research, please cite the DPA4 paper "
        "(https://arxiv.org/abs/2606.02419):"
    )
    log.info(
        "\n"
        "@article{li2026dpa4,\n"
        "  title = {{DPA4}: Pushing the Accuracy-Cost Frontier of Interatomic "
        "Potentials with {EMFA} {SO(2)} Convolution},\n"
        "  author = {Li, Tiancheng and Li, Wentao and Peng, Anyang and "
        "Xue, Jianming and Zhang, Linfeng and Zhang, Duo and Wang, Han},\n"
        "  journal = {arXiv preprint arXiv:2606.02419},\n"
        "  year = {2026},\n"
        "  eprint = {2606.02419},\n"
        "  archivePrefix = {arXiv},\n"
        "  primaryClass = {physics.chem-ph},\n"
        "  doi = {10.48550/arXiv.2606.02419},\n"
        "  url = {https://arxiv.org/abs/2606.02419}\n"
        "}"
    )


def _build_edge_schema_ts(
    fnlist: torch.Tensor,
    extended_coord: torch.Tensor,
    nloc: int,
    nnei: int,
) -> tuple[torch.Tensor, ...]:
    """Build edge schema tensors from a LAMMPS-format neighbour list.

    This function is callable from within a TorchScript-traced
    ``forward_lower`` (unlike ``edge_schema_from_extended``, which returns a
    namedtuple that the tracer cannot follow across module boundaries).  The
    logic must stay in sync with ``edge_schema_from_extended`` in
    ``deepmd.pt_expt.utils.edge_schema``.

    Returns a tuple ``(edge_index, edge_vec, edge_scatter_index, edge_mask,
    nf)``.  The caller is responsible for extracting ``edge_src`` / ``edge_dst``
    or ``dst_coord`` as needed from the index tensors.
    """
    nf = fnlist.shape[0]
    neighbor_flat = fnlist.reshape(-1)
    dst_actual = (
        torch.arange(neighbor_flat.shape[0], device=fnlist.device, dtype=torch.long)
        // nnei
    )
    valid_flat = neighbor_flat >= 0
    neighbor_safe = torch.where(
        valid_flat, neighbor_flat, torch.zeros_like(neighbor_flat)
    )
    neighbor_safe_2d = neighbor_safe.to(dtype=torch.long).view(nf, nloc * nnei)
    # Gather neighbour coordinates.
    neighbor_coord = torch.gather(
        extended_coord,
        1,
        neighbor_safe_2d.unsqueeze(-1).expand(-1, -1, 3),
    ).reshape(-1, 3)
    # Gather destination (central atom) coordinates.
    dst_local = dst_actual % nloc
    dst_coord = torch.gather(
        extended_coord,
        1,
        dst_local.view(nf, nloc * nnei).unsqueeze(-1).expand(-1, -1, 3),
    ).reshape(-1, 3)
    edge_vec = neighbor_coord - dst_coord
    edge_index = torch.stack([neighbor_safe, dst_actual], dim=0)
    edge_scatter_index = edge_index
    edge_mask = valid_flat
    return (edge_index, edge_vec, edge_scatter_index, edge_mask, nf)


def _validate_edge_schema_sync(
    ext_coord: torch.Tensor,
    ext_atype_local: torch.Tensor,
    formatted_nlist: torch.Tensor,
    mapping: torch.Tensor,
) -> None:
    """Verify that ``_build_edge_schema_ts`` and ``edge_schema_from_extended``
    produce identical tensors for the same inputs.

    The two functions must stay in sync; this function is called from
    ``freeze_sezm_to_pth``.  A shape-only comparison runs **unconditionally**
    on every freeze (it is cheap).  Full tensor equality is gated by the
    ``DP_FREEZE_VALIDATE_EDGE_SCHEMA`` environment variable because it requires
    materialising both edge schemas in memory.  Raises ``AssertionError`` on
    mismatch.
    """
    validate_values = os.environ.get(
        "DP_FREEZE_VALIDATE_EDGE_SCHEMA", ""
    ).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    try:
        ref_schema = edge_schema_from_extended(
            ext_coord, ext_atype_local, formatted_nlist, mapping
        )
    except Exception as exc:
        log.warning(
            "Edge-schema sync check: edge_schema_from_extended raised %s; "
            "skipping validation.",
            exc,
        )
        return

    # Compute via _build_edge_schema_ts.
    nloc = int(ext_atype_local.shape[1])
    nnei = int(formatted_nlist.shape[2])
    ts_result = _build_edge_schema_ts(formatted_nlist, ext_coord, nloc, nnei)
    ts_edge_index, ts_edge_vec, ts_edge_scatter_index, ts_edge_mask, ts_nf = ts_result

    # --- unconditional shape check ---
    shape_errors: list[str] = []
    for name, ts_val, ref_val in (
        ("edge_index", ts_edge_index, ref_schema.edge_index),
        ("edge_vec", ts_edge_vec, ref_schema.edge_vec),
        (
            "edge_scatter_index",
            ts_edge_scatter_index,
            ref_schema.edge_scatter_index,
        ),
        ("edge_mask", ts_edge_mask, ref_schema.edge_mask),
    ):
        if ts_val.shape != ref_val.shape:
            shape_errors.append(
                f"  {name}: shape mismatch TS={tuple(ts_val.shape)} "
                f"ref={tuple(ref_val.shape)}"
            )

    if shape_errors:
        msg = (
            "Edge-schema sync check FAILED (shape): _build_edge_schema_ts and "
            "edge_schema_from_extended diverged!\n"
            + "\n".join(shape_errors)
            + "\nThis is a bug -- the .pth freeze may produce incorrect results. "
            "Please report it."
        )
        log.error(msg)
        raise AssertionError(msg)

    # --- opt-in value check ---
    if not validate_values:
        log.info(
            "Edge-schema shape check passed: _build_edge_schema_ts shapes match "
            "edge_schema_from_extended for nloc=%d, nnei=%d.  "
            "Set DP_FREEZE_VALIDATE_EDGE_SCHEMA=1 for a full value comparison.",
            nloc,
            nnei,
        )
        return

    value_errors: list[str] = []
    for name, ts_val, ref_val in (
        ("edge_index", ts_edge_index, ref_schema.edge_index),
        ("edge_vec", ts_edge_vec, ref_schema.edge_vec),
        (
            "edge_scatter_index",
            ts_edge_scatter_index,
            ref_schema.edge_scatter_index,
        ),
        ("edge_mask", ts_edge_mask, ref_schema.edge_mask),
    ):
        if not bool(torch.equal(ts_val, ref_val.to(device=ts_val.device))):
            max_diff = float(
                (ts_val.float() - ref_val.float().to(ts_val.device)).abs().max()
            )
            value_errors.append(
                f"  {name}: values differ (max abs diff={max_diff:.2e})"
            )

    if value_errors:
        msg = (
            "Edge-schema sync check FAILED (values): _build_edge_schema_ts and "
            "edge_schema_from_extended diverged!\n"
            + "\n".join(value_errors)
            + "\nThis is a bug -- the .pth freeze may produce incorrect results. "
            "Please report it."
        )
        log.error(msg)
        raise AssertionError(msg)
    log.info(
        "Edge-schema sync check passed (shapes + values): "
        "_build_edge_schema_ts matches edge_schema_from_extended "
        "for nloc=%d, nnei=%d.",
        nloc,
        nnei,
    )


def _extract_state_and_params(
    ckpt: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Unwrap a ``torch.load`` result into ``(state_dict, model_params)``.

    Accepts both the training-wrapper layout (weights under a top-level
    ``"model"`` key) and a bare state dict.
    """
    inner = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(inner, dict):
        raise ValueError("Unsupported checkpoint: expected a dict-like state dict.")
    extra = inner.get("_extra_state") or {}
    params = extra.get("model_params")
    if not isinstance(params, dict):
        raise ValueError("Unsupported checkpoint: missing '_extra_state.model_params'.")
    return inner, params


def is_sezm_checkpoint(ckpt_path: str) -> bool:
    """Best-effort detection used by the CLI to route DPA4 / SeZM checkpoints.

    Returns ``False`` for unreadable files or non-SeZM checkpoints; no
    exception leaks out so the caller can treat this as a pure routing
    signal.
    """
    try:
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    try:
        _, params = _extract_state_and_params(raw)
    except ValueError:
        return False
    if "model_dict" in params:
        return any(
            str(branch_params.get("type", "")).lower() in ("sezm", "dpa4")
            for branch_params in params["model_dict"].values()
        )
    return str(params.get("type", "")).lower() in ("sezm", "dpa4")


def _select_model_head(
    state_dict: dict[str, Any],
    params: dict[str, Any],
    head: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract a single selected model branch from a checkpoint."""
    if "model_dict" not in params:
        if head is not None:
            raise NotImplementedError(
                "SeZM .pt2 freeze does not yet support head selection for single-task checkpoints; pass head=None."
            )
        return state_dict, params

    model_alias_dict, _ = get_model_dict(params["model_dict"])
    model_keys = list(params["model_dict"])
    if head is None and "Default" in model_alias_dict:
        head = "Default"
        log.info(
            "Using default head %s for multitask SeZM freeze.", model_alias_dict[head]
        )
    if head is None:
        raise ValueError(
            "Head must be set for multitask SeZM/DPA4 freeze. "
            f"Available heads are: {model_keys}."
        )
    if head not in model_alias_dict:
        head_lower = head.lower()
        for key in model_alias_dict:
            if key.lower() == head_lower:
                head = key
                break
    if head not in model_alias_dict:
        raise ValueError(
            f"No head or alias named {head!r} in model. Available heads are: {model_keys}."
        )

    branch = model_alias_dict[head]
    branch_params = deepcopy(params["model_dict"][branch])
    branch_state: dict[str, Any] = {
        "_extra_state": deepcopy(state_dict.get("_extra_state", {})),
    }
    branch_state["_extra_state"]["model_params"] = branch_params
    prefix = f"model.{branch}."
    for key, value in state_dict.items():
        if key.startswith(prefix):
            branch_state[key.replace(prefix, "model.Default.")] = value
    return branch_state, branch_params


def _load_sezm_checkpoint(
    ckpt_path: str,
    head: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], torch.nn.Module]:
    """Load a SeZM/DPA4 checkpoint and return (state_dict, params, model).

    Shared by both ``freeze_sezm_to_pt2`` and ``freeze_sezm_to_pth`` so that
    checkpoint loading and head selection logic stays in one place.
    """
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict, params = _extract_state_and_params(raw)
    state_dict, params = _select_model_head(state_dict, params, head)

    model_type = str(params.get("type", "")).lower()
    if model_type not in ("sezm", "dpa4"):
        raise ValueError(
            f"Expected a SeZM/DPA4 checkpoint, got type={params.get('type')!r}."
        )
    model = get_model(params)
    ModelWrapper(model).load_state_dict(state_dict)
    model.eval()
    model.to("cpu")
    return state_dict, params, model


def _to_py_list(value: Any) -> Any:
    """Coerce torch / numpy scalars into JSON-friendly Python values."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, (int, float, bool, str)):
        return value
    raise TypeError(f"Cannot JSON-serialize value of type {type(value)!r}")


def _collect_metadata(
    model: torch.nn.Module,
    output_keys: list[str],
    is_spin: bool | None = None,
    do_atomic_virial: bool = False,
    has_comm_artifact: bool = False,
) -> dict:
    """Assemble the flat metadata dict expected by :class:`DeepPotPTExpt`.

    Mirrors the reader contract at ``source/api_cc/src/DeepPotPTExpt.cc`` and
    the metadata-only load path in ``deepmd.pt_expt.infer.deep_eval.DeepEval``:
    every field consumed by C++ LAMMPS inference **and** every field
    consumed by ``DeepEval._init_from_metadata`` must be present here.

    ``output_keys`` is the insertion order that the loader zips with
    ``AOTIModelPackageLoader::run``'s flat output vector.
    """
    if is_spin is None:
        is_spin = _model_has_spin(model)
    fitting_output_defs: list[dict[str, Any]] = []
    for vdef in model.atomic_output_def().get_data().values():
        fitting_output_defs.append(
            {
                "name": vdef.name,
                "shape": list(vdef.shape),
                "reducible": vdef.reducible,
                "r_differentiable": vdef.r_differentiable,
                "c_differentiable": vdef.c_differentiable,
                "atomic": vdef.atomic,
                # OutputVariableCategory is an IntEnum; force plain int for
                # deterministic JSON serialisation across Python versions.
                "category": int(vdef.category),
                "r_hessian": vdef.r_hessian,
                "magnetic": bool(vdef.magnetic or (is_spin and vdef.name == "energy")),
                "intensive": vdef.intensive,
            }
        )
    exports_atomic_virial = True if not is_spin else bool(do_atomic_virial)
    metadata = {
        "type_map": list(model.get_type_map()),
        "ntypes": _get_model_ntypes(model),
        "rcut": float(model.get_rcut()),
        "sel": [int(s) for s in model.get_sel()],
        "lower_input_kind": model.export_lower_input_kind(),
        "dim_fparam": int(model.get_dim_fparam()),
        "dim_aparam": int(model.get_dim_aparam()),
        "dim_chg_spin": int(model.get_dim_chg_spin()),
        "mixed_types": bool(model.mixed_types()),
        "has_message_passing": _model_has_message_passing(model),
        "has_comm_artifact": bool(has_comm_artifact),
        "do_atomic_virial": exports_atomic_virial,
        "nnei": int(sum(model.get_sel())),
        "has_default_fparam": bool(model.has_default_fparam()),
        "default_fparam": _to_py_list(model.get_default_fparam()),
        "default_chg_spin": _to_py_list(model.get_default_chg_spin()),
        "output_keys": list(output_keys),
        "fitting_output_defs": fitting_output_defs,
        # sel_type feeds DeepEval.get_sel_type() in metadata-only mode.
        # SeZM energy models return [] (every type selected).
        "sel_type": [int(t) for t in model.get_sel_type()],
        "is_spin": bool(is_spin),
    }
    if is_spin:
        metadata["ntypes_spin"] = int(model.spin.get_ntypes_spin())
        metadata["use_spin"] = [bool(v) for v in model.spin.use_spin]
    return metadata


def _tune_triton_configs(model: torch.nn.Module, target_device: torch.device) -> None:
    """Tune the shape-keyed Triton launch tables for this checkpoint's shapes.

    At ``DP_TRITON_INFER >= 2`` the traced graph bakes launch configurations
    resolved from the tables in ``deepmd.kernels.triton.sezm.tile_configs``.  Shape keys
    absent from the built-in tables (an untuned GPU model, or an untuned
    width/degree) are swept here on the local GPU -- the exact hardware the
    ``.pt2`` will run on, since AOTInductor artifacts are not portable across
    GPU models -- and registered for the current process before tracing.
    Keys already covered cost nothing.

    The fused value-path entries are then rebound: the mixing-stack operator
    selection (fp32 versus fp16x3) is fixed at construction time, which
    predates the registrations made here.
    """
    if triton_infer_level() < 2:
        return
    if target_device.type != "cuda" or not torch.cuda.is_available():
        return
    from deepmd.kernels.triton.sezm.so2_value_path import (
        SO2_VALUE_PATH_TRITON_AVAILABLE,
        make_triton_value_path,
    )

    if not SO2_VALUE_PATH_TRITON_AVAILABLE:
        return
    from deepmd.kernels.triton.sezm.sweep_tile_configs import (
        collect_model_shape_keys,
        tune_missing_configs,
    )
    from deepmd.kernels.triton.sezm.tile_configs import (
        _builtin_tables,
    )

    # The built-in tables and the sweep both resolve against the current
    # device; pin it to the AOTI target so a freeze aimed at a secondary GPU
    # tunes and looks up the right hardware (mixed-model hosts).
    if target_device.index is not None:
        torch.cuda.set_device(target_device)
        _builtin_tables.cache_clear()

    shape_keys = collect_model_shape_keys(model)
    registered = tune_missing_configs(
        shape_keys, level=triton_infer_level(), device=target_device
    )
    if registered:
        log.info(
            "Registered freshly tuned Triton launch configurations: %s",
            {family: sorted(entries) for family, entries in registered.items()},
        )
    else:
        log.info(
            "Triton launch tables already cover this checkpoint's shapes on %s; "
            "no tuning needed.",
            torch.cuda.get_device_name(target_device),
        )
    # Rebind unconditionally: the fp32-versus-fp16x3 stack selection was made
    # at construction time, possibly against a different current device's
    # tables, and must reflect the target device and any fresh registrations.
    for module in model.modules():
        if isinstance(module, SO2Convolution) and module.triton_infer_level >= 2:
            module._triton_value_path = make_triton_value_path(module)


# The trace-time sendlist for the with-comm artifact embeds the address of a
# numpy array (``int**`` contract of ``border_op``). The array must outlive the
# trace + export call; the exported graph never reads it at runtime (the op is
# opaque), so a module-level keepalive is sufficient.
_TRACE_SENDLIST_KEEPALIVE: list[np.ndarray] = []


def _build_sample_extended(
    model: torch.nn.Module,
    nframes: int,
    nloc: int,
    device: torch.device,
    has_spin: bool,
) -> tuple[torch.Tensor | None, ...]:
    """Build the extended-region sample tensors shared by the lower builders.

    Returns ``(ext_coord, ext_atype, nlist, mapping, ext_spin, fparam, aparam,
    charge_spin)``; tensors are float64 / int64 (matching the ``.pt2`` I/O
    contract). ``ext_spin`` is ``None`` unless ``has_spin``.
    """
    rcut = float(model.get_rcut())
    sel = list(model.get_sel())
    ntypes = len(model.get_type_map())
    if ntypes == 0:
        ntypes = int(model.get_descriptor().get_ntypes())
    if ntypes <= 0:
        raise ValueError("SeZM .pt2 freeze requires at least one atom type.")
    dim_fparam = int(model.get_dim_fparam())
    dim_aparam = int(model.get_dim_aparam())
    dim_chg_spin = int(model.get_dim_chg_spin())
    mixed_types = bool(model.mixed_types())

    box_size = rcut * 3.0
    box = np.eye(3, dtype=np.float64) * box_size
    box_np = box.reshape(1, 9)

    rng = np.random.default_rng(42)
    coord_np = rng.random((nframes, nloc, 3), dtype=np.float64) * box_size * 0.5
    coord_np += box_size * 0.25  # centre roughly in the middle of the cell

    atype_np = np.zeros((nframes, nloc), dtype=np.int32)
    for i in range(nloc):
        atype_np[:, i] = i % ntypes
    spin_np = np.zeros((nframes, nloc, 3), dtype=np.float64)
    if has_spin:
        atom_idx = np.arange(nloc, dtype=np.float64).reshape(1, nloc)
        spin_np[:, :, 0] = 0.10 + 0.01 * atom_idx
        spin_np[:, :, 1] = 0.20 + 0.02 * atom_idx
        spin_np[:, :, 2] = 0.05

    coord_normalized = normalize_coord(
        coord_np.reshape(nframes, nloc, 3),
        np.tile(box.reshape(1, 3, 3), (nframes, 1, 1)),
    )
    extended_coord, extended_atype, mapping = extend_coord_with_ghosts(
        coord_normalized, atype_np, np.tile(box_np, (nframes, 1)), rcut
    )
    nlist = build_neighbor_list(
        extended_coord,
        extended_atype,
        nloc,
        rcut,
        sel,
        distinguish_types=not mixed_types,
    )
    extended_coord = extended_coord.reshape(nframes, -1, 3)

    ext_coord = torch.tensor(extended_coord, dtype=torch.float64, device=device)
    ext_atype = torch.tensor(extended_atype, dtype=torch.int64, device=device)
    nlist_t = torch.tensor(nlist, dtype=torch.int64, device=device)
    mapping_t = torch.tensor(mapping, dtype=torch.int64, device=device)
    ext_spin = None
    if has_spin:
        extended_spin = np.take_along_axis(spin_np, mapping[..., None], axis=1)
        ext_spin = torch.tensor(extended_spin, dtype=torch.float64, device=device)
    fparam = (
        torch.zeros(nframes, dim_fparam, dtype=torch.float64, device=device)
        if dim_fparam > 0
        else None
    )
    aparam = (
        torch.zeros(nframes, nloc, dim_aparam, dtype=torch.float64, device=device)
        if dim_aparam > 0
        else None
    )
    charge_spin = (
        torch.zeros(nframes, dim_chg_spin, dtype=torch.float64, device=device)
        if dim_chg_spin > 0
        else None
    )
    return (
        ext_coord,
        ext_atype,
        nlist_t,
        mapping_t,
        ext_spin,
        fparam,
        aparam,
        charge_spin,
    )


def _make_sample_inputs(
    model: torch.nn.Module,
    nframes: int,
    nloc: int,
    device: torch.device,
    has_spin: bool = False,
) -> tuple[torch.Tensor | None, ...]:
    """Build representative ``forward_common_lower`` inputs for tracing.

    Three lower ABIs are produced, selected by ``model.export_lower_input_kind()``
    and whether the model carries spin:

    - virtual spin (``nlist``): the DeepSpin extended-input signature, since the
      graph expands virtual atoms internally;
    - native spin (``edge_vec``): the energy edge schema plus the owned-atom
      spins (the first ``nloc`` extended rows, where ``mapping`` is identity);
    - energy (``edge_vec``): the plain single-domain edge schema.
    """
    (
        ext_coord,
        ext_atype,
        nlist_t,
        mapping_t,
        ext_spin,
        fparam,
        aparam,
        charge_spin,
    ) = _build_sample_extended(model, nframes, nloc, device, has_spin)
    if has_spin and model.export_lower_input_kind() == "nlist":
        return (
            ext_coord,
            ext_atype,
            ext_spin,
            nlist_t,
            mapping_t,
            fparam,
            aparam,
            charge_spin,
        )
    formatted_nlist: torch.Tensor = model.format_nlist(ext_coord, ext_atype, nlist_t)
    edge_schema = edge_schema_from_extended(
        ext_coord,
        ext_atype[:, :nloc],
        formatted_nlist,
        mapping_t,
    )
    if has_spin:
        return (
            edge_schema.coord,
            edge_schema.atype,
            edge_schema.edge_index,
            edge_schema.edge_vec,
            edge_schema.edge_scatter_index,
            edge_schema.edge_mask,
            ext_spin[:, :nloc],
            fparam,
            aparam,
            charge_spin,
        )
    return (
        edge_schema.coord,
        edge_schema.atype,
        edge_schema.edge_index,
        edge_schema.edge_vec,
        edge_schema.edge_scatter_index,
        edge_schema.edge_mask,
        fparam,
        aparam,
        charge_spin,
    )


def _make_edge_comm_tensors(
    mapping: torch.Tensor,
    nloc: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    """Build a single self-send swap so the with-comm trace runs ``border_op``.

    A LAMMPS run supplies the real per-swap communication plan at inference time;
    the trace only needs valid in-range indices so the eager output-key probe can
    execute the opaque op. Ghost slot ``k`` copies its owner's local index
    ``mapping[nloc + k]``.
    """
    nall = int(mapping.shape[1])
    nghost = nall - nloc
    send_count = max(1, nghost)
    owner = mapping[0, nloc:nall].to(dtype=torch.int32).cpu().numpy()
    indices = np.ascontiguousarray(np.resize(owner, send_count).astype(np.int32))
    _TRACE_SENDLIST_KEEPALIVE.append(indices)
    addr = indices.ctypes.data_as(ctypes.c_void_p).value
    return (
        torch.tensor([addr], dtype=torch.int64, device=device),  # send_list (int**)
        torch.zeros(1, dtype=torch.int32, device=device),  # send_proc (self)
        torch.zeros(1, dtype=torch.int32, device=device),  # recv_proc (self)
        torch.tensor([send_count], dtype=torch.int32, device=device),  # send_num
        torch.tensor([send_count], dtype=torch.int32, device=device),  # recv_num
        torch.zeros(1, dtype=torch.int64, device=device),  # communicator
        torch.tensor(nloc, dtype=torch.int32, device=device),  # nlocal
        torch.tensor(nghost, dtype=torch.int32, device=device),  # nghost
    )


def _make_comm_sample_inputs(
    model: torch.nn.Module,
    nloc: int,
    device: torch.device,
) -> tuple[torch.Tensor | None, ...]:
    """Build with-comm edge inputs for tracing the parallel ``.pt2`` artifact.

    The parallel path indexes the extended node set directly, so ``edge_index``
    coincides with ``edge_scatter_index`` (both extended) and ghost features are
    refreshed via ``border_op`` rather than gathered through a folded mapping.
    The frame axis is fixed at one, matching LAMMPS single-frame inference. The
    native spin scheme threads the EXTENDED per-node spin (ghost spins ride the
    same exchange), inserted after ``edge_mask`` to match its with-comm signature.
    """
    has_spin = _model_has_spin(model)
    (
        ext_coord,
        ext_atype,
        nlist_t,
        mapping_t,
        ext_spin,
        fparam,
        aparam,
        charge_spin,
    ) = _build_sample_extended(
        model, nframes=1, nloc=nloc, device=device, has_spin=has_spin
    )
    formatted_nlist: torch.Tensor = model.format_nlist(ext_coord, ext_atype, nlist_t)
    edge_schema = edge_schema_from_extended(
        ext_coord,
        ext_atype[:, :nloc],
        formatted_nlist,
        mapping_t,
    )
    edge_inputs = (
        edge_schema.coord,  # (1, nall, 3)
        edge_schema.atype,  # (1, nloc)
        ext_atype,  # (1, nall)
        edge_schema.edge_scatter_index,  # edge_index: extended (2, E)
        edge_schema.edge_vec,
        edge_schema.edge_scatter_index,  # edge_scatter_index: extended (2, E)
        edge_schema.edge_mask,
    )
    comm_tensors = _make_edge_comm_tensors(mapping_t, nloc, device)
    if has_spin:
        return (*edge_inputs, ext_spin, fparam, aparam, charge_spin, *comm_tensors)
    return (*edge_inputs, fparam, aparam, charge_spin, *comm_tensors)


def _resolve_nframes(
    model: torch.nn.Module,
    nloc: int,
    device: torch.device,
    start: int = 2,
    has_spin: bool = False,
) -> tuple[int, tuple[torch.Tensor | None, ...]]:
    """Pick an ``nframes`` that does not collide with any other dim size.

    ``torch.export``'s duck-sizing unifies symbolic dims whose concrete
    sample values match; if ``nframes`` happens to equal, say, the
    spatial ``3`` or the virial ``9``, the ExportedProgram rejects
    later calls whose ``nframes`` differs.  Bumping ``nframes`` until
    no collision is left keeps the export safe.
    """
    nframes = start
    sample = _make_sample_inputs(
        model,
        nframes=nframes,
        nloc=nloc,
        device=device,
        has_spin=has_spin,
    )
    other_dims: set[int] = set()
    for t in sample:
        if t is not None:
            other_dims.update(t.shape[1:])
    while nframes in other_dims:
        nframes += 1
    if nframes != start:
        sample = _make_sample_inputs(
            model,
            nframes=nframes,
            nloc=nloc,
            device=device,
            has_spin=has_spin,
        )
    return nframes, sample


def _build_dynamic_shapes(
    sample_inputs: tuple[torch.Tensor | None, ...],
) -> tuple:
    """Build positional dynamic-shape constraints for the traced lower input.

    The lower ABI is recovered from the sample structure: a floating-point
    tensor at index 2 is the extended spin of the deepspin-scheme nlist contract,
    while an integer ``edge_index`` there marks the edge contract. A native-spin
    edge sample carries the extra per-local-atom spin tensor, giving it ten
    positional entries against the energy contract's nine.
    """
    nframes_dim = torch.export.Dim("nframes", min=1)
    nloc_dim = torch.export.Dim("nloc", min=1)
    nedge_dim = torch.export.Dim("nedge", min=2)
    is_nlist_spin = (
        len(sample_inputs) >= 3
        and sample_inputs[2] is not None
        and sample_inputs[2].is_floating_point()
    )
    if is_nlist_spin:
        nall_dim = torch.export.Dim("nall", min=4)
        fparam = sample_inputs[5]
        aparam = sample_inputs[6]
        charge_spin = sample_inputs[7] if len(sample_inputs) == 8 else None
        shapes = (
            {0: nframes_dim, 1: nall_dim},  # extended_coord
            {0: nframes_dim, 1: nall_dim},  # extended_atype
            {0: nframes_dim, 1: nall_dim},  # extended_spin
            {0: nframes_dim, 1: nloc_dim},  # nlist
            {0: nframes_dim, 1: nall_dim},  # mapping
            {0: nframes_dim} if fparam is not None else None,
            {0: nframes_dim, 1: nloc_dim} if aparam is not None else None,
        )
        if len(sample_inputs) == 8:
            shapes = (*shapes, {0: nframes_dim} if charge_spin is not None else None)
        return shapes

    nall_dim = torch.export.Dim("nall", min=1)
    edge_shapes = (
        {0: nframes_dim, 1: nall_dim},  # extended_coord: (nframes, nall, 3)
        {0: nframes_dim, 1: nloc_dim},  # atype
        {1: nedge_dim},  # edge_index
        {0: nedge_dim},  # edge_vec
        {1: nedge_dim},  # edge_scatter_index
        {0: nedge_dim},  # edge_mask
    )
    # Native-spin edge contract: extra per-local-atom spin leaf at index 6.
    is_native_spin = len(sample_inputs) == 10
    if is_native_spin:
        fparam, aparam, charge_spin = (
            sample_inputs[7],
            sample_inputs[8],
            sample_inputs[9],
        )
        return (
            *edge_shapes,
            {0: nframes_dim, 1: nloc_dim},  # spin: (nframes, nloc, 3)
            {0: nframes_dim} if fparam is not None else None,
            {0: nframes_dim, 1: nloc_dim} if aparam is not None else None,
            {0: nframes_dim} if charge_spin is not None else None,
        )
    fparam = sample_inputs[6]
    aparam = sample_inputs[7]
    charge_spin = sample_inputs[8] if len(sample_inputs) == 9 else None
    shapes = (
        *edge_shapes,
        {0: nframes_dim} if fparam is not None else None,
        {0: nframes_dim, 1: nloc_dim} if aparam is not None else None,
    )
    if len(sample_inputs) == 9:
        shapes = (*shapes, {0: nframes_dim} if charge_spin is not None else None)
    return shapes


def _build_with_comm_dynamic_shapes(
    sample_inputs: tuple[torch.Tensor | None, ...],
) -> tuple:
    """Build dynamic-shape constraints for the parallel with-comm lower input.

    The frame axis is fixed at one (LAMMPS single-frame inference), so only
    ``nall``, ``nloc`` and ``nedge`` vary. The eight communication tensors are
    static: ``nswap`` is fixed at LAMMPS init and the graph carries no variation
    across its value (``border_op`` is opaque to the exported program). The
    native spin contract inserts the extended (nall) spin after ``edge_mask``,
    giving 19 positional entries against the energy contract's 18.
    """
    nall_dim = torch.export.Dim("nall", min=1)
    nloc_dim = torch.export.Dim("nloc", min=1)
    nedge_dim = torch.export.Dim("nedge", min=2)
    edge_base = (
        {1: nall_dim},  # coord: (1, nall, 3)
        {1: nloc_dim},  # atype: (1, nloc)
        {1: nall_dim},  # extended_atype: (1, nall)
        {1: nedge_dim},  # edge_index: (2, nedge)
        {0: nedge_dim},  # edge_vec: (nedge, 3)
        {1: nedge_dim},  # edge_scatter_index: (2, nedge)
        {0: nedge_dim},  # edge_mask: (nedge,)
    )
    is_native_spin = len(sample_inputs) == 19
    if is_native_spin:
        fparam, aparam, charge_spin = (
            sample_inputs[8],
            sample_inputs[9],
            sample_inputs[10],
        )
        base = (
            *edge_base,
            {1: nall_dim},  # spin: (1, nall, 3)
            None if fparam is None else {},  # fparam: (1, ndf) static
            None if aparam is None else {1: nloc_dim},  # aparam: (1, nloc, nda)
            None if charge_spin is None else {},  # charge_spin: (1, nchg) static
        )
        return (*base, *((None,) * 8))
    fparam = sample_inputs[7]
    aparam = sample_inputs[8]
    charge_spin = sample_inputs[9]
    base = (
        *edge_base,
        None if fparam is None else {},  # fparam: (1, ndf) static
        None if aparam is None else {1: nloc_dim},  # aparam: (1, nloc, nda)
        None if charge_spin is None else {},  # charge_spin: (1, nchg) static
    )
    return (*base, *((None,) * 8))


def _export_with_comm_artifact(
    model: torch.nn.Module,
    *,
    target_device: torch.device,
    compile_options: dict[str, Any],
) -> bytes:
    """Trace, export and compile the parallel with-comm ``.pt2`` artifact.

    The artifact mirrors the regular edge graph but exchanges ghost node
    features across ranks via ``border_op``. Returns the compiled package bytes
    for nesting under ``model/extra/forward_lower_with_comm.pt2``; tracing runs
    on CPU and the package is moved to ``target_device`` before compilation.
    """
    from torch._inductor import (
        aoti_compile_and_package,
    )
    from torch._inductor import config as inductor_config

    sample_inputs = _make_comm_sample_inputs(model, nloc=7, device=torch.device("cpu"))
    traced = model.forward_common_lower_exportable_with_comm(*sample_inputs)
    exported = torch.export.export(
        traced,
        sample_inputs,
        dynamic_shapes=_build_with_comm_dynamic_shapes(sample_inputs),
        strict=False,
        prefer_deferred_runtime_asserts_over_guards=True,
    )
    _strip_shape_assertions(exported.graph_module)
    if target_device.type != "cpu":
        from torch.export.passes import (
            move_to_device_pass,
        )

        exported = move_to_device_pass(exported, target_device)
    with tempfile.TemporaryDirectory() as td:
        wc_path = os.path.join(td, "forward_lower_with_comm.pt2")
        with inductor_config.patch({**compile_options, "triton.max_tiles": 1}):
            aoti_compile_and_package(exported, package_path=wc_path)
        with open(wc_path, "rb") as fh:
            return fh.read()


def freeze_sezm_to_pt2(
    ckpt_path: str,
    out_path: str,
    *,
    device: torch.device | None = None,
    head: str | None = None,
    atomic_virial: bool = True,
) -> None:
    """Freeze a SeZM checkpoint into an AOTInductor ``.pt2`` archive.

    Parameters
    ----------
    ckpt_path
        Path to the SeZM training checkpoint (``.pt``).
    out_path
        Destination file.  A ``.pt2`` suffix is expected.
    device
        Target device for the compiled shared library.  Defaults to
        :data:`DEVICE`.  Tracing itself always runs on CPU.
    head
        Model head to export from a multi-task checkpoint. If omitted, the
        ``Default`` head is used when present; otherwise multi-task checkpoints
        must pass an explicit head. Single-task checkpoints must pass ``None``.
    atomic_virial
        Whether the exported model exposes per-atom virial.  Enabled by
        default: the edge-force scatter assembles the per-atom virial as a
        free by-product of the single backward, so exporting it carries no
        compute cost.
    """
    from torch._inductor import (
        aoti_compile_and_package,
    )
    from torch._inductor import config as inductor_config

    target_device = device if device is not None else DEVICE

    # AOTInductor lowers the SeZM graph through Triton *on CUDA targets*. Triton
    # (shipped with PyTorch 2.11/2.12) supports Volta (sm_70) and newer but
    # cannot compile for Pascal (sm_60/sm_62) -- AOTInductor is confirmed broken
    # there. Fail fast on a Pascal CUDA target with an actionable message instead
    # of crashing inside the Triton compiler several minutes into the freeze.
    # Volta/Turing/Ampere+ CUDA targets are unaffected. CPU targets are also
    # unaffected: AOTInductor uses the Inductor cpp backend there (no Triton),
    # and ``freeze_sezm_to_pt2`` already special-cases ``type == "cpu"``. The
    # SeZM `.pt` checkpoint is already ASE / `dp --pt test` loadable in pure
    # eager mode, so no freeze is needed for Python inference; LAMMPS DPA4/SeZM
    # requires a Volta-or-newer GPU. `DP_FREEZE_FORCE_AOTI=1` is an escape hatch
    # for a custom Triton build that does support a Pascal device.
    force_aoti = os.environ.get("DP_FREEZE_FORCE_AOTI", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if target_device.type == "cuda" and not force_aoti:
        cap = cuda_compute_capability(target_device)
        if cap is None:
            # The capability query failed (out-of-range device index or a
            # driver issue) -- distinct from a genuine Pascal device. Fail
            # fast with a precise message instead of a misleading "Pascal" one.
            raise RuntimeError(
                f"Cannot freeze DPA4/SeZM to .pt2 for CUDA device "
                f"{target_device}: its compute capability could not be "
                f"queried (the device index may be out of range or the CUDA "
                f"driver is unavailable). AOTInductor needs a CUDA device it "
                f"can target. For ASE / `dp --pt test` inference, load the "
                f"`.pt` checkpoint directly (no freeze needed)."
            )
        if cap[0] < 7:
            raise RuntimeError(
                f"Cannot freeze DPA4/SeZM to .pt2 on a Pascal CUDA GPU "
                f"({gpu_capability_description(target_device)}): the AOTInductor "
                f"freeze lowers through Triton, which cannot compile for Pascal "
                f"(sm_60); AOTInductor is confirmed broken there. Volta (sm_70) "
                f"and newer are supported. "
                f"Use freeze_sezm_to_pth() or `dp --pt freeze --legacy-gpu` to "
                f"freeze to a TorchScript .pth file instead -- the .pth format "
                f"works on Pascal/P100 GPUs with no Triton dependency. "
                f"For ASE / `dp --pt test` inference, "
                f"load the `.pt` checkpoint directly -- no freeze is needed, and "
                f"the dense eager path runs on any CUDA device PyTorch itself "
                f"supports. Set DP_FREEZE_FORCE_AOTI=1 to attempt the AOTInductor "
                f"freeze anyway (only if you know your Triton build supports this "
                f"device)."
            )

    _state_dict, params, model = _load_sezm_checkpoint(ckpt_path, head)
    is_spin = _model_has_spin(model)

    # The SO(2) linear mixer selects its block-diagonal vs dense matmul from a
    # Python device branch that make_fx resolves at trace time. Since tracing
    # always runs on CPU, pin the choice to the AOTI target device: non-CPU
    # targets bake the block-diagonal contraction (which skips the structural
    # off-|m| zeros); CPU targets keep the dense einsum that dodges the Inductor
    # AVX2 codegen bug.
    force_block_diag = target_device.type != "cpu"
    for module in model.modules():
        if isinstance(module, SO2Linear):
            module._force_block_diag_matmul = force_block_diag

    # Sweep any Triton launch-table keys this checkpoint needs that are not
    # covered for the local GPU, so the traced graph bakes tuned launches.
    _tune_triton_configs(model, target_device)

    _, sample_inputs_cpu = _resolve_nframes(
        model,
        nloc=7,
        device=torch.device("cpu"),
        has_spin=is_spin,
    )

    # Each model's exportable signature matches its sample tuple positionally
    # (energy / native-spin edge ABI, or virtual-spin nlist ABI), so a single
    # splat covers all three contracts.
    log.info("Tracing the lower graph on CPU (make_fx)...")
    traced = model.forward_common_lower_exportable(*sample_inputs_cpu)

    # Output key order is taken from a concrete run; Python dict order
    # is stable and matches what DeepPotPTExpt::extract_outputs zips
    # against AOTIModelPackageLoader::run's output vector.
    with torch.no_grad():
        sample_out = traced(*sample_inputs_cpu)
    output_keys = list(sample_out.keys())

    log.info("Exporting the traced graph (torch.export)...")
    exported = torch.export.export(
        traced,
        sample_inputs_cpu,
        dynamic_shapes=_build_dynamic_shapes(sample_inputs_cpu),
        strict=False,
        prefer_deferred_runtime_asserts_over_guards=True,
    )
    _strip_shape_assertions(exported.graph_module)

    # move_to_device_pass handles FakeTensor device propagation cleanly;
    # a naive .to(device) on the exported program does not.
    if target_device.type != "cpu":
        from torch.export.passes import (
            move_to_device_pass,
        )

        exported = move_to_device_pass(exported, target_device)

    out_path_str = str(out_path)
    compile_options = build_inductor_compile_options(inference=True)
    # Keep AOTInductor aligned with the eval compile path.  ``triton.max_tiles=1``
    # keeps data-dependent edge axes on Triton's x grid, whose bound is large
    # enough for production-scale neighbor lists.
    log.info(
        "Compiling the AOTInductor package for %s (the slowest freeze stage; "
        "typically several minutes)...",
        target_device,
    )
    with inductor_config.patch({**compile_options, "triton.max_tiles": 1}):
        aoti_compile_and_package(exported, package_path=out_path_str)

    # Second artifact: the LAMMPS multi-rank with-comm graph. It threads the
    # eight border_op communication tensors so cross-rank ghost features are
    # exchanged between interaction blocks. Gated on the edge_vec lower contract
    # (energy and native spin), so virtual spin (nlist interface) is excluded;
    # bridging models report supports_edge_parallel()=False (Source Freeze
    # Propagation is not rank-decomposable). Both fall back to single-rank.
    with_comm = (
        model.export_lower_input_kind() == "edge_vec" and model.supports_edge_parallel()
    )
    with_comm_bytes: bytes | None = None
    if with_comm:
        log.info(
            "Compiling the parallel with-comm artifact (second AOTInductor "
            "compilation)..."
        )
        with_comm_bytes = _export_with_comm_artifact(
            model,
            target_device=target_device,
            compile_options=compile_options,
        )

    metadata = _collect_metadata(
        model,
        output_keys=output_keys,
        is_spin=is_spin,
        do_atomic_virial=atomic_virial,
        has_comm_artifact=with_comm,
    )
    with zipfile.ZipFile(out_path_str, "a") as zf:
        zf.writestr("model/extra/metadata.json", json.dumps(metadata))
        if with_comm_bytes is not None:
            zf.writestr("model/extra/forward_lower_with_comm.pt2", with_comm_bytes)
        # The raw training params are preserved so `dp change-bias` and
        # other downstream tooling can recover the exact training config.
        # ``default=str`` is a safety net for exotic nested values.
        zf.writestr(
            "model/extra/model_def_script.json",
            json.dumps(params, default=str),
        )

    log.info(
        "Saved SeZM .pt2 to %s (device=%s, output_keys=%s)",
        out_path_str,
        target_device,
        output_keys,
    )
    _log_dpa4_citation()


def _format_nlist_to_nnei(nlist: torch.Tensor, nnei_target: int) -> torch.Tensor:
    """Pad or truncate a LAMMPS neighbour list to exactly ``nnei`` columns.

    Shared by the ``forward_lower`` methods of both ``SeZMPTHModel`` and
    ``SeZMSpinPTHModel`` to ensure the traced lower graph receives a
    consistent nnei dimension regardless of the actual LAMMPS neighbour
    count.
    """
    nf, nloc_val, nsel_in = nlist.shape
    if nsel_in < nnei_target:
        pad = torch.full(
            (nf, nloc_val, nnei_target - nsel_in),
            -1,
            dtype=nlist.dtype,
            device=nlist.device,
        )
        fnlist = torch.cat([nlist, pad], dim=2)
    elif nsel_in > nnei_target:
        # Keep nearest nnei_target neighbours (first nnei_target columns).
        fnlist = nlist[:, :, :nnei_target]
    else:
        fnlist = nlist
    return fnlist.contiguous()


class _BaseSeZMPTHModel(torch.nn.Module):
    """Shared base for TorchScript-compatible SeZM frozen .pth model wrappers.

    Stores the traced lower graph and all LAMMPS-facing metadata, and
    exposes the public accessor methods expected by ``DeepPotPT`` /
    ``DeepSpinPT``.  Concrete subclasses provide the appropriate
    ``forward_lower`` contract for their respective LAMMPS interface.

    Parameters
    ----------
    lower_graph
        The ``make_fx``-traced lower graph.
    metadata
        All model metadata values needed by the public API and
        ``forward_lower`` post-processing.
    """

    def __init__(
        self,
        lower_graph: torch.nn.Module,
        *,
        sel: list[int],
        rcut: float,
        ntypes: int,
        type_map: list[str],
        nnei: int,
        nsel: int,
        dim_fparam: int,
        dim_aparam: int,
        dim_chg_spin: int,
        has_mp: bool,
        min_nbor_dist: float | None,
        model_def_json: str,
        model_output_types: list[str],
        has_default_fparam: bool,
        default_fparam: list[float] | None,
        default_chg_spin: list[float] | None,
        mixed_types: bool,
        is_spin: bool,
        ntypes_spin: int,
        use_spin: list[bool],
        lower_input_kind: str,
        lower_nf: int,
        do_grad_r: bool,
        do_grad_c: bool,
    ) -> None:
        super().__init__()
        self.lower_graph = lower_graph
        self._rcut = rcut
        self._ntypes = ntypes
        self._sel = sel
        self._type_map = type_map
        self._nnei = nnei
        self._nsel = nsel
        self._dim_fparam = dim_fparam
        self._dim_aparam = dim_aparam
        self._dim_chg_spin = dim_chg_spin
        self._has_mp = has_mp
        self._min_nbor_dist = min_nbor_dist
        self._model_def_json = model_def_json
        self._model_output_types = model_output_types
        self._has_default_fparam = has_default_fparam
        self._default_fparam = default_fparam
        self._default_chg_spin = default_chg_spin
        self._mixed_types = mixed_types
        self._is_spin = is_spin
        self._ntypes_spin = ntypes_spin
        self._use_spin = use_spin
        self._lower_input_kind = lower_input_kind
        self._lower_nf = lower_nf
        self._do_grad_r = do_grad_r
        self._do_grad_c = do_grad_c

    @torch.jit.export
    def get_rcut(self) -> float:
        return self._rcut

    @torch.jit.export
    def get_ntypes(self) -> int:
        return self._ntypes

    @torch.jit.export
    def get_sel(self) -> list[int]:
        return self._sel

    @torch.jit.export
    def get_type_map(self) -> list[str]:
        return self._type_map

    @torch.jit.export
    def get_nnei(self) -> int:
        return self._nnei

    @torch.jit.export
    def get_nsel(self) -> int:
        return self._nsel

    @torch.jit.export
    def get_dim_fparam(self) -> int:
        return self._dim_fparam

    @torch.jit.export
    def get_dim_aparam(self) -> int:
        return self._dim_aparam

    @torch.jit.export
    def is_aparam_nall(self) -> bool:
        return False

    @torch.jit.export
    def has_message_passing(self) -> bool:
        return self._has_mp

    @torch.jit.export
    def get_dim_chg_spin(self) -> int:
        return self._dim_chg_spin

    @torch.jit.export
    def get_min_nbor_dist(self) -> float | None:
        return self._min_nbor_dist

    @torch.jit.export
    def get_model_def_script(self) -> str:
        return self._model_def_json

    @torch.jit.export
    def model_output_type(self) -> list[str]:
        return self._model_output_types

    @torch.jit.export
    def has_default_fparam(self) -> bool:
        return self._has_default_fparam

    @torch.jit.export
    def get_default_fparam(self) -> torch.Tensor | None:
        if self._default_fparam is None:
            return None
        return torch.tensor(self._default_fparam, dtype=torch.float64)

    @torch.jit.export
    def get_default_chg_spin(self) -> torch.Tensor | None:
        if self._default_chg_spin is None:
            return None
        return torch.tensor(self._default_chg_spin, dtype=torch.float64)

    @torch.jit.export
    def mixed_types(self) -> bool:
        return self._mixed_types

    @torch.jit.export
    def has_spin(self) -> bool:
        return self._is_spin

    @torch.jit.export
    def get_ntypes_spin(self) -> int:
        return self._ntypes_spin

    @torch.jit.export
    def get_use_spin(self) -> list[bool]:
        return self._use_spin


class SeZMPTHModel(_BaseSeZMPTHModel):
    """TorchScript-compatible wrapper for a non-spin SeZM frozen .pth model.

    Exposes ``forward_lower`` with the standard ``DeepPotPT`` LAMMPS
    contract.  The edge-level compute graph (from ``make_fx``) is stored
    on the parent class and invoked after converting the LAMMPS neighbour
    list to the edge schema.
    """

    @torch.jit.export
    def forward_lower(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
        comm_dict: dict[str, torch.Tensor] | None = None,
        charge_spin: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Standard LAMMPS ``forward_lower`` contract (``DeepPotPT``).

        Converts the LAMMPS neighbour list to the edge schema, runs the
        traced lower graph, and post-processes the output into the format
        expected by ``DeepPotPT``.

        .. note::
           Only the energy (non-spin) edge ABI is supported by this wrapper.
           ``DeepPotPT.forward_lower`` does not supply ``extended_spin``.
           Spin models are routed to ``SeZMSpinPTHModel`` instead, which
           implements the ``DeepSpinPT`` contract.
        """
        # --- Step 1: format nlist (pad/truncate to nnei) ---
        nnei_target = self._nnei
        fnlist = _format_nlist_to_nnei(nlist, nnei_target)
        nf, nloc_val, _ = fnlist.shape

        # --- Step 2: build edge schema via shared helper ---
        (
            edge_index,
            edge_vec,
            edge_scatter_index,
            edge_mask,
            _nf,
        ) = _build_edge_schema_ts(
            fnlist, extended_coord, int(nloc_val), int(nnei_target)
        )
        dst_coord = extended_coord.gather(
            1,
            torch.arange(nloc_val, device=fnlist.device)
            .view(1, nloc_val, 1)
            .expand(_nf, nloc_val, 3),
        )

        # --- Step 3: call the traced lower graph ---
        atype_local = extended_atype[:, :nloc_val].to(dtype=torch.long)
        lower_inputs = (
            dst_coord,
            atype_local,
            edge_index.to(dtype=torch.long),
            edge_vec,
            edge_scatter_index.to(dtype=torch.long),
            edge_mask,
            fparam,
            aparam,
            charge_spin,
        )

        result = self.lower_graph(*lower_inputs)

        # --- Step 4: post-process output ---
        model_predict: dict[str, torch.Tensor] = {}
        model_predict["atom_energy"] = result["energy"]
        model_predict["energy"] = result["energy_redu"]
        if self._do_grad_r:
            model_predict["extended_force"] = result["energy_derv_r"].squeeze(-2)
        if self._do_grad_c:
            model_predict["virial"] = result["energy_derv_c_redu"].squeeze(-2)
            if do_atomic_virial:
                model_predict["extended_virial"] = result["energy_derv_c"].squeeze(-2)
        return model_predict


class SeZMSpinPTHModel(_BaseSeZMPTHModel):
    """TorchScript-compatible wrapper for a spin SeZM frozen .pth model.

    Exposes ``forward_lower`` with the ``DeepSpinPT`` LAMMPS contract
    (``extended_spin`` between ``extended_atype`` and ``nlist``) so
    LAMMPS can load and call it via ``pair_style deepmd/spin``.

    Supports the ``nlist`` lower ABI (virtual-spin / deepspin scheme)
    where the traced lower graph places virtual atoms internally from
    the extended inputs and raw neighbour list.  The ``edge_vec`` ABI
    (native spin) is not yet supported by this wrapper.
    """

    @torch.jit.export
    def forward_lower(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        extended_spin: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
        comm_dict: dict[str, torch.Tensor] | None = None,
        charge_spin: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """``DeepSpinPT`` LAMMPS ``forward_lower`` contract.

        The traced lower graph (``nlist`` ABI) places virtual atoms
        internally, so this wrapper formats the LAMMPS neighbour list
        and passes the raw extended inputs straight through.

        .. note::
           ``DeepSpinPT`` does **not** pass ``charge_spin``; the
           parameter is accepted for signature compatibility but is
           always ``None`` at runtime.
        """
        # --- Step 1: format nlist (pad/truncate to nnei) ---
        fnlist = _format_nlist_to_nnei(nlist, self._nnei)

        # --- Step 2: call the traced lower graph (nlist ABI) ---
        lower_inputs = (
            extended_coord,
            extended_atype.to(dtype=torch.long),
            extended_spin,
            fnlist,
            mapping,
            fparam,
            aparam,
            charge_spin,
        )
        result = self.lower_graph(*lower_inputs)

        # --- Step 3: post-process output ---
        model_predict: dict[str, torch.Tensor] = {}
        model_predict["atom_energy"] = result["energy"]
        model_predict["energy"] = result["energy_redu"]
        model_predict["extended_mask_mag"] = result["mask_mag"]
        if self._do_grad_r:
            model_predict["extended_force"] = result["energy_derv_r"].squeeze(-2)
            model_predict["extended_force_mag"] = result["energy_derv_r_mag"].squeeze(
                -2
            )
        if self._do_grad_c:
            model_predict["virial"] = result["energy_derv_c_redu"].squeeze(-2)
            if do_atomic_virial:
                model_predict["extended_virial"] = result["energy_derv_c"].squeeze(-2)
        return model_predict


def freeze_sezm_to_pth(
    ckpt_path: str,
    out_path: str,
    *,
    device: torch.device | None = None,
    head: str | None = None,
    atomic_virial: bool = True,
) -> None:
    """Freeze a SeZM checkpoint into a TorchScript ``.pth`` file.

    This path converts the eager lower graph (traced via ``make_fx``) into a
    TorchScript module that works on **all** CUDA GPUs supported by PyTorch,
    including Pascal / P100 (sm_60).  No Triton or AOTInductor dependency is
    required, making this the freeze path for legacy-GPU LAMMPS deployments.

    Supports both plain energy and spin (virtual-spin / ``nlist`` ABI)
    SeZM/DPA4 models.  Energy models are exported via ``SeZMPTHModel``
    (loaded by ``DeepPotPT`` in LAMMPS); spin models are exported via
    ``SeZMSpinPTHModel`` (loaded by ``DeepSpinPT`` in LAMMPS).

    .. note::
       Only the ``nlist`` lower ABI (virtual-spin scheme, used by SeZM
       spin models) is supported for spin.  The ``edge_vec`` ABI (native
       spin) is not yet supported by this path.

    Parameters
    ----------
    ckpt_path
        Path to the SeZM training checkpoint (``.pt``).
    out_path
        Destination file.  A ``.pth`` suffix is expected.
    device
        Target device.  Defaults to :data:`DEVICE`.  Tracing itself always
        runs on CPU.
    head
        Model head to export from a multi-task checkpoint. If omitted, the
        ``Default`` head is used when present; otherwise multi-task checkpoints
        must pass an explicit head. Single-task checkpoints must pass ``None``.
    atomic_virial
        Whether the exported model exposes per-atom virial.  Enabled by
        default: the edge-force scatter assembles the per-atom virial as a
        free by-product of the single backward pass.

    Environment Variables
    ---------------------
    DP_FREEZE_FORCE_PTH
        Set to ``1`` to force the ``.pth`` freeze path even on Triton-capable
        GPUs (Volta+), suppressing the performance warning.  Useful for
        debugging or heterogeneous GPU clusters where TorchScript portability
        is preferred over peak performance.
    """
    target_device = device if device is not None else DEVICE

    # Check for DP_FREEZE_FORCE_PTH env var override.
    force_pth = os.environ.get("DP_FREEZE_FORCE_PTH", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    # Warn if Triton-capable GPU is being used for .pth freeze (since .pt2
    # would be faster), unless the user explicitly forced .pth.
    if target_device.type == "cuda" and not force_pth:
        cap = cuda_compute_capability(target_device)
        if cap is not None and cap[0] >= 7:
            log.warning(
                "Freezing DPA4/SeZM to .pth (TorchScript) on a Triton-capable "
                "GPU (%s). The .pt2 (AOTInductor) path is recommended for this "
                "hardware as it is typically faster. Set DP_FREEZE_FORCE_PTH=1 "
                "to suppress this warning.",
                gpu_capability_description(target_device),
            )

    _state_dict, params, model = _load_sezm_checkpoint(ckpt_path, head)
    is_spin = _model_has_spin(model)

    # Determine the lower input kind.  Spin models (SeZM) use the "nlist"
    # ABI where virtual atoms are placed inside the traced graph; non-spin
    # models use the "edge_vec" ABI where the wrapper builds the edge
    # schema from the LAMMPS neighbour list.
    try:
        _lower_input_kind = model.export_lower_input_kind()
    except AttributeError as err:
        raise ValueError(
            "This SeZM checkpoint does not expose export_lower_input_kind(). "
            "Please re-save the checkpoint with a newer version of deepmd-kit, "
            "or use the .pt2 freeze path (dp --pt freeze) instead."
        ) from err

    # --- Build sample inputs for the lower (edge-level) graph ---
    _, sample_inputs_cpu = _resolve_nframes(
        model,
        nloc=_PTH_SAMPLE_NLOC,
        device=torch.device("cpu"),
        has_spin=is_spin,
    )

    # --- Trace the lower graph via make_fx ---
    log.info("Tracing the lower graph on CPU (make_fx)...")
    traced = model.forward_common_lower_exportable(*sample_inputs_cpu)

    with torch.no_grad():
        sample_out = traced(*sample_inputs_cpu)
    output_keys = list(sample_out.keys())
    log.info("Lower graph output keys: %s", output_keys)

    # --- Build sample LAMMPS-level inputs for forward_lower tracing ---
    ext_tensors = _build_sample_extended(
        model,
        nframes=1,
        nloc=_PTH_SAMPLE_NLOC,
        device=torch.device("cpu"),
        has_spin=is_spin,
    )
    ext_coord, ext_atype, nlist_t, mapping_t = ext_tensors[:4]

    # --- Pre-compute metadata values (must not reference 'model' in wrapper) ---
    ntypes = _get_model_ntypes(model)
    rcut = float(model.get_rcut())
    sel = [int(s) for s in model.get_sel()]
    dim_fparam = int(model.get_dim_fparam())
    dim_aparam = int(model.get_dim_aparam())
    dim_chg_spin = int(model.get_dim_chg_spin())
    if is_spin and dim_chg_spin > 0:
        raise ValueError(
            "SeZM spin models do not support charge_spin. "
            "dim_chg_spin must be 0 for the .pth freeze path. "
            "Use the .pt2 freeze path (dp --pt freeze without --legacy-gpu) "
            "for models with spin+charge."
        )
    type_map = list(model.get_type_map())
    has_mp = _model_has_message_passing(model)
    nnei = int(sum(sel))
    nsel_val = nnei
    do_grad_r = bool(model.do_grad_r("energy"))
    do_grad_c = bool(model.do_grad_c("energy"))
    mixed_types_val = bool(model.mixed_types())
    if not mixed_types_val:
        raise ValueError(
            "The .pth (TorchScript) freeze path requires mixed_types=True for "
            "SeZM/DPA4 models. This model has mixed_types=False. "
            "Use the .pt2 freeze path (dp --pt freeze without --legacy-gpu) "
            "instead."
        )
    has_default_fparam_val = bool(model.has_default_fparam())
    default_fparam_val = _to_py_list(model.get_default_fparam())
    default_chg_spin_val = _to_py_list(model.get_default_chg_spin())
    min_nbor_dist = model.get_min_nbor_dist()
    min_nbor_dist_val = float(min_nbor_dist) if min_nbor_dist is not None else None
    # Spin metadata (only populated for spin models).
    ntypes_spin_val: int = 0
    use_spin_val: list[bool] = []
    if is_spin and hasattr(model, "spin"):
        try:
            ntypes_spin_val = int(model.spin.get_ntypes_spin())
            use_spin_val = [bool(v) for v in model.spin.use_spin]
        except (AttributeError, NotImplementedError):
            log.debug(
                "Could not extract spin metadata (ntypes_spin / use_spin) "
                "from model.spin.",
                exc_info=True,
            )
    model_output_types: list[str] = []
    try:
        out_def = model.model_output_def()
        for kk, vv in out_def.var_defs.items():
            if int(vv.category) == 0:
                model_output_types.append(kk)
    except Exception:
        log.debug(
            "Could not compute model_output_types from model_output_def(); "
            "using empty list.",
            exc_info=True,
        )
    model_def_json = json.dumps(params, default=str)

    # --- Validate edge-schema sync (gated by env var) ---
    # Only applicable to non-spin models (edge_vec ABI) where the wrapper
    # builds the edge schema from the LAMMPS nlist.  Spin models use the
    # nlist ABI and pass raw inputs straight through.
    if not is_spin:
        try:
            formatted_nlist_sync = model.format_nlist(ext_coord, ext_atype, nlist_t)
            _validate_edge_schema_sync(
                ext_coord,
                ext_atype[:, :_PTH_SAMPLE_NLOC],
                formatted_nlist_sync,
                mapping_t,
            )
        except Exception:
            log.debug(
                "Edge-schema sync check skipped (could not build reference "
                "formatted nlist).",
                exc_info=True,
            )

    # --- Trace / script the wrapper ---
    # Build the appropriate wrapper and LAMMPS-level trace inputs.
    # Spin models → SeZMSpinPTHModel (DeepSpinPT contract: extended_spin
    # between atype and nlist).  Non-spin → SeZMPTHModel (DeepPotPT contract).
    common_kwargs = {
        "sel": sel,
        "rcut": rcut,
        "ntypes": ntypes,
        "type_map": type_map,
        "nnei": nnei,
        "nsel": nsel_val,
        "dim_fparam": dim_fparam,
        "dim_aparam": dim_aparam,
        "dim_chg_spin": dim_chg_spin,
        "has_mp": has_mp,
        "min_nbor_dist": min_nbor_dist_val,
        "model_def_json": model_def_json,
        "model_output_types": model_output_types,
        "has_default_fparam": has_default_fparam_val,
        "default_fparam": default_fparam_val,
        "default_chg_spin": default_chg_spin_val,
        "mixed_types": mixed_types_val,
        "is_spin": is_spin,
        "ntypes_spin": ntypes_spin_val,
        "use_spin": use_spin_val,
        "lower_input_kind": _lower_input_kind,
        "lower_nf": 1,
        "do_grad_r": do_grad_r,
        "do_grad_c": do_grad_c,
    }

    lammps_ext_coord = ext_coord  # (1, nall, 3)
    lammps_ext_atype = ext_atype  # (1, nall)
    lammps_nlist = nlist_t  # (1, nloc, nsel)
    lammps_mapping = mapping_t  # (1, nall)
    lammps_fparam = (
        torch.zeros(1, dim_fparam, dtype=torch.float64) if dim_fparam > 0 else None
    )
    lammps_aparam = (
        torch.zeros(1, _PTH_SAMPLE_NLOC, dim_aparam, dtype=torch.float64)
        if dim_aparam > 0
        else None
    )
    lammps_chg_spin = (
        torch.zeros(1, dim_chg_spin, dtype=torch.float64) if dim_chg_spin > 0 else None
    )

    if is_spin:
        log.info("Converting spin FX graph to TorchScript (torch.jit.trace)...")
        wrapper = SeZMSpinPTHModel(traced, **common_kwargs)
        wrapper.eval()
        # DeepSpinPT passes: coord, atype, spin, nlist, mapping,
        # fparam, aparam, do_atomic_virial[, comm_dict].
        # ext_tensors = (coord, atype, nlist, mapping, spin, fparam, aparam, chg_spin)
        _, _, _, _, lammps_ext_spin, _, _, _ = ext_tensors  # (1, nall, 3)
        trace_inputs = (
            lammps_ext_coord,
            lammps_ext_atype,
            lammps_ext_spin,
            lammps_nlist,
            lammps_mapping,
            lammps_fparam,
            lammps_aparam,
            atomic_virial,
            None,  # comm_dict
        )
    else:
        log.info("Converting FX graph to TorchScript (torch.jit.trace)...")
        wrapper = SeZMPTHModel(traced, **common_kwargs)
        wrapper.eval()
        trace_inputs = (
            lammps_ext_coord,
            lammps_ext_atype,
            lammps_nlist,
            lammps_mapping,
            lammps_fparam,
            lammps_aparam,
            atomic_virial,
            None,  # comm_dict
            lammps_chg_spin,
        )

    try:
        traced_module = torch.jit.trace_module(
            wrapper,
            {"forward_lower": trace_inputs},
        )
        log.info("TorchScript tracing succeeded.")
    except Exception as e:
        log.warning(
            "torch.jit.trace failed: %s. Falling back to torch.jit.script. "
            "The frozen model may have untraced control-flow branches; "
            "verify with `dp --pt test` before deploying to LAMMPS.",
            e,
        )
        try:
            traced_module = torch.jit.script(wrapper)
            log.info("TorchScript scripting succeeded.")
            # Validate that the scripted model produces the same output as
            # the original wrapper on the trace sample inputs.
            with torch.no_grad():
                ref_out = wrapper.forward_lower(*trace_inputs)
                ts_out = traced_module.forward_lower(*trace_inputs)
            mismatches = []
            for key in ref_out:
                if key not in ts_out:
                    mismatches.append(f"  {key}: missing in scripted output")
                    continue
                ref_val = ref_out[key]
                ts_val = ts_out[key]
                if ref_val.shape != ts_val.shape:
                    mismatches.append(
                        f"  {key}: shape mismatch "
                        f"ref={tuple(ref_val.shape)} "
                        f"scripted={tuple(ts_val.shape)}"
                    )
                    continue
                max_diff = float((ref_val.float() - ts_val.float()).abs().max())
                if max_diff > 1e-5:
                    mismatches.append(f"  {key}: max abs diff={max_diff:.2e}")
            if mismatches:
                log.error(
                    "TorchScript fallback output validation FAILED:\n%s\n"
                    "The scripted model produces different results than the "
                    "original. The .pth model may be incorrect -- do not "
                    "deploy to LAMMPS without verifying with `dp --pt test`.",
                    "\n".join(mismatches),
                )
            else:
                log.info(
                    "TorchScript fallback output validation passed "
                    "(all %d keys match within 1e-5).",
                    len(ref_out),
                )
        except Exception as e2:
            raise RuntimeError(
                f"Failed to convert SeZM model to TorchScript: {e2}. "
                f"The model may use operations not supported by TorchScript. "
                f"Try using freeze_sezm_to_pt2() for AOTInductor export instead."
            ) from e2

    # --- Move to target device ---
    if target_device.type != "cpu":
        traced_module.to(target_device)

    # --- Save ---
    out_path_str = str(out_path)
    torch.jit.save(traced_module, out_path_str)
    log.info(
        "Saved SeZM .pth to %s (device=%s, output_keys=%s)",
        out_path_str,
        target_device,
        output_keys,
    )
    loader_name = "DeepSpinPT" if is_spin else "DeepPotPT"
    log.info(
        "The .pth model can be loaded by LAMMPS via %s on any CUDA GPU "
        "(Pascal P100 / sm_60+ supported, no Triton required).",
        loader_name,
    )
    _log_dpa4_citation()


__all__ = [
    "SeZMPTHModel",
    "SeZMSpinPTHModel",
    "freeze_sezm_to_pt2",
    "freeze_sezm_to_pth",
    "is_sezm_checkpoint",
]

r"""Validation helpers for the two-stage ``.pth`` freeze pipeline.

These functions validate the correctness of the TorchScript conversion
at each stage of the pipeline introduced for the legacy-GPU (Pascal/P100)
compatibility fix.  They are designed to be run manually in a
PyTorch-enabled environment with a real DPA4/SeZM checkpoint.

All functions follow ``pytest`` conventions and raise ``AssertionError``
with descriptive messages on failure.

Usage (from the repository root)::

    # Standalone lower-graph validation (requires a SeZM checkpoint):
    pytest source/tests/pt/test_freeze_pth_validate.py::test_validate_lower_graph \\
        --ckpt examples/water/se_e2_a/model.pt -v

    # Full wrapper validation:
    pytest source/tests/pt/test_freeze_pth_validate.py::test_validate_full_wrapper \\
        --ckpt examples/water/se_e2_a/model.pt -v

    # P100 runtime validation (requires a frozen .pth and a P100 GPU):
    pytest source/tests/pt/test_freeze_pth_validate.py::test_validate_p100_runtime \\
        --model /path/to/frozen.pth -v
"""

from __future__ import annotations

# Validation scripts use print for diagnostic output — this is intentional.
# ruff: noqa: T201

import json
import os
import tempfile
from typing import (
    Any,
)

import pytest
import torch

# ---------------------------------------------------------------------------
# Test fixtures and utilities
# ---------------------------------------------------------------------------


def _require_ckpt_path(request: pytest.FixtureRequest) -> str:
    """Return the checkpoint path from the ``--ckpt`` pytest custom option."""
    ckpt = request.config.getoption("--ckpt", default=None)
    if not ckpt:
        pytest.skip(
            "No --ckpt provided.  Pass a SeZM/DPA4 checkpoint path via "
            "`pytest --ckpt /path/to/model.pt`."
        )
    if not os.path.isfile(ckpt):
        pytest.skip(f"Checkpoint not found: {ckpt}")
    return ckpt


def _require_model_path(request: pytest.FixtureRequest) -> str:
    """Return the frozen .pth path from the ``--model`` pytest custom option."""
    model = request.config.getoption("--model", default=None)
    if not model:
        pytest.skip(
            "No --model provided.  Pass a frozen .pth path via "
            "`pytest --model /path/to/frozen.pth`."
        )
    if not os.path.isfile(model):
        pytest.skip(f"Frozen model not found: {model}")
    return model


# ---------------------------------------------------------------------------
# A. Lower-graph validation
# ---------------------------------------------------------------------------


def validate_lower_graph(
    fx_graph: torch.nn.Module,
    dim_fparam: int,
    dim_aparam: int,
    dim_chg_spin: int,
    sample_inputs: tuple[torch.Tensor | None, ...],
    is_spin: bool = False,
) -> torch.jit.ScriptModule:
    """Validate the first stage: trace the FX graph into a ScriptModule.

    Parameters
    ----------
    fx_graph
        The ``make_fx``-traced lower graph (from
        ``model.forward_common_lower_exportable``).
    dim_fparam, dim_aparam, dim_chg_spin
        Model parameter dimensions.
    sample_inputs
        Sample inputs matching the lower ABI (energy edge_vec or spin nlist).
    is_spin
        Whether the graph follows the spin (nlist) ABI.

    Returns
    -------
    torch.jit.ScriptModule
        The traced self-contained lower graph.

    Raises
    ------
    AssertionError
        If tracing fails or the traced output diverges from the eager output.
    """
    from deepmd.pt.entrypoints.freeze_pt2 import (
        _build_lower_trace_inputs,
        _select_lower_adapter,
    )

    adapter_cls = _select_lower_adapter(
        dim_fparam, dim_aparam, dim_chg_spin, is_spin=is_spin
    )
    adapter = adapter_cls(fx_graph).eval()
    lower_inputs = _build_lower_trace_inputs(
        sample_inputs, dim_fparam, dim_aparam, dim_chg_spin, is_spin=is_spin
    )

    # Trace the adapter.
    scripted = torch.jit.trace(adapter, lower_inputs, strict=False, check_trace=True)
    assert isinstance(scripted, torch.jit.ScriptModule), (
        f"Expected ScriptModule, got {type(scripted)}"
    )

    # Compare eager vs traced outputs.
    with torch.no_grad():
        eager_out = adapter(*lower_inputs)
        traced_out = scripted(*lower_inputs)

    assert set(eager_out.keys()) == set(traced_out.keys()), (
        f"Output key mismatch: eager={set(eager_out.keys())}, "
        f"traced={set(traced_out.keys())}"
    )

    mismatches: list[str] = []
    for key in eager_out:
        ref = eager_out[key]
        ts = traced_out[key]
        if ref.shape != ts.shape:
            mismatches.append(
                f"  {key}: shape eager={tuple(ref.shape)} traced={tuple(ts.shape)}"
            )
            continue
        max_diff = float((ref.float() - ts.float()).abs().max())
        if max_diff > 1e-5:
            mismatches.append(f"  {key}: max abs diff={max_diff:.2e}")
    assert not mismatches, "Lower graph output mismatch:\n" + "\n".join(mismatches)

    # Round-trip: save, load, compare again.
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tf:
        tmp_path = tf.name
    try:
        torch.jit.save(scripted, tmp_path)
        loaded = torch.jit.load(tmp_path)
        with torch.no_grad():
            loaded_out = loaded(*lower_inputs)
        for key in eager_out:
            ref = eager_out[key]
            ld = loaded_out[key]
            assert ref.shape == ld.shape, (
                f"Round-trip shape mismatch for {key}: "
                f"{tuple(ref.shape)} vs {tuple(ld.shape)}"
            )
            max_diff = float((ref.float() - ld.float()).abs().max())
            assert max_diff <= 1e-5, (
                f"Round-trip value mismatch for {key}: max abs diff={max_diff:.2e}"
            )
    finally:
        os.unlink(tmp_path)

    return scripted


# ---------------------------------------------------------------------------
# B. Full wrapper validation
# ---------------------------------------------------------------------------


def validate_full_wrapper(
    scripted_lower: torch.jit.ScriptModule,
    common_kwargs: dict[str, Any],
    is_spin: bool = False,
) -> torch.jit.ScriptModule:
    """Validate the second stage: script the LAMMPS-facing wrapper.

    Parameters
    ----------
    scripted_lower
        The Stage-1 ScriptModule (output of :func:`validate_lower_graph`).
    common_kwargs
        Keyword arguments for ``SeZMPTHModel`` / ``SeZMSpinPTHModel``
        (sel, rcut, ntypes, type_map, nnei, nsel, dim_fparam, …).
    is_spin
        Whether to use ``SeZMSpinPTHModel``.

    Returns
    -------
    torch.jit.ScriptModule
        The final scripted LAMMPS-facing model.

    Raises
    ------
    AssertionError
        If scripting or round-trip validation fails.
    """
    from deepmd.pt.entrypoints.freeze_pt2 import (
        SeZMPTHModel,
        SeZMSpinPTHModel,
    )

    if is_spin:
        wrapper = SeZMSpinPTHModel(scripted_lower, **common_kwargs)
    else:
        wrapper = SeZMPTHModel(scripted_lower, **common_kwargs)
    wrapper.eval()

    # Script the wrapper.
    scripted = torch.jit.script(wrapper)
    assert isinstance(scripted, torch.jit.ScriptModule), (
        f"Expected ScriptModule, got {type(scripted)}"
    )

    # Verify accessor methods survive scripting.
    assert abs(scripted.get_rcut() - common_kwargs["rcut"]) < 1e-10, (
        "get_rcut() mismatch after scripting"
    )
    assert scripted.get_ntypes() == common_kwargs["ntypes"], (
        "get_ntypes() mismatch after scripting"
    )
    assert scripted.get_nnei() == common_kwargs["nnei"], (
        "get_nnei() mismatch after scripting"
    )

    # Round-trip: save, load, verify accessors again.
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tf:
        tmp_path = tf.name
    try:
        torch.jit.save(scripted, tmp_path)
        loaded = torch.jit.load(tmp_path)
        assert abs(loaded.get_rcut() - common_kwargs["rcut"]) < 1e-10, (
            "Round-trip get_rcut() mismatch"
        )
        assert loaded.get_ntypes() == common_kwargs["ntypes"], (
            "Round-trip get_ntypes() mismatch"
        )
    finally:
        os.unlink(tmp_path)

    return scripted


# ---------------------------------------------------------------------------
# C. P100 runtime validation
# ---------------------------------------------------------------------------


def validate_p100_runtime(
    model_path: str,
    is_spin: bool = False,
) -> None:
    """Load a frozen ``.pth`` and run diagnostic checks on the current device.

    This is designed to be run on a Pascal P100 (or any CUDA GPU) to verify
    that the frozen model can be loaded and executed without Triton.

    Parameters
    ----------
    model_path
        Path to a ``.pth`` file produced by ``freeze_sezm_to_pth``.
    is_spin
        Whether the model follows the spin contract.

    Raises
    ------
    AssertionError
        If loading or execution fails.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available; P100 runtime test requires a GPU.")

    device_props = torch.cuda.get_device_properties(0)
    print(f"GPU: {device_props.name} (sm_{device_props.major}{device_props.minor})")

    # Load the model.
    model = torch.jit.load(model_path)
    model.eval()

    # Print metadata.
    print(f"  rcut:   {model.get_rcut()}")
    print(f"  ntypes: {model.get_ntypes()}")
    print(f"  nnei:   {model.get_nnei()}")
    print(f"  sel:    {model.get_sel()}")
    print(f"  has_spin: {model.has_spin()}")

    # Verify it can be moved to CUDA.
    model.cuda()
    print("  Moved to CUDA successfully.")

    # Build minimal sample inputs for a forward_lower call.
    ntypes = model.get_ntypes()
    nnei = model.get_nnei()
    rcut = model.get_rcut()
    nloc = 7
    nall = nloc + 2  # minimal ghost count

    device = torch.device("cuda")
    ext_coord = torch.rand(1, nall, 3, dtype=torch.float64, device=device) * rcut
    ext_atype = torch.randint(0, ntypes, (1, nall), device=device).to(torch.int64)
    nlist = torch.randint(0, nall, (1, nloc, nnei), device=device).to(torch.int64)
    mapping = torch.arange(nall, device=device).unsqueeze(0).to(torch.int64)

    dim_fparam = model.get_dim_fparam()
    dim_aparam = model.get_dim_aparam()
    fparam = (
        torch.zeros(1, dim_fparam, dtype=torch.float64, device=device)
        if dim_fparam > 0
        else None
    )
    aparam = (
        torch.zeros(1, nloc, dim_aparam, dtype=torch.float64, device=device)
        if dim_aparam > 0
        else None
    )

    if is_spin:
        ext_spin = torch.rand(1, nall, 3, dtype=torch.float64, device=device)
        args = (
            ext_coord,
            ext_atype,
            ext_spin,
            nlist,
            mapping,
            fparam,
            aparam,
            False,
            None,
        )
    else:
        dim_chg_spin = model.get_dim_chg_spin()
        chg_spin = (
            torch.zeros(1, dim_chg_spin, dtype=torch.float64, device=device)
            if dim_chg_spin > 0
            else None
        )
        args = (
            ext_coord,
            ext_atype,
            nlist,
            mapping,
            fparam,
            aparam,
            False,
            None,
            chg_spin,
        )

    # Run on CUDA.
    with torch.no_grad():
        out_cuda = model.forward_lower(*args)

    print(f"  CUDA forward_lower output keys: {list(out_cuda.keys())}")
    assert "energy" in out_cuda, "Output missing 'energy' key"
    assert out_cuda["energy"].device.type == "cuda", "Output not on CUDA"

    # Run on CPU for comparison.
    model.cpu()
    cpu_args = tuple(a.cpu() if isinstance(a, torch.Tensor) else a for a in args)
    with torch.no_grad():
        out_cpu = model.forward_lower(*cpu_args)

    # Compare CUDA vs CPU.
    for key in out_cuda:
        if key not in out_cpu:
            print(f"  WARNING: {key} missing from CPU output")
            continue
        cuda_val = out_cuda[key].cpu()
        cpu_val = out_cpu[key]
        if cuda_val.shape != cpu_val.shape:
            print(f"  WARNING: {key} shape mismatch CUDA vs CPU")
            continue
        max_diff = float((cuda_val.float() - cpu_val.float()).abs().max())
        status = "OK" if max_diff < 1e-4 else "MISMATCH"
        print(f"  {key}: CUDA vs CPU max diff={max_diff:.2e} [{status}]")

    print("P100 runtime validation complete.")


# ---------------------------------------------------------------------------
# Pytest entry points
# ---------------------------------------------------------------------------


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register custom pytest options for validation tests."""
    parser.addoption(
        "--ckpt",
        action="store",
        default=None,
        help="Path to a SeZM/DPA4 training checkpoint (.pt).",
    )
    parser.addoption(
        "--model",
        action="store",
        default=None,
        help="Path to a frozen .pth model file.",
    )


@pytest.mark.skip(reason="Requires a real SeZM checkpoint; run manually with --ckpt")
def test_validate_lower_graph(request: pytest.FixtureRequest) -> None:
    """End-to-end lower-graph validation (requires ``--ckpt``)."""
    ckpt_path = _require_ckpt_path(request)

    from deepmd.pt.entrypoints.freeze_pt2 import (
        _load_sezm_checkpoint,
        _model_has_spin,
        _resolve_nframes,
    )

    _state_dict, _params, model = _load_sezm_checkpoint(ckpt_path)
    is_spin = _model_has_spin(model)
    _, sample_inputs = _resolve_nframes(
        model, nloc=7, device=torch.device("cpu"), has_spin=is_spin
    )

    fx_graph = model.forward_common_lower_exportable(*sample_inputs)
    dim_fparam = int(model.get_dim_fparam())
    dim_aparam = int(model.get_dim_aparam())
    dim_chg_spin = int(model.get_dim_chg_spin())

    scripted = validate_lower_graph(
        fx_graph,
        dim_fparam,
        dim_aparam,
        dim_chg_spin,
        sample_inputs,
        is_spin=is_spin,
    )
    assert isinstance(scripted, torch.jit.ScriptModule)
    print("test_validate_lower_graph: PASSED")


@pytest.mark.skip(reason="Requires a real SeZM checkpoint; run manually with --ckpt")
def test_validate_full_wrapper(request: pytest.FixtureRequest) -> None:
    """End-to-end wrapper validation (requires ``--ckpt``)."""
    ckpt_path = _require_ckpt_path(request)

    from deepmd.pt.entrypoints.freeze_pt2 import (
        _get_model_ntypes,
        _load_sezm_checkpoint,
        _model_has_message_passing,
        _model_has_spin,
        _resolve_nframes,
        _select_lower_adapter,
        _build_lower_trace_inputs,
        _to_py_list,
    )

    _state_dict, _params, model = _load_sezm_checkpoint(ckpt_path)
    is_spin = _model_has_spin(model)
    _, sample_inputs = _resolve_nframes(
        model, nloc=7, device=torch.device("cpu"), has_spin=is_spin
    )

    fx_graph = model.forward_common_lower_exportable(*sample_inputs)
    dim_fparam = int(model.get_dim_fparam())
    dim_aparam = int(model.get_dim_aparam())
    dim_chg_spin = int(model.get_dim_chg_spin())

    # Stage 1: trace the lower graph.
    adapter_cls = _select_lower_adapter(
        dim_fparam, dim_aparam, dim_chg_spin, is_spin=is_spin
    )
    adapter = adapter_cls(fx_graph).eval()
    lower_inputs = _build_lower_trace_inputs(
        sample_inputs, dim_fparam, dim_aparam, dim_chg_spin, is_spin=is_spin
    )
    scripted_lower = torch.jit.trace(
        adapter, lower_inputs, strict=False, check_trace=True
    )

    # Stage 2: script the wrapper.
    common_kwargs = {
        "sel": [int(s) for s in model.get_sel()],
        "rcut": float(model.get_rcut()),
        "ntypes": _get_model_ntypes(model),
        "type_map": list(model.get_type_map()),
        "nnei": int(sum(model.get_sel())),
        "nsel": int(sum(model.get_sel())),
        "dim_fparam": dim_fparam,
        "dim_aparam": dim_aparam,
        "dim_chg_spin": dim_chg_spin,
        "has_mp": _model_has_message_passing(model),
        "min_nbor_dist": None,
        "model_def_json": json.dumps({}),
        "model_output_types": [],
        "has_default_fparam": bool(model.has_default_fparam()),
        "default_fparam": _to_py_list(model.get_default_fparam()),
        "default_chg_spin": _to_py_list(model.get_default_chg_spin()),
        "mixed_types": bool(model.mixed_types()),
        "is_spin": is_spin,
        "ntypes_spin": 0,
        "use_spin": [],
        "lower_input_kind": model.export_lower_input_kind(),
        "lower_nf": 1,
        "do_grad_r": bool(model.do_grad_r("energy")),
        "do_grad_c": bool(model.do_grad_c("energy")),
    }

    scripted = validate_full_wrapper(scripted_lower, common_kwargs, is_spin=is_spin)
    assert isinstance(scripted, torch.jit.ScriptModule)
    print("test_validate_full_wrapper: PASSED")


@pytest.mark.skip(reason="Requires a frozen .pth; run manually with --model")
def test_validate_p100_runtime(request: pytest.FixtureRequest) -> None:
    """P100 runtime validation (requires ``--model``)."""
    model_path = _require_model_path(request)
    validate_p100_runtime(model_path)
    print("test_validate_p100_runtime: PASSED")

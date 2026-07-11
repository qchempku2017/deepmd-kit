# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Environment-variable gates for the SeZM/DPA4 hardware-accelerated kernels.

This module centralizes the opt-in selectors that route inference through the
custom Triton and CuTe kernel packages. The gates are read once at model
construction time so that they become compile-time constants in the traced
(``make_fx``) graph.
"""

from __future__ import (
    annotations,
)

import os
import threading
import warnings
from typing import (
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    import torch

_INFER_TRUE = ("1", "true", "yes", "on")

TRITON_INFER_LEVELS = (0, 1, 2, 3)


def triton_infer_level() -> int:
    """Return the opt-in Triton inference level from ``DP_TRITON_INFER``.

    The level is read at module construction time so that it becomes a
    compile-time constant in the traced (``make_fx``) graph. It only takes
    effect during inference; training always uses the dense reference path.

    Levels are cumulative:

    - ``0`` -- Triton disabled; every operation uses the dense reference path.
    - ``1`` -- universal kernels that need no launch-configuration table:
      block-diagonal rotation, radial degree mixing, the ``SO2Linear``
      block GEMM, Wigner monomials, flash-attention aggregation, and the
      segmented force assembly. These are either runtime-autotuned or run a
      single shape-independent configuration.
    - ``2`` -- adds kernels whose launch configuration is resolved from the
      swept ``(focus_dim, lmax)`` / ``(C_wide, lmax)`` tables in
      :mod:`.triton.tile_configs`: the fused SO(2) value path and the
      edge-block backward kernels. A key absent from a table falls back to
      the level-1 kernel (or a spill-safe configuration) for that operation,
      so unswept shapes never regress below level 1.
    - ``3`` -- adds the fp16x3 split-compensated mixing-stack GEMMs on
      tensor cores. Entries exist only for table keys whose configuration
      passed the fp64 validation sweep; unswept shapes keep the level-2 fp32
      stack. This level trades a bounded accuracy perturbation for speed
      (see :mod:`.triton.so2_stack_fp16x3`).

    Returns
    -------
    int
        The configured level in ``{0, 1, 2, 3}``.

    Raises
    ------
    ValueError
        If ``DP_TRITON_INFER`` is not an integer in ``{0, 1, 2, 3}``.
    """
    raw = os.environ.get("DP_TRITON_INFER", "0").strip()
    try:
        level = int(raw)
    except ValueError:
        raise ValueError(
            f"DP_TRITON_INFER must be an integer in {TRITON_INFER_LEVELS}, got {raw!r}"
        ) from None
    if level not in TRITON_INFER_LEVELS:
        raise ValueError(
            f"DP_TRITON_INFER must be one of {TRITON_INFER_LEVELS}, got {level}"
        )
    return level


def use_cute_infer() -> bool:
    """Return whether the opt-in CuTe inference operator is enabled.

    The flag is controlled by the ``DP_CUTE_INFER`` environment variable and is
    read at module construction time. It selects the fused CuTe SO(2) value-path
    operator (an independent path from ``DP_TRITON_INFER``) and only takes effect
    during inference; training always uses the dense reference path.

    Returns
    -------
    bool
        ``True`` when ``DP_CUTE_INFER`` is set to a truthy value.
    """
    return os.environ.get("DP_CUTE_INFER", "0").strip().lower() in _INFER_TRUE


def use_amp_infer() -> bool:
    """Return whether bf16 autocast is enabled for inference.

    The flag is controlled by the ``DP_AMP_INFER`` environment variable and is
    read at module construction time. It only affects inference when the
    descriptor's ``use_amp`` option is also enabled; training follows
    ``use_amp`` regardless of this environment variable.

    Returns
    -------
    bool
        ``True`` when ``DP_AMP_INFER`` is set to a truthy value.
    """
    return os.environ.get("DP_AMP_INFER", "0").strip().lower() in _INFER_TRUE


# =============================================================================
# GPU capability detection (legacy-GPU compatibility)
# -----------------------------------------------------------------------------
# The SeZM/DPA4 descriptor splits into a dense fp32 eager reference path and a
# set of opt-in accelerated paths (Triton kernels, CuTe kernels, bf16 AMP,
# TF32, FP16 tensor cores, and the AOTInductor ``.pt2`` freeze). The dense
# path is pure PyTorch and runs on every CUDA device PyTorch itself supports.
# The accelerated paths require modern hardware: native bf16/TF32 need sm_80+
# (Ampere), FP16 tensor cores need sm_70+ (Volta), and Triton/AOTInductor are
# unsupported on Pascal (sm_60). These helpers let the descriptor and the
# freeze entrypoint detect legacy GPUs and auto-select the dense path or fail
# fast with an actionable message instead of crashing opaquely in a kernel.
# =============================================================================
# One-shot warning dedup keyed by (capability, reason) so the bf16 warning and
# the Triton warning each fire once even when they share a capability. The lock
# keeps the check-then-add atomic under free-threaded builds (a race would only
# duplicate a one-time warning, but guard it anyway).
_LEGACY_WARNED: set[tuple[tuple[int, int], str]] = set()
_LEGACY_WARN_LOCK = threading.Lock()


def cuda_compute_capability(
    device: torch.device | int | str | None = None,
) -> tuple[int, int] | None:
    """Return the ``(major, minor)`` compute capability of a CUDA device.

    Parameters
    ----------
    device : torch.device, int, str, or None
        Device to query. ``None`` queries the currently selected CUDA device.
        Any non-CUDA device (CPU, or CUDA unavailable) returns ``None``.

    Returns
    -------
    tuple[int, int] or None
        The compute capability, e.g. ``(6, 0)`` for an NVIDIA Tesla P100, or
        ``None`` when CUDA is unavailable or ``device`` is not a CUDA device.
        The function never raises: a failed capability query returns ``None``,
        so callers can treat it as a pure routing signal.
    """
    import torch

    try:
        if isinstance(device, str):
            device = torch.device(device)
        if device is None:
            if not torch.cuda.is_available():
                return None
            idx = torch.cuda.current_device()
        elif isinstance(device, torch.device):
            if device.type != "cuda":
                return None
            idx = (
                device.index
                if device.index is not None
                else torch.cuda.current_device()
            )
        else:
            idx = int(device)
        return torch.cuda.get_device_capability(idx)
    except Exception:
        return None


def cuda_supports_bf16(
    device: torch.device | int | str | None = None,
) -> bool:
    """Return whether ``device`` has native bfloat16 support.

    Native bfloat16 (used by ``torch.autocast`` with ``torch.bfloat16``)
    requires compute capability >= 8.0 (NVIDIA Ampere and newer). Pascal
    (sm_60), Volta (sm_70) and Turing (sm_75) lack native bf16; enabling
    SeZM ``use_amp`` there raises or silently degrades.
    """
    cap = cuda_compute_capability(device)
    return cap is not None and cap[0] >= 8


def cuda_supports_triton(
    device: torch.device | int | str | None = None,
) -> bool:
    """Return whether ``device`` can run Triton and AOTInductor (sm_70+).

    The Triton compiler (shipped with PyTorch 2.11/2.12) cannot compile for
    Pascal (sm_60/sm_62) -- AOTInductor is confirmed broken there -- but it
    *does* support Volta (sm_70) and newer. ``torch.compile`` / Inductor, the
    SeZM Triton inference kernels, and the ``dp --pt freeze`` ``.pt2`` (AOTI)
    path therefore work on Volta/Turing/Ampere+ and are unavailable only on
    Pascal and older.
    """
    cap = cuda_compute_capability(device)
    return cap is not None and cap[0] >= 7


def gpu_capability_description(
    device: torch.device | int | str | None = None,
) -> str:
    """Return a short human-readable description of the CUDA device capability.

    Returns
    -------
    str
        e.g. ``"NVIDIA Tesla P100 PCIe (compute capability sm_60, native bf16:
        no)"``; ``"no CUDA device available"`` when CUDA is absent.
    """
    import torch

    if isinstance(device, str):
        device = torch.device(device)
    cap = cuda_compute_capability(device)
    if cap is None:
        return "no CUDA device available"
    try:
        if device is None:
            idx = torch.cuda.current_device()
        elif isinstance(device, torch.device):
            idx = (
                device.index
                if device.index is not None
                else torch.cuda.current_device()
            )
        else:
            idx = int(device)
        name = torch.cuda.get_device_name(idx)
    except Exception:
        name = "unknown"
    return (
        f"{name} (compute capability sm_{cap[0] * 10 + cap[1]}, "
        f"native bf16: {'yes' if cap[0] >= 8 else 'no'})"
    )


def warn_legacy_gpu_once(
    reason: str,
    *,
    device: torch.device | int | str | None = None,
) -> None:
    """Emit a one-time-per-(capability, reason) ``UserWarning`` for a legacy GPU.

    Each distinct ``(capability, reason)`` pair fires at most once per process,
    so the bfloat16 warning and the Triton warning each surface independently
    while hot forward loops that repeatedly query the capability stay quiet
    after the first notice.

    Parameters
    ----------
    reason : str
        Human-readable explanation appended to the generic legacy-GPU notice.
        Also scopes the one-shot deduplication, so distinct reasons are not
        mutually suppressed.
    device
        Device whose capability scopes the one-shot deduplication.
    """
    cap = cuda_compute_capability(device)
    if cap is None:
        return
    dedup_key = (cap, reason)
    with _LEGACY_WARN_LOCK:
        if dedup_key in _LEGACY_WARNED:
            return
        _LEGACY_WARNED.add(dedup_key)
    warnings.warn(
        "DeePMD-kit detected compute capability "
        f"sm_{cap[0] * 10 + cap[1]} (native bf16: "
        f"{'yes' if cap[0] >= 8 else 'no'}, Triton/AOTInductor: "
        f"{'yes' if cap[0] >= 7 else 'no'}). {reason} See the 'Install and run "
        "on legacy NVIDIA GPUs' guide in the docs for the supported "
        "configuration.",
        UserWarning,
        stacklevel=2,
    )


def assert_triton_supported_gpu(
    device: torch.device | int | str | None = None,
) -> None:
    """Fail fast when a Triton/AOTInductor path is requested on Pascal.

    The ``torch.compile`` and AOTInductor (``.pt2`` freeze) paths lower the
    SeZM graph through Triton *on CUDA targets*. Triton (as shipped with
    PyTorch 2.11/2.12) supports Volta (sm_70) and newer but **cannot compile
    for Pascal (sm_60/sm_62)** -- AOTInductor is confirmed broken on sm_60. This
    raises a clear, actionable error on a Pascal CUDA device instead of
    crashing inside the Triton compiler several minutes into a freeze.

    Non-CUDA targets (CPU, or no CUDA available) are **allowed**: Inductor
    falls back to its cpp backend there, which does not use Triton, so
    ``torch.compile`` / AOTInductor remain valid. Volta, Turing, Ampere, and
    newer CUDA devices are also unaffected.

    Parameters
    ----------
    device
        Device to check; ``None`` queries the current CUDA device.

    Raises
    ------
    RuntimeError
        If a CUDA device is present with compute capability < 7.0 (Pascal and
        older). CPU / no-CUDA returns silently.
    """
    cap = cuda_compute_capability(device)
    if cap is None or cap[0] >= 7:
        return
    raise RuntimeError(
        "The SeZM/DPA4 torch.compile / AOTInductor path is unsupported on "
        f"Pascal GPUs (compute capability sm_{cap[0] * 10 + cap[1]}). It "
        "lowers through Triton, which cannot compile for this architecture "
        "(AOTInductor is confirmed broken on sm_60); Volta (sm_70) and "
        "newer are supported. Use the dense eager path instead: keep "
        "`use_compile=false`, leave `DP_COMPILE_INFER` unset, and set "
        "`DP_TRITON_INFER=0` (the defaults). For ASE / `dp --pt test` "
        "inference, load the `.pt` checkpoint directly (no freeze needed). "
        "To freeze a model for LAMMPS on Pascal, use `dp --pt freeze "
        "--legacy-gpu` to produce a TorchScript `.pth` file instead -- "
        "the TorchScript export path does not call "
        "`model.validate_compile_device()` and skips the Triton check "
        "entirely, so this error should never appear with `--legacy-gpu`. "
        "For the torch.compile / AOTInductor path itself, there is no "
        "override on Pascal -- use the dense eager path on this GPU, "
        "or run on a Volta-or-newer GPU."
    )

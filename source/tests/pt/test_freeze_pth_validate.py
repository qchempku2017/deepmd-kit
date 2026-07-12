# SPDX-License-Identifier: LGPL-3.0-or-later
r"""CPU-only validation tests for the two-stage ``.pth`` freeze path.

These tests validate the correctness of the TorchScript conversion
at each stage of the pipeline introduced for the legacy-GPU (Pascal/P100)
compatibility fix.

**Edge-schema equivalence tests** (``TestEdgeSchemaEquivalence``) run
without any checkpoint and verify that ``_build_edge_schema_ts`` and
``edge_schema_from_extended`` produce identical outputs for a variety
of input configurations.

**Freeze and runtime tests** require a SeZM/DPA4 checkpoint passed via
``--ckpt`` and are skipped at runtime when no checkpoint is provided.

Usage (from the repository root)::

    # Edge-schema equivalence (no checkpoint needed):
    pytest source/tests/pt/test_freeze_pth_validate.py::TestEdgeSchemaEquivalence -v

    # Full freeze validation (requires a SeZM checkpoint):
    pytest source/tests/pt/test_freeze_pth_validate.py --ckpt model.pt -v

    # Runtime validation (requires a frozen .pth):
    pytest source/tests/pt/test_freeze_pth_validate.py --model frozen.pth -v
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass

import pytest
import torch

# The pt test package __init__.py sets default_device to "cuda:9999999"
# which breaks CPU-only torch builds.  Use an autouse fixture scoped to
# this module to avoid a global side-effect that would affect other test
# files running in the same session.


@pytest.fixture(scope="module", autouse=True)
def _set_cpu_default_device() -> None:
    """Set default device to CPU for this module and restore on teardown.

    ``torch.set_default_device`` was introduced in PyTorch 2.1.  On older
    PyTorch versions this fixture is a no-op; the per-test ``device="cpu"``
    arguments still keep the tests on CPU.
    """
    if not hasattr(torch, "set_default_device"):
        yield
        return
    old_device = torch.get_default_device()
    torch.set_default_device("cpu")
    yield
    torch.set_default_device(old_device)


# ===========================================================================
# Inlined production helpers
# ===========================================================================
# These are exact copies of functions from the deepmd-kit codebase that
# cannot be imported without the C extension (deepmd.lib).  They MUST stay
# in sync with their production counterparts:
#
#   EdgeNeighborList          → deepmd/dpmodel/utils/neighbor_list.py
#   _append_dummy_edges       → deepmd/pt_expt/utils/edge_schema.py
#   edge_schema_from_extended → deepmd/pt_expt/utils/edge_schema.py
#   _build_edge_schema_ts     → deepmd/pt/entrypoints/freeze_pt2.py
#
# Last synced: 2026-07-12 (legacy_gpus_compat branch)
# ===========================================================================

_DUMMY_EDGE_COUNT = 2

# Minimum squared edge length to filter coincident (self-self) pairs.
# Mirrors the inline ``1e-10`` literal used in
# deepmd/pt/entrypoints/freeze_pt2.py.
_MIN_EDGE_LEN2 = 1e-10


@dataclass
class EdgeNeighborList:
    """Edge-vector neighbor-list contract (inlined from deepmd.dpmodel.utils.neighbor_list)."""

    coord: torch.Tensor
    atype: torch.Tensor
    edge_index: torch.Tensor
    edge_vec: torch.Tensor
    edge_scatter_index: torch.Tensor
    edge_mask: torch.Tensor


def _append_dummy_edges(
    edge_index: torch.Tensor,
    edge_vec: torch.Tensor,
    edge_scatter_index: torch.Tensor,
) -> EdgeNeighborList:
    """Append masked in-range edges so exported graphs never see empty inputs.

    Inlined from: deepmd/pt_expt/utils/edge_schema.py
    """
    device = edge_index.device
    dummy_index = torch.zeros(
        (2, _DUMMY_EDGE_COUNT),
        dtype=edge_index.dtype,
        device=device,
    )
    dummy_vec = torch.zeros(
        (_DUMMY_EDGE_COUNT, 3),
        dtype=edge_vec.dtype,
        device=device,
    )
    edge_index = torch.cat([edge_index, dummy_index], dim=1)
    edge_vec = torch.cat([edge_vec, dummy_vec], dim=0)
    edge_scatter_index = torch.cat([edge_scatter_index, dummy_index], dim=1)
    edge_mask = torch.cat(
        [
            torch.ones(
                edge_vec.shape[0] - _DUMMY_EDGE_COUNT, dtype=torch.bool, device=device
            ),
            torch.zeros(_DUMMY_EDGE_COUNT, dtype=torch.bool, device=device),
        ]
    )
    return EdgeNeighborList(
        coord=torch.empty(0, dtype=edge_vec.dtype, device=device),
        atype=torch.empty(0, dtype=torch.long, device=device),
        edge_index=edge_index,
        edge_vec=edge_vec,
        edge_scatter_index=edge_scatter_index,
        edge_mask=edge_mask,
    )


def edge_schema_from_extended(
    coord: torch.Tensor,
    atype: torch.Tensor,
    nlist: torch.Tensor,
    mapping: torch.Tensor | None,
    *,
    scatter_to_local: bool = False,
) -> EdgeNeighborList:
    """Build the unified edge schema from an extended-coordinate neighbor list.

    Inlined from: deepmd/pt_expt/utils/edge_schema.py
    """
    nf, nloc, nsel = nlist.shape
    device = coord.device
    nall = coord.shape[1]

    neighbor_flat = nlist.reshape(-1)
    dst_actual = (
        torch.arange(neighbor_flat.shape[0], device=device, dtype=torch.long) // nsel
    )
    frame_idx = dst_actual // nloc
    dst_local = dst_actual % nloc
    valid_flat = neighbor_flat >= 0
    neighbor_safe = torch.where(
        valid_flat, neighbor_flat, torch.zeros_like(neighbor_flat)
    )
    neighbor_safe_2d = neighbor_safe.to(dtype=torch.long).view(nf, nloc * nsel)

    neighbor_coord = torch.gather(
        coord,
        1,
        neighbor_safe_2d.unsqueeze(-1).expand(-1, -1, 3),
    ).reshape(-1, 3)
    dst_coord = torch.gather(
        coord[:, :nloc, :],
        1,
        dst_local.view(nf, -1).unsqueeze(-1).expand(-1, -1, 3),
    ).reshape(-1, 3)
    edge_vec_all = neighbor_coord - dst_coord
    edge_len2 = torch.sum(edge_vec_all * edge_vec_all, dim=-1)

    if mapping is None:
        src_local = neighbor_safe.to(dtype=torch.long)
    else:
        src_local = torch.gather(mapping, 1, neighbor_safe_2d).reshape(-1)
    src_actual = frame_idx * nloc + src_local.to(dtype=torch.long)
    src_scatter = frame_idx * nall + neighbor_safe.to(dtype=torch.long)
    dst_scatter = frame_idx * nall + dst_local

    edge_keep = (
        valid_flat
        & (src_local >= 0)
        & (src_local < nloc)
        & (edge_len2 > _MIN_EDGE_LEN2)
    )
    valid_idx = torch.where(edge_keep)[0]
    edge_index = torch.stack(
        [
            src_actual.index_select(0, valid_idx),
            dst_actual.index_select(0, valid_idx),
        ],
        dim=0,
    )
    if scatter_to_local:
        edge_scatter_index = edge_index
    else:
        edge_scatter_index = torch.stack(
            [
                src_scatter.index_select(0, valid_idx),
                dst_scatter.index_select(0, valid_idx),
            ],
            dim=0,
        )
    schema = _append_dummy_edges(
        edge_index,
        edge_vec_all.index_select(0, valid_idx),
        edge_scatter_index,
    )
    schema.coord = coord[:, :nloc, :].contiguous() if scatter_to_local else coord
    schema.atype = atype[:, :nloc].contiguous()
    return schema


def _build_edge_schema_ts(
    fnlist: torch.Tensor,
    extended_coord: torch.Tensor,
    nloc: int,
    nnei: int,
    mapping: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Build edge schema tensors from a LAMMPS-format neighbour list.

    Inlined from: deepmd/pt/entrypoints/freeze_pt2.py
    """
    nf = fnlist.shape[0]
    nall = extended_coord.shape[1]
    neighbor_flat = fnlist.reshape(-1)
    dst_actual = (
        torch.arange(neighbor_flat.shape[0], device=fnlist.device, dtype=torch.long)
        // nnei
    )
    frame_idx = dst_actual // nloc
    dst_local = dst_actual % nloc
    valid_flat = neighbor_flat >= 0
    neighbor_safe = torch.where(
        valid_flat, neighbor_flat, torch.zeros_like(neighbor_flat)
    )
    neighbor_safe_2d = neighbor_safe.to(dtype=torch.long).view(nf, nloc * nnei)

    # Gather neighbour coordinates from the extended domain.
    neighbor_coord = torch.gather(
        extended_coord,
        1,
        neighbor_safe_2d.unsqueeze(-1).expand(-1, -1, 3),
    ).reshape(-1, 3)
    # Gather destination (central atom) coordinates from the *local* slice only.
    dst_coord = torch.gather(
        extended_coord[:, :nloc, :],
        1,
        dst_local.view(nf, -1).unsqueeze(-1).expand(-1, -1, 3),
    ).reshape(-1, 3)
    edge_vec_all = neighbor_coord - dst_coord
    edge_len2 = torch.sum(edge_vec_all * edge_vec_all, dim=-1)

    # Source local indices via the ghost→owner mapping.
    src_local = torch.gather(mapping, 1, neighbor_safe_2d).reshape(-1)
    src_actual = frame_idx * nloc + src_local.to(dtype=torch.long)
    src_scatter = frame_idx * nall + neighbor_safe.to(dtype=torch.long)
    dst_scatter = frame_idx * nall + dst_local

    # edge_index uses the *local* domain (src_actual, dst_actual).
    edge_index_all = torch.stack([src_actual, dst_actual], dim=0)
    # edge_scatter_index uses the *extended* domain (src_scatter, dst_scatter).
    edge_scatter_all = torch.stack([src_scatter, dst_scatter], dim=0)

    edge_keep = (
        valid_flat
        & (src_local >= 0)
        & (src_local < nloc)
        & (edge_len2 > _MIN_EDGE_LEN2)
    )
    valid_idx = torch.where(edge_keep)[0]
    edge_index = edge_index_all[:, valid_idx]
    edge_vec = edge_vec_all[valid_idx]
    edge_scatter_index = edge_scatter_all[:, valid_idx]

    # Append 2 dummy edges (matching _append_dummy_edges in edge_schema.py).
    _DUMMY_EDGE_COUNT = 2
    device = fnlist.device
    dummy_index = torch.zeros(
        (2, _DUMMY_EDGE_COUNT), dtype=edge_index.dtype, device=device
    )
    dummy_vec = torch.zeros((_DUMMY_EDGE_COUNT, 3), dtype=edge_vec.dtype, device=device)
    num_valid = valid_idx.shape[0]
    edge_mask = torch.cat(
        [
            torch.ones(num_valid, dtype=torch.bool, device=device),
            torch.zeros(_DUMMY_EDGE_COUNT, dtype=torch.bool, device=device),
        ]
    )
    edge_index = torch.cat([edge_index, dummy_index], dim=1)
    edge_vec = torch.cat([edge_vec, dummy_vec], dim=0)
    edge_scatter_index = torch.cat([edge_scatter_index, dummy_index], dim=1)
    return (edge_index, edge_vec, edge_scatter_index, edge_mask, nf)


# ===========================================================================
# Test helpers
# ===========================================================================


def _make_nlist(
    nloc: int,
    nall: int,
    nnei: int,
) -> torch.Tensor:
    """Build a realistic neighbour list with optional ghost references.

    Each local atom gets neighbours from the range ``[i-1, i+1]`` capped
    within ``[0, nall)``.  Remaining slots are filled with -1.
    """
    nlist = torch.full((nloc, nnei), -1, dtype=torch.int64)
    for i in range(nloc):
        candidates = list(range(max(0, i - 1), min(nall, i + 2)))
        for j, cand in enumerate(candidates[:nnei]):
            nlist[i, j] = cand
    return nlist


def _make_extended_coord(
    nf: int,
    nall: int,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Build deterministic extended coordinates for testing."""
    ntotal = nf * nall * 3
    return torch.arange(ntotal, dtype=dtype).reshape(nf, nall, 3)


def _make_atype(nf: int, nloc: int) -> torch.Tensor:
    """Build deterministic atom types for testing."""
    return torch.zeros((nf, nloc), dtype=torch.int64)


def _make_identity_mapping(nf: int, nall: int) -> torch.Tensor:
    """Build an identity mapping tensor (no ghost atoms)."""
    return torch.arange(nall, dtype=torch.int64).unsqueeze(0).expand(nf, nall)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def ckpt_path(request: pytest.FixtureRequest) -> str:
    """Return the checkpoint path from --ckpt, or skip the test."""
    path = request.config.getoption("--ckpt", default=None)
    if path is None:
        pytest.skip("No --ckpt provided")
    if not os.path.isfile(path):
        pytest.skip(f"Checkpoint not found: {path}")
    return path


@pytest.fixture
def model_path(request: pytest.FixtureRequest) -> str:
    """Return the frozen .pth path from --model, or skip the test."""
    path = request.config.getoption("--model", default=None)
    if path is None:
        pytest.skip("No --model provided")
    if not os.path.isfile(path):
        pytest.skip(f"Frozen model not found: {path}")
    return path


# ===========================================================================
# TestEdgeSchemaEquivalence -- no checkpoint needed
# ===========================================================================


class TestEdgeSchemaEquivalence:
    """Direct unit tests comparing ``_build_edge_schema_ts`` against
    ``edge_schema_from_extended`` using manually constructed tensors.

    These tests require no checkpoint and run on CPU.
    """

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _assert_schema_equal(
        ts_result: tuple,
        ref_schema: EdgeNeighborList,
    ) -> None:
        """Assert that the TS result matches the reference schema."""
        ts_edge_index, ts_edge_vec, ts_edge_scatter_index, ts_edge_mask, ts_nf = (
            ts_result
        )

        # Integer tensors: exact equality.
        for name, ts_val, ref_val in (
            ("edge_index", ts_edge_index, ref_schema.edge_index),
            (
                "edge_scatter_index",
                ts_edge_scatter_index,
                ref_schema.edge_scatter_index,
            ),
            ("edge_mask", ts_edge_mask, ref_schema.edge_mask),
        ):
            assert ts_val.shape == ref_val.shape, (
                f"{name}: shape mismatch TS={tuple(ts_val.shape)} "
                f"ref={tuple(ref_val.shape)}"
            )
            assert torch.equal(ts_val, ref_val), (
                f"{name}: values differ; "
                f"first mismatch at {(ts_val != ref_val).nonzero(as_tuple=False)[0].tolist()}"
            )

        # Float tensor: numerical equality.
        assert ts_edge_vec.shape == ref_schema.edge_vec.shape, (
            f"edge_vec: shape mismatch TS={tuple(ts_edge_vec.shape)} "
            f"ref={tuple(ref_schema.edge_vec.shape)}"
        )
        assert torch.allclose(ts_edge_vec, ref_schema.edge_vec, rtol=1e-5, atol=1e-8), (
            f"edge_vec: values differ; "
            f"max abs diff={(ts_edge_vec - ref_schema.edge_vec).abs().max():.2e}"
        )

    @staticmethod
    def _run_comparison(
        extended_coord: torch.Tensor,
        atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor,
        nloc: int,
        nnei: int,
    ) -> None:
        """Run both functions and assert equivalence."""
        # Production reference path (inlined)
        ref_schema = edge_schema_from_extended(extended_coord, atype, nlist, mapping)

        # TS-compatible path (inlined from freeze_pt2)
        ts_result = _build_edge_schema_ts(nlist, extended_coord, nloc, nnei, mapping)

        TestEdgeSchemaEquivalence._assert_schema_equal(ts_result, ref_schema)

    # ------------------------------------------------------------------
    # Test case 1: No ghost atoms, single frame
    # ------------------------------------------------------------------

    def test_no_ghost_single_frame(self):
        """nloc=4, nall=4, nnei=3, identity mapping."""
        nf, nloc, nall, nnei = 1, 4, 4, 3
        extended_coord = _make_extended_coord(nf, nall)
        atype = _make_atype(nf, nloc)
        nlist = _make_nlist(nloc, nall, nnei).unsqueeze(0)  # (1, nloc, nnei)
        mapping = _make_identity_mapping(nf, nall)

        self._run_comparison(extended_coord, atype, nlist, mapping, nloc, nnei)

    # ------------------------------------------------------------------
    # Test case 2: Multiple frames, no ghosts
    # ------------------------------------------------------------------

    def test_multiple_frames_no_ghosts(self):
        """nframes=2, nloc=3, nall=3, nnei=2."""
        nf, nloc, nall, nnei = 2, 3, 3, 2
        extended_coord = _make_extended_coord(nf, nall)
        atype = _make_atype(nf, nloc)
        nlist0 = _make_nlist(nloc, nall, nnei)
        nlist1 = _make_nlist(nloc, nall, nnei)
        nlist = torch.stack([nlist0, nlist1], dim=0)  # (2, nloc, nnei)
        mapping = _make_identity_mapping(nf, nall)

        self._run_comparison(extended_coord, atype, nlist, mapping, nloc, nnei)

    # ------------------------------------------------------------------
    # Test case 3: One ghost atom
    # ------------------------------------------------------------------

    def test_one_ghost_atom(self):
        """nloc=4, nall=5, ghost at index 4 owned by local atom 1."""
        nf, nloc, nall, nnei = 1, 4, 5, 3
        extended_coord = _make_extended_coord(nf, nall)
        atype = _make_atype(nf, nloc)
        nlist = _make_nlist(nloc, nall, nnei).unsqueeze(0)
        # Ghost at index 4 owned by local atom 1
        mapping = torch.tensor([[0, 1, 2, 3, 1]], dtype=torch.int64)

        self._run_comparison(extended_coord, atype, nlist, mapping, nloc, nnei)

    # ------------------------------------------------------------------
    # Test case 4: Multiple ghosts, repeated owners
    # ------------------------------------------------------------------

    def test_multiple_ghosts_repeated_owners(self):
        """nloc=6, nall=9, ghosts at [6,7,8] owned by [1,3,5]."""
        nf, nloc, nall, nnei = 1, 6, 9, 4
        extended_coord = _make_extended_coord(nf, nall)
        atype = _make_atype(nf, nloc)
        nlist = _make_nlist(nloc, nall, nnei).unsqueeze(0)
        mapping = torch.tensor([[0, 1, 2, 3, 4, 5, 1, 3, 5]], dtype=torch.int64)

        self._run_comparison(extended_coord, atype, nlist, mapping, nloc, nnei)

    # ------------------------------------------------------------------
    # Test case 5: -1 padding in nlist
    # ------------------------------------------------------------------

    def test_neg_one_padding(self):
        """Include -1 values in the neighbor list (padding slots)."""
        nf, nloc, nall, nnei = 1, 4, 4, 5
        extended_coord = _make_extended_coord(nf, nall)
        atype = _make_atype(nf, nloc)
        # Make nlist with extra slots that get -1 padding
        nlist = torch.full((1, nloc, nnei), -1, dtype=torch.int64)
        for i in range(nloc):
            nlist[0, i, 0] = (i - 1) % nall
            nlist[0, i, 1] = i
            nlist[0, i, 2] = (i + 1) % nall
        mapping = _make_identity_mapping(nf, nall)

        self._run_comparison(extended_coord, atype, nlist, mapping, nloc, nnei)

    # ------------------------------------------------------------------
    # Test case 6: Coincident pairs
    # ------------------------------------------------------------------

    def test_coincident_pairs(self):
        """Neighbor at same position as dst atom (edge_len2 == 0)."""
        nf, nloc, nall, nnei = 1, 4, 4, 3
        atype = _make_atype(nf, nloc)
        mapping = _make_identity_mapping(nf, nall)

        # Make extended_coord where some dst atoms and neighbors coincide
        extended_coord = torch.zeros((1, nall, 3), dtype=torch.float64)
        for i in range(nall):
            extended_coord[0, i, 0] = float(i)

        # nlist: every atom lists its neighbours including itself (coincident)
        nlist = torch.full((1, nloc, nnei), -1, dtype=torch.int64)
        for i in range(nloc):
            nlist[0, i, 0] = i  # coincident pair (self)
            if i > 0:
                nlist[0, i, 1] = i - 1
            if i + 1 < nall:
                nlist[0, i, 2] = i + 1

        self._run_comparison(extended_coord, atype, nlist, mapping, nloc, nnei)

    # ------------------------------------------------------------------
    # Test case 7: Varying nloc/nall/nnei
    # ------------------------------------------------------------------

    def test_varying_dims(self):
        """Test different dimension combinations."""
        configs = [
            (1, 5, 7, 2),  # nf=1, nloc=5, nall=7, nnei=2
            (2, 3, 5, 4),  # nf=2, nloc=3, nall=5, nnei=4
            (1, 8, 8, 6),  # nf=1, nloc=8, nall=8, nnei=6
            (3, 2, 4, 3),  # nf=3, nloc=2, nall=4, nnei=3
        ]
        for nf, nloc, nall, nnei in configs:
            extended_coord = _make_extended_coord(nf, nall)
            atype = _make_atype(nf, nloc)

            # Build per-frame ghost→owner mappings so that multi-frame
            # tests exercise cross-frame indexing (each frame has a
            # different ghost configuration).
            if nf > 1:
                mapping_rows = []
                for f in range(nf):
                    offset = f % nloc
                    row = list(range(nloc)) + [
                        (offset + i) % nloc for i in range(nloc, nall)
                    ]
                    mapping_rows.append(row)
                mapping = torch.tensor(mapping_rows, dtype=torch.int64)
            else:
                mapping_list = list(range(nloc)) + [
                    (i % nloc) for i in range(nloc, nall)
                ]
                mapping = torch.tensor([mapping_list], dtype=torch.int64).expand(
                    nf, nall
                )

            nlist = _make_nlist(nloc, nall, nnei).unsqueeze(0).expand(nf, nloc, nnei)

            self._run_comparison(extended_coord, atype, nlist, mapping, nloc, nnei)

    # ------------------------------------------------------------------
    # Runtime-input wrapper equivalence tests (2c)
    # ------------------------------------------------------------------

    def test_wrapper_lower_input_equivalence(self):
        """Verify that ``edge_schema_from_extended`` produces inputs
        suitable for the lower-graph interface and that the wrapper-side
        ``_build_edge_schema_ts`` produces matching edge tensors.

        Checks:
        - coord domain is extended (nall, not nloc)
        - atype is local (nloc)
        - edge_index uses local indexing (< nframes * nloc)
        - edge_scatter_index uses extended indexing (< nframes * nall)
        - edge_vec direction
        - edge_mask
        """
        nf, nloc, nall, nnei = 2, 5, 8, 3
        extended_coord = _make_extended_coord(nf, nall)
        atype = _make_atype(nf, nloc)

        # Build mapping with ghosts
        mapping_list = [*list(range(nloc)), 1, 3, 4]  # 3 ghosts
        mapping = torch.tensor([mapping_list], dtype=torch.int64).expand(nf, nall)

        nlist = _make_nlist(nloc, nall, nnei).unsqueeze(0).expand(nf, nloc, nnei)

        # Reference: edge_schema_from_extended
        ref = edge_schema_from_extended(extended_coord, atype, nlist, mapping)

        # Wrapper side: _build_edge_schema_ts
        ts_edge_index, ts_edge_vec, ts_edge_scatter_index, ts_edge_mask, ts_nf = (
            _build_edge_schema_ts(nlist, extended_coord, nloc, nnei, mapping)
        )

        # --- Verify domain properties of the reference schema ---

        # coord domain is extended (uses nall)
        assert ref.coord.shape == (nf, nall, 3), (
            f"coord should be extended (nf={nf}, nall={nall}, 3), "
            f"got {tuple(ref.coord.shape)}"
        )

        # atype is local (nloc)
        assert ref.atype.shape == (nf, nloc), (
            f"atype should be local (nf={nf}, nloc={nloc}), "
            f"got {tuple(ref.atype.shape)}"
        )

        # edge_index uses local indexing (< nframes * nloc)
        nframes_times_nloc = nf * nloc
        real_mask = ref.edge_mask
        assert (ref.edge_index[:, real_mask] < nframes_times_nloc).all(), (
            "edge_index values should all be < nframes * nloc"
        )

        # edge_scatter_index uses extended indexing (< nframes * nall)
        nframes_times_nall = nf * nall
        assert (ref.edge_scatter_index[:, real_mask] < nframes_times_nall).all(), (
            "edge_scatter_index values should all be < nframes * nall"
        )

        # --- Verify equivalence between reference and TS ---

        assert torch.equal(ts_edge_index, ref.edge_index), (
            "edge_index mismatch between TS and reference"
        )
        assert torch.allclose(ts_edge_vec, ref.edge_vec, rtol=1e-5, atol=1e-8), (
            f"edge_vec mismatch: "
            f"max diff={(ts_edge_vec - ref.edge_vec).abs().max():.2e}"
        )
        assert torch.equal(ts_edge_scatter_index, ref.edge_scatter_index), (
            "edge_scatter_index mismatch between TS and reference"
        )
        assert torch.equal(ts_edge_mask, ref.edge_mask), (
            "edge_mask mismatch between TS and reference"
        )
        assert ts_nf == nf, f"nf mismatch: {ts_nf} != {nf}"


# ===========================================================================
# Tests requiring --ckpt (runtime-skipped when no checkpoint provided)
# ===========================================================================


def _try_import_freeze_helpers() -> tuple | None:
    """Try to import freeze helpers; return None if the C extension is missing."""
    try:
        from deepmd.pt.entrypoints.freeze_pt2 import (
            LowerGraphNoParamAdapter,
            _collect_metadata,
            _get_model_ntypes,
            _load_sezm_checkpoint,
            _model_has_message_passing,
            _model_has_spin,
            _resolve_nframes,
            _to_py_list,
            SeZMPTHModel,
            freeze_sezm_to_pth,
        )
    except ImportError:
        return None

    # SeZMSpinPTHModel may be absent in future versions (spin .pth freeze
    # is deprecated).  Import it separately so that a missing spin wrapper
    # does not cascade-fail all checkpoint-dependent tests.
    try:
        from deepmd.pt.entrypoints.freeze_pt2 import SeZMSpinPTHModel
    except ImportError:
        SeZMSpinPTHModel = None  # type: ignore[assignment]

    return (
        freeze_sezm_to_pth,
        _load_sezm_checkpoint,
        _model_has_spin,
        _resolve_nframes,
        _get_model_ntypes,
        _model_has_message_passing,
        LowerGraphNoParamAdapter,
        _to_py_list,
        _collect_metadata,
        SeZMPTHModel,
        SeZMSpinPTHModel,
    )


def _require_freeze_helpers():
    """Return freeze helpers or skip the test."""
    helpers = _try_import_freeze_helpers()
    if helpers is None:
        pytest.skip(
            "deepmd C extension not available; freeze helpers cannot be imported."
        )
    return helpers


# ---------------------------------------------------------------------------
# Shared helper: build common kwargs for SeZMPTHModel / SeZMSpinPTHModel
# ---------------------------------------------------------------------------


def _build_common_kwargs(
    model,
    *,
    _get_model_ntypes,
    _model_has_message_passing,
    _to_py_list,
    dim_fparam: int,
    dim_aparam: int,
    dim_chg_spin: int,
    is_spin: bool,
) -> dict:
    """Return the kwargs dict shared by ``SeZMPTHModel`` constructors.

    This helper eliminates the ~40-line duplication between
    ``test_freeze_roundtrip`` (test 2d) and
    ``test_torchscript_serialization_roundtrip`` (test 2h).

    Parameters
    ----------
    model
        An unwrapped SeZM model object.
    _get_model_ntypes, _model_has_message_passing, _to_py_list
        Callables obtained from ``_try_import_freeze_helpers``.
    dim_fparam, dim_aparam, dim_chg_spin
        Integer dimension values extracted from the model.
    is_spin
        Whether the model is a spin model.
    """
    return {
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
        "lower_input_kind": model.export_lower_input_kind(),
        "lower_nf": 1,
        "do_grad_r": bool(model.do_grad_r("energy")),
        "do_grad_c": bool(model.do_grad_c("energy")),
    }


# ---------------------------------------------------------------------------
# 2d: CPU freeze and round-trip test
# ---------------------------------------------------------------------------


def test_freeze_roundtrip(ckpt_path: str) -> None:
    """Load a SeZM checkpoint, freeze to .pth on CPU, reload, and verify.

    Requires ``--ckpt`` and the compiled ``deepmd.lib`` C extension.
    """
    helpers = _require_freeze_helpers()
    (
        _freeze_sezm_to_pth,
        _load_sezm_checkpoint,
        _model_has_spin,
        _resolve_nframes,
        _get_model_ntypes,
        _model_has_message_passing,
        LowerGraphNoParamAdapter,
        _to_py_list,
        _collect_metadata,
        SeZMPTHModel,
        SeZMSpinPTHModel,
    ) = helpers

    # Load the checkpoint.
    _state_dict, _params, model = _load_sezm_checkpoint(ckpt_path)
    is_spin = _model_has_spin(model)

    device = torch.device("cpu")

    # Build sample inputs.
    _, sample_inputs = _resolve_nframes(model, nloc=7, device=device, has_spin=is_spin)

    # Stage 1: trace the lower graph.
    fx_graph = model.forward_common_lower_exportable(*sample_inputs)
    dim_fparam = int(model.get_dim_fparam())
    dim_aparam = int(model.get_dim_aparam())
    dim_chg_spin = int(model.get_dim_chg_spin())

    adapter = LowerGraphNoParamAdapter(fx_graph).eval()
    lower_inputs = list(sample_inputs[:6])
    scripted_lower = torch.jit.trace(
        adapter, lower_inputs, strict=False, check_trace=True
    )

    # Stage 2: script the wrapper.
    common_kwargs = _build_common_kwargs(
        model,
        _get_model_ntypes=_get_model_ntypes,
        _model_has_message_passing=_model_has_message_passing,
        _to_py_list=_to_py_list,
        dim_fparam=dim_fparam,
        dim_aparam=dim_aparam,
        dim_chg_spin=dim_chg_spin,
        is_spin=is_spin,
    )

    if is_spin:
        if SeZMSpinPTHModel is None:
            pytest.skip(
                "SeZMSpinPTHModel is not available; spin .pth freeze "
                "has been deprecated."
            )
        wrapper = SeZMSpinPTHModel(scripted_lower, **common_kwargs)
    else:
        wrapper = SeZMPTHModel(scripted_lower, **common_kwargs)
    wrapper.eval()

    scripted = torch.jit.script(wrapper)
    assert isinstance(scripted, torch.jit.ScriptModule)

    # Verify accessor methods survive scripting.
    assert abs(scripted.get_rcut() - common_kwargs["rcut"]) < 1e-10
    assert scripted.get_ntypes() == common_kwargs["ntypes"]
    assert scripted.get_nnei() == common_kwargs["nnei"]

    # Round-trip: save, load, verify metadata.
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tf:
        tmp_path = tf.name
    try:
        torch.jit.save(scripted, tmp_path)
        loaded = torch.jit.load(tmp_path)

        # Metadata accessors after reload.
        assert abs(loaded.get_rcut() - common_kwargs["rcut"]) < 1e-10, (
            "Round-trip get_rcut() mismatch"
        )
        assert loaded.get_ntypes() == common_kwargs["ntypes"], (
            "Round-trip get_ntypes() mismatch"
        )
        assert loaded.get_nnei() == common_kwargs["nnei"], (
            "Round-trip get_nnei() mismatch"
        )
        assert loaded.get_sel() == common_kwargs["sel"], "Round-trip get_sel() mismatch"
        print(f"test_freeze_roundtrip: PASSED — model saved to {tmp_path}")  # noqa: T201
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# 2e: Eager-vs-frozen CPU numerical comparison
# ---------------------------------------------------------------------------


def test_eager_vs_frozen(ckpt_path: str) -> None:
    """Compare eager model outputs against frozen .pth model outputs.

    Requires ``--ckpt`` and the compiled ``deepmd.lib`` C extension.
    """
    helpers = _require_freeze_helpers()
    (
        freeze_sezm_to_pth,
        _load_sezm_checkpoint,
        _model_has_spin,
        _resolve_nframes,
        *_rest,
    ) = helpers

    _state_dict, _params, model = _load_sezm_checkpoint(ckpt_path)
    is_spin = _model_has_spin(model)
    device = torch.device("cpu")

    # Get eager outputs.
    _, sample_inputs = _resolve_nframes(model, nloc=7, device=device, has_spin=is_spin)
    with torch.no_grad():
        eager_out = model.forward_common_lower(*sample_inputs)

    # Freeze the model.
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tf:
        tmp_path = tf.name
    try:
        freeze_sezm_to_pth(ckpt_path, tmp_path, device="cpu")
        frozen_model = torch.jit.load(tmp_path)
        frozen_model.eval()

        # Build forward_lower inputs for the frozen model.
        ntypes = frozen_model.get_ntypes()
        nnei = frozen_model.get_nnei()
        rcut = frozen_model.get_rcut()
        nloc = 7
        nall = nloc + 2

        ext_coord = torch.rand(1, nall, 3, dtype=torch.float64, device=device) * rcut
        ext_atype = torch.randint(0, ntypes, (1, nall), device=device).to(torch.int64)
        nlist = torch.randint(0, nall, (1, nloc, nnei), device=device).to(torch.int64)
        mapping = torch.arange(nall, device=device).unsqueeze(0).to(torch.int64)

        dim_fparam = frozen_model.get_dim_fparam()
        dim_aparam = frozen_model.get_dim_aparam()
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
            dim_chg_spin = frozen_model.get_dim_chg_spin()
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

        with torch.no_grad():
            frozen_out = frozen_model.forward_lower(*args)

        # Compare outputs.
        common_keys = set(eager_out.keys()) & set(frozen_out.keys())
        assert common_keys, "No common output keys between eager and frozen"

        mismatches = []
        for key in sorted(common_keys):
            e = eager_out[key]
            f = frozen_out[key]
            if e.shape != f.shape:
                mismatches.append(
                    f"  {key}: shape eager={tuple(e.shape)} frozen={tuple(f.shape)}"
                )
                continue
            max_diff = float((e.float() - f.float()).abs().max())
            if max_diff > 1e-3:
                mismatches.append(f"  {key}: max abs diff={max_diff:.2e}")

        if mismatches:
            msg = "Eager vs frozen mismatch:\n" + "\n".join(mismatches)
            pytest.fail(msg)

        print("test_eager_vs_frozen: PASSED")  # noqa: T201
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# 2f: Shape-variation tests
# ---------------------------------------------------------------------------


def test_shape_variation(ckpt_path: str) -> None:
    """Run the frozen model on different nloc/nall/ghost configurations.

    Requires ``--ckpt`` and the compiled ``deepmd.lib`` C extension.
    """
    helpers = _require_freeze_helpers()
    (
        freeze_sezm_to_pth,
        _load_sezm_checkpoint,
        _model_has_spin,
        _resolve_nframes,
        *_rest,
    ) = helpers

    _state_dict, _params, model = _load_sezm_checkpoint(ckpt_path)
    is_spin = _model_has_spin(model)
    device = torch.device("cpu")

    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tf:
        tmp_path = tf.name
    try:
        freeze_sezm_to_pth(ckpt_path, tmp_path, device="cpu")
        frozen_model = torch.jit.load(tmp_path)
        frozen_model.eval()

        ntypes = frozen_model.get_ntypes()
        nnei = frozen_model.get_nnei()
        rcut = frozen_model.get_rcut()
        dim_fparam = frozen_model.get_dim_fparam()
        dim_aparam = frozen_model.get_dim_aparam()
        dim_chg_spin = frozen_model.get_dim_chg_spin()

        # Test various shapes.
        shape_configs = [
            {"nloc": 3, "nall": 3, "ghost_count": 0},
            {"nloc": 7, "nall": 9, "ghost_count": 2},
            {"nloc": 19, "nall": 25, "ghost_count": 6},
            {"nloc": 5, "nall": 5, "ghost_count": 0},
        ]

        for cfg in shape_configs:
            nloc = cfg["nloc"]
            nall = cfg["nall"]

            ext_coord = (
                torch.rand(1, nall, 3, dtype=torch.float64, device=device) * rcut
            )
            ext_atype = torch.randint(0, ntypes, (1, nall), device=device).to(
                torch.int64
            )
            nlist = torch.randint(0, nall, (1, nloc, nnei), device=device).to(
                torch.int64
            )
            mapping = torch.arange(nall, device=device).unsqueeze(0).to(torch.int64)

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

            with torch.no_grad():
                out = frozen_model.forward_lower(*args)

            assert "energy" in out, f"forward_lower failed for nloc={nloc}, nall={nall}"
            print(  # noqa: T201
                f"  nloc={nloc}, nall={nall}: energy shape={tuple(out['energy'].shape)} OK"
            )

        print("test_shape_variation: PASSED")  # noqa: T201
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Frozen .pth model runtime test (requires --model)
# ---------------------------------------------------------------------------


def test_model_runtime(model_path: str) -> None:
    """Load a frozen .pth and run forward_lower on CPU.

    Requires ``--model`` and the compiled ``deepmd.lib`` C extension.
    """
    _require_freeze_helpers()  # Ensure deepmd is importable

    model = torch.jit.load(model_path)
    model.eval()

    ntypes = model.get_ntypes()
    nnei = model.get_nnei()
    rcut = model.get_rcut()
    dim_fparam = model.get_dim_fparam()
    dim_aparam = model.get_dim_aparam()
    dim_chg_spin = model.get_dim_chg_spin()
    is_spin = model.has_spin()

    device = torch.device("cpu")
    nloc = 7
    nall = nloc + 2

    ext_coord = torch.rand(1, nall, 3, dtype=torch.float64, device=device) * rcut
    ext_atype = torch.randint(0, ntypes, (1, nall), device=device).to(torch.int64)
    nlist = torch.randint(0, nall, (1, nloc, nnei), device=device).to(torch.int64)
    mapping = torch.arange(nall, device=device).unsqueeze(0).to(torch.int64)

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

    with torch.no_grad():
        out = model.forward_lower(*args)

    assert "energy" in out, "Output missing 'energy' key"
    print(  # noqa: T201
        f"test_model_runtime: PASSED — energy shape={tuple(out['energy'].shape)}"
    )


# ---------------------------------------------------------------------------
# Shared helpers for ghost-atom inference tests
# ---------------------------------------------------------------------------


def _build_ghost_inputs(
    ntypes: int,
    nnei: int,
    rcut: float,
    dim_fparam: int,
    dim_aparam: int,
    dim_chg_spin: int,
    device: torch.device,
    *,
    nloc: int = 7,
    nf: int = 1,
) -> dict:
    """Build LAMMPS-style inputs with ghost atoms for inference testing.

    Returns a dict with keys:
    - nloc, nall, nf (int)
    - ext_coord, ext_atype, nlist, mapping (tensors)
    - fparam, aparam, chg_spin (tensors or None)
    """
    nall = nloc + 2  # 2 ghost atoms at indices nloc, nloc+1

    ext_coord = torch.rand(nf, nall, 3, dtype=torch.float64, device=device) * rcut
    ext_atype = torch.randint(0, ntypes, (nf, nall), device=device).to(torch.int64)
    nlist_t = torch.randint(0, nall, (nf, nloc, nnei), device=device).to(torch.int64)
    # Ghost atoms at indices nloc, nloc+1 owned by local atoms 1, 4.
    identity = list(range(nloc))
    ghost_sources = [1, 4]
    mapping = torch.tensor([identity + ghost_sources], dtype=torch.int64, device=device)

    fparam = (
        torch.zeros(nf, dim_fparam, dtype=torch.float64, device=device)
        if dim_fparam > 0
        else None
    )
    aparam = (
        torch.zeros(nf, nloc, dim_aparam, dtype=torch.float64, device=device)
        if dim_aparam > 0
        else None
    )
    chg_spin = (
        torch.zeros(nf, dim_chg_spin, dtype=torch.float64, device=device)
        if dim_chg_spin > 0
        else None
    )

    return {
        "nloc": nloc,
        "nall": nall,
        "nf": nf,
        "ext_coord": ext_coord,
        "ext_atype": ext_atype,
        "nlist": nlist_t,
        "mapping": mapping,
        "fparam": fparam,
        "aparam": aparam,
        "chg_spin": chg_spin,
    }


def _verify_frozen_outputs(
    outputs: dict[str, torch.Tensor],
    *,
    label: str = "frozen model",
    required_keys: tuple[str, ...] = ("energy", "atom_energy", "force"),
) -> None:
    """Verify that outputs contain required keys with finite values."""
    for required_key in required_keys:
        assert required_key in outputs, (
            f"Required output key '{required_key}' missing from {label} output"
        )
        assert torch.isfinite(outputs[required_key]).all(), (
            f"Output '{required_key}': contains non-finite values in {label}"
        )


# ---------------------------------------------------------------------------
# 2g: CPU freeze → save → load → inference with ghost atoms
# ---------------------------------------------------------------------------


def test_freeze_cpu_inference_with_ghosts(ckpt_path: str) -> None:
    """Freeze a SeZM checkpoint to .pth on CPU, reload, and verify
    inference correctness and output consistency with ghost atoms.

    This test validates that the freeze→save→load→inference pipeline
    produces correctly-shaped, finite, and internally-consistent outputs
    when the LAMMPS-style input includes ghost atoms with non-identity
    mapping.  ``atom_energy`` is verified to sum to ``energy``, and force
    shapes are checked.  Numerical correctness between eager and frozen
    models is verified by the serialization round-trip test (test 2h).

    The random inputs are seeded for deterministic reproducibility.

    Requires ``--ckpt`` and the compiled ``deepmd.lib`` C extension.
    """
    helpers = _require_freeze_helpers()
    (
        freeze_sezm_to_pth,
        _load_sezm_checkpoint,
        _model_has_spin,
        _resolve_nframes,
        *_rest,
    ) = helpers

    _state_dict, _params, model = _load_sezm_checkpoint(ckpt_path)
    is_spin = _model_has_spin(model)
    device = torch.device("cpu")

    if is_spin:
        pytest.skip(
            "test_freeze_cpu_inference_with_ghosts only supports non-spin "
            "models; spin freeze is not yet implemented for the legacy "
            ".pth path."
        )

    # Seeded for deterministic reproducibility across runs.
    torch.manual_seed(42)

    # Freeze to .pth on CPU.
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tf:
        tmp_path = tf.name
    try:
        freeze_sezm_to_pth(ckpt_path, tmp_path, device=device)
        frozen_model = torch.jit.load(tmp_path, map_location="cpu")
        frozen_model.eval()

        ntypes = frozen_model.get_ntypes()
        nnei = frozen_model.get_nnei()
        rcut = frozen_model.get_rcut()
        dim_fparam = frozen_model.get_dim_fparam()
        dim_aparam = frozen_model.get_dim_aparam()
        dim_chg_spin = frozen_model.get_dim_chg_spin()

        gi = _build_ghost_inputs(
            ntypes, nnei, rcut, dim_fparam, dim_aparam, dim_chg_spin, device
        )
        nloc = gi["nloc"]

        # --- Frozen model inference with ghost atoms ---
        frozen_args = (
            gi["ext_coord"],
            gi["ext_atype"],
            gi["nlist"],
            gi["mapping"],
            gi["fparam"],
            gi["aparam"],
            False,  # do_atomic_virial
            None,  # sw_proxy
            gi["chg_spin"],
        )
        with torch.no_grad():
            frozen_out = frozen_model.forward_lower(*frozen_args)

        # --- Output correctness assertions ---
        _verify_frozen_outputs(frozen_out, label="frozen model")

        # atom_energy must sum to energy (per-frame).
        energy = frozen_out["energy"]
        atom_energy = frozen_out["atom_energy"]
        assert energy.shape == (1, 1), (
            f"Expected energy shape (1, 1), got {tuple(energy.shape)}"
        )
        assert atom_energy.shape == (1, nloc, 1), (
            f"Expected atom_energy shape (1, {nloc}, 1), got {tuple(atom_energy.shape)}"
        )
        torch.testing.assert_close(
            atom_energy.sum(dim=1),
            energy,
            rtol=1e-5,
            atol=1e-8,
            msg="atom_energy does not sum to energy",
        )

        # Force must have correct shape: (nf, nloc, 3).
        force = frozen_out["force"]
        assert force.shape == (1, nloc, 3), (
            f"Expected force shape (1, {nloc}, 3), got {tuple(force.shape)}"
        )
        assert torch.isfinite(force).all(), "Force contains non-finite values"

        print(  # noqa: T201
            "test_freeze_cpu_inference_with_ghosts: PASSED"
        )
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# 2h: TorchScript serialization round-trip test
# ---------------------------------------------------------------------------


def test_torchscript_serialization_roundtrip(ckpt_path: str) -> None:
    """Build the wrapper, script it, save, reload, and verify
    ``forward_lower`` produces correct numerical outputs.

    This catches regressions in:
    - Optional Tensor ABI handling
    - ``dict[str, Tensor]`` return types
    - Exported method signatures
    - Metadata serialization

    Requires ``--ckpt`` and the compiled ``deepmd.lib`` C extension.
    """
    helpers = _require_freeze_helpers()
    (
        _,  # freeze_sezm_to_pth — unused in this test
        _load_sezm_checkpoint,
        _model_has_spin,
        _resolve_nframes,
        _get_model_ntypes,
        _model_has_message_passing,
        LowerGraphNoParamAdapter,
        _to_py_list,
        _,  # _collect_metadata — unused in this test
        SeZMPTHModel,
        SeZMSpinPTHModel,
    ) = helpers

    _state_dict, _params, model = _load_sezm_checkpoint(ckpt_path)
    is_spin = _model_has_spin(model)
    device = torch.device("cpu")

    if is_spin:
        if SeZMSpinPTHModel is None:
            pytest.skip(
                "SeZMSpinPTHModel is not available; spin .pth freeze "
                "has been deprecated."
            )

    # Build sample inputs.
    _, sample_inputs = _resolve_nframes(model, nloc=7, device=device, has_spin=is_spin)

    # Stage 1: trace the lower graph.
    fx_graph = model.forward_common_lower_exportable(*sample_inputs)
    dim_fparam = int(model.get_dim_fparam())
    dim_aparam = int(model.get_dim_aparam())
    dim_chg_spin = int(model.get_dim_chg_spin())

    adapter = LowerGraphNoParamAdapter(fx_graph).eval()
    lower_inputs = list(sample_inputs[:6])
    scripted_lower = torch.jit.trace(
        adapter, lower_inputs, strict=False, check_trace=True
    )

    # Stage 2: script the wrapper.
    common_kwargs = _build_common_kwargs(
        model,
        _get_model_ntypes=_get_model_ntypes,
        _model_has_message_passing=_model_has_message_passing,
        _to_py_list=_to_py_list,
        dim_fparam=dim_fparam,
        dim_aparam=dim_aparam,
        dim_chg_spin=dim_chg_spin,
        is_spin=is_spin,
    )

    if is_spin:
        wrapper = SeZMSpinPTHModel(scripted_lower, **common_kwargs)
    else:
        wrapper = SeZMPTHModel(scripted_lower, **common_kwargs)
    wrapper.eval()

    # Script → save → load round-trip.
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tf:
        tmp_path = tf.name
    try:
        scripted = torch.jit.script(wrapper)
        assert isinstance(scripted, torch.jit.ScriptModule), (
            "script() did not produce a ScriptModule"
        )

        torch.jit.save(scripted, tmp_path)
        loaded = torch.jit.load(tmp_path, map_location="cpu")
        loaded.eval()

        # Verify metadata accessors survive the round-trip.
        assert abs(loaded.get_rcut() - common_kwargs["rcut"]) < 1e-10, (
            "Round-trip get_rcut() mismatch"
        )
        assert loaded.get_ntypes() == common_kwargs["ntypes"], (
            "Round-trip get_ntypes() mismatch"
        )
        assert loaded.get_nnei() == common_kwargs["nnei"], (
            "Round-trip get_nnei() mismatch"
        )
        assert loaded.get_sel() == common_kwargs["sel"], "Round-trip get_sel() mismatch"

        # Build LAMMPS-style input with ghost atoms for inference.
        nnei_val = loaded.get_nnei()
        rcut_val = loaded.get_rcut()
        ntypes_val = loaded.get_ntypes()

        gi = _build_ghost_inputs(
            ntypes_val,
            nnei_val,
            rcut_val,
            dim_fparam,
            dim_aparam,
            dim_chg_spin,
            device,
        )
        nall = gi["nall"]
        nf = gi["nf"]

        if is_spin:
            ext_spin = torch.rand(nf, nall, 3, dtype=torch.float64, device=device)
            loaded_args = (
                gi["ext_coord"],
                gi["ext_atype"],
                ext_spin,
                gi["nlist"],
                gi["mapping"],
                gi["fparam"],
                gi["aparam"],
                False,
                None,
            )
        else:
            loaded_args = (
                gi["ext_coord"],
                gi["ext_atype"],
                gi["nlist"],
                gi["mapping"],
                gi["fparam"],
                gi["aparam"],
                False,
                None,
                gi["chg_spin"],
            )

        with torch.no_grad():
            loaded_out = loaded.forward_lower(*loaded_args)

        # Verify output keys and finite values.
        _verify_frozen_outputs(loaded_out, label="round-tripped model")

        # Also verify that the pre-save scripted model produces the
        # same outputs as the post-load model.
        with torch.no_grad():
            scripted_out = scripted.forward_lower(*loaded_args)

        common_keys = set(scripted_out.keys()) & set(loaded_out.keys())
        assert common_keys, "No common keys between pre-save and post-load"

        for key in sorted(common_keys):
            pre = scripted_out[key]
            post = loaded_out[key]
            torch.testing.assert_close(
                post.float(),
                pre.float(),
                rtol=1e-5,
                atol=1e-8,
                msg=(f"TorchScript save/load round-trip mismatch for '{key}'"),
            )

        print(  # noqa: T201
            "test_torchscript_serialization_roundtrip: PASSED"
        )
    finally:
        os.unlink(tmp_path)


# ===========================================================================
# Drift detection: verify inlined copies match production sources
# ===========================================================================


def _normalize_source(src: str) -> str:
    """Strip docstrings and normalize whitespace for comparison."""
    import ast

    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # Remove docstring (first expression statement that is a string)
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                node.body.pop(0)
    return ast.unparse(tree)


def _get_production_source(module_name: str, func_name: str) -> str:
    """Import a function from a production module and return its source."""
    import importlib
    import inspect

    mod = importlib.import_module(module_name)
    func = getattr(mod, func_name)
    return inspect.getsource(func)


def test_inlined_copies_match_production() -> None:
    """Verify the inlined copies in this file match their production sources.

    Skipped when the C extension (``deepmd.lib``) is not available,
    because the production modules cannot be imported without it.
    """
    import importlib

    try:
        import deepmd  # noqa: F401

        # Verify the C extension is actually usable by importing a submodule
        # that directly requires deepmd.lib.
        importlib.import_module("deepmd.pt_expt.utils.edge_schema")
    except ImportError as exc:
        import logging

        logging.warning(
            "Skipping inlined-copies test: deepmd C extension not usable "
            "(%s). This is expected in pure-Python CI environments.",
            exc,
        )
        pytest.skip("deepmd C extension not available; cannot compare sources.")

    import inspect

    # Map of (inlined_function, production_module, production_name).
    # We use the inlined functions defined at module level above.
    checks: list[tuple[object, str, str]] = [
        (
            _append_dummy_edges,
            "deepmd.pt_expt.utils.edge_schema",
            "_append_dummy_edges",
        ),
        (
            edge_schema_from_extended,
            "deepmd.pt_expt.utils.edge_schema",
            "edge_schema_from_extended",
        ),
        (
            _build_edge_schema_ts,
            "deepmd.pt.entrypoints.freeze_pt2",
            "_build_edge_schema_ts",
        ),
    ]

    mismatches: list[str] = []

    for inlined_fn, mod_name, fn_name in checks:
        try:
            prod_source = _get_production_source(mod_name, fn_name)
        except Exception as exc:
            mismatches.append(f"  {mod_name}.{fn_name}: could not get source: {exc}")
            continue

        inlined_source = inspect.getsource(inlined_fn)

        try:
            norm_prod = _normalize_source(prod_source)
            norm_inlined = _normalize_source(inlined_source)
        except SyntaxError as exc:
            mismatches.append(
                f"  {mod_name}.{fn_name}: syntax error during normalization: {exc}"
            )
            continue

        if norm_prod != norm_inlined:
            # Compute a simple diff for the error message.
            prod_lines = norm_prod.splitlines()
            inlined_lines = norm_inlined.splitlines()
            max_len = max(len(prod_lines), len(inlined_lines))
            diff_lines = []
            for i in range(max_len):
                pl = prod_lines[i] if i < len(prod_lines) else "<missing>"
                il = inlined_lines[i] if i < len(inlined_lines) else "<missing>"
                if pl != il:
                    diff_lines.append(f"    line {i + 1}:")
                    diff_lines.append(f"      prod: {pl}")
                    diff_lines.append(f"      test: {il}")
            mismatches.append(
                f"  {mod_name}.{fn_name}: source has diverged:\n"
                + "\n".join(diff_lines[:30])
                + ("\n    ..." if len(diff_lines) > 30 else "")
            )

    if mismatches:
        msg = (
            "Inlined copies have diverged from production sources!  "
            "Update the inlined functions in this test file and the "
            "'Last synced' comment at the top of the block.\n" + "\n".join(mismatches)
        )
        pytest.fail(msg)

    print("test_inlined_copies_match_production: PASSED")  # noqa: T201

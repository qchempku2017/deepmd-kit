# Install and run on legacy NVIDIA GPUs (e.g. Tesla P100)

:::{note}
This page covers NVIDIA GPUs that predate Ampere. There are **two independent
capability tiers** to keep straight:

1. **bfloat16 (bf16) / TF32 tier — requires Ampere (`sm_80+`).** Pascal
   (`sm_60`, e.g. Tesla P100), Volta (`sm_70`, e.g. V100) and Turing (`sm_75`,
   e.g. RTX 20 / T4) all lack native bf16, so DPA-4 bfloat16 autocast is
   auto-disabled there.
2. **Triton / `torch.compile` / `.pt2` freeze tier — requires Volta
   (`sm_70+`).** Triton (shipped with PyTorch 2.11/2.12) supports Volta and
   newer but **cannot compile for Pascal (`sm_60`)** — AOTInductor is confirmed
   broken there. However, DPA-4 now has a **`.pth` (TorchScript) freeze path**
   (`dp --pt freeze --legacy-gpu`) that covers LAMMPS inference on Pascal
   without Triton (see the Quick Start below).

So a **Volta/Turing** user only needs point 1 (bf16 stays off, everything else
works). A **Pascal (P100)** user needs bf16 off and the `.pth` freeze path
for LAMMPS; training and `dp --pt test` still use the dense eager path.
:::

> **⚠️ Deprecation Notice — Spin Model `.pth` Freeze Dropped**
>
> As of this update, **spin model freezing to `.pth` (TorchScript) is no longer
> supported** via the `dp --pt freeze --legacy-gpu` command. The
> `SeZMSpinPTHModel` wrapper class has been disabled, and the `--legacy-gpu`
> flag's `.pth` freeze path no longer handles spin models. Only **energy
> (non-spin) models** can be frozen to `.pth` through this path.
>
> Spin model **`.pt2` (AOTInductor) freezing** via `dp --pt freeze` (without
> `--legacy-gpu`) **continues to work on Volta (`sm_70`) and newer GPUs**,
> including the `DeepSpinPTExpt` C++ loader in LAMMPS. Pascal (`sm_60`) users
> who need spin model deployment must upgrade to a Volta+ GPU.

## Quick Start for P100/Pascal GPUs

:::{tip}
**Good news:** DPA-4 now supports LAMMPS on P100/Pascal GPUs through a new
`.pth` (TorchScript) freeze path. You no longer need Volta+ to run DPA-4
in LAMMPS.
:::

Here is the complete workflow from training to LAMMPS on a P100 or other
Pascal GPU:

```bash
# Step 1: Train normally (works on any GPU)
dp --pt train input.json

# Step 2: Freeze with --legacy-gpu flag (produces .pth instead of .pt2)
# Note: only energy (non-spin) models are supported
dp --pt freeze --legacy-gpu -c checkpoint.pt -o frozen_model.pth

# Step 3: Use in LAMMPS
# In LAMMPS input:
#   pair_style deepmd frozen_model.pth
```

The `--legacy-gpu` flag converts the model to TorchScript (`.pth`) instead of
AOTInductor (`.pt2`). The `.pth` file works with LAMMPS via the ``DeepPotPT``
loader for **energy models only**; spin model freezing is no longer supported
through this path. Performance is about 1.5–3× slower than `.pt2` on
Volta+ GPUs, but the model remains fully GPU-accelerated and does not fall
back to CPU.

If you are on Volta (`sm_70`) or newer, use the standard `.pt2` freeze path
(`dp --pt freeze -c checkpoint.pt -o frozen_model.pt2` without `--legacy-gpu`)
for the best performance.

## 1. Why pre-Ampere GPUs need care

DPA-4 / SeZM is split, by design, into two execution paths:

1. A **dense reference path** — pure PyTorch eager ops (`bmm`, `einsum`,
   `index_add`, `matmul`, `softmax`). It is **architecture-agnostic** and runs
   on any CUDA device that PyTorch itself supports, including `sm_60`.
   **Training always uses this path.**
2. A set of **opt-in inference-only accelerated paths** — Triton fused
   kernels (`DP_TRITON_INFER`), a CuTe fused value-path (`DP_CUTE_INFER`),
   bfloat16 autocast (`DP_AMP_INFER` / `descriptor.use_amp`), TF32
   (`DP_TF32_INFER`), and the `torch.compile` / AOTInductor `.pt2` freeze
   (`DP_COMPILE_INFER`). The bf16/TF32 paths need Ampere (`sm_80+`); the
   Triton/compile/`.pt2` paths need Volta (`sm_70+`).

The defaults already select the dense path (`DP_TRITON_INFER=0`,
`DP_CUTE_INFER=0`, `DP_COMPILE_INFER` unset). The one trap is that
**`descriptor.use_amp` defaults to `true`** and bfloat16 errors on cards
without native bf16 — DeePMD-kit now **auto-disables it with a one-time
warning** on pre-Ampere GPUs.

## 2. Capability tiers at a glance

| Family | Examples | Compute capability | native bf16 | `.pt2` / Triton / compile | `.pth` freeze / LAMMPS | What applies |
|--------|----------|-------------------|-------------|---------------------------|------------------------|--------------| | Pascal | Tesla P100, GTX 10 | `sm_60`/`sm_62` | **no** | **no** | **✅ yes** (energy only) | bf16 auto-off; `.pth` freeze → LAMMPS (energy only, spin not supported); dense eager for train/test |
| Volta | V100, Titan V | `sm_70`/`sm_72` | **no** | **yes** | yes (`.pt2` preferred) | bf16 auto-off; `.pt2`/compile/LAMMPS work |
| Turing | RTX 20, T4 | `sm_75` | **no** | **yes** | yes (`.pt2` preferred) | bf16 auto-off; `.pt2`/compile/LAMMPS work |
| Ampere | A100, RTX 30 | `sm_80`/`sm_86` | yes | yes | yes (`.pt2` preferred) | fully supported (all paths) |
| Ada/Hopper | RTX 40, H100 | `sm_89`/`sm_90` | yes | yes | yes (`.pt2` preferred) | fully supported |

DeePMD-kit queries `torch.cuda.get_device_capability()` and uses `cap >= 8`
for the bf16/TF32 axis and `cap >= 7` for the Triton/AOTInductor axis. Both
checks are **no-ops on CPU**, so CPU-only runs and tests are unaffected.

## 3. Install

{{ pytorch_icon }} Use the **PyTorch** backend (`dp --pt`). Install a CUDA
build of PyTorch whose wheel still ships kernels for your architecture, then
install DeePMD-kit.

```bash
# 1. Create an environment (Python 3.10+)
python -m venv venv && source venv/bin/activate

# 2. Install a CUDA PyTorch wheel that includes kernels for your GPU.
#    - For Pascal (P100, sm_60): pick a PyTorch release whose CUDA wheel still
#      targets sm_60 (newer wheels may have dropped Pascal cubins -- run the
#      smoke test in section 6 before training).
#    - For Volta/Turing/Ampere: any recent CUDA wheel works.
pip install torch --index-url https://download.pytorch.org/whl/cu126

# 3. Install DeePMD-kit. DPA-4's core descriptor math uses no custom C++ ops
#    (only the MPI border-exchange op, for multi-rank LAMMPS), so the CUDA
#    variant of the C++ interface is optional for DPA-4 Python training.
pip install deepmd-kit[cu12,torch]
```

:::{important}
DeePMD-kit's `torch.compile`/AOTInductor support is gated to PyTorch **2.11.x
or 2.12.x**. Those releases bundle a Triton that **cannot compile for Pascal
(`sm_60`)** (AOTInductor is confirmed broken on P100). This blocks the
*optimized/compiled* paths on Pascal only; Volta/Turing/Ampere are unaffected.
On Pascal the dense eager path (used by `dp --pt` training, `dp --pt test`,
and the ASE calculator) does **not** use Triton and is unaffected.
:::

If you build the C++ interface from source and want a smaller/faster build
restricted to a single architecture, set the CUDA architecture explicitly
(e.g. for Pascal):

```bash
DP_VARIANT=cuda \
CMAKE_CUDA_ARCHITECTURES=60 \
uv pip install -e .[cpu,test]
```

(The default `CMAKE_CUDA_ARCHITECTURES` is `all` / `all-major`, which already
includes `sm_60` through `sm_90+`, so this override is optional.) The classic
DeePMD C++ custom operators contain no tensor cores / bf16 / `wmma` and
compile for `sm_60`, so older DP / DPA-1/2/3 models keep working on Pascal.

## 4. Train a DPA-4 model

The single most important setting on a pre-Ampere GPU is to keep bfloat16 off
(no native bf16). DeePMD-kit does this automatically, but it is clearest to
state it in the input:

```json
{
  "model": {
    "type": "SeZM",
    "descriptor": { "type": "sezm", "use_amp": false },
    ...
  }
}
```

Then train with the PyTorch backend:

```bash
dp --pt train input.json
```

Recommended settings per architecture:

| Option | Pascal (`sm_60`) | Volta/Turing (`sm_70/75`) | Reason |
|--------|------------------|----------------------------|--------|
| `descriptor.use_amp` | `false` | `false` | no native bf16 (auto-disabled) |
| `model.use_compile` | `false` (default) | OK if desired | compile lowers through Triton; unsupported on Pascal, works on Volta+ |
| `model.enable_tf32` | any (no-op) | any (no-op) | no TF32 hardware below Ampere; harmless |
| `DP_TRITON_INFER` | `0` (default) | OK `1-3` | Triton unsupported on Pascal; works on Volta+ |
| `DP_CUTE_INFER` | `0` (default) | `0` | CuTe path unverified on Pascal |
| `DP_COMPILE_INFER` | unset (default) | OK if desired | no `torch.compile` on Pascal |
| `DP_AMP_INFER` | `0` (default) | `0` | no inference bf16 autocast |

:::{note}
On any pre-Ampere GPU (Pascal/Volta/Turing) the descriptor's default
`use_amp=true` is automatically downgraded to fp32 at runtime: bfloat16
autocast is disabled with a one-time warning, so **`dp --pt train` runs in
fp32 by default** on these cards. You do not have to set `use_amp=false`, but
doing so silences the warning and skips the per-forward capability check.
This is a deliberate change from older behavior (which attempted bf16 and
errored/degraded on cards without native bf16).
:::

:::{note}
The "works on Volta/Turing" claims for the `.pt2` freeze and LAMMPS DPA-4
rest on direct testing of **Volta V100 (`sm_70`)** only. Turing (`sm_75`) and
Volta `torch.compile` are *expected* to work (Triton supports `sm_70`/`sm_75`,
and the Inductor lowering is the same), but are not separately verified — run
the [smoke test](#6-smoke-test-does-my-pytorch-wheel-support-my-gpu) once on
your card before relying on them.
:::

Force training (backward/force/virial) uses the dense PyTorch
`index_add`/`einsum` assembly and needs no Triton, so force-loss training works
on Pascal.

## 5. Inference with `dp --pt test` and the ASE calculator

For inference, **load the `.pt` training checkpoint directly** — it runs in
pure eager mode (DPA-4 always disables TorchScript/AOTInductor for `.pt`
loading). No freeze step is required, on any architecture.

```bash
# Test / infer from a checkpoint
dp --pt test -m model.ckpt.pt -s system/
```

ASE calculator — point `DP` at the `.pt` checkpoint:

```python
from deepmd.calculator import DP

calc = DP(model="model.ckpt.pt")
# use calc.get_potential_energy(atoms), calc.get_forces(atoms), etc.
```

This path is **Triton-free and architecture-agnostic**: with all the
environment variables at their defaults, DPA-4 inference uses only standard
PyTorch ops and runs on `sm_60` (and any newer card) as long as the PyTorch
wheel includes kernels for that architecture.

## 6. Smoke test: does my PyTorch wheel support my GPU?

Before a long run, confirm the PyTorch wheel ships kernels for your card
(most important on Pascal, where recent wheels may have dropped `sm_60`):

```python
import torch
print("capability:", torch.cuda.get_device_capability())   # e.g. (6, 0) for P100
x = torch.randn(512, 512, device="cuda")
y = torch.matmul(x, x)                                    # must NOT raise
print("matmul OK on", torch.cuda.get_device_name())
```

And confirm DeePMD-kit's capability detection:

```python
from deepmd.kernels.utils import (
    cuda_supports_bf16,
    cuda_supports_triton,
    gpu_capability_description,
)
print(gpu_capability_description())
# P100:  "NVIDIA Tesla P100 ... (sm_60, native bf16: no)"
# V100:  "NVIDIA Tesla V100 ... (sm_70, native bf16: no)"
print("bf16:", cuda_supports_bf16())    # False below Ampere
print("triton:", cuda_supports_triton())  # False on Pascal, True on Volta+
```

If `torch.matmul` raises `no kernel image is available for executing on the
device`, the PyTorch wheel lacks cubins for your architecture — install a
CUDA wheel that still targets it.

## 7. Boundaries and limitations

### Pascal (`sm_60`, e.g. Tesla P100)

What **works**:

- ✅ DPA-4 training (forward + backward + force + virial) via `dp --pt train`
  on the dense float32 eager path.
- ✅ DPA-4 inference via `dp --pt test` and the ASE calculator, loading the
  `.pt` checkpoint eagerly.
- ✅ `dp --pt freeze --legacy-gpu` — converts the model to TorchScript `.pth`
  format, which works in LAMMPS on any CUDA GPU (see
  [Quick Start](#quick-start-for-p100pascal-gpus)).
- ✅ DPA-4 inference in **LAMMPS** via the `.pth` frozen model:
  - **Energy models**: use ``pair_style deepmd frozen_model.pth`` with the
    standard ``DeepPotPT`` loader.
  - **Spin models**: no longer supported by the `.pth` freeze path on Pascal.
    The ``SeZMSpinPTHModel`` wrapper has been disabled. Spin model `.pt2`
    (AOTInductor) freezing works on Volta (`sm_70`) and newer GPUs via
    ``dp --pt freeze`` (without ``--legacy-gpu``).
- ✅ Checkpoint save / resume (`model.ckpt.pt`).
- ✅ Older DP / DPA-1 / DPA-2 / DPA-3 models — unaffected (their C++ ops
  compile for `sm_60`).

What **does not work** on Pascal:

- ❌ `dp --pt freeze` (without `--legacy-gpu`) of a DPA-4 checkpoint to `.pt2`
  — AOTInductor lowers through Triton, which cannot compile for `sm_60`. The
  standard freeze now **fails fast** with an actionable message. Use
  `--legacy-gpu` for Pascal instead.
- ❌ DPA-4 **spin model .pth freeze** (``--legacy-gpu``) — spin models (both
  ``nlist`` ABI / virtual-spin deepspin scheme and ``edge_vec`` ABI / native
  spin) cannot be frozen to ``.pth`` on Pascal; the ``SeZMSpinPTHModel``
  wrapper has been disabled. Spin model **.pt2 (AOTInductor) freeze** works on
  Volta+ GPUs.
- ❌ ``dp --pt freeze`` of a spin model DPA-4 checkpoint to ``.pt2`` on Pascal
  — same AOTInductor/Triton limitation as energy models (see above). Use a
  Volta+ GPU for spin ``.pt2`` freeze.
- ❌ `torch.compile` (`model.use_compile` / `DP_COMPILE_INFER`) — lowers through
  Triton; unsupported on `sm_60`. Setting either on Pascal now raises a clear
  `RuntimeError`.
- ❌ Triton fused kernels (`DP_TRITON_INFER >= 1`) — unsupported on `sm_60`.
  Level 3 additionally needs FP16 tensor cores (`sm_70+`). A warning is
  emitted at model construction recommending `DP_TRITON_INFER=0`.
- ❌ CuTe fused value-path (`DP_CUTE_INFER`) — unverified on `sm_60`; keep off.
- ❌ bfloat16 autocast (`descriptor.use_amp=true` / `DP_AMP_INFER=1`) — no
  native bf16; auto-disabled with a warning.

### Volta (`sm_70`) and Turing (`sm_75`)

What **works**: everything — DPA-4 training, inference, ASE, **and**
`dp --pt freeze` to `.pt2` + LAMMPS DPA-4 inference. (`.pt2`/LAMMPS on Volta
V100 `sm_70` is confirmed by testing; Turing `sm_75` and Volta `torch.compile`
are expected — Triton supports `sm_70`/`sm_75` and the Inductor lowering is the
same — but run the [smoke test](#6-smoke-test-does-my-pytorch-wheel-support-my-gpu)
once before relying on them.) FP16 tensor cores exist, so `DP_TRITON_INFER=3`
is also available.

The **only** caveat: **no native bfloat16**, so `descriptor.use_amp` /
`DP_AMP_INFER` must stay off (auto-disabled; training runs in fp32 by default).
Use the fp32 dense path or the fp32 `.pt2` for inference. `torch.compile`
(`model.use_compile`) works but, since it also lowers through Triton, run the
section-6 smoke test once to confirm your Triton build is healthy on your card.

### Overriding the guards (escape hatch)

If you have a custom Triton build that *does* support Pascal, bypass the
fail-fast guards:

- `DP_FREEZE_FORCE_AOTI=1` — attempt the `.pt2` AOTInductor freeze on Pascal.
- The `torch.compile` guard has no override by design, because an unsupported
  compile path produces broken artifacts; use a Volta+ GPU instead.

:::{note}
**Performance note:** The `.pth` (TorchScript) freeze path is typically
1.5–3× slower than `.pt2` (AOTInductor) on Volta+ GPUs because it lacks
Triton-fused kernels. However, it is still fully GPU-accelerated (no CPU
fallback) and is the only freeze option for Pascal. On Volta+ GPUs, use
the standard `dp --pt freeze` (`.pt2`) for the best performance.
:::

## 8. How the `.pth` freeze path works

The `--legacy-gpu` flag triggers a different freeze pipeline:

1. **`make_fx` materializes `autograd.grad`**: During a standard forward pass,
   DPA-4/SeZM computes forces via `torch.autograd.grad(create_graph=True)`.
   TorchScript cannot trace through `autograd.grad` directly. The `.pth` freeze
   uses `torch.fx.experimental.proxy_tensor.make_fx` to trace the forward pass
   and materialize the gradient computation into pure PyTorch operations
   (`bmm`, `einsum`, `index_add`, etc.) — no `autograd.grad` nodes remain.

2. **TorchScript traces the FX graph**: The resulting flat graph of standard
   PyTorch ops is traced into TorchScript (`.pth`). Because every operation is
   a plain PyTorch op, the trace works on any CUDA architecture that PyTorch
   supports, including `sm_60`.

3. **No Triton required**: Unlike the `.pt2` (AOTInductor) path, the `.pth`
   path never invokes Triton or `torch.compile`. This is what makes it work on
   Pascal GPUs.

4. **Environment variable ``DP_FREEZE_FORCE_PTH``**: Set ``DP_FREEZE_FORCE_PTH=1``
   to force the `.pth` freeze path even on Volta+ GPUs (suppresses the
   performance warning). This is useful for debugging or when you need
   TorchScript portability across heterogeneous GPU clusters.

5. **Energy model wrapper**: The `.pth` freeze produces a TorchScript module
   for **energy models only** (spin model `.pth` freezing is no longer
   supported):

   - **Energy models** → ``SeZMPTHModel`` with a ``forward_lower`` that matches
     the ``DeepPotPT`` contract (``pair_style deepmd`` in LAMMPS): ``(coord,
     atype, nlist, mapping, ...)``. The wrapper converts the LAMMPS neighbour
     list to an edge schema before calling the traced lower graph.

   The previously available ``SeZMSpinPTHModel`` wrapper (for spin models via
   ``pair_style deepmd/spin`` and ``DeepSpinPT``) has been **disabled**. Spin
   models can no longer be frozen to ``.pth`` through this path. However, spin
   model **.pt2 (AOTInductor) freezing** via ``dp --pt freeze`` (without
   ``--legacy-gpu``) continues to work on Volta+ GPUs.

The trade-off is that Triton-fused kernels (used in the `.pt2` path) are
faster than the equivalent sequence of individual PyTorch ops in the `.pth`
path. On Volta+ GPUs, prefer `.pt2`; on Pascal, `.pth` is the only option.

## 9. Troubleshooting the `.pth` freeze

### "TorchScript trace failed"

If `dp --pt freeze --legacy-gpu` fails with a TorchScript trace error, the
most common cause is an unsupported operation in the model's forward pass.
Check that:

- You are using a clean training checkpoint (not a partially compiled or
  Triton-modified model). Reload from `model.ckpt.pt` if needed.
- `DP_COMPILE_INFER` is unset and `model.use_compile` is `false`.
- `DP_TRITON_INFER` is `0` (the default).

### Verify the frozen `.pth` model

Before using the frozen model in LAMMPS, test it with `dp --pt test`:

```bash
dp --pt test -m frozen_model.pth -s system/
```

This should produce energies and forces that match the original checkpoint
within numerical round-off. If the test fails or produces NaN, re-freeze from
a clean checkpoint.

### LAMMPS loads the model but crashes at the first step

This usually means the pair coefficient mapping in LAMMPS does not match the
model's `type_map`. Verify with:

```bash
dp --pt show frozen_model.pth
```

The printed `type_map` must match the element order in the LAMMPS
`pair_coeff` line. For example, if the model's type map is `["O", "H"]`,
the LAMMPS input must be:

```lammps
pair_style deepmd frozen_model.pth
pair_coeff * * O H
```

## 10. Why this design

The dense float32 reference path is the same one DeePMD-kit uses to validate
the optimized kernels for numerical correctness, so running it on a legacy GPU
gives the **same physics** as a modern GPU (modulo floating-point
reduction-order differences). There is no accuracy penalty for using the
legacy path — only a speed penalty relative to the fused Triton/CuTe kernels
on Ampere+. For MD workflows sensitive to a smooth potential-energy surface,
keep `DP_TF32_INFER=0` and `DP_AMP_INFER=0` (the defaults).

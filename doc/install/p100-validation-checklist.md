# P100/Pascal GPU Validation Checklist

This checklist documents the steps required to validate the experimental `.pth` TorchScript freeze path on P100/Pascal-class GPUs. These steps have NOT been performed in the current development environment (which is CPU-only).

## Prerequisites

- A system with a Pascal-class NVIDIA GPU (P100, GTX 1080, etc.)
- CUDA toolkit compatible with your GPU
- PyTorch with CUDA support (must include `sm_60` kernels)
- DeepMD-kit built with PyTorch backend
- A trained DPA4/SeZM energy checkpoint (non-spin, no fparam/aparam/charge_spin)

## Freeze

```bash
dp --pt freeze --legacy-gpu -c checkpoint.pt -o frozen_model.pth
```

## Validation Steps

### 1. Basic Loading and Metadata

- [ ] Load the `.pth` on P100: `model = torch.jit.load("frozen_model.pth", map_location="cuda")`
- [ ] Verify `model.get_rcut()` returns expected cutoff
- [ ] Verify `model.get_ntypes()` returns expected number of types
- [ ] Verify `model.get_sel()` returns expected neighbor selection
- [ ] Verify `model.has_message_passing()` returns True
- [ ] Verify `model.mixed_types` returns True

### 2. Forward Lower (Single Frame)

- [ ] Call `model.forward_lower(coord, atype, nlist, mapping, ...)` on GPU
- [ ] Verify energy is a finite scalar
- [ ] Verify forces are finite
- [ ] Verify virial is finite
- [ ] Compare against eager checkpoint outputs (energy, forces, virial)
- [ ] Check atom_energy if atomic=true

### 3. LAMMPS Single-Rank

- [ ] Run `lmp -in in.lammps` with `pair_style deepmd frozen_model.pth`
- [ ] `run 0` — verify energy matches expectations, no NaN
- [ ] `run 1` — verify energy and forces after one step, no NaN
- [ ] `run 100` — short MD trajectory, verify no NaN, no index errors, no CUDA kernel failures

### 4. Multi-Rank LAMMPS (MPI)

- [ ] Run `mpirun -np 2 lmp -in in.lammps` with `pair_style deepmd frozen_model.pth`
- [ ] Verify energy conservation across MPI ranks
- [ ] Verify force consistency across processor boundaries
- [ ] Check for any `comm_dict` related errors

### 5. Edge Cases

- [ ] Test with systems containing ghost atoms
- [ ] Test with varying numbers of atoms per processor
- [ ] Test with periodic boundary conditions
- [ ] Test with non-orthogonal simulation boxes

## Known Limitations

- Spin models are NOT supported in `.pth` format (use `.pt2` on Volta+)
- fparam/aparam/charge_spin parameters are NOT supported
- Multi-rank correctness has NOT been verified
- `comm_dict` support is NOT implemented
- Performance may be lower than `.pt2` on Volta+ GPUs

## Reporting Results

Please report validation results (pass/fail with details) to the DeepMD-kit development team.

# CLAUDE.md — AI Assistant Guide for GKBA

This file provides guidance for AI assistants (Claude Code, Copilot, etc.) working in this repository.

---

## Project Overview

**GKBA** is a Julia package for simulating quantum transport in open systems using variants of the Generalized Kadanoff-Baym Ansatz (GKBA). It solves equations of motion for the lesser Green function G<(t) of a tight-binding central region (1D chain, nx×ny sites) coupled to fermionic leads, with Rashba spin-orbit coupling, s-d exchange with classical local moments, and optional Landau-Lifshitz-Gilbert (LLG) spin dynamics.

The GKBA reduces the two-time Kadanoff-Baym equations (O(Nt²) cost) to a set of coupled ODEs (O(Nt)), enabling long real-time simulations of spin-transfer torque, spin pumping, and non-equilibrium magnetization dynamics over physically relevant timescales.

**Key references:**
- PRB 107, 155141 (2023)
- PRL 130, 246301 (2023)

---

## Repository Structure

```
GKBA/
├── Project.toml          # Julia package manifest and dependencies
├── README.md             # Physics background, equations, usage examples
├── src/                  # Main package source code (8 modules, ~1,438 lines)
│   ├── GKBA.jl           # Package entry point; imports all submodules; exports 28 symbols
│   ├── types.jl          # Dynamics types + ObservablesVar + LLGParams structs
│   ├── constants.jl      # Physical constants (MBOHR, KB_EV, Pauli matrices)
│   ├── hamiltonians.jl   # Hamiltonian builders: build_hs, build_heα, build_hseα, Ozaki poles
│   ├── eom.jl            # Equations of motion for all 5 dynamics types + unpack! helper
│   ├── observables.jl    # compute_observables! for currents and spin/charge densities
│   ├── llg.jl            # Landau-Lifshitz-Gilbert spin dynamics: PrecSpin, update!, heun
│   └── init.jl           # 5 initialization functions returning (dv, ov) tuples
├── test/
│   ├── runtests.jl       # 18 unit tests (5 test sets, one per dynamics type)
│   └── benchmark.jl      # Performance benchmarks and reference solver comparison
├── scripts/              # 16 driver scripts for running simulations (see below)
├── notebooks/
│   ├── example.ipynb     # End-to-end WBL example: init, time evolution, plots, sanity checks
│   └── explore_results.ipynb  # Notebook for analyzing data/ output files
└── legacy/               # 18 archived original per-simulation notebooks (read-only)
```

---

## Physics Background (brief)

The central system is a 1D tight-binding chain with Hamiltonian:

```
H_s = -γ Σ_{<ij>} c†_i c_j  +  γ_so (Rashba terms)  -  j_sd Σ_i m_i · σ_i
```

Two semi-infinite leads (also 1D tight-binding, hopping γ, coupling γc) are attached to the first and last sites. The system is driven out of equilibrium by a bias voltage (different chemical potentials μ_α per lead). The classical moments m_i evolve under the LLG equation, driven by the quantum spin-transfer torque computed from G<(t).

---

## Technology Stack

- **Language:** Julia (scientific computing)
- **Package Manager:** Julia built-in (`Project.toml`; `Manifest.toml` is git-ignored)
- **Core Dependencies:**
  - `DifferentialEquations` — ODE solvers (Vern7, RK4, etc.)
  - `Tullio` — Einstein summation macro for tensor contractions (`@tullio`)
  - `LinearAlgebra` — Eigen decomposition, matrix operations (stdlib)
  - `DelimitedFiles` — Writing simulation results to `.txt` files (stdlib)

---

## Running the Project

### Install dependencies

```bash
julia --project -e 'using Pkg; Pkg.instantiate()'
```

### Run unit tests

```bash
julia --project test/runtests.jl
```

### Run benchmarks

```bash
julia --project test/benchmark.jl
```

### Run a simulation script

```bash
julia --project scripts/run_gkba_wbl_prece.jl
```

Output is written to `data/` (created at runtime, not tracked by git).

---

## The 5 Dynamics Types

All dynamics types live in `src/types.jl` and are distinguished by how lead Green functions are represented:

| Type | Julia struct | Lead representation | Green functions |
|------|-------------|---------------------|-----------------|
| k-rep GKBA | `GKBADynamics` | k-space, static | Analytic equilibrium gl |
| k-rep eGKBA | `eGKBADynamics` | k-space, dynamic | Dynamic lead correlators |
| pos-rep GKBA | `PosRepDynamics` | Position space | Full matrix leads |
| pos-rep eGKBA | `ePosRepDynamics` | Position space | Dynamic lead correlators |
| WBL GKBA | `WBLDynamics` | Wide-band limit | Ozaki pole expansion |

---

## Simulation Workflow

All scripts follow this pattern. The canonical working example is `notebooks/example.ipynb` (WBL, precessing spins):

```julia
using GKBA, DifferentialEquations

# 1. Initialize: returns (dynamics_vars, observables_vars)
dv, ov = init_gkba_wbl(; nx=2, ny=1, nk=400, γ=0.2, γso=0.1, γc=0.0,
                         j_sd=0.5, Temp=0.01, μ_α=[0.5, -0.5], ...)

# 2. (Optional) Set up precessing spins
pr_spins = [PrecSpin(i; theta_zero=20.0, T=5.0) for i in 1:nx]

# 3. Create ODE problem
prob = ODEProblem(eom_gkba_wbl!, dv.rkvec, (0.0, t_end), dv)

# 4. Create integrator (typically RK4 with fixed step)
integrator = init(prob, RK4(); dt=dt, adaptive=false, save_everystep=false)

# 5. Time evolution loop
for t in t_0:t_step:t_end
    step!(integrator, t_step, true)
    unpack!(dv, integrator.u)           # flat vector → tensor views in dv
    compute_observables!(ov, dv)        # fill ov with currents, densities, etc.

    # Update classical spins and rebuild Hamiltonian if using LLG
    for ps in pr_spins
        update!(ps, t)
        dv.vm_i1x[ps.i, :] .= ps.s
    end
    dv.hs_ij = build_hs(dv.vm_i1x, nx, ny, γ, γso, j_sd)

    # Access results: ov.curr_α, ov.sden_i1x, ov.scurr_xα, ov.cden_i
end
```

### Sanity checks to verify correctness

```julia
# G^< must remain anti-Hermitian throughout the simulation
G = dv.Gls_ij
println("max |G + G†| = ", maximum(abs.(G .+ G')))   # should be ~1e-12

# Charge density ~1 electron/site at half-filling
println("Charge density: ", round.(real(ov.cden_i), digits=4))
```

---

## Key API

### Initialization functions (`src/init.jl`)

All return `(dv::DynamicsType, ov::ObservablesVar)`.

```julia
init_gkba(; nx, ny, nk, γ, γso, γc, j_sd, Temp, μ_α, ϵ0α, vm_i1x, nσ, nα)
init_egkba(; ...)
init_gkba_posrep(; ...)
init_egkba_posrep(; ...)
init_gkba_wbl(; ...)
```

### EOM functions (`src/eom.jl`)

All have signature `eom_*!(du, u, dv::DynamicsType, t)` — in-place, compatible with DifferentialEquations.jl.

```julia
eom_gkba!(du, u, dv, t)
eom_egkba!(du, u, dv, t)
eom_gkba_posrep!(du, u, dv, t)
eom_egkba_posrep!(du, u, dv, t)
eom_gkba_wbl!(du, u, dv, t)
```

### Observables (`src/observables.jl`)

```julia
compute_observables!(ov::ObservablesVar, dv::DynamicsType)
```

Fills `ov` with:
- `ov.curr_α` — charge current per lead `(nα,)`
- `ov.scurr_xα` — spin current per lead `(3, nα)`
- `ov.sden_i1x` — site-resolved spin density `(ns, 3)`
- `ov.sden_xij` — spin density matrix `(3, dim_s, dim_s)`
- `ov.cden_i` — charge density per site `(dim_s,)`

### Spin dynamics (`src/llg.jl`)

```julia
PrecSpin(i; axis_phi, axis_theta, phi_zero, theta_zero, start_time, T)
update!(ps::PrecSpin, time)       # Advance spin orientation in time
heun(vm, sden, dt, lv)            # Heun integrator for LLG
```

---

## Code Conventions

### Naming
- **Types/structs:** PascalCase — `GKBADynamics`, `ObservablesVar`, `PrecSpin`
- **Functions:** snake_case — `build_hs`, `compute_observables!`, `eom_gkba!`
- **Constants:** SCREAMING_SNAKE_CASE — `MBOHR`, `KB_EV`, `K_BOLTZMAN`
- **Mutating functions:** suffixed with `!` — `eom_gkba!`, `unpack!`, `compute_observables!`

### Index conventions
| Symbol | Meaning |
|--------|---------|
| `i, j` | System space (site or orbital) |
| `k, α` | Lead k-point and lead index |
| `x` | Spin direction (1=x, 2=y, 3=z) |
| `s` | System |
| `e` | Electron (lead) |
| `l` | Lesser (G<) |

### Matrix naming
| Variable | Description |
|----------|-------------|
| `hs` | System Hamiltonian |
| `heα` | Lead Hamiltonian (lead α) |
| `hseα` / `heαs` | System-lead coupling |
| `Gls` | Lesser Green function (system) |
| `Gleαs` | Lesser Green function (electron-system) |
| `gl` | Static lead lesser Green function |

### Tensor operations
- Use `@tullio` macro (Tullio.jl) for Einstein-style summation; avoids explicit loops.
- All matrices and vectors are complex-valued (`ComplexF64`).
- State vectors are flattened for the ODE solver (`dv.rkvec`) and unpacked via `unpack!`.

### Code style
- Section dividers: `# ─────────────────────────────`
- Docstrings with argument and return type descriptions on public functions
- Physics equations appear in comments for non-obvious operations
- No CI system; tests are run manually

---

## Scripts Reference

| Script | Dynamics | Scenario |
|--------|----------|----------|
| `run_gkba_krep_static.jl` | k-rep GKBA | Static spins |
| `run_gkba_krep_free.jl` | k-rep GKBA | Free precession (LLG) |
| `run_gkba_krep_prece.jl` | k-rep GKBA | Driven precession |
| `run_gkba_krep_prece_single_lead.jl` | k-rep GKBA | Single lead |
| `run_gkba_krep_leviton.jl` | k-rep GKBA | Leviton pulse |
| `run_egkba_krep_static.jl` | k-rep eGKBA | Static spins |
| `run_egkba_krep_free.jl` | k-rep eGKBA | Free precession |
| `run_egkba_krep_prece.jl` | k-rep eGKBA | Driven precession |
| `run_egkba_krep_prece_extra_bath.jl` | k-rep eGKBA | Extra bath (3+ leads) |
| `run_gkba_posrep_static.jl` | pos-rep GKBA | Static (nx=5) |
| `run_gkba_posrep_large_static.jl` | pos-rep GKBA | Large lead |
| `run_egkba_posrep_static.jl` | pos-rep eGKBA | Static spins |
| `run_egkba_posrep_prece.jl` | pos-rep eGKBA | Driven precession |
| `run_egkba_posrep_prece_single_lead.jl` | pos-rep eGKBA | Single lead |
| `run_gkba_wbl_free.jl` | WBL GKBA | Free precession |
| `run_gkba_wbl_prece.jl` | WBL GKBA | Driven precession |

---

## Tests

Tests in `test/runtests.jl` are organized into 5 test sets (one per dynamics type):

- **What is tested:** initialization correctness, output array dimensions, anti-Hermiticity of Green functions, EOM step completeness, current conservation at equilibrium
- **Small system:** `nx=2, ny=1, nk=4` for fast execution
- **No continuous integration** — run manually as above

---

## Git Workflow

- Main branch: `main`
- Feature branches: `claude/<description>` style
- `Manifest.toml` and `data/` are git-ignored; do not commit them
- No custom git hooks configured

---

## What NOT to Do

- Do not commit `data/` output files — they are large and git-ignored by design
- Do not commit `Manifest.toml` — it is git-ignored; use `Project.toml` only
- Do not modify `legacy/` notebooks — they are archived originals
- Do not add external dependencies without updating `Project.toml` via `Pkg.add`
- Avoid breaking the `eom_*!(du, u, dv, t)` interface — DifferentialEquations.jl requires this exact signature
- Avoid allocations inside EOM functions — they are called at every ODE step and must be fast

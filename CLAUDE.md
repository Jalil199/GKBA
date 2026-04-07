# CLAUDE.md

> For physics background, equations, dynamics types, and the scripts table see **README.md**.
> This file covers what you need to *develop and run* the code correctly.

---

## Commands

```bash
# Install dependencies (only needed once or after Project.toml changes)
julia --project -e 'using Pkg; Pkg.instantiate()'

# Run tests
julia --project test/runtests.jl

# Run benchmarks
julia --project test/benchmark.jl

# Run a simulation (output goes to data/, which is git-ignored)
julia --project scripts/run_gkba_wbl_prece.jl
```

---

## Project layout (short)

```
src/          8 source files — edit here
scripts/      16 thin driver scripts — one per simulation scenario
test/         runtests.jl (unit tests) + benchmark.jl
notebooks/    example.ipynb (best starting point), explore_results.ipynb
legacy/       archived notebooks — do not modify
data/         generated at runtime, git-ignored
```

---

## Conventions

- **Types:** PascalCase — `GKBADynamics`, `ObservablesVar`
- **Functions:** snake_case — `build_hs`, `compute_observables!`
- **Mutating functions:** suffix `!` — `eom_gkba!`, `unpack!`, `compute_observables!`
- **Constants:** SCREAMING_SNAKE_CASE — `MBOHR`, `KB_EV`
- **Units:** natural units, ħ = 1 throughout; energy in eV, temperature in K

Index symbols used in variable names:

| Symbol | Meaning |
|--------|---------|
| `i, j` | system site/orbital |
| `k`    | lead k-point (or Ozaki pole in WBL) |
| `α`    | lead index |
| `x`    | spin direction (1=x, 2=y, 3=z) |
| `l`    | lesser (G<) |
| `s`    | system |
| `e`    | lead electron |

---

## Non-obvious design decisions

**State vector flattening.** `dv.rkvec` is a flat `Vector{ComplexF64}` that concatenates all dynamical arrays so DifferentialEquations.jl can handle them as a single ODE state. `unpack!(dv, u)` restores the tensor views after each solver step — always call it before reading `dv` fields.

**WBL `nk` means Ozaki poles, not k-points.** In `WBLDynamics`, `nk` controls the number of Ozaki poles used to expand the Fermi function. It is not a Brillouin zone resolution. Typical values: `nk=20` (fast/test), `nk=400` (production).

**EOM interface is fixed.** All EOM functions must have the signature `eom_*!(du, u, dv, t)` — this is what DifferentialEquations.jl expects. Do not add arguments or change argument order.

**Fixed-step RK4 is deliberate.** The scripts use `RK4()` with `adaptive=false` and a fixed `dt`. The Hamiltonians can vary on short timescales (e.g., Leviton pulses, precessing spins), so adaptive solvers can miss features. Only use `adaptive=true` when the Hamiltonian is smooth and slowly varying.

**Lead coupling envelope.** Several scripts use a sigmoid `stepp(t)` to ramp up `dv.s_α` (the system-lead coupling) adiabatically from zero. This avoids unphysical transients at t=0. When adding new scripts, copy this pattern.

---

## Typical parameter ranges

| Parameter | Typical range | Notes |
|-----------|--------------|-------|
| `γ` | 0.5–2.0 eV | hopping; sets energy scale |
| `γso` | 0.0–0.5 | Rashba spin-orbit; 0 to disable |
| `γc` | 0.5–2.0 | system-lead coupling |
| `j_sd` | 0.0–1.0 | s-d exchange; keep < γ for weak coupling |
| `Temp` | 0.01–300 K | 0.01 = near zero-T; 300 = room temp |
| `nk` | 4 (tests), 20–400 (production) | k-rep: BZ points; WBL: Ozaki poles |
| `dt` | 0.05–0.1 | time step; decrease if G< loses anti-Hermiticity |
| `nx` | 2–10 | chain length; cost scales as ~nx³ |

---

## Sanity checks

After any non-trivial change, verify:

```julia
# G^< must stay anti-Hermitian (should be < 1e-10 for a healthy simulation)
G = dv.Gls_ij
println(maximum(abs.(G .+ G')))

# Charge conservation: sum of currents into/out of leads ≈ 0 at steady state
println(sum(real(ov.curr_α)))

# Charge density ~1 electron/site at half-filling (μ=0, balanced leads)
println(real(ov.cden_i))
```

---

## Performance rules

- **No allocations in EOM functions.** `eom_*!` is called at every solver step. Use pre-allocated arrays in `dv`; avoid `similar`, `zeros`, or temporary matrices inside EOM.
- **Use `@tullio` for tensor contractions.** It generates efficient in-place code and avoids explicit loops. Do not replace `@tullio` with manual loops unless profiling shows a clear win.
- **`@views` on slices.** When slicing arrays in hot paths, use `@views` to avoid copies.

---

## Adding a new simulation script

1. Copy the closest existing script from `scripts/`.
2. Change parameters at the top — do not restructure the init/ODE/loop pattern.
3. Add a row to the Scripts table in `README.md`.
4. Output files go to `data/` with a descriptive prefix — do not hardcode absolute paths.

## Adding a new dynamics type

1. Define a new struct in `src/types.jl` following the existing pattern.
2. Add an `eom_*!` function in `src/eom.jl` and an `init_*` function in `src/init.jl`.
3. Export all public names from `src/GKBA.jl`.
4. Add a test set in `test/runtests.jl` (init + EOM step + anti-Hermiticity check).

---

## Do not

- Commit `data/` or `Manifest.toml` — both are git-ignored for good reason.
- Modify `legacy/` — archived originals, kept for reference only.
- Add dependencies without `Pkg.add` (which updates `Project.toml`).
- Allocate inside `eom_*!` functions.
- Change the `eom_*!(du, u, dv, t)` signature.

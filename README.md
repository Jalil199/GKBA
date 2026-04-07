# GKBA — Generalized Kadanoff-Baym Ansatz for quantum transport

Julia module for real-time quantum transport simulations of open quantum systems
coupled to fermionic leads. Implements several variants of the GKBA and solves
the equations of motion for the lesser Green function G^<(t).

## Physics

The central quantity is the lesser Green function of the system G^<_s(t),
which encodes the non-equilibrium charge and spin density matrix.
The equations of motion couple G^<_s to the system–lead Green function
G^<_{eαs}(k,t) via the GKBA:

```
i ∂_t G^<_{eαs} = [h_{eα}, G^<_{eαs}] + h_{eαs} G^<_s − g^<_α h_{eαs}
i ∂_t G^<_s     = [h_s, G^<_s] − Σ_α (h_{seα} G^<_{eαs} + h.c.)
```

All equations use natural units with ħ = 1.

## Dynamics types

| Type | Lead representation | Lead GF |
|---|---|---|
| `GKBADynamics` | k-space (diagonal) | static, Fermi-Dirac |
| `eGKBADynamics` | k-space | dynamic (PRL 130, 246301) |
| `PosRepDynamics` | position space (full matrix) | static |
| `ePosRepDynamics` | position space | dynamic |
| `WBLDynamics` | wide-band limit + Ozaki poles | analytic |

The WBL dynamics uses the approximation Σ^r = −iΓ/2 (constant self-energy)
and expands the Fermi function in Ozaki poles, reducing the problem to a set
of auxiliary functions F[k,α](t) without k-space integration.

## Observables

`compute_observables!` fills an `ObservablesVar` with:
- `curr_α` — charge current per lead
- `scurr_xα` — spin current (x,y,z) per lead
- `sden_i1x` — site-resolved spin density
- `cden_i` — charge density per orbital

## Structure

```
src/
  GKBA.jl          module entry point and exports
  constants.jl     physical constants (ħ=1, k_B, μ_B, …)
  types.jl         dynamics and observables structs
  hamiltonians.jl  build_hs, build_heα, build_hseα, ozaki_poles, …
  eom.jl           equations of motion + unpack!
  observables.jl   compute_observables!
  init.jl          init_gkba, init_egkba, init_gkba_wbl, …
  llg.jl           Landau-Lifshitz-Gilbert spin dynamics

scripts/           one driver script per simulation variant (see table below)
test/
  runtests.jl      18 unit tests (init, EOM step, conservation laws)
  benchmark.jl     EOM vs direct reference + physical invariant checks
```

## Scripts

| Script | Method | Notes |
|---|---|---|
| `run_gkba_krep_static.jl` | GKBA k-rep | static spins |
| `run_gkba_krep_free.jl` | GKBA k-rep | free precessing spins |
| `run_gkba_krep_prece.jl` | GKBA k-rep | driven precession |
| `run_gkba_krep_prece_single_lead.jl` | GKBA k-rep | 1 lead |
| `run_gkba_krep_leviton.jl` | GKBA k-rep | Leviton pulse |
| `run_egkba_krep_static.jl` | eGKBA k-rep | static spins |
| `run_egkba_krep_free.jl` | eGKBA k-rep | free precession |
| `run_egkba_krep_prece.jl` | eGKBA k-rep | driven precession |
| `run_egkba_krep_prece_extra_bath.jl` | eGKBA k-rep | extra bath lead |
| `run_gkba_posrep_static.jl` | GKBA pos-rep | nx=5 chain |
| `run_gkba_posrep_large_static.jl` | GKBA pos-rep | nx=2, large lead |
| `run_egkba_posrep_static.jl` | eGKBA pos-rep | static spins |
| `run_egkba_posrep_prece.jl` | eGKBA pos-rep | driven precession |
| `run_egkba_posrep_prece_single_lead.jl` | eGKBA pos-rep | 1 lead |
| `run_gkba_wbl_free.jl` | GKBA WBL | free precession |
| `run_gkba_wbl_prece.jl` | GKBA WBL | driven precession |

## Usage

```julia
using DifferentialEquations
import GKBA: init_gkba_wbl, eom_gkba_wbl!, compute_observables!, unpack!

dv, ov = init_gkba_wbl(; nx=2, ny=1, nk=20, γ=1.0, γso=0.0,
                          γc=1.0, j_sd=0.1, Temp=300.0,
                          vm_i1x = vm_init)

prob       = ODEProblem(eom_gkba_wbl!, dv.rkvec, (0.0, 60.0), dv)
integrator = init(prob, RK4(); dt=0.1, adaptive=true, save_everystep=false)

for t in 0.0:0.1:59.9
    step!(integrator, 0.1, true)
    unpack!(dv, integrator.u)
    compute_observables!(ov, dv)
    # access ov.curr_α, ov.sden_i1x, …
end
```

## Reference solvers

`KBE-Test2-prece.ipynb` — full two-time Kadanoff-Baym equations via
[KadanoffBaym.jl](https://github.com/NonequilibriumDynamics/KadanoffBaym.jl).
Useful as an exact reference to benchmark GKBA accuracy.

`Krylov.ipynb` — exact non-interacting evolution via matrix exponentiation
(`Expokit.jl`). Propagates the single-particle correlation matrix
C(t+dt) = e^{−iH dt} C(t) e^{iH dt}.

## Running the tests

```bash
julia --project test/runtests.jl
julia --project test/benchmark.jl
```

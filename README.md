# GKBA — Generalized Kadanoff-Baym Ansatz for quantum transport

## Motivation

Understanding spin and charge transport in nanoscale magnetic devices requires
solving the real-time quantum dynamics of electrons in an open system: a small
correlated region (the *central system*) connected to macroscopic metallic leads
held at equilibrium. The leads drive the system out of equilibrium and act as
sources and sinks of electrons and angular momentum.

The exact solution — the two-time Kadanoff-Baym equations (KBE) — scales as
$\mathcal{O}(N_t^2)$ in memory and time, making long simulations prohibitively
expensive. The **Generalized Kadanoff-Baym Ansatz (GKBA)** reconstructs the
two-time Green function from its equal-time limit, reducing the problem to a set
of coupled ODEs that scale as $\mathcal{O}(N_t)$. This makes it possible to
simulate spin-transfer torque, spin pumping, and non-equilibrium magnetization
dynamics over physically relevant timescales.

## Goals

This repository unifies a collection of research notebooks into a reusable Julia
module. The core physics is implemented once in `src/` and each simulation
scenario (geometry, driving protocol, approximation level) becomes a thin driver
script in `scripts/`. Concretely:

- Provide a clean, tested implementation of the GKBA and its extensions for
  tight-binding systems with s-d coupled local moments.
- Support multiple lead descriptions: k-space embedding (PRB 107, 155141),
  position-space leads, and the wide-band limit with Ozaki pole expansion.
- Enable coupled electron–spin dynamics (GKBA + Landau-Lifshitz-Gilbert) for
  studying spin-transfer torque and spin pumping.

## System

The central region is a 1D tight-binding chain of $n_x \times n_y$ sites with
Rashba spin-orbit coupling $\gamma_{so}$ and local magnetic moments $\mathbf{m}_i$
coupled to the electron spin via s-d exchange $j_{sd}$:

$$H_s = -\gamma \sum_{\langle ij \rangle} c^\dagger_i c_j + \gamma_{so}(\ldots) - j_{sd} \sum_i \mathbf{m}_i \cdot \hat{\boldsymbol{\sigma}}_i$$

Two semi-infinite leads (modeled as 1D tight-binding chains with hopping $\gamma$
and system-lead coupling $\gamma_c$) are attached to the first and last sites.

---

Julia module for real-time quantum transport simulations of open quantum systems
coupled to fermionic leads. Implements several variants of the GKBA and solves
the equations of motion for the lesser Green function $G^{<}(t)$.

## Physics

The central quantity is the lesser Green function of the system $G^{<}_s(t)$,
which encodes the non-equilibrium charge and spin density matrix.
The equations of motion couple $G^{<}_s$ to the system–lead Green function
$G^{<}_{e\alpha s}(k,t)$ via the GKBA:

$$i\,\partial_t G^{<}_{e\alpha s}(k) = h_{e\alpha}(k)\,G^{<}_{e\alpha s}(k) - G^{<}_{e\alpha s}(k)\,h_s + h_{e\alpha s}(k)\,G^{<}_s - g^{<}_\alpha(k)\,h_{e\alpha s}(k)$$

$$i\,\partial_t G^{<}_s = \left[h_s,\, G^{<}_s\right] - \sum_{\alpha,k} \left(h_{se\alpha}(k)\,G^{<}_{e\alpha s}(k) + \mathrm{h.c.}\right)$$

The first equation holds for each lead mode $k$ independently. $G^{<}_{e\alpha s}(k)$ is a
row vector in system space — $h_{e\alpha}(k)$ (a scalar in k-rep) acts on the lead index
while $h_s$ acts on the system index. All equations use natural units with $\hbar = 1$.

## Dynamics types

| Type | Lead representation | Lead GF |
|---|---|---|
| `GKBADynamics` | k-space (diagonal) | static, Fermi-Dirac |
| `eGKBADynamics` | k-space | dynamic (PRL 130, 246301) |
| `PosRepDynamics` | position space (full matrix) | static |
| `ePosRepDynamics` | position space | dynamic |
| `WBLDynamics` | wide-band limit + Ozaki poles | analytic |

The WBL dynamics uses the approximation $\Sigma^r = -i\Gamma/2$ (constant self-energy)
and expands the Fermi function in Ozaki poles, reducing the problem to a set
of auxiliary functions $F_{k\alpha}(t)$ without $k$-space integration.

## Observables

`compute_observables!` fills an `ObservablesVar` with:
- `curr_α` — charge current $J_\alpha$ per lead
- `scurr_xα` — spin current $J^x_\alpha$ (x,y,z) per lead
- `sden_i1x` — site-resolved spin density $\langle S^x_i \rangle$
- `cden_i` — charge density $\langle n_i \rangle$ per orbital

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

## References

- **PRB 107, 155141** — Core theory: GKBA with embedding for leads with internal structure
  (basis for k-rep and pos-rep dynamics).
- **PRL 130, 246301** — Extended GKBA: time-linear transport with dynamical lead correlators
  (basis for `eGKBADynamics` and `ePosRepDynamics`).

## Reference solvers

`KBE-Test2-prece.ipynb` — full two-time Kadanoff-Baym equations via
[KadanoffBaym.jl](https://github.com/NonequilibriumDynamics/KadanoffBaym.jl).
Useful as an exact reference to benchmark GKBA accuracy.

`Krylov.ipynb` — exact non-interacting evolution via matrix exponentiation
(`Expokit.jl`). Propagates the single-particle correlation matrix
$$C(t+\delta t) = e^{-iH\,\delta t}\,C(t)\,e^{iH\,\delta t}$$

## Running the tests

```bash
julia --project test/runtests.jl
julia --project test/benchmark.jl
```

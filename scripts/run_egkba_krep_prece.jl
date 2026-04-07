"""
Driver: extended GKBA (eGKBA), k-representation, precessing spins (2 sites, 2 leads).
Corresponds to HFGKBA-beyond-bohr_krep-prece.ipynb.

Key differences vs standard GKBA:
  - Lead GF G^<_α(q,k) evolves dynamically (no static approximation)
  - Solver: Vern7 (better for the larger state space)
  - System-lead coupling is constant (no turn-on envelope on hseα)
  - β = 1.0 (natural units, as in the reference notebook)
"""

using DifferentialEquations
using DelimitedFiles
using Tullio
using LinearAlgebra
import GKBA: init_egkba, make_llg_params,
             eom_egkba!, compute_observables!, unpack!,
             PrecSpin, update!, build_hs

# ── Parameters ────────────────────────────────────────────────────────────────
nx, ny   = 2, 1
nk       = 300
γ        = 1.0
γso      = 0.0
γc       = 1.0
j_sd     = 0.1
Temp     = 300.0           # only used for display; β is overridden below
μ_α      = [0.0, 0.0]
ϵ0α      = [0.0, 0.0]

# Time grid
t_0      = 0.0
t_step   = 0.1
t_end    = 120.0

# Precession
theta_1  = 20.0            # polar angle (degrees)
phi_1    = 0.0             # azimuthal offset (degrees)
period   = 5.0             # precession period

# Output
output_dir = joinpath(@__DIR__, "..", "data")
name       = "egkba_krep_prece"

# ── Initialize ────────────────────────────────────────────────────────────────
pr_spins = [PrecSpin(i; theta_zero = theta_1, phi_zero = phi_1, T = period)
            for i in 1:nx]

vm_init = zeros(Float64, nx*ny, 3)
for ps in pr_spins
    update!(ps, 0.0)
    vm_init[ps.i, :] .= ps.s
end

# β_override = 1.0 matches the reference notebook convention
dv, ov = init_egkba(; nx, ny, nk, γ, γso, γc, j_sd, Temp, μ_α, ϵ0α,
                      vm_i1x = vm_init, β_override = 1.0)

lv = make_llg_params(nx, ny)

# ── Set up integrator ─────────────────────────────────────────────────────────
prob       = ODEProblem(eom_egkba!, dv.rkvec, (t_0, t_end), dv)
integrator = init(prob, Vern7(); dt = t_step, save_everystep = false,
                  adaptive = true, dense = false)

# ── Open output files ─────────────────────────────────────────────────────────
mkpath(output_dir)
files = Dict(
    :cc     => open(joinpath(output_dir, "cc_$(name)_jl.txt"),     "w+"),
    :sneq   => open(joinpath(output_dir, "sneq_$(name)_jl.txt"),   "w+"),
    :sc     => open(joinpath(output_dir, "sc_$(name)_jl.txt"),     "w+"),
    :cspins => open(joinpath(output_dir, "cspins_$(name)_jl.txt"), "w+"),
    :cden   => open(joinpath(output_dir, "cden_$(name)_jl.txt"),   "w+"),
    :rho    => open(joinpath(output_dir, "rho_$(name)_jl.txt"),    "w+"),
)

# ── Time evolution ────────────────────────────────────────────────────────────
println("Starting eGKBA time evolution  (t_end = $t_end, dt = $t_step, nk = $nk)")
elapsed = @elapsed begin
    for (_, t) in enumerate(t_0:t_step:t_end - t_step)
        tt = round(t + t_step, digits = 2)

        step!(integrator, t_step, true)

        # Unpack state and compute observables
        unpack!(dv, integrator.u)
        compute_observables!(ov, dv)

        # Update precessing spins
        for ps in pr_spins
            update!(ps, tt)
            dv.vm_i1x[ps.i, :] .= ps.s
        end

        # Update system Hamiltonian (coupling hseα stays constant in eGKBA)
        dv.hs_ij = build_hs(dv.vm_i1x, nx, ny, γ, γso, j_sd)

        # Diagnostic: electron number in lead 1 (should be conserved)
        n_lead1 = -1im * tr(dv.Gleα_qkα[:,:,1]) / nk
        println("t = $tt  |  n_lead1 = $(round(real(n_lead1), digits=5))")
        flush(stdout)

        # Save observables
        writedlm(files[:rho],    transpose(vec(dv.Gls_ij)), ',')
        writedlm(files[:cc],     real(ov.curr_α),           ' ')
        writedlm(files[:sneq],   real(ov.sden_i1x),         ' ')
        writedlm(files[:sc],     real(ov.scurr_xα),         ' ')
        writedlm(files[:cspins], real(dv.vm_i1x),           ' ')
        writedlm(files[:cden],   real(ov.cden_i),           ' ')
    end
end

foreach(close, values(files))
println("Done.  Elapsed: $(round(elapsed, digits=1)) s")

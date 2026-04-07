"""
Driver: extended GKBA, position-representation leads, precessing spins, single lead.
Corresponds to HFGKBA-beyond-bohr_posrep-prece-single_lead.ipynb.
"""

using DifferentialEquations
using DelimitedFiles
import GKBA: init_egkba_posrep, eom_egkba_posrep!, compute_observables!,
             PrecSpin, update!, build_hs, build_hseα_posrep
import GKBA: unpack!
import Tullio: @tullio

# ── Parameters ────────────────────────────────────────────────────────────────
nx, ny   = 2, 1
nk       = 50
nα       = 1            # single lead
γ        = 1.0
γso      = 0.0
γc       = 1.0
j_sd     = 0.1
Temp     = 300.0
μ_α      = [0.0]
ϵ0α      = [0.0]

# Time grid
t_0      = 0.0
t_step   = 0.1
t_end    = 120.0

# Precession
theta_1  = 20.0
phi_1    = 0.0
period   = 5.0

# Turn-on envelope
stepp(t; ti = 3, to = 25) = 1 / (1 + exp(-(t - to) / ti))

# Output
output_dir = joinpath(@__DIR__, "..", "data")
name       = "egkba_posrep_prece_single_lead"

# ── Initialize ────────────────────────────────────────────────────────────────
pr_spins = [PrecSpin(i; theta_zero = theta_1, phi_zero = phi_1, T = period)
            for i in 1:nx]

vm_init = zeros(Float64, nx*ny, 3)
for ps in pr_spins
    update!(ps, 0.0)
    vm_init[ps.i, :] .= ps.s
end

dv, ov = init_egkba_posrep(; nx, ny, nk, γ, γso, γc, j_sd, Temp, μ_α, ϵ0α,
                              vm_i1x = vm_init, nα)

dv.hseα_ikα .*= stepp(t_0)
@tullio dv.heαs_kαi[k,α,i] := conj(dv.hseα_ikα[i,k,α])

import GKBA: make_llg_params
lv = make_llg_params(nx, ny)

# ── Set up integrator ─────────────────────────────────────────────────────────
prob       = ODEProblem(eom_egkba_posrep!, dv.rkvec, (t_0, t_end), dv)
integrator = init(prob, RK4(); dt = t_step, save_everystep = false,
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
println("Starting time evolution  (t_end = $t_end, dt = $t_step)")
elapsed = @elapsed begin
    for t in t_0:t_step:t_end - t_step
        tt = round(t + t_step, digits = 2)

        step!(integrator, t_step, true)
        unpack!(dv, integrator.u)
        compute_observables!(ov, dv)

        for ps in pr_spins
            update!(ps, tt)
            dv.vm_i1x[ps.i, :] .= ps.s
        end

        dv.hs_ij    = build_hs(dv.vm_i1x, nx, ny, γ, γso, j_sd)
        dv.hseα_ikα = build_hseα_posrep(nx*ny, γc, nk, 2, nα) .* stepp(tt)
        @tullio dv.heαs_kαi[k,α,i] := conj(dv.hseα_ikα[i,k,α])

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

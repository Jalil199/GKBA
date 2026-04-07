"""
Driver: extended GKBA, k-representation, precessing spins, extra bath on site 2.
Corresponds to HFGKBA-beyond-bohr_krep-prece-EXTRA_BATH.ipynb.

Extra bath: an additional lead (α=3) coupled to site 2 only, acting as a
dissipative reservoir. Implemented by extending hseα to nα=3.
"""

using DifferentialEquations, DelimitedFiles, Tullio, LinearAlgebra
import GKBA: init_egkba, make_llg_params,
             eom_egkba!, compute_observables!, unpack!,
             PrecSpin, update!, build_hs, build_hseα, fermi, K_BOLTZMAN

# ── Parameters ────────────────────────────────────────────────────────────────
nx, ny   = 2, 1
nk       = 300
nσ       = 2
nα       = 3       # 2 leads + 1 extra bath
γ        = 1.0
γso      = 0.0
γc       = 1.0
γc_bath  = 0.5     # coupling strength to extra bath (tune as needed)
j_sd     = 0.1
Temp     = 300.0
μ_α      = [0.0, 0.0, 0.0]
ϵ0α      = [0.0, 0.0, 0.0]
t_step   = 0.1
t_end    = 120.0
theta_1, phi_1, period = 20.0, 0.0, 5.0
stepp(t; ti = 70) = t < ti ? sin((π/2) * t / ti)^4 : 1.0
output_dir = joinpath(@__DIR__, "..", "data");  name = "egkba_krep_prece_extra_bath"

# ── Initialize ────────────────────────────────────────────────────────────────
pr_spins = [PrecSpin(i; theta_zero = theta_1, phi_zero = phi_1, T = period)
            for i in 1:nx]
vm_init = zeros(Float64, nx*ny, 3)
for ps in pr_spins; update!(ps, 0.0); vm_init[ps.i, :] .= ps.s; end

# Use nα=3 for the extra bath
dv, ov = init_egkba(; nx, ny, nk, γ, γso, γc, j_sd, Temp, μ_α, ϵ0α,
                      vm_i1x = vm_init, nα, β_override = 1.0)

# Add extra bath coupling on site 2 (last site)
ns   = nx * ny
dim_e = nk * nσ
dim_s = ns * nσ
ks   = 1:nk;  ln = nk + 1
w_bath = -γc_bath * sin.(ks * π / ln) * sqrt(2 / ln)
dv.hseα_ikα[end-1, 1:2:end, 3] .= w_bath   # site 2 spin-up
dv.hseα_ikα[end,   2:2:end, 3] .= w_bath   # site 2 spin-down
@tullio dv.heαs_kαi[k,α,i] := conj(dv.hseα_ikα[i,k,α])

lv = make_llg_params(nx, ny)

prob       = ODEProblem(eom_egkba!, dv.rkvec, (0.0, t_end), dv)
integrator = init(prob, Vern7(); dt = t_step, save_everystep = false, adaptive = true, dense = false)

# ── Run ───────────────────────────────────────────────────────────────────────
mkpath(output_dir)
files = Dict(
    :cc     => open(joinpath(output_dir, "cc_$(name)_jl.txt"),     "w+"),
    :sneq   => open(joinpath(output_dir, "sneq_$(name)_jl.txt"),   "w+"),
    :sc     => open(joinpath(output_dir, "sc_$(name)_jl.txt"),     "w+"),
    :cspins => open(joinpath(output_dir, "cspins_$(name)_jl.txt"), "w+"),
    :cden   => open(joinpath(output_dir, "cden_$(name)_jl.txt"),   "w+"),
    :rho    => open(joinpath(output_dir, "rho_$(name)_jl.txt"),    "w+"),
)

println("Starting eGKBA + extra bath  (t_end=$t_end, γc_bath=$γc_bath)")
elapsed = @elapsed begin
    for (_, t) in enumerate(0.0:t_step:t_end - t_step)
        tt = round(t + t_step, digits = 2)
        step!(integrator, t_step, true)
        unpack!(dv, integrator.u)
        compute_observables!(ov, dv)

        for ps in pr_spins; update!(ps, tt); dv.vm_i1x[ps.i, :] .= ps.s; end
        dv.hs_ij = build_hs(dv.vm_i1x, nx, ny, γ, γso, j_sd)

        n_lead1 = -1im * tr(dv.Gleα_qkα[:,:,1]) / nk
        println("t=$tt  n_lead1=$(round(real(n_lead1), digits=5))")
        flush(stdout)

        writedlm(files[:rho],    transpose(vec(dv.Gls_ij)), ',')
        writedlm(files[:cc],     real(ov.curr_α),           ' ')
        writedlm(files[:sneq],   real(ov.sden_i1x),         ' ')
        writedlm(files[:sc],     real(ov.scurr_xα),         ' ')
        writedlm(files[:cspins], real(dv.vm_i1x),           ' ')
        writedlm(files[:cden],   real(ov.cden_i),           ' ')
    end
end
foreach(close, values(files))
println("Done. $(round(elapsed, digits=1)) s")

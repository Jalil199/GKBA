"""
Driver: standard GKBA, k-representation, precessing spins, single lead.
Corresponds to HFGKBA_krep-prece-single_lead.ipynb.

Single lead: only lead 1 (left) is coupled (hseα[:,:,2] = 0).
Everything else identical to run_gkba_krep_prece.jl.
"""

using DifferentialEquations, DelimitedFiles, Tullio
import GKBA: init_gkba, make_llg_params,
             eom_gkba!, compute_observables!, unpack!,
             PrecSpin, update!, build_hs, build_hseα

# ── Parameters ────────────────────────────────────────────────────────────────
nx, ny   = 2, 1
nk       = 400
γ        = 1.0
γso      = 0.0
γc       = 1.0
j_sd     = 0.1
Temp     = 300.0
t_step   = 0.1
t_end    = 120.0
theta_1, phi_1, period = 20.0, 0.0, 5.0
stepp(t; ti = 3, to = 25) = 1 / (1 + exp(-(t - to) / ti))
output_dir = joinpath(@__DIR__, "..", "data")
name       = "gkba_krep_prece_single_lead"

# ── Initialize ────────────────────────────────────────────────────────────────
pr_spins = [PrecSpin(i; theta_zero = theta_1, phi_zero = phi_1, T = period)
            for i in 1:nx]
vm_init = zeros(Float64, nx*ny, 3)
for ps in pr_spins; update!(ps, 0.0); vm_init[ps.i, :] .= ps.s; end

dv, ov = init_gkba(; nx, ny, nk, γ, γso, γc, j_sd, Temp, vm_i1x = vm_init)

# Single lead: zero out coupling to lead 2
dv.hseα_ikα[:, :, 2] .= 0.0
dv.hseα_ikα .*= stepp(0.0)
@tullio dv.heαs_kαi[k,α,i] := conj(dv.hseα_ikα[i,k,α])

lv = make_llg_params(nx, ny)

prob       = ODEProblem(eom_gkba!, dv.rkvec, (0.0, t_end), dv)
integrator = init(prob, RK4(); dt = t_step, save_everystep = false,
                  adaptive = true, dense = false)

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

println("Starting single-lead GKBA  (t_end=$t_end)")
elapsed = @elapsed begin
    for (_, t) in enumerate(0.0:t_step:t_end - t_step)
        tt = round(t + t_step, digits = 2)
        step!(integrator, t_step, true)
        unpack!(dv, integrator.u)
        compute_observables!(ov, dv)

        for ps in pr_spins
            update!(ps, tt); dv.vm_i1x[ps.i, :] .= ps.s
        end

        dv.hs_ij    = build_hs(dv.vm_i1x, nx, ny, γ, γso, j_sd)
        hseα_full   = build_hseα(nx*ny, γc, nk) .* stepp(tt)
        hseα_full[:, :, 2] .= 0.0           # single lead: kill lead 2
        dv.hseα_ikα = hseα_full
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
println("Done. $(round(elapsed, digits=1)) s")

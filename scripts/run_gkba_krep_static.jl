"""
Driver: standard GKBA, k-representation, static case (no driving).
Corresponds to HFGKBA_krep.ipynb.

γc = 0 → leads decoupled (no transport).
Spin evolves freely under LLG with quantum torque from electrons.
"""

using DifferentialEquations, DelimitedFiles, Tullio
import GKBA: init_gkba, make_llg_params,
             eom_gkba!, compute_observables!, unpack!,
             build_hs, heun

# ── Parameters ────────────────────────────────────────────────────────────────
nx, ny   = 2, 1
nk       = 400
γ        = 1.0
γso      = 0.0
γc       = 0.0     # decoupled leads
j_sd     = 0.0
Temp     = 300.0
t_step   = 0.1
t_end    = 50.0
output_dir = joinpath(@__DIR__, "..", "data")
name       = "gkba_krep_static"

# ── Initialize ────────────────────────────────────────────────────────────────
dv, ov = init_gkba(; nx, ny, nk, γ, γso, γc, j_sd, Temp)
lv     = make_llg_params(nx, ny)

prob       = ODEProblem(eom_gkba!, dv.rkvec, (0.0, t_end), dv)
integrator = init(prob, Vern7(); dt = t_step, save_everystep = false,
                  adaptive = true, dense = false)

# ── Run ───────────────────────────────────────────────────────────────────────
mkpath(output_dir)
files = Dict(
    :cc     => open(joinpath(output_dir, "cc_$(name)_jl.txt"),     "w+"),
    :sneq   => open(joinpath(output_dir, "sneq_$(name)_jl.txt"),   "w+"),
    :sc     => open(joinpath(output_dir, "sc_$(name)_jl.txt"),     "w+"),
    :cspins => open(joinpath(output_dir, "cspins_$(name)_jl.txt"), "w+"),
    :cden   => open(joinpath(output_dir, "cden_$(name)_jl.txt"),   "w+"),
)

println("Starting static GKBA  (t_end=$t_end)")
elapsed = @elapsed begin
    for (_, t) in enumerate(0.0:t_step:t_end - t_step)
        tt = round(t + t_step, digits = 2)
        step!(integrator, t_step, true)
        unpack!(dv, integrator.u)
        compute_observables!(ov, dv)

        # LLG: spin evolves under electron torque
        dv.vm_i1x .= heun(dv.vm_i1x, ov.sden_i1x, t_step, lv)
        dv.hs_ij   = build_hs(dv.vm_i1x, nx, ny, γ, γso, j_sd)

        writedlm(files[:cc],     real(ov.curr_α),   ' ')
        writedlm(files[:sneq],   real(ov.sden_i1x), ' ')
        writedlm(files[:sc],     real(ov.scurr_xα), ' ')
        writedlm(files[:cspins], real(dv.vm_i1x),   ' ')
        writedlm(files[:cden],   real(ov.cden_i),   ' ')
    end
end
foreach(close, values(files))
println("Done. $(round(elapsed, digits=1)) s")

"""
Driver: extended GKBA, k-representation, free LLG precession.
Corresponds to HFGKBA-beyond-bohr_krep-Free.ipynb.
"""

using DifferentialEquations, DelimitedFiles, Tullio
import GKBA: init_egkba, make_llg_params,
             eom_egkba!, compute_observables!, unpack!,
             build_hs, build_hseα, heun

nx, ny   = 2, 1;  nk = 300;  γ = 1.0;  γso = 0.0;  γc = 1.0;  j_sd = 0.1
t_step = 0.1;  t_end = 1000.0;  t_llg_start = 200.0
stepp(t; ti = 70) = t < ti ? sin((π/2) * t / ti)^4 : 1.0
output_dir = joinpath(@__DIR__, "..", "data");  name = "egkba_krep_free"

vm_init = zeros(Float64, nx*ny, 3)
vm_init[1, 1] = 1.0;  vm_init[2, 3] = 1.0

dv, ov = init_egkba(; nx, ny, nk, γ, γso, γc, j_sd, β_override = 1.0, vm_i1x = vm_init)
lv = make_llg_params(nx, ny)

prob       = ODEProblem(eom_egkba!, dv.rkvec, (0.0, t_end), dv)
integrator = init(prob, Vern7(); dt = t_step, save_everystep = false, adaptive = true, dense = false)

mkpath(output_dir)
files = Dict(
    :cc     => open(joinpath(output_dir, "cc_$(name)_jl.txt"),     "w+"),
    :sneq   => open(joinpath(output_dir, "sneq_$(name)_jl.txt"),   "w+"),
    :sc     => open(joinpath(output_dir, "sc_$(name)_jl.txt"),     "w+"),
    :cspins => open(joinpath(output_dir, "cspins_$(name)_jl.txt"), "w+"),
    :cden   => open(joinpath(output_dir, "cden_$(name)_jl.txt"),   "w+"),
    :rho    => open(joinpath(output_dir, "rho_$(name)_jl.txt"),    "w+"),
)

println("Starting free-precession eGKBA  (t_end=$t_end)")
elapsed = @elapsed begin
    for (_, t) in enumerate(0.0:t_step:t_end - t_step)
        tt = round(t + t_step, digits = 2)
        step!(integrator, t_step, true)
        unpack!(dv, integrator.u)
        compute_observables!(ov, dv)

        if tt > t_llg_start
            dv.vm_i1x .= heun(dv.vm_i1x, ov.sden_i1x, t_step, lv)
            dv.vm_i1x[1, :] .= [1.0, 0.0, 0.0]
        end

        dv.hs_ij = build_hs(dv.vm_i1x, nx, ny, γ, γso, j_sd)

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

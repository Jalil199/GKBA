"""
Driver: standard GKBA, k-representation, free LLG precession.
Corresponds to HFGKBA_krep-free.ipynb.

Initial condition: site 1 spin-x, site 2 spin-z.
Turn-on: sin^4 envelope over ti=60.
LLG activated after t > t_llg_start (default 200).
After LLG kicks in, site 1 is constrained to x-axis (fixed external moment).
"""

using DifferentialEquations, DelimitedFiles, Tullio
import GKBA: init_gkba, make_llg_params,
             eom_gkba!, compute_observables!, unpack!,
             build_hs, build_hseα, heun

# ── Parameters ────────────────────────────────────────────────────────────────
nx, ny   = 2, 1
nk       = 1000
γ        = 1.0
γso      = 0.0
γc       = 1.0
j_sd     = 0.1
Temp     = 300.0
t_step   = 0.1
t_end    = 1000.0
t_llg_start = 200.0    # LLG torque activates after this time

stepp(t; ti = 60) = t < ti ? sin((π/2) * t / ti)^4 : 1.0

output_dir = joinpath(@__DIR__, "..", "data")
name       = "gkba_krep_free"

# ── Initialize ────────────────────────────────────────────────────────────────
# Site 1 pointing in x, site 2 pointing in z
vm_init       = zeros(Float64, nx*ny, 3)
vm_init[1, 1] = 1.0    # site 1 → x
vm_init[2, 3] = 1.0    # site 2 → z

dv, ov = init_gkba(; nx, ny, nk, γ, γso, γc, j_sd, Temp, vm_i1x = vm_init)

# Apply initial turn-on to coupling
dv.hseα_ikα .*= stepp(0.0)
@tullio dv.heαs_kαi[k,α,i] := conj(dv.hseα_ikα[i,k,α])

lv = make_llg_params(nx, ny)

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

println("Starting free-precession GKBA  (t_end=$t_end, LLG after t=$t_llg_start)")
elapsed = @elapsed begin
    for (_, t) in enumerate(0.0:t_step:t_end - t_step)
        tt = round(t + t_step, digits = 2)
        step!(integrator, t_step, true)
        unpack!(dv, integrator.u)
        compute_observables!(ov, dv)

        if tt > t_llg_start
            # Free LLG evolution with electron spin-torque
            dv.vm_i1x .= heun(dv.vm_i1x, ov.sden_i1x, t_step, lv)
            # Constrain site 1 to x-axis (fixed external reference moment)
            dv.vm_i1x[1, :] .= [1.0, 0.0, 0.0]
        end

        dv.hs_ij    = build_hs(dv.vm_i1x, nx, ny, γ, γso, j_sd)
        dv.hseα_ikα = build_hseα(nx*ny, γc, nk) .* stepp(tt)
        @tullio dv.heαs_kαi[k,α,i] := conj(dv.hseα_ikα[i,k,α])

        writedlm(files[:cc],     real(ov.curr_α),   ' ')
        writedlm(files[:sneq],   real(ov.sden_i1x), ' ')
        writedlm(files[:sc],     real(ov.scurr_xα), ' ')
        writedlm(files[:cspins], real(dv.vm_i1x),   ' ')
        writedlm(files[:cden],   real(ov.cden_i),   ' ')
    end
end
foreach(close, values(files))
println("Done. $(round(elapsed, digits=1)) s")

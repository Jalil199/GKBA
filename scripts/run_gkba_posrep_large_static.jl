"""
Driver: standard GKBA, position-representation leads, large k-grid, static spins.
Corresponds to HFGKBA-posrep-large.ipynb  (nx=2, nk=200).
"""

using DifferentialEquations
using DelimitedFiles
import GKBA: init_gkba_posrep, eom_gkba_posrep!, compute_observables!, unpack!

# ── Parameters ────────────────────────────────────────────────────────────────
nx, ny   = 2, 1
nk       = 200        # large lead grid (the "large" variant)
γ        = 1.0
γso      = 0.0
γc       = 1.0
j_sd     = 0.0
Temp     = 300.0
μ_α      = [0.0, 0.0]
ϵ0α      = [0.0, 0.0]

# Time grid
t_0      = 0.0
t_step   = 0.1
t_end    = 100.0

# Output
output_dir = joinpath(@__DIR__, "..", "data")
name       = "gkba_posrep_large_static"

# ── Initialize ────────────────────────────────────────────────────────────────
dv, ov = init_gkba_posrep(; nx, ny, nk, γ, γso, γc, j_sd, Temp, μ_α, ϵ0α)

# ── Set up integrator ─────────────────────────────────────────────────────────
prob       = ODEProblem(eom_gkba_posrep!, dv.rkvec, (t_0, t_end), dv)
integrator = init(prob, RK4(); dt = t_step, save_everystep = false,
                  adaptive = true, dense = false)

# ── Open output files ─────────────────────────────────────────────────────────
mkpath(output_dir)
files = Dict(
    :cc     => open(joinpath(output_dir, "cc_$(name)_jl.txt"),     "w+"),
    :sneq   => open(joinpath(output_dir, "sneq_$(name)_jl.txt"),   "w+"),
    :sc     => open(joinpath(output_dir, "sc_$(name)_jl.txt"),     "w+"),
    :cden   => open(joinpath(output_dir, "cden_$(name)_jl.txt"),   "w+"),
    :rho    => open(joinpath(output_dir, "rho_$(name)_jl.txt"),    "w+"),
)

# ── Time evolution ────────────────────────────────────────────────────────────
println("Starting time evolution  (t_end = $t_end, dt = $t_step)")
elapsed = @elapsed begin
    for t in t_0:t_step:t_end - t_step
        step!(integrator, t_step, true)
        unpack!(dv, integrator.u)
        compute_observables!(ov, dv)

        writedlm(files[:rho],  transpose(vec(dv.Gls_ij)), ',')
        writedlm(files[:cc],   real(ov.curr_α),           ' ')
        writedlm(files[:sneq], real(ov.sden_i1x),         ' ')
        writedlm(files[:sc],   real(ov.scurr_xα),         ' ')
        writedlm(files[:cden], real(ov.cden_i),           ' ')
    end
end

foreach(close, values(files))
println("Done.  Elapsed: $(round(elapsed, digits=1)) s")

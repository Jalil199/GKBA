"""
Driver: standard GKBA, k-representation, voltage pulse driving.
Corresponds to HFGKBA_krep-leviton.ipynb.

The lead energy levels are shifted by a time-dependent voltage pulse applied
directly to heα_kα. Two pulse shapes are available:
  - Gaussian pulse (active in the reference notebook)
  - Lorentzian (Leviton) pulse (commented out, set pulse_shape = :leviton)

j_sd = 0 → no magnetic moments, pure charge transport.
"""

using DifferentialEquations, DelimitedFiles, Tullio
import GKBA: init_gkba, make_llg_params,
             eom_gkba!, compute_observables!, unpack!,
             build_hs, heun

# ── Parameters ────────────────────────────────────────────────────────────────
nx, ny   = 2, 1
nk       = 500
γ        = 1.0
γso      = 0.0
γc       = 0.0     # γc=0 in the reference notebook init_params
j_sd     = 0.0
Temp     = 300.0
t_step   = 0.1
t_end    = 100.0

# Pulse on left lead (lead 1)
pulse_shape = :gaussian   # :gaussian or :leviton
t0_pulse    = 50.0
τ_pulse     = 1.0
A_pulse     = 0.5

gauss(t; A = A_pulse, t0 = t0_pulse, τ = τ_pulse) = A * exp(-(t - t0)^2 / (2τ^2))
lev(t;   t0 = t0_pulse, τ = τ_pulse)               = 2τ / ((t - t0)^2 + τ^2)

output_dir = joinpath(@__DIR__, "..", "data")
name       = "gkba_krep_leviton"

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

println("Starting Leviton/pulse GKBA  (pulse=$pulse_shape, t0=$t0_pulse, τ=$τ_pulse)")
elapsed = @elapsed begin
    for (_, t) in enumerate(0.0:t_step:t_end - t_step)
        tt = round(t + t_step, digits = 2)
        step!(integrator, t_step, true)
        unpack!(dv, integrator.u)
        compute_observables!(ov, dv)

        # LLG (j_sd=0 → no torque, but kept for consistency)
        dv.vm_i1x .= heun(dv.vm_i1x, ov.sden_i1x, t_step, lv)
        dv.hs_ij   = build_hs(dv.vm_i1x, nx, ny, γ, γso, j_sd)

        # Time-dependent voltage pulse on lead 1
        V = pulse_shape == :leviton ? lev(tt) : gauss(tt)
        dv.heα_kα[1:2:end, 1] .= V
        dv.heα_kα[2:2:end, 1] .= V

        writedlm(files[:cc],     real(ov.curr_α),   ' ')
        writedlm(files[:sneq],   real(ov.sden_i1x), ' ')
        writedlm(files[:sc],     real(ov.scurr_xα), ' ')
        writedlm(files[:cspins], real(dv.vm_i1x),   ' ')
        writedlm(files[:cden],   real(ov.cden_i),   ' ')
    end
end
foreach(close, values(files))
println("Done. $(round(elapsed, digits=1)) s")

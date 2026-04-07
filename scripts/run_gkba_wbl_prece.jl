"""
Driver: GKBA in the wide-band limit, precessing spins (2 sites, 2 leads).
Corresponds to HFGKBA_prece-WBL.ipynb.

Same as run_gkba_wbl_free.jl — both notebooks share the same structure.
Adjust theta_1, phi_1, period, t_end and stepp parameters as needed.
"""

using DifferentialEquations
using DelimitedFiles
import GKBA: init_gkba_wbl, eom_gkba_wbl!, compute_observables!,
             PrecSpin, update!, build_hs
import GKBA: unpack!

# ── Parameters ────────────────────────────────────────────────────────────────
nx, ny   = 2, 1
nk       = 20          # Ozaki pole parameter; n_oz = nk*nσ = 40 poles
γ        = 1.0
γso      = 0.0
γc       = 1.0
j_sd     = 0.1
Temp     = 300.0

# Time grid
t_0      = 0.0
t_step   = 0.1
t_end    = 120.0       # longer run to resolve the steady-state pumping

# Precession
theta_1  = 20.0
phi_1    = 0.0
period   = 5.0

# Coupling turn-on envelope
stepp(t; ti = 3, to = 25) = 1 / (1 + exp(-(t - to) / ti))

# Output
output_dir = joinpath(@__DIR__, "..", "data")
name       = "gkba_wbl_prece"

# ── Initialize ────────────────────────────────────────────────────────────────
pr_spins = [PrecSpin(i; theta_zero = theta_1, phi_zero = phi_1, T = period)
            for i in 1:nx]

vm_init = zeros(Float64, nx*ny, 3)
for ps in pr_spins
    update!(ps, 0.0)
    vm_init[ps.i, :] .= ps.s
end

dv, ov = init_gkba_wbl(; nx, ny, nk, γ, γso, γc, j_sd, Temp, vm_i1x = vm_init)
dv.s_α .= stepp(t_0)

# ── Set up integrator ─────────────────────────────────────────────────────────
prob       = ODEProblem(eom_gkba_wbl!, dv.rkvec, (t_0, t_end), dv)
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

        # Update precessing spins and system Hamiltonian
        for ps in pr_spins
            update!(ps, tt)
            dv.vm_i1x[ps.i, :] .= ps.s
        end
        dv.hs_ij = build_hs(dv.vm_i1x, nx, ny, γ, γso, j_sd)

        # Update coupling envelope
        dv.s_α .= stepp(tt)

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

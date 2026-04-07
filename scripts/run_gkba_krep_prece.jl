"""
Driver: standard GKBA, k-representation, precessing spins (2 sites, 2 leads).
Corresponds to HFGKBA_krep-prece.ipynb.
"""

using DifferentialEquations
using DelimitedFiles
import GKBA: init_gkba, make_llg_params,
             eom_gkba!, compute_observables!,
             PrecSpin, update!, build_hs, build_hseα

# ── Parameters ────────────────────────────────────────────────────────────────
nx, ny   = 2, 1
nk       = 400
γ        = 1.0
γso      = 0.0
γc       = 1.0
j_sd     = 0.1
Temp     = 300.0
μ_α      = [0.0, 0.0]
ϵ0α      = [0.0, 0.0]

# Time grid
t_0      = 0.0
t_step   = 0.1
t_end    = 120.0

# Precession
theta_1  = 20.0      # polar angle (degrees)
phi_1    = 0.0       # azimuthal offset (degrees)
period   = 5.0       # precession period

# Turn-on envelope  γc(t) → smooth step from 0 to 1
stepp(t; ti = 3, to = 25) = 1 / (1 + exp(-(t - to) / ti))

# Output
output_dir = joinpath(@__DIR__, "..", "data")
name       = "gkba_krep_prece"

# ── Initialize ────────────────────────────────────────────────────────────────
pr_spins = [PrecSpin(i; theta_zero = theta_1, phi_zero = phi_1, T = period)
            for i in 1:nx]

vm_init = zeros(Float64, nx*ny, 3)
for ps in pr_spins
    update!(ps, 0.0)
    vm_init[ps.i, :] .= ps.s
end

dv, ov = init_gkba(; nx, ny, nk, γ, γso, γc, j_sd, Temp, μ_α, ϵ0α,
                     vm_i1x = vm_init)

# Apply initial turn-on envelope to coupling
dv.hseα_ikα .*= stepp(t_0)
import Tullio: @tullio
@tullio dv.heαs_kαi[k,α,i] := conj(dv.hseα_ikα[i,k,α])

lv = make_llg_params(nx, ny)

# ── Set up integrator ─────────────────────────────────────────────────────────
prob       = ODEProblem(eom_gkba!, dv.rkvec, (t_0, t_end), dv)
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
    for (_, t) in enumerate(t_0:t_step:t_end - t_step)
        tt = round(t + t_step, digits = 2)

        step!(integrator, t_step, true)

        import GKBA: unpack!
        unpack!(dv, integrator.u)
        compute_observables!(ov, dv)

        # Update precessing spins
        for ps in pr_spins
            update!(ps, tt)
            dv.vm_i1x[ps.i, :] .= ps.s
        end

        # Update Hamiltonians
        dv.hs_ij     = build_hs(dv.vm_i1x, nx, ny, γ, γso, j_sd)
        dv.hseα_ikα  = build_hseα(nx*ny, γc, nk) .* stepp(tt)
        @tullio dv.heαs_kαi[k,α,i] := conj(dv.hseα_ikα[i,k,α])

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

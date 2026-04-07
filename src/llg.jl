# ─────────────────────────────────────────────────────────────────────────────
# Precessing spin
# ─────────────────────────────────────────────────────────────────────────────

"""
    PrecSpin

Rigid precessing spin used as an external driving field for the s-d coupling.
The spin precesses around an axis defined by (axis_theta, axis_phi) with
polar angle theta_zero and azimuthal offset phi_zero at t=start_time.
"""
mutable struct PrecSpin
    i          :: Int
    axis_phi   :: Float64
    axis_theta :: Float64
    phi_zero   :: Float64
    theta_zero :: Float64
    start_time :: Float64
    T          :: Float64      # period
    s          :: Vector{Float64}
    function PrecSpin(i::Int = 0;
                      axis_phi   = 0.0, axis_theta = 0.0,
                      phi_zero   = 0.0, theta_zero = 0.0,
                      start_time = 0.0, T = 1.0)
        new(i, axis_phi, axis_theta, phi_zero, theta_zero, start_time, T, [0.0, 0.0, 1.0])
    end
end

function update!(p::PrecSpin, time::Float64)
    t = max(0.0, time - p.start_time)
    ω = 2π / p.T
    θ = π * p.theta_zero / 180
    ϕ = π * p.phi_zero   / 180

    sz = cos(θ)
    sx = cos(ϕ + ω*t) * sin(θ)
    sy = sin(ϕ + ω*t) * sin(θ)

    # Rotate around axis (axis_theta, axis_phi)
    aθ = π * p.axis_theta / 180
    aϕ = π * p.axis_phi   / 180

    sx1 =  sx*cos(aθ) - sz*sin(aθ)
    sy1 =  sy
    sz1 =  sx*sin(aθ) + sz*cos(aθ)

    p.s[1] =  sx1*cos(aϕ) + sy1*sin(aϕ)
    p.s[2] = -sx1*sin(aϕ) + sy1*cos(aϕ)
    p.s[3] =  sz1
    nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# LLG dynamics (Landau-Lifshitz-Gilbert)
# ─────────────────────────────────────────────────────────────────────────────

"""
    heff(vm_a1x, vs_a1x, lp) → hef_a1x

Effective field for the LLG equation including exchange, s-d, anisotropy and
demagnetization contributions.
"""
function heff(vm_a1x::Matrix{Float64}, vs_a1x::Matrix{Float64}, lp::LLGParams)
    one_x = Matrix{Float64}(I, lp.nx, lp.nx)
    one_y = Matrix{Float64}(I, lp.ny, lp.ny)
    J_exc = (diagm(-1 => lp.jxs_exc, 1 => lp.jxs_exc) ⊗ one_y +
              one_x ⊗ diagm(-1 => lp.jys_exc, 1 => lp.jys_exc))
    hef = zeros(Float64, lp.nx*lp.ny, 3)
    @tullio hef[a2,x]  = J_exc[a1,a2] * vm_a1x[a1,x] / MBOHR
    @tullio hef[a2,x] += lp.js_sd_a1[a2] * vs_a1x[a2,x] / MBOHR
    @tullio hef[a2,x] += lp.js_ani_a1[a2] * vm_a1x[a2,x1] * lp.e_x[x1] * lp.e_x[x] / MBOHR
    @tullio hef[a2,x] += -lp.js_dem_a1[a2] * vm_a1x[a2,x1] * lp.e_demag_x[x1] * lp.e_demag_x[x] / MBOHR
    @tullio hef[a2,x] += lp.h0_a1x[a2,x]
    return hef
end

function corrector(vm_a1x::Matrix{Float64}, vs_a1x::Matrix{Float64}, lp::LLGParams)
    hef = heff(vm_a1x, vs_a1x, lp)
    @tullio sh[a,x]  := vm_a1x[a,i] * hef[a,j] * lp.ε[i,j,x]
    @tullio shh[a,x] := vm_a1x[a,i] * sh[a,j]  * lp.ε[i,j,x]
    return (-GAMMA_R / (1 + lp.g_lambda^2)) * (sh + lp.g_lambda * shh)
end

"""
    normalize_rows(m) → m_normalized

Normalize each row of m to unit length.
"""
function normalize_rows(m::Matrix{Float64})
    Matrix(hcat(normalize.(eachrow(m))...)')
end

"""
    heun(vm_a1x, vs_a1x, dt, lp) → vm_new

Propagate classical spin configuration one time step dt using Heun (RK2) method.
"""
function heun(vm_a1x::Matrix{Float64}, vs_a1x::Matrix{Float64},
              dt::Float64, lp::LLGParams)
    vm      = normalize_rows(vm_a1x)
    k1      = corrector(vm, vs_a1x, lp)
    vm_tmp  = normalize_rows(vm + k1 * dt)
    k2      = corrector(vm_tmp, vs_a1x, lp)
    normalize_rows(vm + 0.5 * (k1 + k2) * dt)
end

# ─────────────────────────────────────────────────────────────────────────────
# LLG parameter constructor
# ─────────────────────────────────────────────────────────────────────────────

"""
    make_llg_params(nx, ny; kwargs...) → LLGParams

Build an LLGParams struct. All magnetic couplings default to zero (pure
precession driver use-case). Override via keyword arguments as needed.
"""
function make_llg_params(nx::Int, ny::Int;
                          jx_exc   = 0.0,
                          jy_exc   = 0.0,
                          g_lambda = 0.0,
                          j_sd     = 0.0,
                          j_dmi    = 0.0,
                          j_ani    = 0.0,
                          j_dem    = 0.0,
                          js_pol   = 0.0,
                          js_ana   = 0.0,
                          thop     = 0.0,
                          h0_a1x   = zeros(Float64, nx*ny, 3),
                          e_x      = [0.0, 0.0, 0.0],
                          e_demag_x = [0.0, 0.0, 0.0],
                          dt       = 0.1,
                          nt       = 1)
    ns = nx * ny
    ε = zeros(Int, 3, 3, 3)
    ε[1,2,3] = ε[2,3,1] = ε[3,1,2] =  1
    ε[3,2,1] = ε[2,1,3] = ε[1,3,2] = -1
    LLGParams(
        nx = nx, ny = ny, nt = nt, dt = dt,
        h0_a1x    = h0_a1x,
        jx_exc    = jx_exc,  jy_exc    = jy_exc,
        g_lambda  = g_lambda,
        j_sd      = j_sd,    j_dmi     = j_dmi,
        j_ani     = j_ani,   j_dem     = j_dem,
        js_pol    = js_pol,  js_ana    = js_ana,
        thop      = thop,
        e_x       = e_x,     e_demag_x = e_demag_x,
        js_sd_a1  = fill(j_sd,  ns),
        js_ani_a1 = fill(j_ani, ns),
        js_dem_a1 = fill(j_dem, ns),
        jxs_exc   = fill(jx_exc, max(nx-1, 1)),
        jys_exc   = fill(jy_exc, max(ny-1, 1)),
        ε         = ε,
    )
end

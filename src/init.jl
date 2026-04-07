# ─────────────────────────────────────────────────────────────────────────────
# Standard GKBA initialization
# ─────────────────────────────────────────────────────────────────────────────

"""
    init_gkba(; kwargs...) → (dv::GKBADynamics, ov::ObservablesVar)

Initialize all Green functions and Hamiltonians for a standard GKBA simulation.

The lead lesser GF is treated as static (equilibrium Fermi-Dirac).

# Keyword arguments
- `nx`, `ny`  : lattice dimensions (default 2, 1)
- `nk`        : number of k-points per lead (default 400)
- `γ`, `γso`  : hopping and spin-orbit coupling (default 1.0, 0.0)
- `γc`        : system-lead coupling (default 1.0)
- `j_sd`      : s-d exchange coupling (default 0.1)
- `Temp`      : temperature in K (default 300.0)
- `μ_α`       : chemical potentials [μ_L, μ_R] (default [0.0, 0.0])
- `ϵ0α`       : lead on-site energies (default [0.0, 0.0])
- `vm_i1x`    : initial spin configuration (ns×3). Defaults to zeros.
- `nσ`, `nα`  : spin and lead multiplicity (default 2, 2)
"""
function init_gkba(;
        nx   :: Int     = 2,
        ny   :: Int     = 1,
        nk   :: Int     = 400,
        γ    :: Float64 = 1.0,
        γso  :: Float64 = 0.0,
        γc   :: Float64 = 1.0,
        j_sd :: Float64 = 0.1,
        Temp :: Float64 = 300.0,
        μ_α  :: Vector  = [0.0, 0.0],
        ϵ0α  :: Vector  = [0.0, 0.0],
        vm_i1x :: Union{Matrix{Float64}, Nothing} = nothing,
        nσ   :: Int     = 2,
        nα   :: Int     = 2)

    ns    = nx * ny
    dim_s = ns * nσ
    dim_e = nk * nσ

    dims_Gleαs = (dim_e, nα, dim_s)
    dims_Gls   = (dim_s, dim_s)
    sz_Gleαs   = prod(dims_Gleαs)
    sz_Gls     = prod(dims_Gls)

    vm = vm_i1x !== nothing ? vm_i1x : zeros(Float64, ns, 3)

    # ── Hamiltonians ──────────────────────────────────────────────────────────
    hs_ij    = build_hs(vm, nx, ny, γ, γso, j_sd)
    heα_kα   = build_heα(ϵ0α, γ, nk, nσ, nα)
    hseα_ikα = build_hseα(ns, γc, nk, nσ, nα)
    @tullio heαs_kαi[k,α,i] := conj(hseα_ikα[i,k,α])

    # ── Initial Green functions ───────────────────────────────────────────────
    β   = 1.0 / (K_BOLTZMAN * Temp)
    ks  = 1:nk
    ln  = nk + 1
    ε_k = -2γ * cos.(ks * π / ln)

    # G^<_s(t=0): equilibrium density matrix  G^< = i ρ_eq
    ε_s, U_s = eigen(hs_ij)
    f_s = fermi.((ε_s) * β)
    Gls_ij = ComplexF64.(1im * (U_s' * Diagonal(f_s) * U_s))

    # Static lead GF  g^<_α(k) = i f(ε_k - μ_α)
    gl_kα = zeros(ComplexF64, dim_e, nα)
    for α in 1:nα
        gl_kα[1:2:end, α] .= 1im * fermi.((ε_k .- μ_α[α]) * β)
        gl_kα[2:2:end, α] .= 1im * fermi.((ε_k .- μ_α[α]) * β)
    end

    Gleαs_kαi = zeros(ComplexF64, dims_Gleαs)

    rkvec = zeros(ComplexF64, sz_Gleαs + sz_Gls)
    rkvec[1:sz_Gleαs]              .= vec(Gleαs_kαi)
    rkvec[sz_Gleαs+1:end]          .= vec(Gls_ij)

    σ_ijx = pauli_extended(ns, nσ)

    dv = GKBADynamics(
        hs_ij      = hs_ij,
        heα_kα     = heα_kα,
        heαs_kαi   = heαs_kαi,
        hseα_ikα   = hseα_ikα,
        Gleαs_kαi  = Gleαs_kαi,
        Gls_ij     = Gls_ij,
        gl_kα      = gl_kα,
        rkvec      = rkvec,
        vm_i1x     = vm,
        σ_ijx      = σ_ijx,
        dims_Gleαs = dims_Gleαs,
        dims_Gls   = dims_Gls,
        sz_Gleαs   = sz_Gleαs,
        sz_Gls     = sz_Gls,
    )

    ov = ObservablesVar(
        sden_i1x  = zeros(Float64, ns, 3),
        curr_α    = zeros(Float64, nα),
        scurr_xα  = zeros(Float64, 3, nα),
        sden_xij  = zeros(ComplexF64, 3, dim_s, dim_s),
        cden_i    = zeros(Float64, dim_s),
    )

    return dv, ov
end

# ─────────────────────────────────────────────────────────────────────────────
# Extended GKBA initialization
# ─────────────────────────────────────────────────────────────────────────────

"""
    init_egkba(; kwargs...) → (dv::eGKBADynamics, ov::ObservablesVar)

Initialize all Green functions and Hamiltonians for an extended GKBA simulation.

The lead lesser GF G^<_α(q,k) is initialized as diagonal (Fermi-Dirac) and
then evolves dynamically according to its own EOM.

Same keyword arguments as `init_gkba`, plus:
- `β_override` : if provided, overrides 1/(K_BOLTZMAN*Temp) for lead GF init
  (the eGKBA reference notebook uses β=1.0 instead of physical β)
"""
function init_egkba(;
        nx          :: Int     = 2,
        ny          :: Int     = 1,
        nk          :: Int     = 300,
        γ           :: Float64 = 1.0,
        γso         :: Float64 = 0.0,
        γc          :: Float64 = 1.0,
        j_sd        :: Float64 = 0.1,
        Temp        :: Float64 = 300.0,
        μ_α         :: Vector  = [0.0, 0.0],
        ϵ0α         :: Vector  = [0.0, 0.0],
        vm_i1x :: Union{Matrix{Float64}, Nothing} = nothing,
        nσ          :: Int     = 2,
        nα          :: Int     = 2,
        β_override  :: Union{Float64, Nothing} = nothing)

    ns    = nx * ny
    dim_s = ns * nσ
    dim_e = nk * nσ

    dims_Gleαs = (dim_e, nα, dim_s)
    dims_Gls   = (dim_s, dim_s)
    dims_Gleα  = (dim_e, dim_e, nα)
    sz_Gleαs   = prod(dims_Gleαs)
    sz_Gls     = prod(dims_Gls)
    sz_Gleα    = prod(dims_Gleα)

    vm = vm_i1x !== nothing ? vm_i1x : zeros(Float64, ns, 3)

    # ── Hamiltonians ──────────────────────────────────────────────────────────
    hs_ij    = build_hs(vm, nx, ny, γ, γso, j_sd)
    heα_kα   = build_heα(ϵ0α, γ, nk, nσ, nα)
    hseα_ikα = build_hseα(ns, γc, nk, nσ, nα)
    @tullio heαs_kαi[k,α,i] := conj(hseα_ikα[i,k,α])

    # ── Initial Green functions ───────────────────────────────────────────────
    β_phys = 1.0 / (K_BOLTZMAN * Temp)
    β      = β_override !== nothing ? β_override : β_phys
    ks     = 1:nk
    ln     = nk + 1
    ε_k    = -2γ * cos.(ks * π / ln)

    Gleαs_kαi = zeros(ComplexF64, dims_Gleαs)
    Gls_ij    = zeros(ComplexF64, dims_Gls)

    # G^<_α(q,k) = i f(ε_k) δ_{qk}  (initial equilibrium)
    Gleα_qkα = zeros(ComplexF64, dims_Gleα)
    for α in 1:nα
        f_α = fermi.((ε_k .- μ_α[α]) * β)
        Gleα_qkα[1:2:end, 1:2:end, α] .= Diagonal(ComplexF64.(1im * f_α))
        Gleα_qkα[2:2:end, 2:2:end, α] .= Diagonal(ComplexF64.(1im * f_α))
    end

    rkvec = zeros(ComplexF64, sz_Gleαs + sz_Gls + sz_Gleα)
    rkvec[1:sz_Gleαs]                          .= vec(Gleαs_kαi)
    rkvec[sz_Gleαs+1:sz_Gleαs+sz_Gls]         .= vec(Gls_ij)
    rkvec[sz_Gleαs+sz_Gls+1:end]              .= vec(Gleα_qkα)

    σ_ijx = pauli_extended(ns, nσ)

    dv = eGKBADynamics(
        hs_ij      = hs_ij,
        heα_kα     = heα_kα,
        heαs_kαi   = heαs_kαi,
        hseα_ikα   = hseα_ikα,
        Gleαs_kαi  = Gleαs_kαi,
        Gls_ij     = Gls_ij,
        Gleα_qkα   = Gleα_qkα,
        rkvec      = rkvec,
        vm_i1x     = vm,
        t          = 0.0,
        σ_ijx      = σ_ijx,
        dims_Gleαs = dims_Gleαs,
        dims_Gls   = dims_Gls,
        dims_Gleα  = dims_Gleα,
        sz_Gleαs   = sz_Gleαs,
        sz_Gls     = sz_Gls,
        sz_Gleα    = sz_Gleα,
    )

    ov = ObservablesVar(
        sden_i1x  = zeros(Float64, ns, 3),
        curr_α    = zeros(Float64, nα),
        scurr_xα  = zeros(Float64, 3, nα),
        sden_xij  = zeros(ComplexF64, 3, dim_s, dim_s),
        cden_i    = zeros(Float64, dim_s),
    )

    return dv, ov
end

# ─────────────────────────────────────────────────────────────────────────────
# Standard GKBA initialization — position-representation leads
# ─────────────────────────────────────────────────────────────────────────────

"""
    init_gkba_posrep(; kwargs...) → (dv::PosRepDynamics, ov::ObservablesVar)

Initialize for standard GKBA with leads described as finite tight-binding chains
in position space.  The lead lesser GF is static (equilibrium Fermi-Dirac,
computed from eigendecomposition of the lead Hamiltonian).
"""
function init_gkba_posrep(;
        nx   :: Int     = 2,
        ny   :: Int     = 1,
        nk   :: Int     = 200,
        γ    :: Float64 = 1.0,
        γso  :: Float64 = 0.0,
        γc   :: Float64 = 1.0,
        j_sd :: Float64 = 0.0,
        Temp :: Float64 = 300.0,
        μ_α  :: Vector  = [0.0, 0.0],
        ϵ0α  :: Vector  = [0.0, 0.0],
        vm_i1x :: Union{Matrix{Float64}, Nothing} = nothing,
        nσ   :: Int     = 2,
        nα   :: Int     = 2)

    ns    = nx * ny
    dim_s = ns * nσ
    dim_e = nk * nσ

    dims_Gleαs = (dim_e, nα, dim_s)
    dims_Gls   = (dim_s, dim_s)
    sz_Gleαs   = prod(dims_Gleαs)
    sz_Gls     = prod(dims_Gls)

    vm = vm_i1x !== nothing ? vm_i1x : zeros(Float64, ns, 3)

    # ── Hamiltonians ──────────────────────────────────────────────────────────
    hs_ij    = build_hs(vm, nx, ny, γ, γso, j_sd)
    heα_qkα  = build_heα_posrep(ϵ0α, γ, nk, nσ, nα)
    hseα_ikα = build_hseα_posrep(ns, γc, nk, nσ, nα)
    @tullio heαs_kαi[k,α,i] := conj(hseα_ikα[i,k,α])

    # ── Initial Green functions ───────────────────────────────────────────────
    β = 1.0 / (K_BOLTZMAN * Temp)

    # G^<_s(t=0): equilibrium density matrix of isolated system
    ε_s, U_s = eigen(hs_ij)
    f_s      = fermi.((ε_s) * β)
    Gls_ij   = ComplexF64.(1im * (U_s' * Diagonal(f_s) * U_s))

    # Static lead GF: g^<_α(q,k) = i U f(ε - μ_α) U†
    gl_qkα = zeros(ComplexF64, dim_e, dim_e, nα)
    for α in 1:nα
        ε_e, U_e = eigen(heα_qkα[:,:,α])
        f_e      = fermi.((ε_e .- μ_α[α]) * β)
        gl_qkα[:,:,α] = 1im * U_e * Diagonal(ComplexF64.(f_e)) * U_e'
    end

    Gleαs_kαi = zeros(ComplexF64, dims_Gleαs)
    rkvec = zeros(ComplexF64, sz_Gleαs + sz_Gls)
    rkvec[1:sz_Gleαs]     .= vec(Gleαs_kαi)
    rkvec[sz_Gleαs+1:end] .= vec(Gls_ij)

    σ_ijx = pauli_extended(ns, nσ)

    dv = PosRepDynamics(
        hs_ij      = hs_ij,
        heα_qkα    = heα_qkα,
        heαs_kαi   = heαs_kαi,
        hseα_ikα   = hseα_ikα,
        Gleαs_kαi  = Gleαs_kαi,
        Gls_ij     = Gls_ij,
        gl_qkα     = gl_qkα,
        rkvec      = rkvec,
        vm_i1x     = vm,
        σ_ijx      = σ_ijx,
        dims_Gleαs = dims_Gleαs,
        dims_Gls   = dims_Gls,
        sz_Gleαs   = sz_Gleαs,
        sz_Gls     = sz_Gls,
    )

    ov = ObservablesVar(
        sden_i1x  = zeros(Float64, ns, 3),
        curr_α    = zeros(Float64, nα),
        scurr_xα  = zeros(Float64, 3, nα),
        sden_xij  = zeros(ComplexF64, 3, dim_s, dim_s),
        cden_i    = zeros(Float64, dim_s),
    )

    return dv, ov
end

# ─────────────────────────────────────────────────────────────────────────────
# Extended GKBA initialization — position-representation leads
# ─────────────────────────────────────────────────────────────────────────────

"""
    init_egkba_posrep(; kwargs...) → (dv::ePosRepDynamics, ov::ObservablesVar)

Initialize for extended GKBA with leads in position representation.
The lead lesser GF G^<_α(q,k) is initialized from equilibrium (Fermi-Dirac)
and evolves dynamically.
"""
function init_egkba_posrep(;
        nx          :: Int     = 2,
        ny          :: Int     = 1,
        nk          :: Int     = 50,
        γ           :: Float64 = 1.0,
        γso         :: Float64 = 0.0,
        γc          :: Float64 = 1.0,
        j_sd        :: Float64 = 0.1,
        Temp        :: Float64 = 300.0,
        μ_α         :: Vector  = [0.0, 0.0],
        ϵ0α         :: Vector  = [0.0, 0.0],
        vm_i1x :: Union{Matrix{Float64}, Nothing} = nothing,
        nσ          :: Int     = 2,
        nα          :: Int     = 2,
        β_override  :: Union{Float64, Nothing} = nothing)

    ns    = nx * ny
    dim_s = ns * nσ
    dim_e = nk * nσ

    dims_Gleαs = (dim_e, nα, dim_s)
    dims_Gls   = (dim_s, dim_s)
    dims_Gleα  = (dim_e, dim_e, nα)
    sz_Gleαs   = prod(dims_Gleαs)
    sz_Gls     = prod(dims_Gls)
    sz_Gleα    = prod(dims_Gleα)

    vm = vm_i1x !== nothing ? vm_i1x : zeros(Float64, ns, 3)

    # ── Hamiltonians ──────────────────────────────────────────────────────────
    hs_ij    = build_hs(vm, nx, ny, γ, γso, j_sd)
    heα_qkα  = build_heα_posrep(ϵ0α, γ, nk, nσ, nα)
    hseα_ikα = build_hseα_posrep(ns, γc, nk, nσ, nα)
    @tullio heαs_kαi[k,α,i] := conj(hseα_ikα[i,k,α])

    # ── Initial Green functions ───────────────────────────────────────────────
    β_phys = 1.0 / (K_BOLTZMAN * Temp)
    β      = β_override !== nothing ? β_override : β_phys

    # G^<_α(q,k) = i U f(ε - μ_α) U†  (equilibrium lead GF)
    Gleα_qkα = zeros(ComplexF64, dims_Gleα)
    for α in 1:nα
        ε_e, U_e = eigen(heα_qkα[:,:,α])
        f_e      = fermi.((ε_e .- μ_α[α]) * β)
        Gleα_qkα[:,:,α] = 1im * U_e * Diagonal(ComplexF64.(f_e)) * U_e'
    end

    # G^<_s(t=0): equilibrium density matrix of isolated system
    ε_s, U_s = eigen(hs_ij)
    f_s      = fermi.((ε_s) * β_phys)
    Gls_ij   = ComplexF64.(1im * (U_s' * Diagonal(f_s) * U_s))

    Gleαs_kαi = zeros(ComplexF64, dims_Gleαs)
    rkvec = zeros(ComplexF64, sz_Gleαs + sz_Gls + sz_Gleα)
    rkvec[1:sz_Gleαs]                  .= vec(Gleαs_kαi)
    rkvec[sz_Gleαs+1:sz_Gleαs+sz_Gls] .= vec(Gls_ij)
    rkvec[sz_Gleαs+sz_Gls+1:end]      .= vec(Gleα_qkα)

    σ_ijx = pauli_extended(ns, nσ)

    dv = ePosRepDynamics(
        hs_ij      = hs_ij,
        heα_qkα    = heα_qkα,
        heαs_kαi   = heαs_kαi,
        hseα_ikα   = hseα_ikα,
        Gleαs_kαi  = Gleαs_kαi,
        Gls_ij     = Gls_ij,
        Gleα_qkα   = Gleα_qkα,
        rkvec      = rkvec,
        vm_i1x     = vm,
        t          = 0.0,
        σ_ijx      = σ_ijx,
        dims_Gleαs = dims_Gleαs,
        dims_Gls   = dims_Gls,
        dims_Gleα  = dims_Gleα,
        sz_Gleαs   = sz_Gleαs,
        sz_Gls     = sz_Gls,
        sz_Gleα    = sz_Gleα,
    )

    ov = ObservablesVar(
        sden_i1x  = zeros(Float64, ns, 3),
        curr_α    = zeros(Float64, nα),
        scurr_xα  = zeros(Float64, 3, nα),
        sden_xij  = zeros(ComplexF64, 3, dim_s, dim_s),
        cden_i    = zeros(Float64, dim_s),
    )

    return dv, ov
end

# ─────────────────────────────────────────────────────────────────────────────
# GKBA initialization — wide-band limit with Ozaki poles
# ─────────────────────────────────────────────────────────────────────────────

"""
    init_gkba_wbl(; kwargs...) → (dv::WBLDynamics, ov::ObservablesVar)

Initialize for GKBA in the wide-band limit.

The lead self-energy is Σ^r = −iΓ/2 (constant), and the Fermi function
is decomposed with `n_oz = nk*nσ` Ozaki poles.
`s_α` is initialized to zero and must be updated externally in the time loop.

# Keyword arguments
- `nx`, `ny`  : lattice dimensions (default 2, 1)
- `nk`        : Ozaki pole parameter; n_oz = nk*nσ poles (default 20)
- `γ`, `γso`  : hopping and spin-orbit of the system (default 1.0, 0.0)
- `γc`        : system-lead coupling (default 1.0)
- `j_sd`      : s-d exchange coupling (default 0.1)
- `Temp`      : temperature in K (default 300.0)
- `nσ`, `nα`  : spin and lead multiplicity (default 2, 2)
- `vm_i1x`    : initial spin configuration (ns×3). Defaults to zeros.
"""
function init_gkba_wbl(;
        nx   :: Int     = 2,
        ny   :: Int     = 1,
        nk   :: Int     = 20,
        γ    :: Float64 = 1.0,
        γso  :: Float64 = 0.0,
        γc   :: Float64 = 1.0,
        j_sd :: Float64 = 0.1,
        Temp :: Float64 = 300.0,
        nσ   :: Int     = 2,
        nα   :: Int     = 2,
        vm_i1x :: Union{Matrix{Float64}, Nothing} = nothing)

    ns    = nx * ny
    dim_s = ns * nσ
    n_oz  = nk * nσ           # number of Ozaki poles

    dims_Gleαs = (n_oz, nα, dim_s, dim_s)
    dims_Gls   = (dim_s, dim_s)
    sz_Gleαs   = prod(dims_Gleαs)
    sz_Gls     = prod(dims_Gls)

    vm = vm_i1x !== nothing ? vm_i1x : zeros(Float64, ns, 3)

    # ── Hamiltonians ──────────────────────────────────────────────────────────
    hs_ij = build_hs(vm, nx, ny, γ, γso, j_sd)

    # ── WBL parameters ───────────────────────────────────────────────────────
    β     = 1.0 / (K_BOLTZMAN * Temp)
    ξ_k, η_k = ozaki_poles(n_oz)
    Γ_αij = build_Gamma_wbl(ns, γ, γc, nσ, nα)
    s_α   = zeros(Float64, nα)   # starts at zero; updated externally via stepp

    # ── Initial Green functions ───────────────────────────────────────────────
    ε_s, U_s = eigen(hs_ij)
    f_s      = fermi.((ε_s) * β)
    Gls_ij   = ComplexF64.(1im * (U_s' * Diagonal(f_s) * U_s))

    Gleαs_kαij = zeros(ComplexF64, dims_Gleαs)

    rkvec = zeros(ComplexF64, sz_Gleαs + sz_Gls)
    rkvec[1:sz_Gleαs]     .= vec(Gleαs_kαij)
    rkvec[sz_Gleαs+1:end] .= vec(Gls_ij)

    σ_ijx = pauli_extended(ns, nσ)

    dv = WBLDynamics(
        hs_ij      = hs_ij,
        Gleαs_kαij = Gleαs_kαij,
        Gls_ij     = Gls_ij,
        rkvec      = rkvec,
        vm_i1x     = vm,
        σ_ijx      = σ_ijx,
        ξ_k        = ξ_k,
        η_k        = η_k,
        s_α        = s_α,
        Γ_αij      = Γ_αij,
        β          = β,
        dims_Gleαs = dims_Gleαs,
        dims_Gls   = dims_Gls,
        sz_Gleαs   = sz_Gleαs,
        sz_Gls     = sz_Gls,
    )

    ov = ObservablesVar(
        sden_i1x  = zeros(Float64, ns, 3),
        curr_α    = zeros(Float64, nα),
        scurr_xα  = zeros(Float64, 3, nα),
        sden_xij  = zeros(ComplexF64, 3, dim_s, dim_s),
        cden_i    = zeros(Float64, dim_s),
    )

    return dv, ov
end

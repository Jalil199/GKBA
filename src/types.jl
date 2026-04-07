# ─────────────────────────────────────────────────────────────────────────────
# Observables
# ─────────────────────────────────────────────────────────────────────────────
Base.@kwdef mutable struct ObservablesVar
    sden_i1x  :: Matrix{Float64}          # (ns, 3)       site-resolved spin density
    curr_α    :: Vector{Float64}          # (nα,)         charge current per lead
    scurr_xα  :: Matrix{Float64}          # (3, nα)       spin current per lead
    sden_xij  :: Array{ComplexF64,3}      # (3, dim_s, dim_s)
    cden_i    :: Vector{Float64}          # (dim_s,)      charge density
end

# ─────────────────────────────────────────────────────────────────────────────
# Standard GKBA dynamics
#   Lead lesser GF  →  gl_kα[k,α]  (static, initialized from Fermi-Dirac)
# ─────────────────────────────────────────────────────────────────────────────
Base.@kwdef mutable struct GKBADynamics
    # Hamiltonians
    hs_ij      :: Matrix{ComplexF64}       # (dim_s, dim_s)
    heα_kα     :: Matrix{ComplexF64}       # (dim_e, nα)
    heαs_kαi   :: Array{ComplexF64,3}      # (dim_e, nα, dim_s)
    hseα_ikα   :: Array{ComplexF64,3}      # (dim_s, dim_e, nα)
    # Green functions
    Gleαs_kαi  :: Array{ComplexF64,3}      # (dim_e, nα, dim_s)
    Gls_ij     :: Matrix{ComplexF64}       # (dim_s, dim_s)
    gl_kα      :: Matrix{ComplexF64}       # (dim_e, nα)  equilibrium lead GF (static)
    # State vector (flattened Gleαs + Gls)
    rkvec      :: Vector{ComplexF64}
    # Classical spin
    vm_i1x     :: Matrix{Float64}          # (ns, 3)
    # Extended Pauli matrices σ_ijx[i,j,x] = <i|σ_x|j>
    σ_ijx      :: Array{ComplexF64,3}      # (dim_s, dim_s, 3)
    # Cached dimensions
    dims_Gleαs :: NTuple{3,Int}
    dims_Gls   :: NTuple{2,Int}
    sz_Gleαs   :: Int
    sz_Gls     :: Int
end

# ─────────────────────────────────────────────────────────────────────────────
# Extended GKBA dynamics (PRL 130, 246301)
#   Lead lesser GF  →  Gleα_qkα[q,k,α]  (evolves dynamically)
# ─────────────────────────────────────────────────────────────────────────────
Base.@kwdef mutable struct eGKBADynamics
    # Hamiltonians
    hs_ij      :: Matrix{ComplexF64}
    heα_kα     :: Matrix{ComplexF64}
    heαs_kαi   :: Array{ComplexF64,3}
    hseα_ikα   :: Array{ComplexF64,3}
    # Green functions
    Gleαs_kαi  :: Array{ComplexF64,3}      # (dim_e, nα, dim_s)
    Gls_ij     :: Matrix{ComplexF64}       # (dim_s, dim_s)
    Gleα_qkα   :: Array{ComplexF64,3}      # (dim_e, dim_e, nα)  dynamic lead GF
    # State vector (flattened Gleαs + Gls + Gleα)
    rkvec      :: Vector{ComplexF64}
    # Classical spin
    vm_i1x     :: Matrix{Float64}
    t          :: Float64                  # current time (updated by EOM)
    # Extended Pauli matrices
    σ_ijx      :: Array{ComplexF64,3}
    # Cached dimensions
    dims_Gleαs :: NTuple{3,Int}
    dims_Gls   :: NTuple{2,Int}
    dims_Gleα  :: NTuple{3,Int}
    sz_Gleαs   :: Int
    sz_Gls     :: Int
    sz_Gleα    :: Int
end

# ─────────────────────────────────────────────────────────────────────────────
# Standard GKBA, position-representation leads
#   heα_qkα[q,k,α]  — full lead Hamiltonian matrix in position space
#   gl_qkα[q,k,α]   — static equilibrium lead lesser GF (full matrix)
# ─────────────────────────────────────────────────────────────────────────────
Base.@kwdef mutable struct PosRepDynamics
    # Hamiltonians
    hs_ij      :: Matrix{ComplexF64}
    heα_qkα    :: Array{ComplexF64,3}      # (dim_e, dim_e, nα)
    heαs_kαi   :: Array{ComplexF64,3}      # (dim_e, nα, dim_s)
    hseα_ikα   :: Array{ComplexF64,3}      # (dim_s, dim_e, nα)
    # Green functions
    Gleαs_kαi  :: Array{ComplexF64,3}      # (dim_e, nα, dim_s)
    Gls_ij     :: Matrix{ComplexF64}       # (dim_s, dim_s)
    gl_qkα     :: Array{ComplexF64,3}      # (dim_e, dim_e, nα)  static lead GF
    # State vector
    rkvec      :: Vector{ComplexF64}
    # Classical spin
    vm_i1x     :: Matrix{Float64}
    # Extended Pauli matrices
    σ_ijx      :: Array{ComplexF64,3}
    # Cached dimensions
    dims_Gleαs :: NTuple{3,Int}
    dims_Gls   :: NTuple{2,Int}
    sz_Gleαs   :: Int
    sz_Gls     :: Int
end

# ─────────────────────────────────────────────────────────────────────────────
# Extended GKBA, position-representation leads  (PRL 130, 246301)
#   Gleα_qkα[q,k,α]  — dynamic lead lesser GF (full matrix, evolves in time)
# ─────────────────────────────────────────────────────────────────────────────
Base.@kwdef mutable struct ePosRepDynamics
    # Hamiltonians
    hs_ij      :: Matrix{ComplexF64}
    heα_qkα    :: Array{ComplexF64,3}
    heαs_kαi   :: Array{ComplexF64,3}
    hseα_ikα   :: Array{ComplexF64,3}
    # Green functions
    Gleαs_kαi  :: Array{ComplexF64,3}
    Gls_ij     :: Matrix{ComplexF64}
    Gleα_qkα   :: Array{ComplexF64,3}      # (dim_e, dim_e, nα)  dynamic
    # State vector
    rkvec      :: Vector{ComplexF64}
    # Classical spin
    vm_i1x     :: Matrix{Float64}
    t          :: Float64
    # Extended Pauli matrices
    σ_ijx      :: Array{ComplexF64,3}
    # Cached dimensions
    dims_Gleαs :: NTuple{3,Int}
    dims_Gls   :: NTuple{2,Int}
    dims_Gleα  :: NTuple{3,Int}
    sz_Gleαs   :: Int
    sz_Gls     :: Int
    sz_Gleα    :: Int
end

# ─────────────────────────────────────────────────────────────────────────────
# GKBA in the wide-band limit (WBL) with Ozaki pole expansion
#
#   Lead self-energy Σ^r = -iΓ/2  (constant in frequency, WBL)
#   Fermi function expanded as sum of Ozaki poles ξ_k with weights η_k
#   Gleαs_kαij[k,α,i,j]: auxiliary GF for pole k, lead α (full system matrix)
#   s_α[α]: time-dependent coupling scale per lead (can be updated externally)
# ─────────────────────────────────────────────────────────────────────────────
Base.@kwdef mutable struct WBLDynamics
    # System Hamiltonian
    hs_ij      :: Matrix{ComplexF64}         # (dim_s, dim_s)
    # Green functions
    Gleαs_kαij :: Array{ComplexF64,4}        # (n_oz, nα, dim_s, dim_s)
    Gls_ij     :: Matrix{ComplexF64}         # (dim_s, dim_s)
    # State vector
    rkvec      :: Vector{ComplexF64}
    # Classical spin
    vm_i1x     :: Matrix{Float64}            # (ns, 3)
    # Extended Pauli matrices
    σ_ijx      :: Array{ComplexF64,3}        # (dim_s, dim_s, 3)
    # WBL parameters
    ξ_k        :: Vector{Float64}            # (n_oz,) Ozaki poles
    η_k        :: Vector{Float64}            # (n_oz,) Ozaki weights
    s_α        :: Vector{Float64}            # (nα,) coupling scale (time-varying)
    Γ_αij      :: Array{Float64,3}           # (nα, dim_s, dim_s) hybridization
    β          :: Float64                    # inverse temperature (1/kT)
    # Cached dimensions
    dims_Gleαs :: NTuple{4,Int}
    dims_Gls   :: NTuple{2,Int}
    sz_Gleαs   :: Int
    sz_Gls     :: Int
end

# ─────────────────────────────────────────────────────────────────────────────
# LLG parameters
# ─────────────────────────────────────────────────────────────────────────────
Base.@kwdef mutable struct LLGParams
    nx         :: Int
    ny         :: Int
    nt         :: Int
    dt         :: Float64
    h0_a1x     :: Matrix{Float64}          # (ns, 3) external field
    jx_exc     :: Float64
    jy_exc     :: Float64
    g_lambda   :: Float64                  # Gilbert damping
    j_sd       :: Float64
    j_dmi      :: Float64
    j_ani      :: Float64
    j_dem      :: Float64
    js_pol     :: Float64
    js_ana     :: Float64
    thop       :: Float64
    e_x        :: Vector{Float64}          # easy axis
    e_demag_x  :: Vector{Float64}          # demagnetization axis
    js_sd_a1   :: Vector{Float64}          # (ns,)
    js_ani_a1  :: Vector{Float64}          # (ns,)
    js_dem_a1  :: Vector{Float64}          # (ns,)
    jxs_exc    :: Vector{Float64}          # (nx-1,)
    jys_exc    :: Vector{Float64}          # (ny-1,)
    ε          :: Array{Int,3}             # (3,3,3) Levi-Civita
end

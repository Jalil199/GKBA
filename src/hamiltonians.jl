⊗(A, B) = kron(A, B)

"""
    pauli_extended(ns, nσ=2) → σ_ijx

Build the spin operator matrices extended to the full lattice⊗spin space.
σ_ijx[i,j,x] gives the x-component of σ in the (dim_s × dim_s) basis.
"""
function pauli_extended(ns::Int, nσ::Int = 2)
    I_ns = Matrix{ComplexF64}(I, ns, ns)
    cat(I_ns ⊗ σ_x, I_ns ⊗ σ_y, I_ns ⊗ σ_z; dims = 3)
end

"""
    block_h(ny, γ, γso) → (H0, T)

Intra-cell (H0) and inter-cell (T) tight-binding blocks for a strip of width ny.
Includes Rashba SOC with strength γso.
"""
function block_h(ny::Int, γ::Real, γso::Real)
    dim = ny * 2
    One_y = Diagonal(ones(ny))
    Ty    = diagm(-1 => ones(ny - 1))
    T0    = Ty ⊗ (-γ * σ_0 - 1im * γso * σ_x)
    H0    = T0 + T0'
    T     = One_y ⊗ (-γ * σ_0 + 1im * γso * σ_y)
    return H0, T
end

"""
    build_hs(vm_i1x, nx, ny, γ, γso, j_sd) → H_s

System (central region) Hamiltonian on an nx×ny lattice with local magnetic
moments vm_i1x[i,x] coupled via s-d exchange j_sd.
"""
function build_hs(vm_i1x::Matrix{Float64}, nx::Int, ny::Int,
                  γ::Real, γso::Real, j_sd::Real)
    dim   = nx * ny * 2
    HC    = zeros(ComplexF64, dim, dim)
    One_x = Diagonal(ones(nx))
    H0, T = block_h(ny, γ, γso)
    Tx    = diagm(-1 => ones(nx - 1)) ⊗ T
    HC    = (One_x ⊗ H0) + Tx + Tx'
    tmp   = zeros(ComplexF64, nx, nx)
    for i in 1:nx
        tmp[i, i] = 1.0
        HC .+= -j_sd * tmp ⊗ (vm_i1x[i,1]*σ_x + vm_i1x[i,2]*σ_y + vm_i1x[i,3]*σ_z)
        tmp[i, i] = 0.0
    end
    return HC
end

"""
    build_heα(ϵα0, γ, nk, nσ=2, nα=2) → heα_kα

Lead Hamiltonian in k-representation.
heα_kα[k, α] = ε_k + ϵα0[α]  (spin-degenerate, both spin channels).
k-grid: half-open BZ using open boundary conditions → k = 1…nk, ε_k = -2γ cos(kπ/(nk+1)).
"""
function build_heα(ϵα0::AbstractVector, γ::Real, nk::Int,
                   nσ::Int = 2, nα::Int = 2)
    dim_e   = nk * nσ
    heα_kα  = zeros(ComplexF64, dim_e, nα)
    ks      = 1:nk
    ln      = nk + 1
    ε_k     = -2γ * cos.(ks * π / ln)
    for α in 1:nα
        heα_kα[1:2:end, α] .= ε_k .+ ϵα0[α]   # spin-up
        heα_kα[2:2:end, α] .= ε_k .+ ϵα0[α]   # spin-down
    end
    return heα_kα
end

"""
    build_hseα(ns, γc, nk, nσ=2, nα=2) → hseα_ikα

System–lead coupling matrix in k-representation (embedding).
hseα_ikα[i, k, α]: site i ← mode k of lead α.
Lead 1 couples to the first site; lead 2 couples to the last site.
"""
function build_hseα(ns::Int, γc::Real, nk::Int,
                    nσ::Int = 2, nα::Int = 2)
    dim_s    = ns * nσ
    dim_e    = nk * nσ
    hseα_ikα = zeros(ComplexF64, dim_s, dim_e, nα)
    ks       = 1:nk
    ln       = nk + 1
    w        = -γc * sin.(ks * π / ln) * sqrt(2 / ln)
    # Lead 1 → first site (indices 1,2 for spin up/down)
    hseα_ikα[1, 1:2:end, 1] .= w
    hseα_ikα[2, 2:2:end, 1] .= w
    # Lead 2 → last site (indices dim_s-1, dim_s)
    hseα_ikα[end-1, 1:2:end, 2] .= w
    hseα_ikα[end,   2:2:end, 2] .= w
    return hseα_ikα
end

"""
    ozaki_poles(M) → (ξ, η)

Compute M Ozaki poles (imaginary-axis locations ξ) and their weights η
for the Padé expansion of the Fermi function:
    f(ε) ≈ 1/2 − Σ_k (η_k/β) [1/(iξ_k/β − ε) + 1/(−iξ_k/β − ε)]

Built from the eigendecomposition of a 2M×2M symmetric tridiagonal matrix.
"""
function ozaki_poles(M::Int)
    M2 = 2M
    b  = [1.0 / (2*sqrt(4n^2 - 1.0)) for n in 1:M2-1]
    T  = SymTridiagonal(zeros(M2), b)
    F  = eigen(T)
    λ  = F.values
    V  = F.vectors
    pos = findall(>(0.0), λ)
    ξ  = 1.0 ./ λ[pos]
    η  = abs2.(V[1, pos]) ./ (4.0 * λ[pos].^2)
    return ξ, η
end

"""
    build_Gamma_wbl(ns, γ, γc, nσ=2, nα=2) → Γ_αij

Hybridization matrix for WBL leads at ε=0.
Γ_αij[α,i,j] is nonzero only at the edge sites:
  lead 1 → site 1, lead 2 → site ns (spin-diagonal).
Γ(ε=0) = 2γc²/γ  (from the 1D semi-infinite chain self-energy).
"""
function build_Gamma_wbl(ns::Int, γ::Real, γc::Real,
                          nσ::Int = 2, nα::Int = 2)
    dim_s  = ns * nσ
    Γ0     = 2 * γc^2 / γ
    Γ_αij  = zeros(Float64, nα, dim_s, dim_s)
    Γ_αij[1, 1, 1] = Γ0
    Γ_αij[1, 2, 2] = Γ0
    if nα >= 2
        Γ_αij[2, dim_s-1, dim_s-1] = Γ0
        Γ_αij[2, dim_s,   dim_s  ] = Γ0
    end
    return Γ_αij
end

"""
    build_heα_posrep(ϵ0α, γ, nk, nσ=2, nα=2) → heα_qkα

Lead Hamiltonian in position-representation: a 1D tight-binding chain of nk sites.
heα_qkα[:,:,α] is a (nk*nσ)×(nk*nσ) tridiagonal matrix (spin-degenerate, no SOC).
Built by reusing build_hs with zero local moments.
"""
function build_heα_posrep(ϵ0α::AbstractVector, γ::Real, nk::Int,
                           nσ::Int = 2, nα::Int = 2)
    dim_e    = nk * nσ
    heα_qkα  = zeros(ComplexF64, dim_e, dim_e, nα)
    vm0      = zeros(Float64, nk, 3)
    H0       = build_hs(vm0, nk, 1, γ, 0.0, 0.0)   # 1D chain, no SOC, no moments
    for α in 1:nα
        heα_qkα[:, :, α] = H0 + ϵ0α[α] * I(dim_e)
    end
    return heα_qkα
end

"""
    build_hseα_posrep(ns, γc, nk, nσ=2, nα=2) → hseα_ikα

System–lead coupling in position-representation.
Lead 1 couples system site 1 to lead site 1; lead 2 couples system site ns to lead site 1.
Each lead site has nσ=2 spin channels (spin-degenerate coupling).
"""
function build_hseα_posrep(ns::Int, γc::Real, nk::Int,
                            nσ::Int = 2, nα::Int = 2)
    dim_s    = ns * nσ
    dim_e    = nk * nσ
    hseα_ikα = zeros(ComplexF64, dim_s, dim_e, nα)
    # Lead 1 → system site 1
    hseα_ikα[1, 1, 1] = γc
    hseα_ikα[2, 2, 1] = γc
    if nα >= 2
        # Lead 2 → system site ns
        hseα_ikα[dim_s - 1, 1, 2] = γc
        hseα_ikα[dim_s,     2, 2] = γc
    end
    return hseα_ikα
end

"""
    fermi(ϵ) → f

Fermi-Dirac function with clamping to avoid overflow.
"""
function fermi(ϵ::Real)
    ϵ > 36 ? 0.0 : (ϵ < -36 ? 1.0 : 1.0 / (1.0 + exp(ϵ)))
end

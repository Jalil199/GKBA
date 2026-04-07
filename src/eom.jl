# ─────────────────────────────────────────────────────────────────────────────
# Unpack flat state vector → tensor views stored in the dynamics struct
# ─────────────────────────────────────────────────────────────────────────────

function unpack!(dv::GKBADynamics, u::AbstractVector)
    a, b = dv.sz_Gleαs, dv.sz_Gls
    dv.Gleαs_kαi = reshape(u[1:a],       dv.dims_Gleαs)
    dv.Gls_ij    = reshape(u[a+1:a+b],   dv.dims_Gls)
    nothing
end

function unpack!(dv::WBLDynamics, u::AbstractVector)
    a, b = dv.sz_Gleαs, dv.sz_Gls
    dv.Gleαs_kαij = reshape(u[1:a],     dv.dims_Gleαs)
    dv.Gls_ij     = reshape(u[a+1:a+b], dv.dims_Gls)
    nothing
end

function unpack!(dv::PosRepDynamics, u::AbstractVector)
    a, b = dv.sz_Gleαs, dv.sz_Gls
    dv.Gleαs_kαi = reshape(u[1:a],     dv.dims_Gleαs)
    dv.Gls_ij    = reshape(u[a+1:a+b], dv.dims_Gls)
    nothing
end

function unpack!(dv::ePosRepDynamics, u::AbstractVector)
    a, b, c = dv.sz_Gleαs, dv.sz_Gls, dv.sz_Gleα
    dv.Gleαs_kαi = reshape(u[1:a],           dv.dims_Gleαs)
    dv.Gls_ij    = reshape(u[a+1:a+b],       dv.dims_Gls)
    dv.Gleα_qkα  = reshape(u[a+b+1:a+b+c],  dv.dims_Gleα)
    nothing
end

function unpack!(dv::eGKBADynamics, u::AbstractVector)
    a, b, c = dv.sz_Gleαs, dv.sz_Gls, dv.sz_Gleα
    dv.Gleαs_kαi = reshape(u[1:a],           dv.dims_Gleαs)
    dv.Gls_ij    = reshape(u[a+1:a+b],       dv.dims_Gls)
    dv.Gleα_qkα  = reshape(u[a+b+1:a+b+c],  dv.dims_Gleα)
    nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# Standard GKBA equations of motion
#
#   i dG^<_eαs/dt = [h_eα, G^<_eαs] + h_eαs·G^<_s − g^<_α·h_eαs
#   i dG^<_s/dt   = [h_s, G^<_s]   − Σ_α (h_seα·G^<_eαs + h.c.)
#
# Lead GF g^<_α(k) is static (equilibrium Fermi-Dirac, stored in gl_kα).
# ─────────────────────────────────────────────────────────────────────────────
function eom_gkba!(du, u, dv::GKBADynamics, t)
    unpack!(dv, u)
    Gleαs = dv.Gleαs_kαi
    Gls   = dv.Gls_ij
    hs    = dv.hs_ij
    heα   = dv.heα_kα
    heαs  = dv.heαs_kαi
    hseα  = dv.hseα_ikα
    gl    = dv.gl_kα

    dGleαs = similar(Gleαs)
    dGls   = similar(Gls)

    @tullio dGleαs[k,α,i]  = -1im *  heα[k,α] * Gleαs[k,α,i]
    @tullio dGleαs[k,α,i] +=  1im *  Gleαs[k,α,j] * hs[j,i]
    @tullio dGleαs[k,α,i] += -1im *  heαs[k,α,j] * Gls[j,i]
    @tullio dGleαs[k,α,i] +=  1im *  gl[k,α] * heαs[k,α,i]

    @tullio dGls[i,j]  = -1im * hs[i,j1] * Gls[j1,j]
    @tullio dGls[i,j] +=  1im * Gls[i,j1] * hs[j1,j]
    @tullio dGls[i,j] += -1im * hseα[i,k1,α] * Gleαs[k1,α,j]
    @tullio dGls[i,j] += -1im * conj(hseα[j,k1,α] * Gleαs[k1,α,i])

    a, b = dv.sz_Gleαs, dv.sz_Gls
    du[1:a]     .= vec(dGleαs)
    du[a+1:a+b] .= vec(dGls)
    nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# Standard GKBA EOM — position representation
#
#   i dG^<_eαs/dt = Σ_q [h_eα(k,q) G^<_eαs(q) - G^<_α(k,q) h_eαs(q)]
#                 + G^<_eαs(k) h_s - h_eαs(k) G^<_s
#   i dG^<_s/dt   = [h_s, G^<_s] - Σ_α (h_seα G^<_eαs + h.c.)
#
# Lead GF g^<_α(q,k) is static (equilibrium, stored in gl_qkα).
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# GKBA EOM — wide-band limit with Ozaki pole expansion
#
#   ∂_t F[k,α] = i s_α I + i F[k,α]·heff† − (ξ_k/β) F[k,α]
#   ∂_t G^<_s  = −i heff·G^<_s + i Γ/4 + i Σ_{k,α} s_α η_k Γ_α·F[k,α]/β − h.c.
#
# where F[k,α] ≡ Gleαs_kαij[k,α,:,:], heff = hs − i Γ_tot/2,
# and Γ_tot[i,j] = Σ_α s_α² Γ_αij[α,i,j].
# s_α is updated externally (coupling turn-on envelope).
# ─────────────────────────────────────────────────────────────────────────────
function eom_gkba_wbl!(du, u, dv::WBLDynamics, t)
    unpack!(dv, u)
    Gleαs = dv.Gleαs_kαij
    Gls   = dv.Gls_ij
    hs    = dv.hs_ij
    s_α   = dv.s_α
    Γ_αij = dv.Γ_αij
    η_k   = dv.η_k
    ξ_k   = dv.ξ_k
    β     = dv.β

    @tullio Γ_ij[i,j] := Γ_αij[α,i,j] * s_α[α]^2
    heff = hs - 1im/2 * Γ_ij

    # dGleαs[k,α,i,j] = i s_α δ_{ij} + i Gleαs[k,α,i,l]·heff†[l,j] − (ξ_k/β) Gleαs[k,α,i,j]
    dGleαs = zeros(ComplexF64, size(Gleαs))
    n_oz, nα_, dim_s_, _ = size(Gleαs)
    for α in 1:nα_, k in 1:n_oz, i in 1:dim_s_
        dGleαs[k, α, i, i] += 1im * s_α[α]
    end
    @tullio dGleαs[k,α,i,j] += 1im * Gleαs[k,α,i,l] * conj(heff[j,l])
    @tullio dGleαs[k,α,i,j] += -Gleαs[k,α,i,j] * ξ_k[k] / β

    # dGls[i,j] = −i heff·Gls + i Γ/4 + i Σ s_α η_k Γ_α·Gleαs[k,α]/β
    @tullio dGls[i,j] := -1im * heff[i,l] * Gls[l,j]
    @tullio dGls[i,j] +=  1im * Γ_ij[i,j] / 4
    @tullio dGls[i,j] +=  1im * s_α[α] * η_k[k] * Γ_αij[α,i,l] * Gleαs[k,α,l,j] / β
    dGls .-= dGls'                      # enforce anti-Hermitian ∂_t G^<

    a, b = dv.sz_Gleαs, dv.sz_Gls
    du[1:a]     .= vec(dGleαs)
    du[a+1:a+b] .= vec(dGls)
    nothing
end

function eom_gkba_posrep!(du, u, dv::PosRepDynamics, t)
    unpack!(dv, u)
    Gleαs = dv.Gleαs_kαi
    Gls   = dv.Gls_ij
    hs    = dv.hs_ij
    heα   = dv.heα_qkα
    heαs  = dv.heαs_kαi
    hseα  = dv.hseα_ikα
    gl    = dv.gl_qkα

    dGleαs = similar(Gleαs)
    dGls   = similar(Gls)

    @tullio dGleαs[k,α,i]  = -1im *  heα[k,q,α] * Gleαs[q,α,i]
    @tullio dGleαs[k,α,i] +=  1im *  Gleαs[k,α,j] * hs[j,i]
    @tullio dGleαs[k,α,i] += -1im *  heαs[k,α,j] * Gls[j,i]
    @tullio dGleαs[k,α,i] +=  1im *  gl[k,q,α] * heαs[q,α,i]

    @tullio dGls[i,j]  = -1im * hs[i,j1] * Gls[j1,j]
    @tullio dGls[i,j] +=  1im * Gls[i,j1] * hs[j1,j]
    @tullio dGls[i,j] += -1im * hseα[i,k1,α] * Gleαs[k1,α,j]
    @tullio dGls[i,j] += -1im * conj(hseα[j,k1,α] * Gleαs[k1,α,i])

    a, b = dv.sz_Gleαs, dv.sz_Gls
    du[1:a]     .= vec(dGleαs)
    du[a+1:a+b] .= vec(dGls)
    nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# Extended GKBA EOM — position representation  (PRL 130, 246301)
#
#   Same as above but G^<_α(q,k) evolves dynamically.
# ─────────────────────────────────────────────────────────────────────────────
function eom_egkba_posrep!(du, u, dv::ePosRepDynamics, t)
    dv.t = t
    unpack!(dv, u)
    Gleαs = dv.Gleαs_kαi
    Gls   = dv.Gls_ij
    Gleα  = dv.Gleα_qkα
    hs    = dv.hs_ij
    heα   = dv.heα_qkα
    heαs  = dv.heαs_kαi
    hseα  = dv.hseα_ikα

    dGleαs = similar(Gleαs)
    dGls   = similar(Gls)
    dGleα  = similar(Gleα)

    @tullio dGleαs[k,α,i]  = -1im * heα[k,q,α] * Gleαs[q,α,i]
    @tullio dGleαs[k,α,i] +=  1im * Gleαs[k,α,j] * hs[j,i]
    @tullio dGleαs[k,α,i] += -1im * heαs[k,α,j] * Gls[j,i]
    @tullio dGleαs[k,α,i] +=  1im * Gleα[k,q,α] * heαs[q,α,i]

    @tullio dGls[i,j]  = -1im * hs[i,j1] * Gls[j1,j]
    @tullio dGls[i,j] +=  1im * Gls[i,j1] * hs[j1,j]
    @tullio dGls[i,j] += -1im * hseα[i,k1,α] * Gleαs[k1,α,j]
    @tullio dGls[i,j] += -1im * conj(hseα[j,k1,α] * Gleαs[k1,α,i])

    @tullio dGleα[q,k,α]  = -1im * heα[q,k1,α] * Gleα[k1,k,α]
    @tullio dGleα[q,k,α] +=  1im * Gleα[q,k1,α] * heα[k1,k,α]
    @tullio dGleα[q,k,α] +=  1im * conj(Gleαs[k,α,i] * hseα[i,q,α])
    @tullio dGleα[q,k,α] +=  1im * Gleαs[q,α,i] * hseα[i,k,α]

    a, b, c = dv.sz_Gleαs, dv.sz_Gls, dv.sz_Gleα
    du[1:a]         .= vec(dGleαs)
    du[a+1:a+b]     .= vec(dGls)
    du[a+b+1:a+b+c] .= vec(dGleα)
    nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# Extended GKBA equations of motion  (PRL 130, 246301)
#
#   i dG^<_eαs/dt = [h_eα, G^<_eαs] + h_eαs·G^<_s − G^<_α·h_eαs
#   i dG^<_s/dt   = [h_s, G^<_s]   − Σ_α (h_seα·G^<_eαs + h.c.)
#   i dG^<_α/dt   = [h_eα, G^<_α]  + (G^<_eαs·h_seα + h.c.)
#
# The full lead lesser GF G^<_α(q,k) evolves dynamically (no GKBA approximation
# for the leads).
# ─────────────────────────────────────────────────────────────────────────────
function eom_egkba!(du, u, dv::eGKBADynamics, t)
    dv.t = t
    unpack!(dv, u)
    Gleαs = dv.Gleαs_kαi
    Gls   = dv.Gls_ij
    Gleα  = dv.Gleα_qkα
    hs    = dv.hs_ij
    heα   = dv.heα_kα
    heαs  = dv.heαs_kαi
    hseα  = dv.hseα_ikα

    dGleαs = similar(Gleαs)
    dGls   = similar(Gls)
    dGleα  = similar(Gleα)

    @tullio dGleαs[k,α,i]  = -1im * heα[k,α] * Gleαs[k,α,i]
    @tullio dGleαs[k,α,i] +=  1im * Gleαs[k,α,j] * hs[j,i]
    @tullio dGleαs[k,α,i] += -1im * heαs[k,α,j] * Gls[j,i]
    @tullio dGleαs[k,α,i] +=  1im * Gleα[k,k1,α] * heαs[k1,α,i]

    @tullio dGls[i,j]  = -1im * hs[i,j1] * Gls[j1,j]
    @tullio dGls[i,j] +=  1im * Gls[i,j1] * hs[j1,j]
    @tullio dGls[i,j] += -1im * hseα[i,k1,α] * Gleαs[k1,α,j]
    @tullio dGls[i,j] += -1im * conj(hseα[j,k1,α] * Gleαs[k1,α,i])

    @tullio dGleα[q,k,α]  = -1im * heα[q,α] * Gleα[q,k,α]
    @tullio dGleα[q,k,α] +=  1im * Gleα[q,k,α] * heα[k,α]
    @tullio dGleα[q,k,α] +=  1im * conj(Gleαs[k,α,i] * hseα[i,q,α])
    @tullio dGleα[q,k,α] +=  1im * Gleαs[q,α,i] * hseα[i,k,α]

    a, b, c = dv.sz_Gleαs, dv.sz_Gls, dv.sz_Gleα
    du[1:a]         .= vec(dGleαs)
    du[a+1:a+b]     .= vec(dGls)
    du[a+b+1:a+b+c] .= vec(dGleα)
    nothing
end

# WBL observables — defined before the k-rep version so the docstring
# of the main method covers both via Julia's method union doc.
#
# I[α,i,j] = −i s_α² Γ_α·G^<_s/2 − s_α² Γ_α/4 + s_α Σ_k η_k Γ_α·F[k,α]/β
function compute_observables!(ov::ObservablesVar, dv::WBLDynamics)
    s_α   = dv.s_α
    Γ_αij = dv.Γ_αij
    η_k   = dv.η_k
    β     = dv.β
    Gleαs = dv.Gleαs_kαij
    Gls   = dv.Gls_ij
    σ     = dv.σ_ijx
    ns    = size(dv.vm_i1x, 1)

    # Build current tensor I[α,i,j]
    @tullio I_αij[α,i,j] := -1im * s_α[α]^2 * Γ_αij[α,i,l] * Gls[l,j] / 2
    @tullio I_αij[α,i,j] += -1 * s_α[α]^2 * Γ_αij[α,i,j] / 4
    @tullio I_αij[α,i,j] += s_α[α] * η_k[k] * Γ_αij[α,i,l] * Gleαs[k,α,l,j] / β

    # Charge current  J_α = 2 Re[Tr I[α]]
    @tullio ov.curr_α[α] = real(2 * I_αij[α,i,i])

    # Spin current  J^x_{α} = 4π Re[Tr(σ_x · I[α])]
    @tullio ov.scurr_xα[x,α] = real(4π * σ[l,j,x] * I_αij[α,j,l])

    # Spin density (same formula as k-rep)
    @tullio sden_xij[x,i,j] := -1im * Gls[i,j1] * σ[j1,j,x]
    ov.sden_xij .= sden_xij
    for i1 in 1:ns, x in 1:3
        ov.sden_i1x[i1,x] = real(sden_xij[x, 2i1-1, 2i1-1] + sden_xij[x, 2i1, 2i1])
    end

    # Charge density
    @tullio ov.cden_i[i] = real(-1im * Gls[i,i])
    nothing
end

"""
    compute_observables!(ov, dv)

Update `ov` in-place with observables from the current state in `dv`.
Works for GKBADynamics, eGKBADynamics, PosRepDynamics, ePosRepDynamics, and WBLDynamics.
"""
function compute_observables!(ov::ObservablesVar,
                               dv::Union{GKBADynamics, eGKBADynamics,
                                         PosRepDynamics, ePosRepDynamics})
    σ = dv.σ_ijx
    ns = size(dv.vm_i1x, 1)

    # Charge current:  J_α = -2 Re[ Tr(h_seα · G^<_eαs) ]
    @tullio ov.curr_α[α] = real(-conj(dv.hseα_ikα[i,k1,α] * dv.Gleαs_kαi[k1,α,i]))

    # Spin current:  J^x_{α} = 2 Im[ Tr(σ_x · h_seα · G^<_eαs) ]
    @tullio ov.scurr_xα[x,α] = real(-conj(2 * σ[l,j,x] * dv.hseα_ikα[j,k1,α] * dv.Gleαs_kαi[k1,α,l]))

    # Spin density matrix  S_xij = -i G^<_s σ_x
    @tullio sden_xij[x,i,j] := -1im * dv.Gls_ij[i,j1] * σ[j1,j,x]
    ov.sden_xij .= sden_xij

    # Site-resolved spin density  ⟨S^x_i⟩ = Tr_{spin}(S_x at site i)
    for i1 in 1:ns, x in 1:3
        ov.sden_i1x[i1,x] = real(sden_xij[x, 2i1-1, 2i1-1] + sden_xij[x, 2i1, 2i1])
    end

    # Charge density  ⟨n_i⟩ = -i G^<_s[i,i]
    @tullio ov.cden_i[i] = real(-1im * dv.Gls_ij[i,i])
    nothing
end

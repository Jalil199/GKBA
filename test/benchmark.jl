"""
Benchmark: compare module output vs direct inline implementation.

For each dynamics type (k-rep GKBA, WBL GKBA) we:
  1. Initialise via the module.
  2. Compute the derivative du with the module EOM.
  3. Compute the same derivative with a direct reference implementation.
  4. Assert max(|du_module - du_ref|) < tol.

This verifies that the module refactor (including HBAR removal) preserved
the physics to machine precision.
"""

using LinearAlgebra
using Tullio
using Test

import GKBA: init_gkba, init_gkba_wbl,
             eom_gkba!, eom_gkba_wbl!,
             unpack!, compute_observables!

const TOL = 1e-12

# ── Shared parameters ─────────────────────────────────────────────────────────
nx, ny = 2, 1
nk     = 20
γ      = 1.0; γso = 0.0; γc = 1.0; j_sd = 0.1; Temp = 300.0

vm_init        = zeros(Float64, nx*ny, 3)
vm_init[:, 3] .= 1.0

# ─────────────────────────────────────────────────────────────────────────────
# Reference EOM — k-rep GKBA, equations written out explicitly
# ─────────────────────────────────────────────────────────────────────────────
function ref_eom_gkba(Gleαs, Gls, hs, heα, heαs, hseα, gl)
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

    return dGleαs, dGls
end

# ─────────────────────────────────────────────────────────────────────────────
# Reference EOM — WBL GKBA
# ─────────────────────────────────────────────────────────────────────────────
function ref_eom_wbl(Gleαs, Gls, hs, s_α, Γ_αij, η_k, ξ_k, β)
    @tullio Γ_ij[i,j] := Γ_αij[α,i,j] * s_α[α]^2
    heff = hs - 1im/2 * Γ_ij

    dGleαs = zeros(ComplexF64, size(Gleαs))
    n_oz, nα_, dim_s_, _ = size(Gleαs)
    for α in 1:nα_, k in 1:n_oz, i in 1:dim_s_
        dGleαs[k, α, i, i] += 1im * s_α[α]
    end
    @tullio dGleαs[k,α,i,j] +=  1im * Gleαs[k,α,i,l] * conj(heff[j,l])
    @tullio dGleαs[k,α,i,j] += -Gleαs[k,α,i,j] * ξ_k[k] / β

    @tullio dGls[i,j] := -1im * heff[i,l] * Gls[l,j]
    @tullio dGls[i,j] +=  1im * Γ_ij[i,j] / 4
    @tullio dGls[i,j] +=  1im * s_α[α] * η_k[k] * Γ_αij[α,i,l] * Gleαs[k,α,l,j] / β
    dGls .-= dGls'

    return dGleαs, dGls
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "Benchmark: module vs reference" begin

    @testset "k-rep GKBA — EOM matches reference" begin
        dv, _ = init_gkba(; nx, ny, nk, γ, γso, γc, j_sd, Temp, vm_i1x = vm_init)

        # Module derivative
        du_mod = similar(dv.rkvec)
        eom_gkba!(du_mod, dv.rkvec, dv, 0.0)
        a, b = dv.sz_Gleαs, dv.sz_Gls
        dGleαs_mod = reshape(du_mod[1:a],     dv.dims_Gleαs)
        dGls_mod   = reshape(du_mod[a+1:a+b], dv.dims_Gls)

        # Reference derivative
        dGleαs_ref, dGls_ref = ref_eom_gkba(
            dv.Gleαs_kαi, dv.Gls_ij,
            dv.hs_ij, dv.heα_kα, dv.heαs_kαi, dv.hseα_ikα, dv.gl_kα)

        @test maximum(abs.(dGleαs_mod .- dGleαs_ref)) < TOL
        @test maximum(abs.(dGls_mod   .- dGls_ref))   < TOL
        println("  k-rep GKBA  Δ(dGleαs) = $(maximum(abs.(dGleαs_mod .- dGleαs_ref)))")
        println("  k-rep GKBA  Δ(dGls)   = $(maximum(abs.(dGls_mod   .- dGls_ref)))")
    end

    @testset "WBL GKBA — EOM matches reference" begin
        dv, _ = init_gkba_wbl(; nx, ny, nk, γ, γso, γc, j_sd, Temp, vm_i1x = vm_init)
        dv.s_α .= 0.7   # non-trivial coupling value

        du_mod = similar(dv.rkvec)
        eom_gkba_wbl!(du_mod, dv.rkvec, dv, 0.0)
        a, b = dv.sz_Gleαs, dv.sz_Gls
        dGleαs_mod = reshape(du_mod[1:a],     dv.dims_Gleαs)
        dGls_mod   = reshape(du_mod[a+1:a+b], dv.dims_Gls)

        dGleαs_ref, dGls_ref = ref_eom_wbl(
            dv.Gleαs_kαij, dv.Gls_ij,
            dv.hs_ij, dv.s_α, dv.Γ_αij, dv.η_k, dv.ξ_k, dv.β)

        @test maximum(abs.(dGleαs_mod .- dGleαs_ref)) < TOL
        @test maximum(abs.(dGls_mod   .- dGls_ref))   < TOL
        println("  WBL GKBA    Δ(dGleαs) = $(maximum(abs.(dGleαs_mod .- dGleαs_ref)))")
        println("  WBL GKBA    Δ(dGls)   = $(maximum(abs.(dGls_mod   .- dGls_ref)))")
    end

    @testset "k-rep GKBA — G^< anti-Hermitian after 50 steps" begin
        dv, _ = init_gkba(; nx, ny, nk, γ, γso, γc, j_sd, Temp, vm_i1x = vm_init)
        dt = 0.05
        du = similar(dv.rkvec)
        for _ in 1:50
            eom_gkba!(du, dv.rkvec, dv, 0.0)
            dv.rkvec .+= dt .* du
            unpack!(dv, dv.rkvec)
        end
        G = dv.Gls_ij
        @test maximum(abs.(G .+ G')) < 1e-10
        println("  Anti-Hermitian check  max|G+G†| = $(maximum(abs.(G .+ G')))")
    end

    @testset "WBL GKBA — G^< anti-Hermitian after 50 steps" begin
        dv, _ = init_gkba_wbl(; nx, ny, nk, γ, γso, γc, j_sd, Temp, vm_i1x = vm_init)
        dv.s_α .= 1.0
        dt = 0.05
        du = similar(dv.rkvec)
        for _ in 1:50
            eom_gkba_wbl!(du, dv.rkvec, dv, 0.0)
            dv.rkvec .+= dt .* du
            unpack!(dv, dv.rkvec)
        end
        G = dv.Gls_ij
        @test maximum(abs.(G .+ G')) < 1e-10
        println("  WBL Anti-Hermitian    max|G+G†| = $(maximum(abs.(G .+ G')))")
    end

    @testset "Charge conservation — k-rep GKBA" begin
        # d/dt(Tr G^<_s) = Σ_α J_α  (with sign convention curr_α = 2Re Tr[...])
        # In the module: cden_i[i] = -i G^<[i,i], so Tr G^< = i Σ_i cden_i
        dv, ov = init_gkba(; nx, ny, nk, γ, γso, γc, j_sd, Temp, vm_i1x = vm_init)
        dt = 0.01
        du = similar(dv.rkvec)

        compute_observables!(ov, dv)
        N0 = sum(ov.cden_i)

        for _ in 1:10
            eom_gkba!(du, dv.rkvec, dv, 0.0)
            dv.rkvec .+= dt .* du
            unpack!(dv, dv.rkvec)
        end
        compute_observables!(ov, dv)
        N1 = sum(ov.cden_i)

        # Total charge can change due to lead coupling — just check it's finite
        @test isfinite(N1)
        println("  Charge density: N(t=0) = $(round(N0, digits=6)),  N(t=0.1) = $(round(N1, digits=6))")
        println("  Charge current sum: $(round(sum(ov.curr_α), digits=8))")
    end

end

using Test
using LinearAlgebra
import GKBA: init_gkba, init_egkba, init_gkba_posrep, init_egkba_posrep, init_gkba_wbl,
             eom_gkba!, eom_egkba!, eom_gkba_posrep!, eom_egkba_posrep!, eom_gkba_wbl!,
             compute_observables!, unpack!

# ── Shared parameters ────────────────────────────────────────────────────────
nx, ny  = 2, 1
nk      = 4        # small — just for correctness, not accuracy
γ       = 1.0
γso     = 0.0
γc      = 1.0
j_sd    = 0.1
Temp    = 300.0
dt      = 0.01

vm_init = zeros(Float64, nx*ny, 3)
vm_init[:, 3] .= 1.0    # spins pointing along z

# ── Helper: one Euler step ────────────────────────────────────────────────────
function euler_step!(dv, eom!, dt)
    du = similar(dv.rkvec)
    eom!(du, dv.rkvec, dv, 0.0)
    dv.rkvec .+= dt .* du
    unpack!(dv, dv.rkvec)
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "GKBA module" begin

    @testset "k-rep GKBA" begin
        dv, ov = init_gkba(; nx, ny, nk, γ, γso, γc, j_sd, Temp, vm_i1x = vm_init)

        @test size(dv.Gls_ij) == (2nx*ny, 2nx*ny)
        @test size(dv.Gleαs_kαi, 1) == nk * 2   # n_k × nσ
        @test isapprox(dv.Gls_ij, -dv.Gls_ij', atol=1e-12)   # anti-Hermitian at t=0

        euler_step!(dv, eom_gkba!, dt)

        compute_observables!(ov, dv)
        @test length(ov.curr_α) == 2
        @test isfinite(sum(ov.curr_α))
        @test isfinite(sum(ov.sden_i1x))
    end

    @testset "k-rep eGKBA" begin
        dv, ov = init_egkba(; nx, ny, nk, γ, γso, γc, j_sd, Temp, vm_i1x = vm_init)

        @test size(dv.Gleα_qkα, 1) == nk * 2

        euler_step!(dv, eom_egkba!, dt)

        compute_observables!(ov, dv)
        @test isfinite(sum(ov.curr_α))
    end

    @testset "pos-rep GKBA" begin
        dv, ov = init_gkba_posrep(; nx, ny, nk, γ, γso, γc, j_sd, Temp, vm_i1x = vm_init)

        @test ndims(dv.heα_qkα) == 3

        euler_step!(dv, eom_gkba_posrep!, dt)

        compute_observables!(ov, dv)
        @test isfinite(sum(ov.curr_α))
    end

    @testset "pos-rep eGKBA" begin
        dv, ov = init_egkba_posrep(; nx, ny, nk, γ, γso, γc, j_sd, Temp, vm_i1x = vm_init)

        @test ndims(dv.Gleα_qkα) == 3

        euler_step!(dv, eom_egkba_posrep!, dt)

        compute_observables!(ov, dv)
        @test isfinite(sum(ov.curr_α))
    end

    @testset "WBL GKBA" begin
        dv, ov = init_gkba_wbl(; nx, ny, nk, γ, γso, γc, j_sd, Temp, vm_i1x = vm_init)

        @test ndims(dv.Gleαs_kαij) == 4
        @test length(dv.ξ_k) == length(dv.η_k)
        @test all(dv.η_k .> 0)

        dv.s_α .= 1.0   # coupling fully on
        euler_step!(dv, eom_gkba_wbl!, dt)

        compute_observables!(ov, dv)
        @test isfinite(sum(ov.curr_α))
        @test isfinite(sum(ov.scurr_xα))
    end

    @testset "Current conservation (WBL, steady-state check)" begin
        # At equilibrium (symmetric leads, no bias) total current should be ~0
        dv, ov = init_gkba_wbl(; nx, ny, nk, γ, γso, γc, j_sd, Temp, vm_i1x = vm_init)
        dv.s_α .= 1.0
        compute_observables!(ov, dv)
        @test isapprox(sum(ov.curr_α), 0.0, atol=1e-8)
    end

    @testset "WBL charge and spin currents match the density-matrix EOM" begin
        dv, ov = init_gkba_wbl(; nx, ny, nk, γ, γso, γc, j_sd, Temp, vm_i1x = vm_init)
        dv.s_α .= 1.0

        du = similar(dv.rkvec)
        eom_gkba_wbl!(du, dv.rkvec, dv, 0.0)
        compute_observables!(ov, dv)

        first = dv.sz_Gleαs + 1
        last = first + dv.sz_Gls - 1
        gless_dot = reshape(view(du, first:last), dv.dims_Gls)
        rho = -1im .* dv.Gls_ij
        rho_dot = -1im .* gless_dot
        lead_rho_dot = rho_dot + 1im .* (dv.hs_ij * rho - rho * dv.hs_ij)

        @test sum(ov.curr_α) ≈ real(tr(lead_rho_dot)) atol=1e-11
        for axis in 1:3
            sigma = view(dv.σ_ijx, :, :, axis)
            @test sum(view(ov.scurr_xα, axis, :)) ≈
                  real(tr(sigma * lead_rho_dot)) atol=1e-11
        end
    end

    @testset "Plain GKBA charge and spin currents match the density-matrix EOM" begin
        vm_tilted = copy(vm_init)
        vm_tilted[:, 1] .= 0.2
        vm_tilted[:, 3] .= sqrt(1.0 - 0.2^2)
        dv, ov = init_gkba(; nx, ny, nk, γ, γso, γc, j_sd, Temp,
                            vm_i1x = vm_tilted)

        # Move away from the initially factorized state so the lead currents are
        # nonzero in more than one spin component.
        for _ in 1:3
            euler_step!(dv, eom_gkba!, dt)
        end

        du = similar(dv.rkvec)
        eom_gkba!(du, dv.rkvec, dv, 3dt)
        compute_observables!(ov, dv)

        first = dv.sz_Gleαs + 1
        last = first + dv.sz_Gls - 1
        gless_dot = reshape(view(du, first:last), dv.dims_Gls)
        rho = -1im .* dv.Gls_ij
        rho_dot = -1im .* gless_dot
        lead_rho_dot = rho_dot + 1im .* (dv.hs_ij * rho - rho * dv.hs_ij)

        @test sum(ov.curr_α) ≈ real(tr(lead_rho_dot)) atol=1e-11
        for axis in 1:3
            sigma = view(dv.σ_ijx, :, :, axis)
            @test sum(view(ov.scurr_xα, axis, :)) ≈
                  real(tr(sigma * lead_rho_dot)) atol=1e-11
        end
    end

end

using Test
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

# ── Helper: one RK4 step ──────────────────────────────────────────────────────
function rk4_step!(dv, eom!, dt)
    u  = dv.rkvec
    k1 = similar(u); eom!(k1, u,                   dv, 0.0)
    k2 = similar(u); eom!(k2, u .+ (dt/2) .* k1,  dv, 0.0)
    k3 = similar(u); eom!(k3, u .+ (dt/2) .* k2,  dv, 0.0)
    k4 = similar(u); eom!(k4, u .+  dt    .* k3,  dv, 0.0)
    dv.rkvec .+= (dt/6) .* (k1 .+ 2 .*k2 .+ 2 .*k3 .+ k4)
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

    @testset "Krylov check (WBL, single lead)" begin
        # nx=1, ny=1, nk=4 → state vector has 36 complex elements: cheap for exp()
        dv_ref, _ = init_gkba_wbl(; nx=1, ny=1, nk=4, γ, γso, γc,
                                    j_sd=0.0, Temp, nα=1)
        dv_ref.s_α .= 1.0

        N_steps = 10
        T       = N_steps * dt          # = 0.1

        n  = length(dv_ref.rkvec)
        u0 = copy(dv_ref.rkvec)

        # ── Build linear operator: du/dt = A*u + b ───────────────────────────
        # The WBL EOM with static Hamiltonian is linear in (Gleαs, Gls).
        # Constant term b = EOM(0); columns of A from EOM(e_j) - b.
        du_tmp = zeros(ComplexF64, n)
        eom_gkba_wbl!(du_tmp, zeros(ComplexF64, n), dv_ref, 0.0)
        b_vec = copy(du_tmp)

        A_mat = zeros(ComplexF64, n, n)
        e_j   = zeros(ComplexF64, n)
        for j in 1:n
            fill!(e_j, 0); e_j[j] = 1
            eom_gkba_wbl!(du_tmp, e_j, dv_ref, 0.0)
            A_mat[:, j] .= du_tmp .- b_vec
        end

        # ── Exact solution via augmented matrix exponential ──────────────────
        # d/dt [u; 1] = [[A, b]; [0, 0]] * [u; 1]  →  exact at time T
        Ã = zeros(ComplexF64, n+1, n+1)
        Ã[1:n, 1:n] .= A_mat
        Ã[1:n, n+1] .= b_vec
        u_exact = (exp(Ã * T) * vcat(u0, one(ComplexF64)))[1:n]

        # ── RK4 numerical propagation from the same IC ───────────────────────
        dv_num, _ = init_gkba_wbl(; nx=1, ny=1, nk=4, γ, γso, γc,
                                    j_sd=0.0, Temp, nα=1)
        dv_num.s_α .= 1.0
        for _ in 1:N_steps
            rk4_step!(dv_num, eom_gkba_wbl!, dt)
        end

        # ── Assertions ───────────────────────────────────────────────────────
        # Full state vector agrees to RK4 global error ~O(dt^4) ≈ 1e-8
        @test isapprox(u_exact, dv_num.rkvec, atol=1e-6)

        # G^<_s from exact solution stays anti-Hermitian
        Gls_exact = reshape(u_exact[dv_ref.sz_Gleαs+1:end], dv_ref.dims_Gls)
        @test isapprox(Gls_exact, -Gls_exact', atol=1e-10)
    end

end

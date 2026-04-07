module GKBA

using LinearAlgebra
using Tullio
using DifferentialEquations

include("constants.jl")
include("types.jl")
include("hamiltonians.jl")
include("eom.jl")
include("observables.jl")
include("llg.jl")
include("init.jl")

export
    # Constants
    MBOHR, KB_EV, GAMMA_R, HBAR, K_BOLTZMAN,
    σ_0, σ_x, σ_y, σ_z,

    # Types
    ObservablesVar,
    GKBADynamics,
    eGKBADynamics,
    PosRepDynamics,
    ePosRepDynamics,
    WBLDynamics,
    LLGParams,

    # Hamiltonians
    pauli_extended,
    block_h,
    build_hs,
    build_heα,
    build_hseα,
    build_heα_posrep,
    build_hseα_posrep,
    ozaki_poles,
    build_Gamma_wbl,
    fermi,

    # EOM
    unpack!,
    eom_gkba!,
    eom_egkba!,
    eom_gkba_wbl!,
    eom_gkba_posrep!,
    eom_egkba_posrep!,

    # Observables
    compute_observables!,

    # LLG
    PrecSpin,
    update!,
    heff,
    corrector,
    normalize_rows,
    heun,
    make_llg_params,

    # Init
    init_gkba,
    init_egkba,
    init_gkba_posrep,
    init_egkba_posrep,
    init_gkba_wbl

end # module

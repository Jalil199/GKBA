# Physical constants
const MBOHR      = 5.788381e-5          # Bohr magneton (eV/T)
const KB_EV      = 8.6173324e-5         # Boltzmann constant (eV/K)
const GAMMA_R    = 1.760859644e-4       # Gyromagnetic ratio
const HBAR       = 1.0                  # ℏ (natural units, eV·fs)
const K_BOLTZMAN = 8.617343e-5          # eV/K

# Base Pauli matrices (2×2, ComplexF64)
const σ_0 = Matrix{ComplexF64}([1 0; 0 1])
const σ_x = Matrix{ComplexF64}([0 1; 1 0])
const σ_y = Matrix{ComplexF64}([0 -im; im 0])
const σ_z = Matrix{ComplexF64}([1 0; 0 -1])

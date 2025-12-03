#!/usr/bin/env python
# coding: utf-8

# In[2]:


#get_ipython().system('pip install qutip')
import numpy as np
import matplotlib.pyplot as plt
from qutip import (
    tensor, basis, qeye, Qobj,
    sigmax, sigmay, sigmaz, sigmam, sigmap,
    bloch_redfield_tensor, liouvillian, steadystate, expect,
    operator_to_vector,   # |ρ⟩⟩ mapping (unused here but imported for completeness)
    vector_to_operator
)
from matplotlib import rc 


# In[3]:


epsilon = 1e-13

# -----------------------------------------------------------------------------
#                    Two qubit operator algebra (tensor basis)
# -----------------------------------------------------------------------------
# Single qubit Pauli operators embedded in the full four level Hilbert space.
# Example: σ_x^{(1)} ≡ σ_x ⊗ 𝟙, σ_x^{(2)} ≡ 𝟙 ⊗ σ_x.

# Define Pauli operators for each qubit
sx1, sy1, sz1 = (tensor(op, qeye(2)) for op in (sigmax(), sigmay(), sigmaz()))
sx2, sy2, sz2 = (tensor(qeye(2), op) for op in (sigmax(), sigmay(), sigmaz()))
sm1 = tensor(sigmam(), qeye(2))
sp1 = tensor(sigmap(), qeye(2))
sm2 = tensor(qeye(2), sigmam())
sp2 = tensor(qeye(2), sigmap())

# -----------------------------------------------------------------------------
#                    Thermal occupation and effective temperature
# -----------------------------------------------------------------------------

def nB(omega: float, T: float) -> float:
    """Bose–Einstein occupation n_B(ω, T) with overflow protection."""
    if abs(omega) <= epsilon:
        return 0.0
    beta = 1.0 / T
    if beta * abs(omega) > 700:  # exp(700) is close to float overflow
        return 0.0
    return 1.0 / (np.exp(beta * omega) - 1.0)


def T_star(T_c: float, gamma_c: float, T_h: float, gamma_h: float, w0: float) -> float:
    """Effective temperature of the composite resonator bath."""
    den = (
        (nB(w0, T_h) + 1) * gamma_h + (nB(w0, T_c) + 1) * gamma_c
    ) / (
        nB(w0, T_h) * gamma_h + nB(w0, T_c) * gamma_c
    )
    return w0 / np.log(den)

# -----------------------------------------------------------------------------
#                       Bath spectral densities (Lorentzian)
# -----------------------------------------------------------------------------
# Each bath spectrum J(ω) is formed by a Bose factor times a Lorentzian filter
# centred at the resonator frequency w0 with quality factor Q1.


def J_h(w: float, T_h: float, gamma_h: float, w0: float, Q1: float) -> float:
    """Hot bath spectrum J_h(ω)."""
    w_abs = np.abs(w)
    nb = nB(w_abs, T_h)
    lam = w0 / Q1  # Full width at half maximum Γ = ω0 / Q
    Lor = (0.5 * lam)**2 / ((w_abs - w0) ** 2 + (0.5 * lam) ** 2)
    return (
        gamma_h * (1.0 + nb) * (w > epsilon) +
        gamma_h * nb * (w < -epsilon)
    ) * Lor 


def J_c(w: float, T_c: float, gamma_c: float, w0: float, Q1: float) -> float:
    """Cold bath spectrum J_c(ω)."""
    w_abs = np.abs(w)
    nb = nB(w_abs, T_c)
    lam = w0 / Q1
    Lor = (0.5 * lam)**2 / ((w_abs - w0) ** 2 + (0.5 * lam) ** 2)
    return (
        gamma_c * (1.0 + nb) * (w > epsilon) +
        gamma_c * nb * (w < -epsilon)
    ) * Lor


def J_global(w: float, T_c: float, gamma_c: float, T_h: float, gamma_h: float,
             w0: float, Q1: float) -> float:
    """Composite spectrum J_global(ω) = J_c + J_h."""
    return J_c(w, T_c, gamma_c, w0, Q1) + J_h(w, T_h, gamma_h, w0, Q1)

# -----------------------------------------------------------------------------
#                            Heat current estimators
# -----------------------------------------------------------------------------
# *Current_ind* computes the heat current into the hot bath when each qubit
# couples independently.  *Current_coll* does the same for collective coupling.

def Current_ind(w1: float, w2: float, rho, T_h: float, gamma_h: float,
                w0: float, Q1: float) -> float:
    """Independent coupling heat current for a given steady state ρ."""
    # Transition rates for each qubit
    rate_sm1 = J_h(+w1, T_h, gamma_h, w0, Q1)
    rate_sp1 = J_h(-w1, T_h, gamma_h, w0, Q1)
    rate_sm2 = J_h(+w2, T_h, gamma_h, w0, Q1)
    rate_sp2 = J_h(-w2, T_h, gamma_h, w0, Q1)
    # Two point correlators ⟨σ_± σ_∓⟩
    Sp1Sm1 = expect(sp1 * sm1, rho)
    Sm1Sp1 = expect(sm1 * sp1, rho)
    Sp2Sm2 = expect(sp2 * sm2, rho)
    Sm2Sp2 = expect(sm2 * sp2, rho)
    # Energy balance
    Q_dot  = w1 * (-rate_sm1 * Sp1Sm1 + rate_sp1 * Sm1Sp1)
    Q_dot += w2 * (-rate_sm2 * Sp2Sm2 + rate_sp2 * Sm2Sp2)
    return Q_dot

#Current_ind_Asym takes in one additional parameter gamma_h_q2 to take into account that for when g1 ≠ g2

def Current_ind_Asym(w1: float, w2: float, rho, T_h: float, gamma_h_q1: float, gamma_h_q2: float,
                w0: float, Q1: float) -> float:
    """Independent coupling heat current for a given steady state ρ."""
    # Transition rates for each qubit
    rate_sm1 = J_h(+w1, T_h, gamma_h_q1, w0, Q1)
    rate_sp1 = J_h(-w1, T_h, gamma_h_q1, w0, Q1)
    rate_sm2 = J_h(+w2, T_h, gamma_h_q2, w0, Q1)
    rate_sp2 = J_h(-w2, T_h, gamma_h_q2, w0, Q1)
    # Two point correlators ⟨σ_± σ_∓⟩
    Sp1Sm1 = expect(sp1 * sm1, rho)
    Sm1Sp1 = expect(sm1 * sp1, rho)
    Sp2Sm2 = expect(sp2 * sm2, rho)
    Sm2Sp2 = expect(sm2 * sp2, rho)
    # Energy balance
    Q_dot  = w1 * (-rate_sm1 * Sp1Sm1 + rate_sp1 * Sm1Sp1)
    Q_dot += w2 * (-rate_sm2 * Sp2Sm2 + rate_sp2 * Sm2Sp2)
    return Q_dot

# Collective Heat Current without Asymmetry
def Current_coll(w1: float, w2: float, rho, T_h: float, gamma_h: float,
                 w0: float, Q1: float, Jm_h: Qobj, Jp_h: Qobj) -> float:
    """Collective coupling heat current for a given steady state ρ."""
    Q_dot = 0.0
    if w1 == w2:  # Degenerate qubits ⇒ collective J_± jumps only
        En = np.array([w1, -w1])
        S = [Jm_h, Jp_h]
        for i in range(2):
            for j in range(2):
                Q_dot += -J_h(+En[i], T_h, gamma_h, w0, Q1) * En[j] * \
                         expect((S[i].dag() * S[j] + S[j].dag() * S[i]), rho) / 2
    else:  # Non degenerate case: revert to individual σ_± operators
        En = np.array([w1, w2, -w1, -w2])
        S = [sm1, sm2, sp1, sp2]
        for i in range(4):
            for j in range(4):
                Q_dot += -J_h(+En[i], T_h, gamma_h, w0, Q1) * En[j] * \
                         expect((S[i].dag() * S[j] + S[j].dag() * S[i]), rho) / 2
    return Q_dot


# Current_coll now takes in two additional parameters J_-^(h) and J_+^(h) instead of calling them inside the function
def Current_coll_Asym(w1: float, w2: float, rho, T_h: float, gamma_h:float,
                 w0: float, Q1: float, Jm_h: Qobj, Jp_h: Qobj) -> float:
    """Collective coupling heat current for a given steady state ρ."""
    Q_dot = 0.0
    if w1 == w2:  # Degenerate qubits ⇒ collective J_± jumps only
        En = np.array([w1, -w1])
        S = [Jm_h, Jp_h]
        for i in range(2):
            for j in range(2):
                Q_dot += -J_h(+En[i], T_h, gamma_h, w0, Q1) * En[j] * \
                         expect((S[i].dag() * S[j] + S[j].dag() * S[i]), rho) / 2
    else:  # Non degenerate case: revert to individual σ_± operators
        En = np.array([w1, w2, -w1, -w2])
        S = [sm1, sm2, sp1, sp2]
        for i in range(4):
            for j in range(4):
                Q_dot += -J_h(+En[i], T_h, gamma_h, w0, Q1) * En[j] * \
                         expect((S[i].dag() * S[j] + S[j].dag() * S[i]), rho) / 2
    return Q_dot


# -----------------------------------------------------------------------------
#                  Steady state solvers (collective vs independent)
# -----------------------------------------------------------------------------
# Collective Steady State Density Matrix without Asymmetry for Even (+1) Case
def rho_ss_termic_collective_sup(w1: float, w2: float, gamma_local: float,
                             T_local: float, gamma_deph: float,
                             T_h: float, gamma_h: float,
                             T_c: float, gamma_c: float,
                             w0: float, Q1: float):
    """Steady state with collective resonator coupling."""
    # System Hamiltonian H = ½(ω1 σ_z^{(1)} + ω2 σ_z^{(2)})
    H = 0.5 * (w1 * sz1 + w2 * sz2)
    # Collective system operator that couples to the resonator mode
    sigma_sum = sx1 + sx2
    a_ops_coll = [[sigma_sum,
                   lambda w: J_global(w, T_c, gamma_c, T_h, gamma_h, w0, Q1)]]
    R = bloch_redfield_tensor(H, a_ops_coll, fock_basis=True, sec_cutoff=-1)
    # Local Lindblad channels: thermal relaxation plus pure dephasing
    c_ops = [
        np.sqrt(gamma_local * (1 + nB(w1, T_local))) * sm1,
        np.sqrt(gamma_local * nB(w1, T_local))       * sp1,
        np.sqrt(gamma_local * (1 + nB(w2, T_local))) * sm2,
        np.sqrt(gamma_local * nB(w2, T_local))       * sp2,
        np.sqrt(gamma_deph) * sz1,
        np.sqrt(gamma_deph) * sz2,
    ]
    L_full = R + liouvillian(0 * sz1, c_ops)  # No additional Hamiltonian term
    return steadystate(L_full, method='direct')

# Collective Steady State Density Matrix without Asymmetry for Even (-1) Case
def rho_ss_termic_collective_sup_2(w1: float, w2: float, gamma_local: float,
                             T_local: float, gamma_deph: float,
                             T_h: float, gamma_h: float,
                             T_c: float, gamma_c: float,
                             w0: float, Q1: float):
    """Steady state with collective resonator coupling.
    The idea here is that we have collective system 
    operator of different pairities for each resonator mode.  """
    # System Hamiltonian H = ½(ω1 σ_z^{(1)} + ω2 σ_z^{(2)})
    H = 0.5 * (w1 * sz1 + w2 * sz2)
    # Collective system operator that couples to the resonator mode
    sigma_h = sx1 - sx2
    sigma_c = sx1 - sx2
    a_ops_coll_1 = [[sigma_h,
                   lambda w: J_h(w, T_h, gamma_h, w0, Q1)]]
    R1 = bloch_redfield_tensor(H, a_ops_coll_1, fock_basis=True, sec_cutoff=-1)
    a_ops_coll_2 = [[sigma_c,
                   lambda w: J_c(w, T_c, gamma_c, w0, Q1)]]
    R2 = bloch_redfield_tensor(H, a_ops_coll_2, fock_basis=True, sec_cutoff=-1)
    # Local Lindblad channels: thermal relaxation plus pure dephasing
    c_ops = [
        np.sqrt(gamma_local * (1 + nB(w1, T_local))) * sm1,
        np.sqrt(gamma_local * nB(w1, T_local))       * sp1,
        np.sqrt(gamma_local * (1 + nB(w2, T_local))) * sm2,
        np.sqrt(gamma_local * nB(w2, T_local))       * sp2,
        np.sqrt(gamma_deph) * sz1,
        np.sqrt(gamma_deph) * sz2,
    ]
    L_full = R1 + R2 + liouvillian(0 * sz1, c_ops)  # No additional Hamiltonian term
    return steadystate(L_full, method='direct')

# Collective Steady State Density Matrix without Asymmetry for Odd Case
def rho_ss_termic_collective_sub(w1: float, w2: float, gamma_local: float,
                             T_local: float, gamma_deph: float,
                             T_h: float, gamma_h: float,
                             T_c: float, gamma_c: float,
                             w0: float, Q1: float):
    """Steady state with collective resonator coupling.
    The idea here is that we have collective system 
    operator of different pairities for each resonator mode.  """
    # System Hamiltonian H = ½(ω1 σ_z^{(1)} + ω2 σ_z^{(2)})
    H = 0.5 * (w1 * sz1 + w2 * sz2)
    # Collective system operator that couples to the resonator mode
    sigma_h = sx1 + sx2
    sigma_c = sx1 - sx2
    a_ops_coll_1 = [[sigma_h,
                   lambda w: J_h(w, T_h, gamma_h, w0, Q1)]]
    R1 = bloch_redfield_tensor(H, a_ops_coll_1, fock_basis=True, sec_cutoff=-1)
    a_ops_coll_2 = [[sigma_c,
                   lambda w: J_c(w, T_c, gamma_c, w0, Q1)]]
    R2 = bloch_redfield_tensor(H, a_ops_coll_2, fock_basis=True, sec_cutoff=-1)
    # Local Lindblad channels: thermal relaxation plus pure dephasing
    c_ops = [
        np.sqrt(gamma_local * (1 + nB(w1, T_local))) * sm1,
        np.sqrt(gamma_local * nB(w1, T_local))       * sp1,
        np.sqrt(gamma_local * (1 + nB(w2, T_local))) * sm2,
        np.sqrt(gamma_local * nB(w2, T_local))       * sp2,
        np.sqrt(gamma_deph) * sz1,
        np.sqrt(gamma_deph) * sz2,
    ]
    L_full = R1 + R2 + liouvillian(0 * sz1, c_ops)  # No additional Hamiltonian term
    return steadystate(L_full, method='direct')

# the collective density matrix solver now takes in one additional parameter sigma_sum = 0.5*(σ_x^(1) + σ_x^(2)) = J_x

def rho_ss_termic_collective_Asym(w1: float, w2: float, gamma_local: float,
                             T_local: float, gamma_deph: float,
                             T_h: float, gamma_h:float,
                             T_c: float, gamma_c:float,
                             w0: float, Q1: float, sigma_sum: Qobj):
    """Steady state with collective resonator coupling."""
    # System Hamiltonian H = ½(ω1 σ_z^{(1)} + ω2 σ_z^{(2)})
    H = 0.5 * (w1 * sz1 + w2 * sz2)
    # Collective system operator that couples to the resonator mode
    #sigma_sum = sx1 + sx2
    a_ops_coll = [[0.5*sigma_sum,
                   lambda w: J_global(w, T_c, gamma_c, T_h, gamma_h, w0, Q1)]]
    R = bloch_redfield_tensor(H, a_ops_coll, fock_basis=True, sec_cutoff=-1)
    # Local Lindblad channels: thermal relaxation plus pure dephasing
    c_ops = [
        np.sqrt(gamma_local * (1 + nB(w1, T_local))) * sm1,
        np.sqrt(gamma_local * nB(w1, T_local))       * sp1,
        np.sqrt(gamma_local * (1 + nB(w2, T_local))) * sm2,
        np.sqrt(gamma_local * nB(w2, T_local))       * sp2,
        np.sqrt(gamma_deph) * sz1,
        np.sqrt(gamma_deph) * sz2,
    ]
    L_full = R + liouvillian(0 * sz1, c_ops)  # No additional Hamiltonian term
    return steadystate(L_full, method='direct')

def rho_ss_termic_collective_sub_Asym(w1: float, w2: float, gamma_local: float,
                             T_local: float, gamma_deph: float,
                             T_h: float, gamma_h:float,
                             T_c: float, gamma_c:float,
                             w0: float, Q1: float, sigma_sum: Qobj, sigma_sub: Qobj):
    """Steady state with collective resonator coupling."""
    # System Hamiltonian H = ½(ω1 σ_z^{(1)} + ω2 σ_z^{(2)})
    H = 0.5 * (w1 * sz1 + w2 * sz2)
    # Collective system operator that couples to the resonator mode
    #sigma_h = sx1 + sx2
    #sigma_c = sx1 - sx2
    a_ops_coll_1 = [[sigma_sum,
                   lambda w: J_h(w, T_h, gamma_h, w0, Q1)]]
    R1 = bloch_redfield_tensor(H, a_ops_coll_1, fock_basis=True, sec_cutoff=-1)
    a_ops_coll_2 = [[sigma_sub,
                   lambda w: J_c(w, T_c, gamma_c, w0, Q1)]]
    R2 = bloch_redfield_tensor(H, a_ops_coll_2, fock_basis=True, sec_cutoff=-1)
    # Local Lindblad channels: thermal relaxation plus pure dephasing
    c_ops = [
        np.sqrt(gamma_local * (1 + nB(w1, T_local))) * sm1,
        np.sqrt(gamma_local * nB(w1, T_local))       * sp1,
        np.sqrt(gamma_local * (1 + nB(w2, T_local))) * sm2,
        np.sqrt(gamma_local * nB(w2, T_local))       * sp2,
        np.sqrt(gamma_deph) * sz1,
        np.sqrt(gamma_deph) * sz2,
    ]
    L_full = R1 + R2 + liouvillian(0 * sz1, c_ops)  # No additional Hamiltonian term
    return steadystate(L_full, method='direct')


# Independent Steady State Density Matrix without Asymmetry

def rho_ss_termic_indepentend(
    w1, w2,
    gamma_local, T_local, gamma_deph,
    T_h, gamma_h, T_c, gamma_c,
    w0, Q1
):
    """
    Steady state for the *independent-bath* configuration.

    Each qubit feels the composite resonator spectrum J_global(ω) **locally**
    (i.e. as its own Lindblad channel).  In addition, both qubits couple to a
    parasitic thermal bath at T_local and to a pure-dephasing bath with rate
    gamma_deph.
    """
    # Resonator-induced relaxation / absorption rates
    rate_sm1 = J_global(+w1, T_c, gamma_c, T_h, gamma_h, w0, Q1)
    rate_sp1 = J_global(-w1, T_c, gamma_c, T_h, gamma_h, w0, Q1)
    rate_sm2 = J_global(+w2, T_c, gamma_c, T_h, gamma_h, w0, Q1)
    rate_sp2 = J_global(-w2, T_c, gamma_c, T_h, gamma_h, w0, Q1)

    # Full list of collapse operators:
    #   ─ resonator (first four),
    #   ─ parasitic thermalisation (next four),
    #   ─ pure dephasing (last two).
    c_global_local = [
        np.sqrt(rate_sm1) * sm1,
        np.sqrt(rate_sp1) * sp1,
        np.sqrt(rate_sm2) * sm2,
        np.sqrt(rate_sp2) * sp2,
        np.sqrt(gamma_local * (1 + nB(w1, T_local))) * sm1,
        np.sqrt(gamma_local * nB(w1, T_local))       * sp1,
        np.sqrt(gamma_local * (1 + nB(w2, T_local))) * sm2,
        np.sqrt(gamma_local * nB(w2, T_local))       * sp2,
        np.sqrt(gamma_deph) * sz1,
        np.sqrt(gamma_deph) * sz2,
    ]

    # Bare Hamiltonian: non-interacting qubits with splittings w1, w2.
    H = 0.5 * (w1 * sz1 + w2 * sz2)

    # Liouvillian and steady state.
    L_full = liouvillian(H, c_global_local)
    rho_ss = steadystate(L_full, method="direct")
    return rho_ss


# Takes in two new parameter gamma_h_q2 and gamma_c_q2 to when g1 ≠ g2, g3 ≠ g4
def rho_ss_termic_indepentend_Asym(
    w1, w2,
    gamma_local, T_local, gamma_deph,
    T_h, gamma_h_q1, gamma_h_q2, T_c, gamma_c_q1, gamma_c_q2,
    w0, Q1
):
    """
    Steady state for the *independent-bath* configuration.

    Each qubit feels the composite resonator spectrum J_global(ω) **locally**
    (i.e. as its own Lindblad channel).  In addition, both qubits couple to a
    parasitic thermal bath at T_local and to a pure-dephasing bath with rate
    gamma_deph.
    """
    # Resonator-induced relaxation / absorption rates
    rate_sm1 = J_global(+w1, T_c, gamma_c_q1, T_h, gamma_h_q1, w0, Q1)
    rate_sp1 = J_global(-w1, T_c, gamma_c_q1, T_h, gamma_h_q1, w0, Q1)
    rate_sm2 = J_global(+w2, T_c, gamma_c_q2, T_h, gamma_h_q2, w0, Q1)
    rate_sp2 = J_global(-w2, T_c, gamma_c_q2, T_h, gamma_h_q2, w0, Q1)

    # Full list of collapse operators:
    #   ─ resonator (first four),
    #   ─ parasitic thermalisation (next four),
    #   ─ pure dephasing (last two).
    c_global_local = [
        np.sqrt(rate_sm1) * sm1,
        np.sqrt(rate_sp1) * sp1,
        np.sqrt(rate_sm2) * sm2,
        np.sqrt(rate_sp2) * sp2,
        np.sqrt(gamma_local * (1 + nB(w1, T_local))) * sm1,
        np.sqrt(gamma_local * nB(w1, T_local))       * sp1,
        np.sqrt(gamma_local * (1 + nB(w2, T_local))) * sm2,
        np.sqrt(gamma_local * nB(w2, T_local))       * sp2,
        np.sqrt(gamma_deph) * sz1,
        np.sqrt(gamma_deph) * sz2,
    ]

    # Bare Hamiltonian: non-interacting qubits with splittings w1, w2.
    H = 0.5 * (w1 * sz1 + w2 * sz2)

    # Liouvillian and steady state.
    L_full = liouvillian(H, c_global_local)
    rho_ss = steadystate(L_full, method="direct")
    return rho_ss


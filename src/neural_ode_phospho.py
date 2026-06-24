"""
Ordinary Differential Equation (ODE) dynamics systems.
Consolidates all 19 mathematical differential equation models used in the simulations.
"""

import numpy as np
from src.preprocessing import HODGKIN_HUXLEY_PARAMS

# =====================================================================
# 1. Module 1: Murburn foundations
# =====================================================================
def radical_chain(t, y, k1, k2, k3, k4, k5):
    """
    Coupled ODE for radical chain dynamics inside neuronal mitochondria.
    y = [O2m, H2O2, OH, GSSG, ATP]
    """
    O2m, H2O2, OH, GSSG, ATP = y
    dO2m  = k1 * 0.8 - k2 * O2m
    dH2O2 = 0.5 * k2 * O2m - k3 * H2O2
    dOH   = k3 * H2O2 - k4 * OH
    dGSSG = k4 * OH - k5 * GSSG
    dATP  = k5 * GSSG * 2.5
    return [dO2m, dH2O2, dOH, dGSSG, dATP]

# =====================================================================
# 2. Module 2: Neural EM
# =====================================================================
def hodgkin_huxley(t_end=150.0, I_ext_base=10.0, I_tms=0.0, dros_level=1.0):
    """
    Hodgkin-Huxley single-neuron simulation with DROS-dependent reversal potential shift.
    dros_level modulates ENa and baseline background depolarisation.
    """
    dt = 0.01
    t = np.arange(0, t_end, dt)

    # Gating kinetics equations (Hodgkin-Huxley 1952)
    def alpha_m(V): return 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10) + 1e-12)
    def beta_m(V):  return 4.0 * np.exp(-(V + 65) / 18)
    def alpha_h(V): return 0.07 * np.exp(-(V + 65) / 20)
    def beta_h(V):  return 1.0 / (1.0 + np.exp(-(V + 35) / 10))
    def alpha_n(V): return 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10) + 1e-12)
    def beta_n(V):  return 0.125 * np.exp(-(V + 65) / 80)

    # Parameters from configs
    gNa_max = HODGKIN_HUXLEY_PARAMS['gNa_max_base'] * dros_level   # DROS scales gNa
    gK_max  = HODGKIN_HUXLEY_PARAMS['gK_max']
    gL      = HODGKIN_HUXLEY_PARAMS['gL']
    EK      = HODGKIN_HUXLEY_PARAMS['EK']
    EL      = HODGKIN_HUXLEY_PARAMS['EL']
    Cm      = HODGKIN_HUXLEY_PARAMS['Cm']
    
    # Murburn: DROS-dependent shift to Na+ reversal potential
    ENa_shift = HODGKIN_HUXLEY_PARAMS['dros_voltage_shift'] * (dros_level - 1.0)
    ENa = HODGKIN_HUXLEY_PARAMS['ENa'] + ENa_shift

    V, m, h, n = -65.0, 0.05, 0.6, 0.32
    Vs, ms, hs, ns = [], [], [], []

    for ti in t:
        I_ext = I_ext_base
        # TMS pulse duration 0.3ms at t=50ms
        if I_tms > 0 and 50.0 < ti < 50.3:
            I_ext += I_tms
        # DROS-induced background depolarisation (tonic effect)
        if dros_level > 1.0:
            I_ext += (dros_level - 1.0) * 3.0

        am, bm = alpha_m(V), beta_m(V)
        ah, bh = alpha_h(V), beta_h(V)
        an, bn = alpha_n(V), beta_n(V)

        INa = gNa_max * (m**3) * h * (V - ENa)
        IK  = gK_max  * (n**4)     * (V - EK)
        IL  = gL                  * (V - EL)

        dV = (I_ext - INa - IK - IL) / Cm
        dm = am * (1 - m) - bm * m
        dh = ah * (1 - h) - bh * h
        dn = an * (1 - n) - bn * n

        V += dV * dt
        m += dm * dt
        h += dh * dt
        n += dn * dt
        Vs.append(V)
        ms.append(m)
        hs.append(h)
        ns.append(n)

    return t, np.array(Vs), np.array(ms), np.array(hs), np.array(ns)

# =====================================================================
# 3. Module 3: Gamma entrainment
# =====================================================================
def microglia_ode(t, y, k12, k21, k23, k30, stim_amp=0.0):
    """
    ODE for microglia phenotypic transitions: M0 resting, M1 active, M2 phagocytic.
    y = [M0, M1, M2]
    """
    M0, M1, M2 = y
    stim = stim_amp * np.exp(-((t - 6.0)**2) / 4.0)  # stimulus at t=6h
    dM0 = -k12 * M0 * stim + k30 * M2
    dM1 =  k12 * M0 * stim - k21 * M1 - k23 * M1
    dM2 =  k23 * M1 - k30 * M2
    return [dM0, dM1, dM2]

# =====================================================================
# 4. Module 5: BBI, Levitation, Connectome
# =====================================================================
def epileptor(t, y, x0=-1.6, r=1e-3, s=4.0, gamma=0.01):
    """
    Six-dimensional Epileptor model for seizure dynamics (Jirsa et al. 2014).
    y = [x1, y1, z, x2, y2, u]
    """
    x1, y1, z, x2, y2, u = y
    if x1 < 0:
        f1 = x1**3 - 3.0 * x1**2
    else:
        f1 = (x2 - 0.6 * (z - 4.0)**2) * x1
    I1 = 3.1
    I2 = 0.45
    dx1 = y1 - f1 - z + I1
    dy1 = 1.0 - 5.0 * x1**2 - y1
    dz  = r * (4.0 * (x1 - x0) - z)
    dx2 = -y2 + x2 - x2**3 + I2 + 2.0 * u - 0.3 * (z - 3.5)
    dy2 = (1.0 / 0.1) * (-y2 + f1 if x2 < -0.25 else 6.0 * x2 + 1.5 * x2**2 - y2)
    du  = gamma * (-u + x2)
    return [dx1, dy1, dz, dx2, dy2, du]

def thal_cortex(t, A, B_stim, k_tc=0.3, k_ct=0.2, k_self=0.1, tau=5.0):
    """
    Two-compartment thalamocortical arousal loop.
    A = [T_input, C_act] (Thalamic input, Cortical activation)
    """
    T_input, C_act = A
    stim = B_stim * np.exp(-((t - 20.0)**2) / 10.0) if 18.0 < t < 35.0 else 0.0
    dT = -k_self * T_input + k_ct * C_act + stim
    dC = -k_self * C_act + k_tc * T_input - 0.1 * C_act**3 + 0.05
    return [dT, dC]

# =====================================================================
# 5. Module 6: Phantom limb pain
# =====================================================================
def central_sensitisation(t, y, pain_amp, murburn=True):
    """
    ODE for spinal cord central sensitisation under Murburn DROS.
    y = [W, DROS] (Wind-up factor, local DROS)
    """
    W, DROS = y
    k_wind  = 0.4
    k_decay = 0.15
    k_dros  = 0.25 if murburn else 0.0
    k_mito  = 0.05
    k_SOD   = 0.3
    amp_d   = 0.2 if murburn else 0.0
    pain    = pain_amp * np.exp(-0.05 * max(0.0, t - 5.0)**2)
    dW    = k_wind * pain - k_decay * W + k_dros * DROS
    dDROS = k_mito + amp_d * pain * W - k_SOD * DROS
    return [dW, dDROS]

def tc_loop(t_vec, y, g_tc, g_ct, tau_t, tau_c, I_ext):
    """
    Wilson-Cowan loop simplified for Thalamocortical loops in PLP.
    y = [E_t, E_c] (thalamic excitation, cortical excitation)
    """
    E_t, E_c = y
    def sigma(x): return 1.0 / (1.0 + np.exp(-5.0 * (x - 0.5)))
    dE_t = (-E_t + sigma(g_ct * E_c + I_ext)) / tau_t
    dE_c = (-E_c + sigma(g_tc * E_t)) / tau_c
    return [dE_t, dE_c]

# =====================================================================
# 6. Module 7: Neuroplasticity & Synaptic Buffering
# =====================================================================
def bdnf_cascade(t, y, stim_amp, murburn=False):
    """
    4-state BDNF cascade.
    y = [BDNF, TrkB, ERK, CREB]
    """
    BDNF, TrkB, ERK, CREB = y
    stim = stim_amp * np.exp(-((t - 5.0)**2) / 8.0)  # rTMS stimulus at t=5min
    dros_boost = 0.3 if murburn else 0.0
    dBDNF = 0.8 * stim - 0.4 * BDNF
    dTrkB = (0.6 + dros_boost) * BDNF - 0.3 * TrkB
    dERK  = 0.7 * TrkB - 0.35 * ERK
    dCREB = 0.5 * ERK - 0.2 * CREB
    return [dBDNF, dTrkB, dERK, dCREB]

def excitotoxic_cascade(t, y, glu_dose, murburn_dros=False):
    """
    ODE modeling glutamate-induced Ca2+ toxicity and mPTP mitochondrial collapse.
    y = [Glu, Ca, mPTP, DROS, cell_health]
    """
    Glu, Ca, mPTP, DROS, cell_health = y
    dGlu  = -0.3 * Glu + glu_dose * np.exp(-0.5 * max(0.0, t - 2.0))
    k_nmda = 0.4 + (0.25 if murburn_dros else 0.0)
    dCa   = k_nmda * Glu * (10.0 - Ca) / 10.0 - 0.15 * Ca
    dros_drive = 0.2 if murburn_dros else 0.0
    dmPTP = 0.1 * Ca**2 / (1.0 + Ca**2) + dros_drive * DROS - 0.05 * mPTP
    dDROS = 0.3 * Ca * mPTP - 0.4 * DROS
    dHealth = -0.08 * Ca - 0.2 * mPTP - 0.15 * DROS
    return [dGlu, dCa, dmPTP, dDROS, dHealth]

def growth_cone_ode(t, y, NGF_amp, BDNF_amp, sema_amp, TMS_guide=False):
    """
    Goodhill-style (1998) axon growth cone dynamics.
    y = [pos_x, pos_y, vel_x, vel_y]
    """
    pos_x, pos_y, vel_x, vel_y = y
    dist_x = 10.0 - pos_x
    dist_y = -pos_y
    d = max(np.sqrt(dist_x**2 + dist_y**2), 0.1)
    F_NGF  = NGF_amp  * np.array([dist_x, dist_y]) / d
    F_BDNF = BDNF_amp * np.array([dist_x, dist_y]) / d * 0.7
    
    scar_dist = pos_x - 3.5
    F_sema = np.array([-sema_amp * np.exp(-scar_dist**2 / 2.0), 0.0]) if abs(scar_dist) < 3.0 else np.array([0.0, 0.0])
    
    F_tms = np.array([0.3, 0.0]) if TMS_guide else np.array([0.0, 0.0])
    # Reproducible noise is handled in calling loops via RNG seed
    F_total = F_NGF + F_BDNF + F_sema + F_tms - 0.4 * np.array([vel_x, vel_y])
    return [vel_x, vel_y, F_total[0], F_total[1]]

def glu_clearance(t, y, Vmax, stim_freq=20.0):
    """
    GLT-1 astrocytic glutamate clearance.
    y = [Glu_ext]
    """
    Glu = y[0]
    Km_glt1 = 0.4
    pulses = sum(1.5 * np.exp(-0.1 * (t - k * (1000.0 / stim_freq)))
                 for k in range(1, 8) if t > k * (1000.0 / stim_freq))
    return [-Vmax * Glu / (Km_glt1 + Glu) + pulses]

def tripartite_ca(t, y, spine_pump, astro_buff):
    """
    Calcium buffering dynamics in tripartite synapse.
    y = [Ca_spine, Ca_astro]
    """
    Ca_spine, Ca_astro = y
    Ca_input = 2.0 * np.exp(-((t - 5.0)**2) / 4.0) + 0.8 * np.exp(-((t - 30.0)**2) / 6.0)
    dCa_spine = Ca_input - spine_pump * Ca_spine - 0.3 * (Ca_spine - Ca_astro)
    dCa_astro = astro_buff * (Ca_spine - Ca_astro) - 0.1 * Ca_astro
    return [dCa_spine, dCa_astro]

def vesicle_pools(t, y, stim_hz, refill_rate):
    """
    Vesicle pool dynamics: readily releasable (RRP), recycling (RP), and reserve (RevP).
    y = [RRP, RP, RevP]
    """
    RRP, RP, RevP = y
    stim_prob = stim_hz / 1000.0
    dRRP  = refill_rate * RP - stim_prob * RRP
    dRP   = 0.002 * RevP - refill_rate * RP + stim_prob * RRP * 0.7
    dRevP = -0.002 * RevP + 0.001 * (1.0 - RevP)
    return [dRRP, dRP, dRevP]

# =====================================================================
# 7. Module 8: Consciousness & Memory Systems
# =====================================================================
def global_workspace(t, y, stim, broadcast=True):
    """
    Dehaene global workspace theory ignition.
    y = [S, P, Pa, T] (Sensory, Prefrontal, Parietal, Thalamic arousal)
    """
    S, P, Pa, T = y
    dS  = -0.3 * S + stim * np.exp(-((t - 5.0)**2) / 4.0) + 0.1 * T
    ignition = 1.0 if (S > 0.5 and broadcast) else 0.0
    dP  = -0.25 * P + 0.6 * ignition * S + 0.2 * Pa
    dPa = -0.3 * Pa + 0.5 * P
    dT  = -0.4 * T + 0.3 * S + 0.2 * P
    return [dS, dP, dPa, dT]

def lucid_dream_ode(t, y):
    """
    Lucid dreaming self-awareness feedback loop.
    y = [V, A, P, D] (vividness, self-awareness, PFC re-engagement, DROS)
    """
    V, A, P, D = y
    dV = 0.3 * D * (1.0 - V) - 0.1 * A * V
    dA = 0.4 * P - 0.2 * A + 0.05 * V
    dP = 0.2 * A * V - 0.3 * P
    dD = 0.15 * (1.5 - D) - 0.05 * P
    return [dV, dA, dP, dD]

def rubber_hand(t, y):
    """
    Rubber hand body ownership illusion (synchronous stimulation).
    y = [O, V, Tact, M] (Ownership, visual, tactile, multisensory binding)
    """
    O, V, Tact, M = y
    V_stim = 0.8 * np.exp(-((t - 5.0)**2) / 4.0)
    T_stim = 0.8 * np.exp(-((t - 5.1)**2) / 4.0)  # synchronous (0.1s delay)
    dV    = -0.3 * V + V_stim
    dTact = -0.3 * Tact + T_stim
    dM    = 0.6 * V * Tact - 0.4 * M
    dO    = 0.7 * M - 0.3 * O
    return [dO, dV, dTact, dM]

def rubber_hand_async(t, y):
    """
    Rubber hand body ownership illusion (asynchronous stimulation).
    y = [O, V, Tact, M] (Ownership, visual, tactile, multisensory binding)
    """
    O, V, Tact, M = y
    V_stim = 0.8 * np.exp(-((t - 5.0)**2) / 4.0)
    T_stim = 0.8 * np.exp(-((t - 5.8)**2) / 4.0)  # asynchronous (0.8s delay)
    dV    = -0.3 * V + V_stim
    dTact = -0.3 * Tact + T_stim
    dM    = 0.6 * V * Tact - 0.4 * M
    dO    = 0.7 * M - 0.3 * O
    return [dO, dV, dTact, dM]

def engram_ode(t, y, input_strength, murburn=False):
    """
    Memory engram trace formation and consolidation.
    y = [E_L, E_S, E_P, DROS] (Labile engram, stable engram, structural protein, DROS)
    """
    E_L, E_S, E_P, DROS = y
    stim = input_strength * sum(np.exp(-((t - k)**2) / 0.5) for k in [2, 4, 6, 8])
    dros_boost = 0.3 if murburn else 0.0
    dE_L  = stim + dros_boost * DROS - 0.8 * E_L
    dE_S  = 0.4 * E_L - 0.15 * E_S
    dE_P  = 0.05 * E_S - 0.01 * E_P
    dDROS = 0.2 * stim - 0.3 * DROS + 0.1
    return [dE_L, dE_S, dE_P, dDROS]

def reconsolidation(t, y, reactivate_t, modify_strength):
    """
    Memory reconsolidation kinetics during the reactivation lability window.
    y = [M, R, M2, protein] (Original trace, labile copy, modified trace, plastic protein)
    """
    M, R, M2, protein = y
    # Reactivation trigger window
    react = 1.0 * (abs(t - reactivate_t) < 0.1)
    window = np.exp(-((t - reactivate_t)**2) / (2 * 2**2)) * (t > reactivate_t)
    
    dM  = -0.05 * M - window * M + 0.3 * protein
    dR  = react * M - 0.5 * R
    dM2 = modify_strength * window * R - 0.02 * M2
    dprot = 0.3 * R - 0.1 * protein
    return [dM, dR, dM2, dprot]

def memory_dros(t, y, dros_applied):
    """
    DROS-dependent state-dependent memory weakening/fortification.
    y = [mem_trace, dros_local, camp]
    """
    mem_trace, dros_local, camp = y
    dros_target = dros_applied * np.exp(-((t - 5.0)**2) / 4.0)
    ddros = dros_target - 0.4 * dros_local
    
    if dros_applied < 0.5:
        trace_mod = -0.08 * mem_trace
    elif dros_applied < 2.0:
        trace_mod = +0.06 * dros_local * mem_trace * (1 - mem_trace)
    else:
        trace_mod = -0.3 * mem_trace
        
    dcamp  = 0.2 * dros_local - 0.15 * camp
    dtrace = trace_mod + 0.02 * camp
    return [dtrace, ddros, dcamp]

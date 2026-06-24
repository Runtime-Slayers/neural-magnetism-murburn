"""
Biochemical activity, cellular responses, and physical energy harvesting models.
Consolidates STDP rules, excitotoxicity curves, thermoelectric/biofuel/piezo harvesting,
synaptic buffering safety margins, and forensic memory P300 signatures.
"""

import numpy as np
from src.preprocessing import ENERGY_PARAMS

# =====================================================================
# 1. Module 4: Energy Harvesting Activity
# =====================================================================
def get_thermoelectric_output(dT, S_val, legs=50):
    """
    Computes Seebeck output voltage in mV for a given temperature difference.
    """
    return S_val * dT * 1e3 * legs

def get_thermoelectric_power_density(R_L, S_val, dT=17.0, R_int=10.0, legs=50, area_cm2=10.0):
    """
    Computes TEG output power density in uW/cm² vs load resistance R_L.
    """
    V_oc = S_val * dT * legs
    P = (V_oc**2 * R_L) / ((R_int + R_L)**2) * 1e6  # uW
    return P / area_cm2

def get_biofuel_cell_curves(J):
    """
    Computes sweat biofuel cell polarization (Voltage) and Power Density (uW/cm²).
    """
    E_oc = 0.57                     # V open-circuit
    R_cell_gluc = 80                # ohm.cm2
    n_gluc = 0.045                  # Tafel slope factor
    n_lac  = 0.055

    V_gluc = E_oc - n_gluc * np.log(J + 0.01) - J * R_cell_gluc * 1e-3
    V_lac  = (E_oc - 0.05) - n_lac * np.log(J + 0.01) - J * (R_cell_gluc * 1.2) * 1e-3
    V_gluc = np.clip(V_gluc, 0.0, E_oc)
    V_lac  = np.clip(V_lac,  0.0, E_oc - 0.05)

    P_gluc = V_gluc * J * 1000.0   # uW/cm²
    P_lac  = V_lac  * J * 1000.0   # uW/cm²
    return V_gluc, P_gluc, V_lac, P_lac

def get_biofuel_cell_vs_glucose(glucose_conc):
    """
    Computes GOx biofuel cell power density vs sweat glucose concentration in mM.
    """
    Km_GOx = 0.18
    Jmax   = 1.8
    J_gox = Jmax * glucose_conc / (Km_GOx + glucose_conc)
    V_cell = 0.4 - 0.05 * np.log(J_gox + 0.01)
    P_biof = J_gox * V_cell * 1000.0   # uW/cm²
    
    GOx_PEDOT_factor = 1.45
    P_enhanced = P_biof * GOx_PEDOT_factor
    return P_biof, P_enhanced

def get_piezo_voltages(t, seed=0):
    """
    Computes PVDF piezoelectric voltage from body motion types (heartbeat, breathing, walking).
    """
    # Stresses (kPa)
    heart_stress = 0.5 * np.abs(np.sin(np.pi * 1.1 * t)) * (1.0 + 0.1 * np.sin(2.0 * np.pi * 0.25 * t))
    breath_stress = 0.3 * np.sin(2.0 * np.pi * 0.25 * t) + 0.3
    
    rng = np.random.RandomState(seed)
    walk_stress = 0.8 * np.abs(np.sin(2.0 * np.pi * 2.0 * t)) * (1.0 + 0.3 * rng.randn(len(t)) * 0.1)
    
    g31_PVDF = 216e-3   # Vm/N
    thick = 50e-6       # 50 um
    
    V_heart  = g31_PVDF * heart_stress  * 1e3 * thick * 1000.0   # mV
    V_breath = g31_PVDF * breath_stress * 1e3 * thick * 1000.0
    V_walk   = g31_PVDF * walk_stress   * 1e3 * thick * 500.0
    return V_heart, V_breath, V_walk

# =====================================================================
# 2. Module 7: Neuroplasticity & Excitotoxicity Activity
# =====================================================================
def get_stdp_curve(dt_range, murburn=False):
    """
    Computes STDP synaptic weight change vs spike timing difference dt (ms).
    """
    A_plus, A_minus = 0.8, 0.7
    tau_plus, tau_minus = 20.0, 25.0
    if not murburn:
        dw = np.where(dt_range > 0,
                      A_plus * np.exp(-dt_range / tau_plus),
                      -A_minus * np.exp(dt_range / tau_minus))
    else:
        # Murburn DROS shifts tau_plus (widens LTP window) and reduces LTD
        dw = np.where(dt_range > 0,
                      A_plus * 1.25 * np.exp(-dt_range / (tau_plus * 1.4)),
                      -A_minus * 0.85 * np.exp(dt_range / tau_minus))
    return dw

def get_excitotoxic_survival(glu_doses):
    """
    Computes survival fraction vs glutamate dose under three conditions.
    """
    LD50_normal = 2.2
    LD50_murb = 1.5
    LD50_protected = 3.5
    hill_n = 3
    survival_normal = 1.0 / (1.0 + (glu_doses / LD50_normal)**hill_n)
    survival_murb   = 1.0 / (1.0 + (glu_doses / LD50_murb)**hill_n)
    survival_protected   = 1.0 / (1.0 + (glu_doses / LD50_protected)**hill_n)
    return survival_normal, survival_murb, survival_protected

def get_glt1_clearance_kinetics(Glu_ext):
    """
    Michaelis-Menten glutamate uptake rate for GLT-1 and GLAST.
    """
    Vmax_glt1  = 3.5   # mM/s
    Km_glt1    = 0.4   # mM
    Vmax_glast = 1.2
    Km_glast   = 0.5
    Vmax_murb  = Vmax_glt1 * 0.55   # 45% reduction under high DROS
    Vmax_rtms  = Vmax_glt1 * 1.4    # 40% enhancement under rTMS

    J_glt1   = Vmax_glt1   * Glu_ext / (Km_glt1  + Glu_ext)
    J_glast  = Vmax_glast  * Glu_ext / (Km_glast + Glu_ext)
    J_murb   = Vmax_murb   * Glu_ext / (Km_glt1  + Glu_ext)
    J_rtms   = Vmax_rtms   * Glu_ext / (Km_glt1  + Glu_ext)
    return J_glt1, J_glast, J_murb, J_rtms

def get_tsodyks_markram_stp(spike_times, U_0, tau_rec, tau_fac, A=1.0):
    """
    Tsodyks-Markram short-term synaptic plasticity model (facilitation/depression).
    """
    x, u, T_hist = 1.0, U_0, []
    t_prev = 0.0
    for t_sp in spike_times:
        dt = t_sp - t_prev
        x = 1.0 - (1.0 - x * (1.0 - u)) * np.exp(-dt / tau_rec)
        u = U_0 + (u - U_0) * np.exp(-dt / tau_fac)
        T_hist.append(A * u * x)
        u = u + U_0 * (1.0 - u)
        t_prev = t_sp
    return np.array(T_hist)

def get_buffer_failure_probability(F, D):
    """
    Computes synapse buffer failure probability contour map.
    """
    failure = 1.0 - np.exp(-(F / 80.0)**2) * np.exp(-D * 2.0)
    return np.clip(failure, 0.0, 1.0)

def get_dros_hormesis_response(dros):
    """
    Computes Calabrese-Baldwin inverted-U hormetic curve and processes.
    """
    hormetic = 2.0 * dros * np.exp(-dros / 0.8) - 1.5 * dros**2 * np.exp(-dros / 1.5)
    plasticity = 1.5 * dros * np.exp(-dros / 1.0)
    regen = 1.2 * dros * np.exp(-dros / 0.9)
    toxicity = -0.5 * np.maximum(0.0, dros - 1.5)**2
    return hormetic, plasticity, regen, toxicity

# =====================================================================
# 3. Module 8: Cognitive & Forensic Activity
# =====================================================================
def get_bayesian_inference(x_range):
    """
    Computes prior, likelihood, and posteriors for Bayesian coding.
    """
    prior_mean, prior_std = 1.0, 1.5
    prior = np.exp(-0.5 * ((x_range - prior_mean) / prior_std)**2)
    
    like_mean, like_std = -0.5, 1.0
    likelihood = np.exp(-0.5 * ((x_range - like_mean) / like_std)**2)
    
    posterior = prior * likelihood
    posterior /= np.max(posterior)
    
    # Murburn DROS shifts prior mean
    prior_murb = np.exp(-0.5 * ((x_range - (prior_mean + 0.5)) / (prior_std * 0.8))**2)
    posterior_murb = prior_murb * likelihood
    posterior_murb /= np.max(posterior_murb)
    
    return prior, likelihood, posterior, posterior_murb

def get_forensic_p300_amplitude(time_since):
    """
    Computes decaying P300 truth recognition amplitude over days since event.
    """
    p300_trusted = 0.85 * np.exp(-time_since / 180.0) + 0.15
    p300_reTMS   = 0.90 * np.exp(-time_since / 250.0) + 0.20
    p300_murb    = 0.92 * np.exp(-time_since / 300.0) + 0.25
    return p300_trusted, p300_reTMS, p300_murb

def get_schumann_spectra(freq_env, seed=55):
    """
    Computes environmental Schumann resonance EM location contexts.
    """
    rng = np.random.RandomState(seed)
    schumann_peaks = [7.83, 14.3, 20.8, 27.3, 33.8]
    env_field = 0.05 * np.ones(len(freq_env)) + 0.02 * rng.randn(len(freq_env))
    for f_peak in schumann_peaks:
        env_field += 0.3 * np.exp(-((freq_env - f_peak)**2) / 0.2)
        
    env_A = env_field + 0.5 * np.exp(-((freq_env - 50.0)**2) / 0.05)
    env_B = env_field.copy()
    env_A_murb = env_A * (1.0 + 0.3 * np.exp(-freq_env / 30.0))
    return env_A, env_B, env_A_murb

def get_forgetting_curves(t_years):
    """
    Ebbinghaus, consolidated, periodically reinforced, and Murburn memory curves.
    """
    ebbinghaus  = np.exp(-t_years / (365.0 * 2.0))
    consolidated = np.exp(-t_years / (365.0 * 20.0))
    lt_reinforced = np.exp(-t_years / (365.0 * 50.0))
    murb_prot   = np.exp(-t_years / (365.0 * 65.0))
    return ebbinghaus, consolidated, lt_reinforced, murb_prot

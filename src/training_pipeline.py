"""
Simulation training and execution pipeline coordinating all 8 modules.
Solves all ODEs and computes physical curves, then caches results to data/synthetic/.
"""

import os
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import spectrogram, welch, butter, filtfilt
import networkx as nx

# Import custom solvers and activity equations
from src.data_download import get_tissue_conductivity_params
from src.preprocessing import CONFIG_PATH, BIO_CONSTANTS, HODGKIN_HUXLEY_PARAMS, KURAMOTO_PARAMS
from src.protein_inference_gnn import (
    get_dros_rates, get_rpm_yields, get_murburn_rates_vs_b,
    get_dros_diffusion_profile, get_energy_landscapes, get_antioxidant_response
)
from src.pointer_network import (
    run_kuramoto_simulation, run_bbi_simulation, run_hive_mind_simulation,
    get_connectome_network, simulate_signal_propagation, run_virtual_surgery_outcomes,
    get_microtubule_mode_profile
)
from src.neural_ode_phospho import (
    radical_chain, hodgkin_huxley, microglia_ode, epileptor, thal_cortex,
    central_sensitisation, tc_loop, bdnf_cascade, excitotoxic_cascade,
    growth_cone_ode, glu_clearance, tripartite_ca, vesicle_pools,
    global_workspace, lucid_dream_ode, rubber_hand, rubber_hand_async,
    engram_ode, reconsolidation, memory_dros
)
from src.kinase_activity import (
    get_thermoelectric_output, get_thermoelectric_power_density,
    get_biofuel_cell_curves, get_biofuel_cell_vs_glucose, get_piezo_voltages,
    get_stdp_curve, get_excitotoxic_survival, get_glt1_clearance_kinetics,
    get_tsodyks_markram_stp, get_buffer_failure_probability,
    get_dros_hormesis_response, get_bayesian_inference,
    get_forensic_p300_amplitude, get_schumann_spectra, get_forgetting_curves
)

def run_simulation_pipeline(force_run=False):
    """
    Orchestrates and runs all simulation steps, caching the outcomes
    to data/synthetic/simulation_data.npz. Returns a dictionary of results.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_path = os.path.join(base_dir, 'data', 'synthetic', 'simulation_data.npz')
    
    if os.path.exists(cache_path) and not force_run:
        print(f"--> Loading cached simulation results from: {cache_path}")
        try:
            # We load the NPZ file as a dictionary of arrays
            data = np.load(cache_path, allow_pickle=True)
            return {key: data[key] for key in data.files}
        except Exception as e:
            print(f"--> Error loading cache: {e}. Re-running simulations...")
            
    print("--> Running full multi-scale simulation suite (this may take up to 40 seconds)...")
    results = {}
    
    # -----------------------------------------------------------------
    # MODULE 1: Murburn foundations
    # -----------------------------------------------------------------
    O2 = np.linspace(0.0, 300.0, 300)
    v_classic, v_murburn = get_dros_rates(O2)
    results['mod1_O2'] = O2
    results['mod1_v_classic'] = v_classic
    results['mod1_v_murburn'] = v_murburn
    
    t_span = (0.0, 60.0)
    y0_rc = [0.01, 0.001, 0.0001, 0.0, 0.0]
    sol_rc = solve_ivp(radical_chain, t_span, y0_rc, args=(0.5, 0.8, 0.3, 1.2, 0.4), max_step=0.1)
    results['mod1_rc_t'] = sol_rc.t
    results['mod1_rc_y'] = sol_rc.y
    
    B = np.linspace(0.0, 200.0, 500)
    phi_S, phi_T = get_rpm_yields(B)
    results['mod1_B'] = B
    results['mod1_phi_S'] = phi_S
    results['mod1_phi_T'] = phi_T
    
    B_range = np.linspace(0.001, 100.0, 400)
    k_dros_norm, k_atp_norm = get_murburn_rates_vs_b(B_range)
    results['mod1_B_range'] = B_range
    results['mod1_k_dros_norm'] = k_dros_norm
    results['mod1_k_atp_norm'] = k_atp_norm
    
    x_diff = np.linspace(0.0, 50.0, 300)
    results['mod1_x_diff'] = x_diff
    t_vals_diff = [0.5, 2.0, 5.0, 15.0, 50.0]
    results['mod1_t_vals_diff'] = np.array(t_vals_diff)
    for i, t_val in enumerate(t_vals_diff):
        results[f'mod1_diff_profile_{i}'] = get_dros_diffusion_profile(x_diff, t_val)
        
    rxn_coord = np.linspace(0.0, 10.0, 200)
    G_classic, G_murburn = get_energy_landscapes(rxn_coord)
    results['mod1_rxn_coord'] = rxn_coord
    results['mod1_G_classic'] = G_classic
    results['mod1_G_murburn'] = G_murburn
    
    B_field = np.linspace(0.0, 300.0, 300)
    GSH_GSSG, DROS_level = get_antioxidant_response(B_field)
    results['mod1_B_field'] = B_field
    results['mod1_GSH_GSSG'] = GSH_GSSG
    results['mod1_DROS_level'] = DROS_level

    # -----------------------------------------------------------------
    # MODULE 2: Neural EM
    # -----------------------------------------------------------------
    # Panel B: E-field depth profile
    depth = np.linspace(0.0, 30.0, 300)
    E_primary = 120 * np.exp(-depth / 25.0)
    E_shunting = 40 * np.exp(-((depth - 10.0)**2) / 4.0)
    E_total = E_primary - E_shunting + 5 * np.random.RandomState(42).randn(300) * 0.1
    results['mod2_depth'] = depth
    results['mod2_E_primary'] = E_primary
    results['mod2_E_total'] = E_total
    
    # Panel C: Strength-duration (Chronaxie-Rheobase)
    pulse_dur = np.logspace(-2, 2, 300)
    results['mod2_pulse_dur'] = pulse_dur
    
    # HH simulations
    t_hh, V_base, m_base, h_base, n_base = hodgkin_huxley(150.0, I_ext_base=10.0)
    _, V_tms, _, _, _ = hodgkin_huxley(150.0, I_ext_base=2.0, I_tms=80.0)
    
    results['mod2_t_hh'] = t_hh
    results['mod2_V_base'] = V_base
    results['mod2_V_tms'] = V_tms
    results['mod2_m_base'] = m_base
    results['mod2_h_base'] = h_base
    results['mod2_n_base'] = n_base
    
    # F-I curve
    I_range = np.arange(0, 50, 2)
    rates_baseline = []
    rates_dros = []
    for I_val in I_range:
        _, V_b, _, _, _ = hodgkin_huxley(500.0, I_ext_base=float(I_val), dros_level=1.0)
        spikes_b = np.sum((V_b[:-1] < 20.0) & (V_b[1:] >= 20.0))
        rates_baseline.append(spikes_b * 2.0)
        
        _, V_d, _, _, _ = hodgkin_huxley(500.0, I_ext_base=float(I_val), dros_level=1.4)
        spikes_d = np.sum((V_d[:-1] < 20.0) & (V_d[1:] >= 20.0))
        rates_dros.append(spikes_d * 2.0)
    results['mod2_I_range'] = I_range
    results['mod2_rates_baseline'] = np.array(rates_baseline)
    results['mod2_rates_dros'] = np.array(rates_dros)
    
    # Phase portraits
    _, V_phase_ctrl, _, _, _ = hodgkin_huxley(200.0, I_ext_base=12.0, dros_level=1.0)
    _, V_phase_dros, _, _, _ = hodgkin_huxley(200.0, I_ext_base=12.0, dros_level=1.5)
    results['mod2_V_phase_ctrl'] = V_phase_ctrl
    results['mod2_V_phase_dros'] = V_phase_dros
    
    # Cole-Cole tissue curves
    freq = np.logspace(0, 6, 500)
    results['mod2_freq'] = freq
    
    # -----------------------------------------------------------------
    # MODULE 3: Gamma, VNS, Memory
    # -----------------------------------------------------------------
    # Kuramoto synchrony scans
    t_kur, r_05, _ = run_kuramoto_simulation(0.5, T=5.0)
    _, r_20, _ = run_kuramoto_simulation(2.0, T=5.0)
    _, r_40, _ = run_kuramoto_simulation(4.0, T=5.0)
    _, r_40_stim, _ = run_kuramoto_simulation(4.0, ext_amp=2.0, ext_freq=40.0, T=5.0)
    
    results['mod3_t_kur'] = t_kur
    results['mod3_r_05'] = r_05
    results['mod3_r_20'] = r_20
    results['mod3_r_40'] = r_40
    results['mod3_r_40_stim'] = r_40_stim
    
    # Microglia phenotypic transitions
    t_h_micro = np.linspace(0, 36, 360)
    y0_micro = [1.0, 0.0, 0.0]
    sol_micro_base = solve_ivp(microglia_ode, (0, 36), y0_micro, args=(0.1, 0.1, 0.05, 0.02, 0.0), max_step=0.1, dense_output=True)
    sol_micro_40hz = solve_ivp(microglia_ode, (0, 36), y0_micro, args=(0.3, 0.1, 0.15, 0.04, 1.5), max_step=0.1, dense_output=True)
    results['mod3_t_h_micro'] = t_h_micro
    results['mod3_micro_base'] = sol_micro_base.sol(t_h_micro)
    results['mod3_micro_40hz'] = sol_micro_40hz.sol(t_h_micro)
    
    # Kuramoto frequency tuning curve
    freq_scan = np.arange(5, 80, 1)
    order_scan = []
    for f_scan in freq_scan:
        _, r_scan, _ = run_kuramoto_simulation(3.0, ext_amp=3.0, ext_freq=float(f_scan), T=2.0)
        order_scan.append(np.mean(r_scan[-500:]))
    results['mod3_freq_scan'] = freq_scan
    results['mod3_order_scan'] = np.array(order_scan)
    
    # VNS CAIP cytokine suppression
    t_h_vns = np.linspace(0, 48, 480)
    results['mod3_t_h_vns'] = t_h_vns
    
    # ACh concentrations and macrophage inhibition
    ACh_conc = np.logspace(-10, -4, 300)
    results['mod3_ACh_conc'] = ACh_conc
    
    # Memory reconsolidation lability curves
    t_h_recon = np.linspace(0, 24, 240)
    results['mod3_t_h_recon'] = t_h_recon

    # -----------------------------------------------------------------
    # MODULE 4: Energy Harvesting
    # -----------------------------------------------------------------
    dT_teg = np.linspace(0, 20, 200)
    results['mod4_dT_teg'] = dT_teg
    
    R_L = np.logspace(-1, 3, 300)
    results['mod4_R_L'] = R_L
    
    J = np.linspace(0, 2.0, 300)
    results['mod4_J'] = J
    
    glucose_conc = np.linspace(0.01, 5, 300)
    results['mod4_glucose_conc'] = glucose_conc
    
    t_piezo = np.linspace(0, 10, 1000)
    results['mod4_t_piezo'] = t_piezo
    
    t_cap = np.linspace(0, 120, 1200)
    results['mod4_t_cap'] = t_cap
    
    years_teg = [2020, 2022, 2024, 2026, 2028, 2030, 2035]
    results['mod4_years_teg'] = np.array(years_teg)

    # -----------------------------------------------------------------
    # MODULE 5: BBI, Levitation, Connectome
    # -----------------------------------------------------------------
    # BBI synchrony curves
    t_bbi, EEG_A, EEG_B_free, EEG_B_coupled, phi_A, phi_B_free, phi_B_coupled = run_bbi_simulation()
    results['mod5_t_bbi'] = t_bbi
    results['mod5_EEG_A'] = EEG_A
    results['mod5_EEG_B_free'] = EEG_B_free
    results['mod5_EEG_B_coupled'] = EEG_B_coupled
    results['mod5_phi_A'] = phi_A
    results['mod5_phi_B_free'] = phi_B_free
    results['mod5_phi_B_coupled'] = phi_B_coupled
    
    # Hive mind alpha coherence and game accuracy
    t_hive, order_hive = run_hive_mind_simulation()
    results['mod5_t_hive'] = t_hive
    results['mod5_order_hive'] = order_hive
    
    # Diamagnetic levitation curves
    B_lev = np.linspace(0, 25, 300)
    results['mod5_B_lev'] = B_lev
    
    # Connectome grid and Epileptor ODE
    y0_ep = [0.015, -10.0, 2.5, 0.0, -0.4, 0.017]
    sol_ep = solve_ivp(epileptor, (0, 4000), y0_ep, method='RK45', max_step=0.1)
    results['mod5_t_ep'] = sol_ep.t[-10000:]
    results['mod5_x1_ep'] = sol_ep.y[0, -10000:]
    
    # Thalamocortical arousal
    y0_tc = [0.02, 0.02]
    sol_coma = solve_ivp(thal_cortex, (0, 60), y0_tc, args=(0.0,), max_step=0.1, dense_output=True)
    sol_arousal = solve_ivp(thal_cortex, (0, 60), y0_tc, args=(0.8,), max_step=0.1, dense_output=True)
    t_plot_tc = np.linspace(0, 60, 600)
    results['mod5_t_plot_tc'] = t_plot_tc
    results['mod5_tc_coma'] = sol_coma.sol(t_plot_tc)[1]
    results['mod5_tc_arousal'] = sol_arousal.sol(t_plot_tc)[1]
    
    # Microtubule biophoton mode profile
    r_mt = np.linspace(0, 25, 300)
    results['mod5_r_mt'] = r_mt
    results['mod5_E_mt'] = get_microtubule_mode_profile(r_mt)
    
    t_us = np.linspace(0, 100, 500)
    results['mod5_t_us'] = t_us

    # -----------------------------------------------------------------
    # MODULE 6: Phantom limb pain
    # -----------------------------------------------------------------
    months_plp = np.linspace(0, 24, 200)
    results['mod6_months_plp'] = months_plp
    
    displacement_mm = np.linspace(0, 25, 200)
    results['mod6_displacement_mm'] = displacement_mm
    
    freq_plp = np.linspace(1, 80, 500)
    results['mod6_freq_plp'] = freq_plp
    
    t_eval_cs = np.linspace(0, 60, 600)
    y0_cs = [0.0, 0.05]
    sol_cs_low = solve_ivp(central_sensitisation, (0, 60), y0_cs, t_eval=t_eval_cs, args=(0.4, False))
    sol_cs_high = solve_ivp(central_sensitisation, (0, 60), y0_cs, t_eval=t_eval_cs, args=(1.2, False))
    sol_cs_murb = solve_ivp(central_sensitisation, (0, 60), y0_cs, t_eval=t_eval_cs, args=(1.2, True))
    
    results['mod6_t_eval_cs'] = t_eval_cs
    results['mod6_cs_low_W'] = sol_cs_low.y[0]
    results['mod6_cs_high_W'] = sol_cs_high.y[0]
    results['mod6_cs_murb_W'] = sol_cs_murb.y[0]
    results['mod6_cs_low_DROS'] = sol_cs_low.y[1]
    results['mod6_cs_high_DROS'] = sol_cs_high.y[1]
    results['mod6_cs_murb_DROS'] = sol_cs_murb.y[1]
    
    # TC loop resonance in phantom pain
    t_ev_tc = np.linspace(0, 5, 2000)
    s_tc_n = solve_ivp(tc_loop, (0, 5), [0.3, 0.3], t_eval=t_ev_tc, args=(1.5, 1.5, 0.05, 0.04, 0.2))
    s_tc_ph = solve_ivp(tc_loop, (0, 5), [0.3, 0.3], t_eval=t_ev_tc, args=(2.8, 2.8, 0.05, 0.04, 0.6))
    s_tc_tr = solve_ivp(tc_loop, (0, 5), [0.3, 0.3], t_eval=t_ev_tc, args=(1.4, 1.4, 0.05, 0.04, 0.15))
    
    results['mod6_t_ev_tc'] = t_ev_tc
    results['mod6_tc_n_Et'] = s_tc_n.y[0]
    results['mod6_tc_ph_Et'] = s_tc_ph.y[0]
    results['mod6_tc_tr_Et'] = s_tc_tr.y[0]
    results['mod6_tc_n_Ec'] = s_tc_n.y[1]
    results['mod6_tc_ph_Ec'] = s_tc_ph.y[1]
    
    t2_gate = np.linspace(0, 10, 500)
    results['mod6_t2_gate'] = t2_gate
    
    scs_amp_mA = np.linspace(0, 10, 200)
    results['mod6_scs_amp_mA'] = scs_amp_mA
    
    t_mo_plp = np.linspace(0, 60, 500)
    results['mod6_t_mo_plp'] = t_mo_plp
    
    sessions_plp = np.arange(1, 31)
    results['mod6_sessions_plp'] = sessions_plp
    
    t_sc_fmri = np.linspace(0, 20, 400)
    results['mod6_t_sc_fmri'] = t_sc_fmri
    
    t_vr_plp = np.linspace(0, 60, 600)
    results['mod6_t_vr_plp'] = t_vr_plp
    
    week_plp = np.linspace(0, 6, 600)
    results['mod6_week_plp'] = week_plp
    
    t_plas_plp = np.linspace(0, 30, 300)
    results['mod6_t_plas_plp'] = t_plas_plp
    
    t_tms_plp = np.linspace(0, 30, 3000)
    results['mod6_t_tms_plp'] = t_tms_plp
    
    t_dc_plp = np.linspace(0, 25, 500)
    results['mod6_t_dc_plp'] = t_dc_plp
    
    t_cl_plp = np.linspace(0, 10, 2000)
    results['mod6_t_cl_plp'] = t_cl_plp
    
    t_adj_plp = np.linspace(0, 90, 900)
    results['mod6_t_adj_plp'] = t_adj_plp

    # -----------------------------------------------------------------
    # MODULE 7: Neuroplasticity, Excitotoxicity, Regeneration
    # -----------------------------------------------------------------
    t_ev_plas = np.linspace(0, 30, 600)
    sol_bdnf_base = solve_ivp(bdnf_cascade, (0, 30), [0]*4, t_eval=t_ev_plas, args=(1.0, False))
    sol_bdnf_murb = solve_ivp(bdnf_cascade, (0, 30), [0]*4, t_eval=t_ev_plas, args=(1.0, True))
    
    results['mod7_t_ev_plas'] = t_ev_plas
    results['mod7_bdnf_base'] = sol_bdnf_base.y
    results['mod7_bdnf_murb'] = sol_bdnf_murb.y
    
    dt_range_plas = np.linspace(-80, 80, 400)
    results['mod7_dt_range_plas'] = dt_range_plas
    
    t_sim_plas = np.arange(0, 120, 0.5)
    results['mod7_t_sim_plas'] = t_sim_plas
    
    bdnf_conc_plas = np.linspace(0, 100, 300)
    results['mod7_bdnf_conc_plas'] = bdnf_conc_plas
    
    age_plas = np.linspace(0, 80, 400)
    results['mod7_age_plas'] = age_plas
    
    days_plas = np.linspace(0, 30, 300)
    results['mod7_days_plas'] = days_plas
    
    # Excitotoxicity cascade
    t_ev_exc = np.linspace(0, 40, 800)
    y0_exc = [0.0, 0.1, 0.0, 0.01, 1.0]
    sol_exc_low = solve_ivp(excitotoxic_cascade, (0, 40), y0_exc, t_eval=t_ev_exc, args=(0.5, False))
    sol_exc_high = solve_ivp(excitotoxic_cascade, (0, 40), y0_exc, t_eval=t_ev_exc, args=(2.0, False))
    sol_exc_murb = solve_ivp(excitotoxic_cascade, (0, 40), y0_exc, t_eval=t_ev_exc, args=(2.0, True))
    sol_exc_treat = solve_ivp(excitotoxic_cascade, (0, 40), y0_exc, t_eval=t_ev_exc, args=(0.3, False))
    
    results['mod7_t_ev_exc'] = t_ev_exc
    results['mod7_exc_low_health'] = sol_exc_low.y[4]
    results['mod7_exc_high_health'] = sol_exc_high.y[4]
    results['mod7_exc_murb_health'] = sol_exc_murb.y[4]
    results['mod7_exc_treat_health'] = sol_exc_treat.y[4]
    results['mod7_exc_low_Ca'] = sol_exc_low.y[1]
    results['mod7_exc_high_Ca'] = sol_exc_high.y[1]
    results['mod7_exc_murb_Ca'] = sol_exc_murb.y[1]
    results['mod7_exc_high_DROS'] = sol_exc_high.y[3]
    results['mod7_exc_murb_DROS'] = sol_exc_murb.y[3]
    results['mod7_exc_high_mPTP'] = sol_exc_high.y[2]
    results['mod7_exc_murb_mPTP'] = sol_exc_murb.y[2]
    
    glu_doses_exc = np.linspace(0, 5, 100)
    results['mod7_glu_doses_exc'] = glu_doses_exc
    
    # Neuro-regeneration axon growth
    t_grow = np.linspace(0, 20, 2000)
    sol_gc_notms = solve_ivp(growth_cone_ode, (0, 20), [0.5, 0.5, 0.0, 0.0], t_eval=t_grow, args=(2.0, 1.5, 1.2, False))
    sol_gc_tms    = solve_ivp(growth_cone_ode, (0, 20), [0.5, 0.5, 0.0, 0.0], t_eval=t_grow, args=(2.0, 1.5, 1.2, True))
    sol_gc_nongf = solve_ivp(growth_cone_ode, (0, 20), [0.5, 0.5, 0.0, 0.0], t_eval=t_grow, args=(0.5, 0.3, 1.8, False))
    
    results['mod7_t_grow'] = t_grow
    results['mod7_gc_notms_xy'] = np.array([sol_gc_notms.y[0], sol_gc_notms.y[1]])
    results['mod7_gc_tms_xy'] = np.array([sol_gc_tms.y[0], sol_gc_tms.y[1]])
    results['mod7_gc_nongf_xy'] = np.array([sol_gc_nongf.y[0], sol_gc_nongf.y[1]])
    
    t_scar = np.linspace(0, 180, 600)
    results['mod7_t_scar'] = t_scar
    
    t_days_regen = np.linspace(0, 60, 400)
    results['mod7_t_days_regen'] = t_days_regen
    
    days_diff_regen = np.linspace(0, 21, 300)
    results['mod7_days_diff_regen'] = days_diff_regen
    
    months_regen = np.linspace(0, 6, 300)
    results['mod7_months_regen'] = months_regen
    
    # Synaptic clearance and buffering
    Glu_ext_buf = np.linspace(0, 5, 400)
    results['mod7_Glu_ext_buf'] = Glu_ext_buf
    
    t_ms_clear = np.linspace(0, 200, 2000)
    results['mod7_t_ms_clear'] = t_ms_clear
    
    t_buf_ca = np.linspace(0, 100, 1000)
    sol_ca_healthy = solve_ivp(tripartite_ca, (0, 100), [0.1, 0.05], t_eval=t_buf_ca, args=(0.8, 0.4))
    sol_ca_impaired = solve_ivp(tripartite_ca, (0, 100), [0.1, 0.05], t_eval=t_buf_ca, args=(0.4, 0.15))
    sol_ca_rtms     = solve_ivp(tripartite_ca, (0, 100), [0.1, 0.05], t_eval=t_buf_ca, args=(1.1, 0.6))
    
    results['mod7_t_buf_ca'] = t_buf_ca
    results['mod7_ca_healthy_spine'] = sol_ca_healthy.y[0]
    results['mod7_ca_impaired_spine'] = sol_ca_impaired.y[0]
    results['mod7_ca_rtms_spine'] = sol_ca_rtms.y[0]
    results['mod7_ca_healthy_astro'] = sol_ca_healthy.y[1]
    
    t_vp = np.arange(0, 2000, 0.5)
    y0_vp = [1.0, 0.7, 1.0]
    sol_vp_20hz  = solve_ivp(vesicle_pools, (0, 2000), y0_vp, t_eval=t_vp, args=(20, 0.05))
    sol_vp_100hz = solve_ivp(vesicle_pools, (0, 2000), y0_vp, t_eval=t_vp, args=(100, 0.05))
    sol_vp_murb  = solve_ivp(vesicle_pools, (0, 2000), y0_vp, t_eval=t_vp, args=(100, 0.02))
    
    results['mod7_t_vp'] = t_vp
    results['mod7_vp_20hz_RRP'] = sol_vp_20hz.y[0]
    results['mod7_vp_100hz_RRP'] = sol_vp_100hz.y[0]
    results['mod7_vp_murb_RRP'] = sol_vp_murb.y[0]
    results['mod7_vp_20hz_RP'] = sol_vp_20hz.y[1]
    
    dros_horm = np.linspace(0, 5, 500)
    results['mod7_dros_horm'] = dros_horm
    
    intensity_pct_dros = np.linspace(60, 140, 100)
    results['mod7_intensity_pct_dros'] = intensity_pct_dros
    
    t_sess_dros = np.linspace(0, 60, 600)
    results['mod7_t_sess_dros'] = t_sess_dros

    # -----------------------------------------------------------------
    # MODULE 8: Consciousness & Memory
    # -----------------------------------------------------------------
    t_states_cons = np.linspace(0, 60, 600)
    results['mod8_t_states_cons'] = t_states_cons
    
    t_gw_cons = np.linspace(0, 30, 600)
    sol_gw_aware = solve_ivp(global_workspace, (0, 30), [0, 0, 0, 0.3], t_eval=t_gw_cons, args=(1.2, True))
    sol_gw_sub   = solve_ivp(global_workspace, (0, 30), [0, 0, 0, 0.3], t_eval=t_gw_cons, args=(0.3, True))
    sol_gw_anesth= solve_ivp(global_workspace, (0, 30), [0, 0, 0, 0.2], t_eval=t_gw_cons, args=(1.2, False))
    
    results['mod8_t_gw_cons'] = t_gw_cons
    results['mod8_gw_aware'] = sol_gw_aware.y
    results['mod8_gw_sub'] = sol_gw_sub.y
    results['mod8_gw_anesth'] = sol_gw_anesth.y
    
    t_tc_cons = np.linspace(0, 1.0, 4000)
    results['mod8_t_tc_cons'] = t_tc_cons
    
    dros_level_cons = np.linspace(0, 3, 400)
    results['mod8_dros_level_cons'] = dros_level_cons
    
    t_anesth_cons = np.linspace(0, 120, 600)
    results['mod8_t_anesth_cons'] = t_anesth_cons
    
    # Dreams hypnogram
    t_night_dream = np.linspace(0, 480, 4800)
    results['mod8_t_night_dream'] = t_night_dream
    
    t_lfp_dream = np.linspace(0, 1.5, 6000)
    results['mod8_t_lfp_dream'] = t_lfp_dream
    
    t_rem_dream = np.linspace(0, 60, 600)
    results['mod8_t_rem_dream'] = t_rem_dream
    
    t_luc_dream = np.linspace(0, 60, 1200)
    sol_luc_dream = solve_ivp(lucid_dream_ode, (0, 60), [0.3, 0.0, 0.0, 1.2], t_eval=t_luc_dream)
    sol_ord_dream = solve_ivp(lucid_dream_ode, (0, 60), [0.3, 0.0, 0.0, 0.5], t_eval=t_luc_dream)
    results['mod8_t_luc_dream'] = t_luc_dream
    results['mod8_luc_dream_y'] = sol_luc_dream.y
    results['mod8_ord_dream_y'] = sol_ord_dream.y
    
    # Predictive coding & Illusion
    t_pc_ill = np.linspace(0, 10, 1000)
    results['mod8_t_pc_ill'] = t_pc_ill
    
    t_rhi_ill = np.linspace(0, 30, 600)
    sol_rhi_sync = solve_ivp(rubber_hand, (0, 30), [0, 0, 0, 0], t_eval=t_rhi_ill)
    sol_rhi_async = solve_ivp(rubber_hand_async, (0, 30), [0, 0, 0, 0], t_eval=t_rhi_ill)
    results['mod8_t_rhi_ill'] = t_rhi_ill
    results['mod8_rhi_sync_O'] = sol_rhi_sync.y[0]
    results['mod8_rhi_async_O'] = sol_rhi_async.y[0]
    results['mod8_rhi_sync_M'] = sol_rhi_sync.y[3]
    
    t_predict_time = np.linspace(0, 6, 600)
    results['mod8_t_predict_time'] = t_predict_time
    
    x_range_bayes = np.linspace(-5, 5, 500)
    results['mod8_x_range_bayes'] = x_range_bayes
    
    t_before_erp = np.linspace(-3, 5, 400)
    results['mod8_t_before_erp'] = t_before_erp
    
    # Memory engrams and engram kinetics
    t_eng_mem = np.linspace(0, 40, 800)
    sol_eng_strong = solve_ivp(engram_ode, (0, 40), [0, 0, 0, 0.2], t_eval=t_eng_mem, args=(1.5, False))
    sol_eng_weak   = solve_ivp(engram_ode, (0, 40), [0, 0, 0, 0.2], t_eval=t_eng_mem, args=(0.5, False))
    sol_eng_murb   = solve_ivp(engram_ode, (0, 40), [0, 0, 0, 0.2], t_eval=t_eng_mem, args=(1.5, True))
    
    results['mod8_t_eng_mem'] = t_eng_mem
    results['mod8_eng_strong_y'] = sol_eng_strong.y
    results['mod8_eng_weak_y'] = sol_eng_weak.y
    results['mod8_eng_murb_y'] = sol_eng_murb.y
    
    context_match_mem = np.linspace(0, 1, 300)
    results['mod8_context_match_mem'] = context_match_mem
    
    t_years_mem = np.linspace(0, 80 * 365, 2000)
    results['mod8_t_years_mem'] = t_years_mem
    
    t_ltp_mem = np.linspace(0, 360, 720)
    results['mod8_t_ltp_mem'] = t_ltp_mem
    
    # Memory reconsolidation lability
    t_rc_recon = np.linspace(0, 30, 600)
    sol_rc_normal = solve_ivp(reconsolidation, (0, 30), [1, 0, 0, 0], t_eval=t_rc_recon, args=(8, 0.0))
    sol_rc_modify = solve_ivp(reconsolidation, (0, 30), [1, 0, 0, 0], t_eval=t_rc_recon, args=(8, 0.8))
    sol_rc_erase  = solve_ivp(reconsolidation, (0, 30), [1, 0, 0, 0], t_eval=t_rc_recon, args=(8, -0.5))
    
    results['mod8_t_rc_recon'] = t_rc_recon
    results['mod8_rc_normal_M'] = sol_rc_normal.y[0]
    results['mod8_rc_modify_M'] = sol_rc_modify.y[0]
    results['mod8_rc_modify_M2'] = sol_rc_modify.y[2]
    results['mod8_rc_erase_M'] = sol_rc_erase.y[0]
    
    t_dm_dros = np.linspace(0, 30, 600)
    sol_dm_02 = solve_ivp(memory_dros, (0, 30), [0.5, 0.1, 0.1], t_eval=t_dm_dros, args=(0.2,))
    sol_dm_10 = solve_ivp(memory_dros, (0, 30), [0.5, 0.1, 0.1], t_eval=t_dm_dros, args=(1.0,))
    sol_dm_25 = solve_ivp(memory_dros, (0, 30), [0.5, 0.1, 0.1], t_eval=t_dm_dros, args=(2.5,))
    sol_dm_15 = solve_ivp(memory_dros, (0, 30), [0.5, 0.1, 0.1], t_eval=t_dm_dros, args=(1.5,))
    
    results['mod8_t_dm_dros'] = t_dm_dros
    results['mod8_dm_02_trace'] = sol_dm_02.y[0]
    results['mod8_dm_10_trace'] = sol_dm_10.y[0]
    results['mod8_dm_25_trace'] = sol_dm_25.y[0]
    results['mod8_dm_15_trace'] = sol_dm_15.y[0]
    
    t_int_em = np.linspace(0, 20, 2000)
    results['mod8_t_int_em'] = t_int_em
    
    track_pc = np.linspace(0, 200, 500)
    results['mod8_track_pc'] = track_pc
    
    t_tg_pc = np.linspace(0, 0.5, 5000)
    results['mod8_t_tg_pc'] = t_tg_pc
    
    time_since_p300 = np.linspace(0, 365, 300)
    results['mod8_time_since_p300'] = time_since_p300
    
    freq_schumann = np.linspace(1, 100, 500)
    results['mod8_freq_schumann'] = freq_schumann

    # Save results dictionary to compressed NPZ
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, **results)
    print(f"--> Saved simulation results cache successfully at: {cache_path}")
    
    return results

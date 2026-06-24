"""
Network simulations: Kuramoto oscillator models (Gamma entrainment, BBI coupling)
and Small-World connectome networks (Digital Twin model).
"""

import numpy as np
import networkx as nx
from src.preprocessing import KURAMOTO_PARAMS

def run_kuramoto_simulation(K, ext_amp=0.0, ext_freq=40.0, T=5.0, dt=0.001, seed=42):
    """
    Simulates a Kuramoto phase-oscillator network of hippocampal interneurons.
    """
    N = KURAMOTO_PARAMS['N_interneurons']
    t = np.arange(0, T, dt)
    rng = np.random.RandomState(seed)
    
    # Natural frequencies centered around 40 Hz (2*pi*40 rad/s)
    center_freq = KURAMOTO_PARAMS['f_genus'] * 2 * np.pi
    half_width = KURAMOTO_PARAMS['f_half_width'] * 2 * np.pi
    omega_dist = rng.normal(center_freq, half_width, N)
    
    theta = rng.uniform(-np.pi, np.pi, N)
    order_params = []
    
    for ti in t:
        # Sum coupling over all node pairs
        coupling = (K / N) * np.sum(np.sin(theta[:, None] - theta[None, :]), axis=1)
        ext = ext_amp * np.sin(ext_freq * 2 * np.pi * ti)
        theta += dt * (omega_dist + coupling + ext)
        r = np.abs(np.mean(np.exp(1j * theta)))
        order_params.append(r)
        
    return t, np.array(order_params), theta

def run_bbi_simulation(T=10.0, dt=0.001, seed=42):
    """
    Simulates Brain-to-Brain Interface coupling of alpha rhythms between Brain A and B.
    """
    t = np.arange(0, T, dt)
    rng = np.random.RandomState(seed)
    
    f_A = 10.0  # Hz
    f_B = 10.2  # Hz (slightly mismatched)
    
    phi_A = np.cumsum(2 * np.pi * f_A * dt * np.ones(len(t)) + rng.randn(len(t)) * 0.05) % (2 * np.pi)
    phi_B_free = np.cumsum(2 * np.pi * f_B * dt * np.ones(len(t)) + rng.randn(len(t)) * 0.05) % (2 * np.pi)
    
    K_bbi = 0.0
    phi_B_coupled = phi_B_free.copy()
    
    for i in range(1, len(t)):
        if t[i] > 3.0:
            K_bbi = min(K_bbi + 0.0003, 2.0)
        phi_B_coupled[i] = phi_B_coupled[i-1] + dt * (
            2 * np.pi * f_B + K_bbi * np.sin(phi_A[i-1] - phi_B_coupled[i-1]) + rng.randn() * 0.02
        )
        
    EEG_A = np.sin(phi_A)
    EEG_B_free = np.sin(phi_B_free)
    EEG_B_coupled = np.sin(phi_B_coupled % (2 * np.pi))
    
    return t, EEG_A, EEG_B_free, EEG_B_coupled, phi_A, phi_B_free, phi_B_coupled

def run_hive_mind_simulation(N_brains=5, K_net=1.5, T=30.0, dt=0.001, seed=42):
    """
    Simulates synchronization (order parameter r) across a network of coupled brains.
    """
    t = np.arange(0, T, dt)
    rng = np.random.RandomState(seed)
    freqs_b = np.array([9.8, 10.0, 10.3, 9.9, 10.1])
    theta_net = rng.uniform(-np.pi, np.pi, N_brains)
    order_params = []
    
    for ti in t:
        coupling_sum = np.zeros(N_brains)
        for i in range(N_brains):
            coupling_sum[i] = (K_net / N_brains) * np.sum(np.sin(theta_net - theta_net[i]))
            
        theta_net += dt * (freqs_b * 2 * np.pi + coupling_sum + rng.randn(N_brains) * 0.02)
        r = np.abs(np.mean(np.exp(1j * theta_net)))
        order_params.append(r)
        
    return t, np.array(order_params)

def get_connectome_network(N_conn=80, k_neighbors=6, beta=0.15, seed=42):
    """
    Generates a Watts-Strogatz small-world network representing the connectome.
    """
    G = nx.watts_strogatz_graph(N_conn, k_neighbors, beta, seed=seed)
    return G

def simulate_signal_propagation(G, seed_node=5, N_t=50, seed=7):
    """
    Simulates standard random walk / activity propagation through the connectome.
    """
    rng = np.random.RandomState(seed)
    N_nodes = len(G.nodes)
    A = nx.to_numpy_array(G)
    
    # Row-normalize adjacency matrix
    row_sums = A.sum(axis=1, keepdims=True)
    A_norm = A / np.where(row_sums > 0, row_sums, 1.0)
    
    signal = np.zeros((N_nodes, N_t))
    signal[seed_node, 0] = 1.0
    
    for t_step in range(1, N_t):
        signal[:, t_step] = 0.85 * (A_norm @ signal[:, t_step-1]) + 0.1 * signal[:, t_step-1]
        signal[:, t_step] += rng.randn(N_nodes) * 0.005
        
    return signal

def run_virtual_surgery_outcomes(G):
    """
    Evaluates the impact of resecting each node on connectome connectivity/seizure spread.
    """
    N_nodes = len(G.nodes)
    outcomes = []
    for node in range(N_nodes):
        G_res = G.copy()
        G_res.remove_node(node)
        if len(G_res.nodes) > 0:
            lcc = max(nx.connected_components(G_res), key=len)
            spread = len(lcc) / N_nodes
        else:
            spread = 0.0
        outcomes.append(spread)
    return outcomes

def get_microtubule_mode_profile(r, R_in=7.5, R_out=12.5):
    """
    TE01-like Bessel mode profile of optical wave guided in microtubule wall.
    """
    from scipy.special import j0
    n_tubulin = 1.55
    n_water = 1.33
    k_out = 2 * np.pi * n_water / 500
    k_in = 2 * np.pi * n_tubulin / 500
    
    E = np.zeros_like(r)
    for i, ri in enumerate(r):
        if ri <= R_in:
            E[i] = j0(k_out * ri) * 0.3
        elif ri <= R_out:
            E[i] = j0(k_in * ri) * 0.8 + np.exp(-(ri - 10.0)**2 / 5.0) * 0.4
        else:
            E[i] = np.exp(-(ri - R_out) * 0.2) * 0.2
    return E

"""
Murburn foundations, Radical Pair Mechanism (RPM), and Redox Web network modeling.
"""

import numpy as np
import networkx as nx
from src.preprocessing import BIO_CONSTANTS, COLORS

def get_dros_rates(O2):
    """
    Computes substrate-inhibited Michaelis-Menten DROS rates.
    """
    Vmax_classic = 12.0
    Vmax_murburn = 28.0
    Km_classic = 80.0
    Km_murburn = 35.0
    Ki_murburn = 250.0

    v_classic = Vmax_classic * O2 / (Km_classic + O2)
    v_murburn = Vmax_murburn * O2 / (Km_murburn + O2 + O2**2 / Ki_murburn)
    return v_classic, v_murburn

def get_rpm_yields(B):
    """
    Computes singlet (Phi_S) and triplet (Phi_T) radical pair yields vs B-field.
    """
    hfc = BIO_CONSTANTS['b_hfc_mT']
    k_esc = 0.5
    k_rxn = 2.0

    B_hfc = hfc * np.sqrt(3)
    phi_S = (k_rxn / (k_rxn + k_esc)) * (1 - B**2 / (B**2 + B_hfc**2 + 1e-9))
    phi_T = 1.0 - phi_S
    return phi_S, phi_T

def get_murburn_rates_vs_b(B_range):
    """
    Computes Zeeman modulation of normalized DROS and ATP rate constants.
    """
    # Zeeman modulation of DROS formation rate (normalised)
    k_dros_norm = 1 + 0.35 * np.sin(np.log(B_range) * 1.5) * np.exp(-B_range / 60)
    k_atp_norm  = 1 / k_dros_norm
    return k_dros_norm, k_atp_norm

def get_dros_diffusion_profile(x_nm, t_ms):
    """
    Computes normalized DROS concentration profile from mitochondrial surface.
    """
    D_dros = BIO_CONSTANTS['dros_diffusion_coef']
    x_m = x_nm * 1e-9
    t_s = t_ms * 1e-3
    sigma = np.sqrt(2 * D_dros * t_s)
    conc = np.exp(-x_m**2 / (2 * sigma**2))
    return conc

def get_energy_landscapes(x):
    """
    Computes NADH->ATP Gibbs free energy profiles.
    """
    def energy_profile(coord, barriers, depths):
        G = np.zeros_like(coord)
        for xi, b, d in zip(barriers, depths, [1]*len(barriers)):
            G += b * np.exp(-((coord - xi)**2) / 0.15) - d * np.exp(-((coord - xi)**2) / 0.6)
        return G

    G_classic = energy_profile(x, [2, 4, 6, 8], [4.5, 3.2, 5.0, 2.8]) - 8
    G_murburn = energy_profile(x, [2.5, 4.5, 6.5], [2.1, 1.8, 2.3]) - 9.5
    return G_classic, G_murburn

def get_redox_web_graph():
    """
    Constructs the directed graph representing the Murburn Redox Web.
    """
    G_net = nx.DiGraph()
    nodes = {
        'NADH': (0, 2), 'complex_I': (1, 2), 'CoQ': (2, 1.5),
        'complex_III': (3, 2), 'Cyt_c': (4, 2), 'complex_IV': (5, 2), 'H₂O': (6, 2),
        'O₂•⁻': (2, 0), 'H₂O₂': (3.5, 0), 'OH•': (4.5, 0.7),
        'GSH': (5.5, 0.3), 'ATP': (3, 3.5)
    }
    node_colors = {
        'NADH': '#3498DB', 'complex_I': '#9B59B6', 'CoQ': '#F39C12',
        'complex_III': '#9B59B6', 'Cyt_c': '#E74C3C', 'complex_IV': '#9B59B6',
        'H₂O': '#2ECC71', 'O₂•⁻': '#E74C3C', 'H₂O₂': '#FF8C00',
        'OH•': '#FF0000', 'GSH': '#27AE60', 'ATP': '#27AE60'
    }
    for n, pos in nodes.items():
        G_net.add_node(n, pos=pos, color=node_colors.get(n, '#888888'))

    edges_classical = [
        ('NADH','complex_I'), ('complex_I','CoQ'),
        ('CoQ','complex_III'), ('complex_III','Cyt_c'),
        ('Cyt_c','complex_IV'), ('complex_IV','H₂O')
    ]
    edges_murburn = [
        ('complex_I','O₂•⁻'), ('CoQ','O₂•⁻'),
        ('O₂•⁻','H₂O₂'), ('H₂O₂','OH•'),
        ('OH•','GSH'), ('GSH','ATP'), ('H₂O₂','ATP')
    ]
    
    return G_net, edges_classical, edges_murburn

def get_antioxidant_response(B_field):
    """
    Computes GSH/GSSG ratio and DROS levels vs static B-field strength.
    """
    GSH_GSSG = 10 * np.exp(-B_field * 0.005) + 2 * np.exp(-((B_field - 200)**2)/3000)
    DROS_level = 1.0 + 0.8 * (1.0 - np.exp(-B_field * 0.01))
    return GSH_GSSG, DROS_level

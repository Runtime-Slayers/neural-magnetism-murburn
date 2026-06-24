"""
Reference biological datasets and tissue parameters (e.g. Cole-Cole properties).
"""

# Frequency-dependent dielectric Cole-Cole parameters for biological tissues (Gabriel et al. 1996)
TISSUE_COLE_COLE_PARAMS = {
    'Grey Matter':     {'sigma_dc': 0.33, 'eps_inf': 45,  'tau': 7.9e-3, 'alpha': 0.1},
    'White Matter':    {'sigma_dc': 0.14, 'eps_inf': 32,  'tau': 7.9e-3, 'alpha': 0.1},
    'CSF':             {'sigma_dc': 1.79, 'eps_inf': 109, 'tau': 5e-3,   'alpha': 0.0},
    'Skull (cortical)':{'sigma_dc': 0.01, 'eps_inf': 12,  'tau': 15e-3,  'alpha': 0.2},
    'Scalp':           {'sigma_dc': 0.43, 'eps_inf': 55,  'tau': 7.9e-3, 'alpha': 0.1},
}

def get_tissue_conductivity_params():
    """Returns Cole-Cole parameters for cranial tissue layers."""
    return TISSUE_COLE_COLE_PARAMS

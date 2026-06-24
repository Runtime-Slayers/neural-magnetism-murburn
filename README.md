# Neural Magnetism × Murburn Simulation Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/downloads/)

A professional, modular scientific computing and simulation framework investigating the biophysical interface of **neural electromagnetic fields**, **mitochondrial Murburn bioenergetics**, and the quantum **radical pair mechanism (RPM)**. 

This repository models 19 coupled differential equation systems and network architectures to bridge the gap between macroscopic neural dynamics (e.g., EEG rhythms, rTMS neuromodulation) and sub-neuronal/molecular chemistry.

---

## 🔬 Core Modeling Modules

The framework compiles and evaluates the following systems:
* **Murburn Foundations**: Dehydroascorbate reductase (DROS) Michaelis-Menten kinetics, radical pair mechanism branching yields, and Redox Web graph representations.
* **Neural Electromagnetic Fields**: Hodgkin-Huxley membrane dynamics under DROS modification, cable equations for transcranial magnetic stimulation (rTMS), and Cole-Cole tissue dielectric conductivity.
* **Brain-Computer Interfaces & Cognitive Dynamics**: Kuramoto networks for inter-brain synchrony, Watts-Strogatz small-world connectomes (Digital Twin), and microtubule-biophoton feedback.
* **Energy Harvesting & Forensics**: Thermoelectric generation (Seebeck effect), sweat-glucose biofuel cell kinetics, PVDF piezoelectric harvesting, and Schumann location memory (forensic P300).
* **Neuroplasticity & Toxicity**: BDNF/TrkB STDP time-windows, NMDA-mediated excitotoxicity curves, and astrocyte-mediated synaptic buffering safety thresholds.
* **Consciousness, Sleep & Memory**: Integrated Information Theory (IIT) Phi transitions under anesthesia, sleep-wake state switching, predictive coding error dynamics, state-dependent DROS memory encryption, and electromagnetic hippocampal ripple corruption.

---

## 📂 Repository Structure

The project is organized according to professional python engineering guidelines:

```directory
├── configs/
│   └── hyperparams.yaml        # Shared biological/physical constants and plotting style tokens
├── data/
│   ├── processed/              # Generated high-resolution dark-theme plot panels (fig01 to fig32)
│   └── synthetic/              # Local cache storing intermediate ODE integration results
├── src/
│   ├── __init__.py
│   ├── data_download.py        # Static database parameters (e.g., Cole-Cole dielectric parameters)
│   ├── preprocessing.py        # YAML configuration parsing and custom Matplotlib style initializers
│   ├── pointer_network.py      # Graph pointer networks, Kuramoto, and Connectome models
│   ├── protein_inference_gnn.py# Redox networks, NetworkX graphs, and Murburn rate equations
│   ├── neural_ode_phospho.py   # Coupled ODE systems (HH, CAIP kinetics, memory states)
│   ├── kinase_activity.py      # Biophysical activity models (STDP, Excitotoxicity, Energy Harvesting)
│   ├── training_pipeline.py    # Master simulation runner executing the 19 models and saving caching
│   └── visualization.py        # Rendering engine for the 32 figure panels (high-end HSL dark-theme)
├── main.py                     # Command-line entry orchestrator (runs simulations, plots, and summaries)
├── requirements.txt            # Project dependencies
├── setup.sh                    # Automation shell script to configure python virtual environments
├── LICENSE                     # MIT License
└── CONTRIBUTING.md             # Developer contribution guidelines
```

---

## 🛠️ Prerequisites & Setup

### System Requirements
* **Python 3.10+** (Python 3.13 recommended)
* Standard C/C++ compilation toolchain (optional, for acceleration in scientific libs)

### Installation
Clone this repository and configure a virtual environment:

```bash
# Clone the repository
git clone https://github.com/Runtime-Slayers/neural-magnetism-murburn.git
cd neural-magnetism-murburn

# Create and activate virtual environment (Windows)
py -3 -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

*(Alternatively, run the automated setup script on Unix-like shells: `./setup.sh`)*

---

## 🚀 Execution & Usage

The master entry point is [main.py](file:///c:/Users/MUTHURAMANRAMANATHAN/Downloads/NeuralMagnetism_Murburn/main.py). It provides flexible CLI options:

### Run Full Pipeline
Solve all ODE/network systems, cache intermediate metrics, and render all 32 figure panels:
```bash
py -3 main.py --all
```

### Run Simulations Only
Execute all 19 model integrations and update the cached NumPy archive (`data/synthetic/simulation_data.npz`):
```bash
py -3 main.py --run-simulations
```

### Render Plots Only
Render figure panels from the cached simulation runs:
```bash
py -3 main.py --plot
```

---

## ⚖️ License & Contributions

This project is licensed under the **MIT License** — see the [LICENSE](file:///c:/Users/MUTHURAMANRAMANATHAN/Downloads/NeuralMagnetism_Murburn/LICENSE) file for details. Contributions are freely welcomed under the terms specified in [CONTRIBUTING.md](file:///c:/Users/MUTHURAMANRAMANATHAN/Downloads/NeuralMagnetism_Murburn/CONTRIBUTING.md).

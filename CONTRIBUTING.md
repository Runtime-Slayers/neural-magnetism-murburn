# Contributing to Neural Magnetism × Murburn Project

We welcome contributions to this interdisciplinary research project! Whether you want to fix a bug, improve simulation algorithms, enhance figure layouts, or extend the physical/biological models, this guide is here to help you get started.

## Project Structure

- `configs/` - Project and simulation hyperparameters (`hyperparams.yaml`).
- `data/`
  - `processed/` - Where the generated 32 figures are stored.
  - `synthetic/` - Cached NumPy data from simulation runs (`simulation_data.npz`).
- `figures/` - Copy of the generated figures (used for compiling the LaTeX manuscript).
- `src/`
  - `data_download.py` - Reference data structures and static biological properties.
  - `preprocessing.py` - Configuration loading and styling setup.
  - `pointer_network.py` - Network connectivity and phase-oscillator models.
  - `protein_inference_gnn.py` - Biochemical rates and radical spin dynamics.
  - `neural_ode_phospho.py` - Consolidated differential solvers (coupled systems).
  - `kinase_activity.py` - Activity curves and energy harvesting calculations.
  - `training_pipeline.py` - Runs all simulations and caches calculated arrays.
  - `visualization.py` - Renders all 32 figures with premium styling.
  - `latex_generator.py` - Gracefully runs pdflatex compilation of the manuscript.
- `main.py` - Main orchestration entry point.
- `paper.tex` - LaTeX source code for the research manuscript.

## Getting Started

1. **Setup Environment**:
   Initialize the Python virtual environment:
   ```bash
   ./setup.sh
   ```
   Or manually:
   ```bash
   python -m venv venv
   # On Windows:
   source venv/Scripts/activate
   # On macOS/Linux:
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run Orchestrator**:
   Compile the entire workflow (simulations, figures, paper PDF):
   ```bash
   py -3 main.py --all
   ```

3. **CLI Arguments**:
   - `--run-simulations`: Force recalculation of all mathematical systems and refresh the local cache.
   - `--plot`: Re-render all 32 figures from the cache (great for fast visualization styling iterations).
   - `--compile-pdf`: Compile the LaTeX manuscript.

## Development Workflow

- **Preserve Quantitative Integrity**: All 21 novel claims and 14 ODE models must remain quantitatively consistent with references. If you adjust any coefficient, verify that it does not break limit cycles or blow up integrations.
- **Code Style**: Standard PEP 8 formatting. Avoid adding ad-hoc formatting rules; use the shared configuration parameters in `configs/hyperparams.yaml` for plotting styles.
- **Adding New Figures**: If you add new figures or panels, update both `src/visualization.py` and `paper.tex` to maintain complete document synchronization.

Thank you for helping push boundaries in neuro-electromagnetic research!

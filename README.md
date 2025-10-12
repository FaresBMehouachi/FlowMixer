# FlowMixer
[NeurIPS 2025] Depth-agnostic neural architecture for spatiotemporal forecasting with interpretable Kronecker-Koopman eigenmodes. Sometimes, One layer is all you need.



# 🌊 FlowMixer: Depth-Agnostic Neural Architecture for Interpretable Spatiotemporal Forecasting

[![NeurIPS 2024](https://img.shields.io/badge/NeurIPS-2024-blue.svg)](https://neurips.cc/)
[![arXiv](https://img.shields.io/badge/arXiv-2024.xxxxx-b31b1b.svg)](https://arxiv.org/abs/xxxx)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8+-ff6f00.svg)](https://tensorflow.org/)

**[Paper](link) | [Project Page](link) | [Video](link) | [Slides](link) | [Poster](link)**

## 🎯 TL;DR

**FlowMixer** introduces a mathematically constrained neural architecture where **a single operational layer can represent any depth** through semi-group composition. This eliminates neural architecture search while achieving state-of-the-art performance on time series forecasting, chaos prediction, and turbulent flow modeling.

<p align="center">
  <img src="assets/architecture_overview.png" width="100%">
  <br>
  <em>FlowMixer architecture: Reversible normalization (φ) wraps constrained mixing operations (W_t, W_f) to create interpretable spatiotemporal patterns</em>
</p>

## ✨ Key Features

- **🏗️ Depth-Agnostic**: Single layer with semi-group property - no depth tuning needed
- **🔬 Interpretable**: Direct extraction of Kronecker-Koopman spatiotemporal eigenmodes  
- **📈 State-of-the-Art**: Outperforms deep models on 7 benchmark datasets
- **⚡ Efficient**: 16× faster than spectral methods for fluid dynamics
- **🎯 Versatile**: Unified framework for statistics and dynamics

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/FlowMixer.git
cd FlowMixer

# Install dependencies
pip install -r requirements.txt

# Optional: GPU acceleration for fluid dynamics
pip install cupy-cuda11x  # Adjust for your CUDA version
Basic Usage
pythonimport torch
from flowmixer import FlowMixer

# Initialize model - no depth search needed!
model = FlowMixer(
    seq_len=336,        # Input sequence length
    feature_dim=7,      # Number of features  
    pred_len=96,        # Prediction horizon
    mixer_type='standard'
)

# Forward pass
output = model(input_sequence)

# Extract interpretable eigenmodes
Wt, Wf = model.get_mixing_matrices()
eigenmodes = model.get_kronecker_koopman_modes()
📊 Main Results
Time Series Forecasting
Performance on multivariate benchmarks (MSE):
DatasetFlowMixerBest BaselineImprovementETTh10.3550.361 (TSMixer)1.7%ETTh20.2640.297 (Koopa)11.1%Weather0.1430.145 (TSMixer)1.4%Traffic0.3770.395 (iTransformer)4.6%
Chaotic Systems
<p align="center">
  <img src="assets/chaos_results.png" width="80%">
  <br>
  <em>Long-term predictions (1024 steps) for Lorenz, Rössler, and Aizawa attractors</em>
</p>
Turbulent Flows
<p align="center">
  <img src="assets/turbulence_results.png" width="80%">
  <br>
  <em>Vorticity field prediction for flow past cylinder (Re=150) and NACA airfoil (Re=1000)</em>
</p>
🏃 Running Experiments
Interactive Notebooks
Start with our Jupyter notebooks for hands-on exploration:
bashjupyter notebook notebooks/1_interactive_demo.ipynb

1_interactive_demo.ipynb - Quick start with visualization
2_time_series.ipynb - Forecasting experiments
3_chaos_prediction.ipynb - Chaotic attractors
4_fluid_dynamics.ipynb - Turbulent flow prediction

Command Line
Reproduce paper results:
bash# Time series benchmarks
python experiments/run_forecasting.py --dataset ETTh1 --horizon 96

# Chaotic systems
python experiments/run_chaos.py --system lorenz --steps 1024

# Fluid dynamics
python experiments/run_fluids.py --case cylinder --reynolds 150
🔬 Kronecker-Koopman Analysis
FlowMixer uniquely enables direct extraction of interpretable spatiotemporal patterns:
pythonfrom flowmixer.analysis import analyze_eigenmodes

# Extract eigenmodes
eigenmodes = analyze_eigenmodes(model, sample_data)

# Visualize spatiotemporal patterns
eigenmodes.plot_kronecker_koopman_modes()
eigenmodes.plot_stability_analysis()
<p align="center">
  <img src="assets/eigenmodes.png" width="100%">
  <br>
  <em>Kronecker-Koopman eigenmodes revealing interpretable spatiotemporal structures</em>
</p>
📁 Repository Structure
FlowMixer/
├── flowmixer/              # Core library
│   ├── models.py          # FlowMixer architectures
│   ├── layers.py          # RevIN, mixing modules
│   └── analysis.py        # Eigenmode tools
├── experiments/           # Reproduction scripts
├── notebooks/            # Interactive demos
├── data/                # Dataset directory
└── checkpoints/         # Pretrained models
📖 Citation
If you find this code useful, please cite our paper:
bibtex@inproceedings{mehouachi2024flowmixer,
  title={FlowMixer: A Depth-Agnostic Neural Architecture for 
         Interpretable Spatiotemporal Forecasting},
  author={Mehouachi, Fares B. and Jabari, Saif Eddin},
  booktitle={Advances in Neural Information Processing Systems},
  year={2024}
}
🤝 Contributors

Fares B. Mehouachi - NYU Abu Dhabi
Saif Eddin Jabari - NYU Tandon

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
🙏 Acknowledgments
This work was supported by the NYUAD Center for Interacting Urban Networks (CITIES), funded by Tamkeen under the NYUAD Research Institute Award CG001.
📧 Contact
For questions or collaborations:

Fares B. Mehouachi: fm2620@nyu.edu
Saif Eddin Jabari: sej7@nyu.edu


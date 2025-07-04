# End-to-End Graph Neural Networks for Rainfall-Driven Real-Time Hydraulic Prediction in Stormwater Systems

# Overview
This repository presents Urban-Stormwater-GNN-Surrogate, a novel end-to-end Graph Neural Network (GNN) model designed for rainfall-driven, real-time hydraulic prediction in urban stormwater systems. It accurately emulates runoff generation and flow routing processes, providing predictions for hydraulic states (inflow, depth, and flow) at both network junctions and conduits. The model incorporates physics-guided constraints and a pushforward training strategy for enhanced stability and physical consistency.

![Fig1](https://github.com/user-attachments/assets/1dd2b933-4ffe-485a-9bae-762dc68b713b)


# Key Contributions
- A novel end-to-end GNN surrogate model (GNN-SWS) is developed for rainfall-driven hydraulic prediction in stormwater systems.
- The model jointly learns rainfall-driven inflow dynamics and flow routing at junctions and conduits directly from rainfall inputs.
- Physics-guided constraints and the pushforward training strategy are integrated to improve physical consistency and long-term forecast stability.
- A heterogeneous message-passing architecture distinguishes node types based on subcatchment connectivity, providing a structured hydrologic representation.


# Data Preparation
This module automates the creation of training and evaluation data for the GNN-SWS model using the SWMM hydraulic simulator.

- **Simulations**: Runs SWMM for multiple rainfall events by modifying the base *.inp* file and attaching corresponding *.dat* rainfall files.
- **Static Features**: Extracts geometric and hydraulic properties of subcatchments, junctions, outfalls, and conduits from the SWMM model.
- **Time Series Outputs**: Collects time-dependent variables (e.g., inflow, depth, runoff, flow) from simulation results for each event.
  
All outputs are saved in structured text files under **export_data/**, which serve as inputs to the GNN model.

GNN-SWS/
├── data_preparation/
│   ├── swmm_run.py                  # Main script to run SWMM and extract data
│   ├── swmm_utils.py               # Supporting utilities for simulation and parsing
│   └── rainfall_data/              # Input .dat files for rainfall events
│
├── export_data/                    # Generated outputs from data_preparation/
│   ├── constant_features/          # Static geometry + hydraulic features
│   └── time_series/                # Dynamic simulation outputs (depth, flow, etc.)

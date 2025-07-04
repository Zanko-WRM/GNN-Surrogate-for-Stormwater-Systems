# End-to-End Graph Neural Networks for Rainfall-Driven Real-Time Hydraulic Prediction in Stormwater Systems

# Overview
This repository presents Urban-Stormwater-GNN-Surrogate, a novel end-to-end Graph Neural Network (GNN) model designed for rainfall-driven, real-time hydraulic prediction in urban stormwater systems. It accurately emulates runoff generation and flow routing processes, providing predictions for hydraulic states (inflow, depth, and flow) at both network junctions and conduits. The model incorporates physics-guided constraints and a pushforward training strategy for enhanced stability and physical consistency.

![Fig1](https://github.com/user-attachments/assets/1dd2b933-4ffe-485a-9bae-762dc68b713b)


# Key Contributions
- A novel end-to-end GNN surrogate model (GNN-SWS) is developed for rainfall-driven hydraulic prediction in stormwater systems.
- The model jointly learns rainfall-driven inflow dynamics and flow routing at junctions and conduits directly from rainfall inputs.
- Physics-guided constraints and the pushforward training strategy are integrated to improve physical consistency and long-term forecast stability.
- A heterogeneous message-passing architecture distinguishes node types based on subcatchment connectivity, providing a structured hydrologic representation.

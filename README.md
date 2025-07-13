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

The data preparation process involves several key steps orchestrated by `data_processing_conduits.py`:
1.  **SWMM Data Loading**: Raw SWMM simulation outputs (e.g., `junctions_head.txt`, `conduits_flow.txt`) and static network configuration files (`subcatchments.txt`, `junctions.txt`, `conduits.txt`, `outfalls.txt`) are loaded.
2.  **Feature Engineering**:
    * The `conduits.txt` file is augmented with an 'Area' column derived from 'MaxDepth' (assuming a circular cross-section).
    * **Node-Level Features**: Static properties of junctions and outfalls are combined with aggregated features from connected subcatchments (e.g., total area, average imperviousness). This process distinguishes between "type_a" nodes (those with contributing subcatchments) and "type_b" nodes (those without), allowing for differentiated feature handling in the GNN encoder.
    * **Edge-Level Features**: Static properties of conduits (e.g., Length, Roughness, MaxDepth, Area) are used. Additionally, physics-guided features are computed, such as relative coordinates (`dx`, `dy`) between connected nodes, and differences in hydraulic head (`diff_head`) and inflow (`diff_inflow`) between an edge's 'From_Node' and 'To_Node'.
    * **Dynamic Feature Windows**: Time-series data (e.g., rainfall, accumulated rainfall, head, inflow, flow, depth) is organized into sliding windows of `n_time_step` past observations for both nodes and edges.
    * **Future Exogenous Inputs**: Separate "future rainfall" features are extracted for both nodes and edges, spanning `max_steps_ahead` into the future. These are critical for the pushforward training strategy as they provide necessary exogenous information when predicting beyond the initial time step.
3.  **Graph Construction**: For each time step within an event, a `torch_geometric.data.Data` object is constructed. Each graph contains:
    * `x`: Node features (concatenation of constant and dynamic features).
    * `edge_index`: Graph connectivity (adjacency list).
    * `edge_attr`: Edge features (concatenation of constant and dynamic features).
    * `y` and `y_edges`: Multi-step-ahead ground truth targets for nodes (head, inflow) and edges (flow, depth), respectively.
    * `future_rainfall` and `future_edge_rainfall`: The future exogenous rainfall inputs for nodes and edges, aligned with the prediction horizon.
    * `node_type`: A categorical label indicating whether a node is 'type_a' (with subcatchment contribution) or 'type_b' (without).
4.  **Data Splitting**: The generated graphs are split into training, validation, and test sets. A key aspect is the **event-based split for the test set**, where entire rainfall events are held out to rigorously evaluate the model's generalization capabilities on unseen hydraulic conditions. The remaining events are then split on a graph (time-step) basis for training and validation.
5.  **Persistence**: All generated graphs are serialized and saved as `.pkl` files to enable rapid loading for subsequent training and evaluation runs, avoiding redundant preprocessing.

## Model Architecture
The GNN-SWS model (implemented in `GNN_models.py`) adopts an **Encode-Process-Decode** architecture, specifically designed for joint prediction of both node and edge states in stormwater networks.

<img width="3191" height="2528" alt="Fig4_V5" src="https://github.com/user-attachments/assets/a69f9e8e-d524-4a0f-9c76-e65f93b6b848" />

![Model Structure Diagram](https://github.com/YourGitHubUsername/YourRepoName/blob/main/assets/model_structure.png)
*Figure 2: Conceptual overview of the GNN-SWS model architecture.*


* **Encoder (`Encoder` class)**: This module projects the raw input node and edge features into a higher-dimensional latent space.
    * It features **separate processing branches for different node types**: "type_a" nodes (those with subcatchment contributions) use their full feature set, while "type_b" nodes (those without) have specific, irrelevant static features dynamically removed before encoding. This allows the model to effectively learn distinct representations for different hydrological roles within the network.
* **Processor (`Processor` class)**: This is the core of the GNN, consisting of multiple stacked **Interaction Network** layers (`InteractionNetwork` class).
    * Each `InteractionNetwork` layer performs message passing, where information is exchanged between connected nodes and edges.
    * Messages for nodes are computed based on the concatenated latent representations of the source node, target node, and the connecting edge. These messages are then aggregated (summed) at each node.
    * Node features are updated by combining their original latent state with the aggregated messages.
    * **Crucially, edge features are also updated within each message-passing step**: The new edge latent representations are derived from the *updated* latent states of their connected nodes and the original edge features. This enables the model to jointly refine node and edge states iteratively.
    * Residual connections are applied at each layer of the `InteractionNetwork` to aid training stability and convergence.
* **Decoders (`Decoder` and `EdgeDecoder` classes)**: These modules map the refined latent representations from the Processor back to the desired physical outputs.
    * The `Decoder` predicts node-level hydraulic states (e.g., hydraulic head and inflow).
    * The `EdgeDecoder` predicts edge-level hydraulic states (e.g., flow and depth in conduits).
* **Output Residual Connections**: The overall `EncodeProcessDecodeDual` model incorporates residual connections at the output layer. The GNN predicts the *change* in hydraulic states, which is then added to the last known state from the input features to produce the final prediction. This technique significantly improves numerical stability and physical consistency in time-series predictions.

## Training Process
The training of the GNN-SWS model (orchestrated by `GNN_train_final.py` and utilizing `GNN_utils_final.py`) employs a sophisticated **pushforward training strategy** designed for robust long-term hydraulic prediction.

The training objective combines two distinct loss components:

1.  **Real Loss Branch**:
    * In this branch, for each time step in the multi-step prediction horizon (`max_steps_ahead`), the model makes a prediction.
    * For subsequent predictions within the same horizon, the input features are updated by "shifting" the time window and appending the **ground truth** values for the actual next state from the SWMM simulation. This ensures the model learns to predict accurately when provided with true historical context.
2.  **Stability Loss Branch**:
    * This branch is critical for enhancing the model's stability and performance during long-term, recursive rollouts (where ground truth is unavailable).
    * For the *first* prediction step in this branch, the input features are prepared as if the model were making a real-time forecast (i.e., using the model's *own* previously predicted state for the current time `t` to inform the prediction for `t+1`).
    * For all subsequent steps within the `max_steps_ahead` horizon, the model recursively uses its **own predictions** from the previous step as input for the next, mimicking a true rollout scenario.
    * The loss from this branch explicitly penalizes deviations when the model is forced to rely on its own errors, thereby improving its long-term stability.

**Loss Function (`swmm_gnn_loss` in `GNN_utils_final.py`):**
The total loss is a weighted sum of the Real Loss and the Stability Loss. The `swmm_gnn_loss` function provides flexibility, allowing configuration of `mse`, `mae`, or a `hybrid` loss (combining relative MSE and MAE).
Key physics-guided components of the loss function include:
* **Negative Value Penalties**: A penalty term (`F.relu(-prediction)`) is added to discourage the model from predicting non-physical negative values for hydraulic head and depth, promoting physical consistency.
* **Difference Loss (`diff_loss`)**: This term penalizes discrepancies in the *differences* of predicted hydraulic states (head, inflow) between connected nodes. This encourages the model to learn accurate hydraulic gradients and flow patterns across conduits, further enforcing physical laws.

**Optimization:**
The model is trained using the Adam optimizer with a configured learning rate and weight decay, and a `StepLR` scheduler is used to adjust the learning rate during training. Wandb (Weights & Biases) is integrated for comprehensive experiment tracking and visualization of training and validation metrics. Checkpointing mechanisms allow saving and loading model states for resuming training or deploying the best-performing model.

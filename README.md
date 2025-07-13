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

The data preparation process involves several key steps orchestrated by `data_processing.py`:
1.  **SWMM Data Loading**: Raw SWMM simulation outputs (e.g., `junctions_depth.txt`, `conduits_flow.txt`) and static network configuration files (`subcatchments.txt`, `junctions.txt`, `conduits.txt`, `outfalls.txt`) are loaded.
2.  **Feature Engineering**:
    * The `conduits.txt` file is augmented with an 'Area' column derived from 'MaxDepth' (assuming a circular cross-section).
    * **Node-Level Features**: Static properties of junctions and outfalls are combined with aggregated features from connected subcatchments (e.g., total area, average imperviousness). This process distinguishes between "type_a" nodes (those with contributing subcatchments) and "type_b" nodes (those without), allowing for differentiated feature handling in the GNN encoder.
    * **Edge-Level Features**: Static properties of conduits (e.g., Length, Roughness, MaxDepth, Area) are used. Additionally, physics-guided features are computed, such as relative coordinates (`dx`, `dy`) between connected nodes, and differences in hydraulic head (`diff_head`) and inflow (`diff_inflow`) between an edge's 'From_Node' and 'To_Node'.
    * **Dynamic Feature Windows**: Time-series data (e.g., rainfall, accumulated rainfall, depth, inflow, flow) is organized into sliding windows of `n_time_step` past observations for both nodes and edges.
    * **Future Exogenous Inputs**: Separate "future rainfall" features are extracted for both nodes and edges, spanning `max_steps_ahead` into the future. These are critical for the pushforward training strategy as they provide necessary exogenous information when predicting beyond the initial time step.
3.  **Graph Construction**: For each time step within an event, a `torch_geometric.data.Data` object is constructed. Each graph contains:
    * `x`: Node features (concatenation of constant and dynamic features).
    * `edge_index`: Graph connectivity (adjacency list).
    * `edge_attr`: Edge features (concatenation of constant and dynamic features).
    * `y` and `y_edges`: Multi-step-ahead ground truth targets for nodes (depth, inflow) and edges (flow, depth), respectively.
    * `future_rainfall` and `future_edge_rainfall`: The future exogenous rainfall inputs for nodes and edges, aligned with the prediction horizon.
    * `node_type`: A categorical label indicating whether a node is 'type_a' (with subcatchment contribution) or 'type_b' (without).
4.  **Data Splitting**: The generated graphs are split into training, validation, and test sets. A key aspect is the **event-based split for the test set**, where entire rainfall events are held out to rigorously evaluate the model's generalization capabilities on unseen hydraulic conditions. The remaining events are then split on a graph (time-step) basis for training and validation.
5.  **Persistence**: All generated graphs are serialized and saved as `.pkl` files to enable rapid loading for subsequent training and evaluation runs, avoiding redundant preprocessing.

## Model Architecture
The GNN-SWS model (implemented in `GNN_models.py`) adopts an **Encode-Process-Decode** architecture, specifically designed for joint prediction of both node and edge states in stormwater networks.

![Model Structure Diagram](https://github.com/user-attachments/assets/a69f9e8e-d524-4a0f-9c76-e65f93b6b848)
**Conceptual overview of the GNN-SWS model architecture.**
<be>

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

The training of the GNN-SWS model (orchestrated by `GNN_train.py` and utilizing `GNN_utils.py`) employs a sophisticated **pushforward training strategy** designed for robust long-term hydraulic prediction. This strategy addresses the challenge of accumulating errors in recursive multi-step forecasts by combining two distinct learning objectives:

* **Ground-Truth Guided Learning**: The model is trained to accurately predict future hydraulic states based on sequences where input features are updated with **ground truth** data at each step. This ensures the model learns the correct short-term dynamics when provided with accurate historical context.
* **Recursive Stability Learning**: Simultaneously, the model is trained on scenarios where it must recursively use its **own predictions** from previous steps as input for subsequent predictions within the forecast horizon. This explicitly penalizes error propagation and enhances the model's ability to maintain stability and accuracy during extended rollouts where ground truth is unavailable.

**Loss Function:**
The detailed definition and components of the loss function can be found in `GNN_utils.py`.

**Optimization:**
The model is trained using the Adam optimizer with a configured learning rate and weight decay, and a `StepLR` scheduler is used to adjust the learning rate during training. Wandb (Weights & Biases) is integrated for comprehensive experiment tracking and visualization of training and validation metrics. Checkpointing mechanisms allow saving and loading model states for resuming training or deploying the best-performing model.

## Testing Process

The GNN-SWS model's performance on unseen events is evaluated using a **multi-step rollout prediction** approach, implemented in `GNN_test.py` and powered by `rollout_prediction.py`. This process simulates real-world forecasting where the model recursively uses its own predictions to generate long-term forecasts.

![Rollout Prediction Process](https://github.com/user-attachments/assets/ffb8cb90-fe78-4c39-babb-f2870e3dd13a)
**Illustration of the Rollout Prediction Process.** 


The testing workflow generally involves:

1.  **Loading Initial State**: For each held-out test event, an initial graph representing the system state at `t=0` is loaded, along with the full time series of future exogenous inputs (e.g., rainfall).
2.  **Recursive Forecasting**: The trained model generates predictions one time step at a time. For each subsequent step, the model's own output from the previous step is fed back as input, along with the corresponding future exogenous data, to predict the next state. This continues for the entire duration of the event.
3.  **Result Processing**: The raw model outputs are converted into a readable format (e.g., pandas DataFrames) and denormalized to their original physical scales if applicable.
4.  **Saving Results**: The final, denormalized rollout predictions are saved for further analysis and comparison with ground truth data.

This methodology provides a rigorous assessment of the model's ability to provide stable and accurate long-term forecasts in dynamic stormwater environments.

## Interactive Dashboard

To facilitate the exploration and analysis of the GNN-SWS model's predictions, an interactive web-based dashboard has been developed using Plotly Dash (`Dashboard.py`). This dashboard provides a comprehensive platform for visualizing model performance against SWMM simulations across various storm events and network elements.

![Overview of the Interactive SWMM-GNN Dashboard](https://github.com/user-attachments/assets/9f57d92e-2d7b-4179-bb71-be5435c9791f)
**Overview of the Interactive SWMM-GNN Dashboard.** 

The dashboard offers the following key functionalities:

* **Dynamic Data Selection**: Users can select the dataset split (Training, Validation, Testing), specific rainfall events, and hydraulic domains (Junctions or Conduits) to analyze.
* **Network-Wide Error Visualization**: The main network graph spatially visualizes the model's performance. Nodes or conduits are colored according to selected error metrics (e.g., RMSE, NRMSE, Relative MAE, Correlation Coefficient, Nash–Sutcliffe Efficiency (NSE)), providing an immediate overview of areas with higher or lower model accuracy.
* **Individual Element Time Series**: Upon clicking a specific junction or conduit in the network graph, a detailed time series plot is displayed, showing a direct comparison between the GNN-SWS predictions and the SWMM simulated "ground truth" for the selected hydraulic variable (e.g., depth, inflow, flow). For junctions, corresponding rainfall inputs are also visualized.
* **Real-Time Error Metrics**: For the selected network element, a panel dynamically displays precise numerical values for all computed error metrics, offering quantitative insights into local model accuracy.
* **Temporal Error Animation**: An interactive plot visualizes the evolution of average error (and its standard deviation) across the entire active network throughout a storm event. A time slider and animation controls allow users to observe how model performance varies dynamically over time.
* **Comprehensive Help Guide**: A built-in modal provides detailed information about the research framework, a guide on how to use the dashboard's features, and clear explanations of all the error metrics used, including their mathematical formulas.

The dashboard is designed to be a powerful tool for researchers and practitioners to intuitively understand the GNN-SWS model's behavior, identify areas of strong performance or potential limitations, and gain deeper insights into urban stormwater system dynamics.

To run the dashboard, execute `Dashboard.py` directly after training the model and generating predictions.
## Getting Started

This section guides you through setting up the environment and running the GNN-SWS model, from data preparation to training, testing, and interactive visualization.

### Prerequisites

* **Python**: Version 3.9 or higher is recommended.
* **SWMM**: EPA Storm Water Management Model (SWMM) is required to generate the initial simulation data. Ensure you have a functional SWMM installation available on your system, as the data preparation step involves running SWMM simulations.
* **NVIDIA GPU with CUDA**: For training the model efficiently, an NVIDIA GPU with CUDA Toolkit installed is highly recommended. The provided `requirements.txt` includes PyTorch and PyTorch Geometric versions compiled with CUDA support.

### Environment Setup

1.  **Clone the Repository**:
    ```bash
    git clone [https://github.com/YourUsername/Urban-Stormwater-GNN-Surrogate.git](https://github.com/YourUsername/Urban-Stormwater-GNN-Surrogate.git)
    cd Urban-Stormwater-GNN-Surrogate
    ```
2.  **Install Dependencies**:
    It is highly recommended to use a virtual environment. Once inside your project directory and with your virtual environment active (or managing dependencies globally), install all required Python packages:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: If you plan to use a GPU, ensure your CUDA Toolkit version is compatible with the `torch` version specified in `requirements.txt`. If you encounter issues or do not have a compatible GPU, consider installing the CPU-only version of PyTorch and PyTorch Geometric packages by manually adjusting `requirements.txt` before installation.*

### Configuration

The `config.yml` file is central to customizing the entire workflow. It defines paths, feature sets, model architecture, training hyperparameters, and more.

* **Review `config.yml`**: Before running any scripts, open `config.yml` and review the paths under the `data` and `checkpoint` sections. These are set using **relative paths** to the repository root but might need adjustment based on your local data storage.
* **Wandb Integration**: The `wandb` section allows you to configure Weights & Biases for experiment tracking. Update `project` and `entity` as needed, or comment out the `wandb.init()` call in `GNN_train_final.py` if you don't wish to use it.

---

## Citation

If you use this code or the concepts from this paper in your research, please cite our work:

**Zandsalimi, Z.**, Taghizadeh, M., Lee Lynn, S., Goodall, J.L., Shafiee-J., M., Alemazkoor, N., 2025. **End-to-End Graph Neural Networks for Rainfall-Driven Real-Time Hydraulic Prediction in Stormwater Systems.** *Water Research*. (Under Review).

```bibtex
@article{zandsalimi2025GNN-SWS,
  title={{End-to-End Graph Neural Networks for Rainfall-Driven Real-Time Hydraulic Prediction in Stormwater Systems}},
  author={Zandsalimi, Zanko and Taghizadeh, Mehdi and Lee Lynn, S. and Goodall, Jonathan L. and Shafiee-Jood, Majid and Alemazkoor, Negin},
  journal={Water Research},
  year={2025},
  note={Under Review},
  publisher={Elsevier}
}

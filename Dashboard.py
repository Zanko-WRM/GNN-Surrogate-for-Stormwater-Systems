import dash
from dash import dcc, html
import dash.dependencies as dd
import plotly.graph_objects as go
import networkx as nx
import os
import pickle
import yaml
import torch
import numpy as np
import pandas as pd
import dash_bootstrap_components as dbc

from data_processing_conduits import (
    load_constant_features, load_dynamic_data, calculate_relative_coordinates,
    convert_conduits_to_graph, denormalize_dynamic_data, transform_subcatchment_dynamic_data
)

# ===========================================================
# 1) Load Configuration
# ===========================================================
CONFIG_PATH = "config.yml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

data_config = config["data"]
dynamic_features = config["dynamic_features"]
constant_features = config["constant_features"]
computed_features = config["computed_features"]
n_time_step = config["model"]["n_time_step"]  # Number of warm-up steps
max_steps_ahead = config["model"]["max_steps_ahead"]

data_dirs = {
    "Training": data_config["train_graph_dir"],
    "Validation": data_config["val_graph_dir"],
    "Testing": data_config["test_graph_dir"]
}

# ===========================================================
# 2) Load GNN Predictions (Raw)
# ===========================================================
gnn_predictions_path = os.path.join(data_config["gnn_predictions_dir"], "gnn_rollout_predictions.pkl")
if os.path.exists(gnn_predictions_path):
    with open(gnn_predictions_path, "rb") as f:
        gnn_predictions = pickle.load(f)
    print("✅ GNN Predictions Loaded Successfully!")
else:
    gnn_predictions = {}
    print("⚠️ No GNN Predictions Found.")

# Gather event names from precomputed graph files
available_events = {
    key: [
        file.replace("_graphs.pkl", "")
        for file in os.listdir(path)
        if file.endswith("_graphs.pkl")
    ]
    for key, path in data_dirs.items()
}

# ===========================================================
# 3) Load Observed Data + Node Mapping
# ===========================================================
static_data, original_subcatchment_areas = load_constant_features(
    data_config["constant_data_dir"],
    config=constant_features,
    normalize=data_config["normalize_features"]
)
node_coordinates = static_data["node_coordinates"]

dynamic_data, raw_dynamic_data, outfall_dynamic_data, raw_outfall_dynamic_data, time_steps_per_event, global_min_max, flow_direction_data = load_dynamic_data(
    data_config["dynamic_data_dir"],
    config=dynamic_features,
    outfall_names=static_data["outfall_names"],
    apply_normalization=data_config["normalize_features"],
    save_min_max_path=data_config["global_min_max_dir"],
    use_abs_flow=True
)

transformed_rainfall = transform_subcatchment_dynamic_data(static_data, dynamic_data, original_subcatchment_areas)
denormalized_data = denormalize_dynamic_data(dynamic_data, outfall_dynamic_data, global_min_max)

updated_conduits_df, edge_to_nodes = calculate_relative_coordinates(
    static_data["conduits"],
    static_data["junctions"],
    static_data["outfalls"],
    computed_config=computed_features
)

graph, edge_index, node_mapping, edge_info_list = convert_conduits_to_graph(
    updated_conduits_df, node_coordinates
)

# Create inverse mappings for nodes and edges
inv_node_mapping = {v: k for k, v in node_mapping.items()}
inv_edge_mapping = {
    i: edge_info_list[i].get("conduit_name", f"Edge_{i}")
    for i in range(len(edge_info_list))
}


# ===========================================================
# 4) Post-process GNN Predictions to Rename Row Indices
# ===========================================================
def rename_gnn_predictions_nodes(gnn_preds, inv_map):
    """Rename row indices in GNN predictions from numeric index to node names."""
    for event, event_data in gnn_preds.items():
        if "junctions" not in event_data:
            continue
        for var, df in event_data["junctions"].items():
            new_index = [inv_map.get(idx, f"Unknown_{idx}") for idx in df.index]
            df.index = new_index
    return gnn_preds


def rename_gnn_predictions_edges(gnn_preds, inv_map):
    """Rename row indices in GNN predictions from numeric index to conduit names."""
    for event, event_data in gnn_preds.items():
        if "conduits" not in event_data:
            continue
        for var, df in event_data["conduits"].items():
            new_index = [inv_map.get(idx, f"Edge_{idx}") for idx in df.index]
            df.index = new_index
    return gnn_preds


gnn_predictions = rename_gnn_predictions_nodes(gnn_predictions, inv_node_mapping)
gnn_predictions = rename_gnn_predictions_edges(gnn_predictions, inv_edge_mapping)


# ===========================================================
# 4.1) Define Error Metric Functions
# ===========================================================
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def nrmse(y_true, y_pred):
    eps = 1e-6
    mean_y = np.mean(np.abs(y_true))
    return rmse(y_true, y_pred) / (mean_y + eps)


def relative_mae(y_true, y_pred):
    eps = 1e-6
    return np.mean(np.abs(y_true - y_pred)) / (np.mean(np.abs(y_true)) + eps)


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 1e-6 else np.nan


def r_score(y_true, y_pred):
    if len(y_true) < 2: return np.nan
    return np.corrcoef(y_true, y_pred)[0, 1]


def nse(y_true, y_pred):
    return 1 - np.sum((y_true - y_pred) ** 2) / (np.sum((y_true - np.mean(y_true)) ** 2) + 1e-6)


# ===========================================================
# 4.2) Precompute Error Metrics for Each (Event, Domain, Variable)
# ===========================================================
error_metrics = {}


def compute_all_metrics(event, domain, variable):
    """
    Compute error metrics on denormalized data for the specified event/domain/variable
    on a per-node/edge basis (aggregated over time).
    """
    if (event not in denormalized_data or
            domain not in denormalized_data[event] or
            variable not in denormalized_data[event][domain]):
        return {}

    observed_df = denormalized_data[event][domain][variable]

    if (event not in gnn_predictions or
            domain not in gnn_predictions[event] or
            variable not in gnn_predictions[event][domain]):
        return {}

    pred_df = gnn_predictions[event][domain][variable]
    observed_points = set(observed_df.columns)

    # MODIFIED: Ensure the number of time steps in observed data and GNN predictions are aligned.
    # Remove the warm-up period (n_time_step) from the observed data before alignment.
    if set(pred_df.index).intersection(observed_points):
        common_points = list(observed_points.intersection(pred_df.index))
        obs_array = observed_df[common_points].values[n_time_step:]  # Excludes warm-up steps
        pred_array = pred_df.loc[common_points].values.T
    else:
        common_points = list(observed_points.intersection(pred_df.columns))
        obs_array = observed_df[common_points].values[n_time_step:]  # Excludes warm-up steps
        pred_array = pred_df[common_points].values

    if len(common_points) == 0:
        return {}

    T_obs, N_obs = obs_array.shape
    T_pred, N_pred = pred_array.shape
    T_min = min(T_obs, T_pred)
    obs_array = obs_array[:T_min, :]
    pred_array = pred_array[:T_min, :]

    rmse_vals = {}
    nrmse_vals = {}
    relmae_vals = {}
    r_vals = {}
    nse_vals = {}

    # Define thresholds for NSE calculation (mean absolute observed value)
    # These are heuristic values, you might want to tune them based on your data.
    nse_thresholds = {
        "depth": 0.01,  # e.g., 1 cm
        "inflow": 0.001,  # e.g., 1 L/s
        "flow": 0.001  # e.g., 1 L/s
    }
    current_variable_threshold = nse_thresholds.get(variable, 0.0)  # Default to 0 if not specified

    for i, pt in enumerate(common_points):
        y_true = obs_array[:, i]
        y_pred = pred_array[:, i]

        # Calculate mean absolute observed value for thresholding
        mean_abs_y_true = np.mean(np.abs(y_true))

        # Only calculate NSE if the mean absolute observed value is above the threshold
        if mean_abs_y_true >= current_variable_threshold:
            rmse_vals[pt] = rmse(y_true, y_pred)
            nrmse_vals[pt] = nrmse(y_true, y_pred)
            relmae_vals[pt] = relative_mae(y_true, y_pred)
            r_vals[pt] = r_score(y_true, y_pred)
            nse_vals[pt] = nse(y_true, y_pred)
        else:
            # Assign NaN if threshold is not met, so it can be handled for coloring (e.g., gray)
            rmse_vals[pt] = np.nan
            nrmse_vals[pt] = np.nan
            relmae_vals[pt] = np.nan
            r_vals[pt] = np.nan
            nse_vals[pt] = np.nan

    return {
        "rmse": pd.Series(rmse_vals),
        "nrmse": pd.Series(nrmse_vals),
        "relmae": pd.Series(relmae_vals),
        "r": pd.Series(r_vals),
        "nse": pd.Series(nse_vals)
    }


# Build error_metrics for each event/domain/variable
for dataset_type, events_list in available_events.items():
    for event_name in events_list:
        if event_name not in denormalized_data or event_name not in gnn_predictions:
            continue
        if event_name not in error_metrics:
            error_metrics[event_name] = {"junctions": {}, "conduits": {}}

        for domain_name in ["junctions", "conduits"]:
            if domain_name not in denormalized_data[event_name]:
                continue
            if domain_name not in gnn_predictions[event_name]:
                continue

            var_list = list(denormalized_data[event_name][domain_name].keys())
            for var_name in var_list:
                if var_name not in gnn_predictions[event_name][domain_name]:
                    continue
                metrics_dict = compute_all_metrics(event_name, domain_name, var_name)
                if len(metrics_dict) == 0:
                    continue

                if var_name not in error_metrics[event_name][domain_name]:
                    error_metrics[event_name][domain_name][var_name] = {}
                error_metrics[event_name][domain_name][var_name]["rmse"] = metrics_dict["rmse"]
                error_metrics[event_name][domain_name][var_name]["nrmse"] = metrics_dict["nrmse"]
                error_metrics[event_name][domain_name][var_name]["relmae"] = metrics_dict["relmae"]
                error_metrics[event_name][domain_name][var_name]["r"] = metrics_dict["r"]
                error_metrics[event_name][domain_name][var_name]["nse"] = metrics_dict["nse"]  # Add NSE


# ===========================================================
# 5) Convert PyG Graph to NetworkX (For the Dashboard)
# ===========================================================
def pyg_to_networkx(pyg_graph, node_mapping, edge_info_list):
    """Convert PyG graph to NetworkX DiGraph with node positions."""
    G = nx.DiGraph()
    if not hasattr(pyg_graph, "pos") or pyg_graph.pos is None:
        print("❌ ERROR: `pos` attribute is missing in PyG graph!")
        return G

    pos_array = pyg_graph.pos.cpu().numpy()
    index_to_name = {v: k for k, v in node_mapping.items()}

    for i in range(pyg_graph.num_nodes):
        original_name = index_to_name.get(i, f"Node_{i}")
        G.add_node(original_name, pos=tuple(pos_array[i]))

    edge_list = pyg_graph.edge_index.cpu().numpy().T
    for idx, edge in enumerate(edge_list):
        from_node = index_to_name.get(edge[0], f"Node_{edge[0]}")
        to_node = index_to_name.get(edge[1], f"Node_{edge[1]}")
        edge_name = edge_info_list[idx].get("conduit_name", f"Edge_{idx}")
        G.add_edge(from_node, to_node, edge_name=edge_name)
    return G


graph_nx = pyg_to_networkx(graph, node_mapping, edge_info_list)
node_positions = nx.get_node_attributes(graph_nx, "pos")

# Precompute edge marker positions
edge_marker_x, edge_marker_y, edge_names = [], [], []
for idx, edge in enumerate(graph_nx.edges(data=True)):
    from_node, to_node = edge[0], edge[1]
    pos_from = np.array(node_positions[from_node])
    pos_to = np.array(node_positions[to_node])
    midpoint = (pos_from + pos_to) / 2
    edge_marker_x.append(midpoint[0])
    edge_marker_y.append(midpoint[1])
    edge_name = edge[2].get("edge_name", f"Edge_{idx}")
    edge_names.append(edge_name)

# # ===========================================================
# # 6) Dash App Layout
# # ===========================================================
### HELP MODAL ### Initialize app with Bootstrap stylesheet
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    external_scripts=[ "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"]
)
#"https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML"

# Define a variable for header height to use in calc()
HEADER_HEIGHT_PX = "120px"  # Approximate height for H1 + top controls + margins
APP_MARGIN_PX = "40px"  # Total vertical margin for the whole app container (20px top + 20px bottom)

app.layout = html.Div([
    html.H1("SWMM-GNN Dashboard", style={"textAlign": "center", "fontFamily": "Times New Roman"}),

    # ---------- (A) Top Controls ----------
    html.Div([
        html.Label("Select Dataset:", style={"marginRight": "10px", "fontFamily": "Times New Roman"}),
        dcc.Dropdown(
            id="dataset-dropdown",
            options=[{"label": key, "value": key} for key in available_events.keys()],
            value="Training",
            style={"width": "180px", "marginRight": "40px", "fontFamily": "Times New Roman"}
        ),

        html.Label("Select Event:", style={"marginRight": "10px", "fontFamily": "Times New Roman"}),
        dcc.Dropdown(
            id="event-dropdown",
            style={"width": "180px", "marginRight": "40px", "fontFamily": "Times New Roman"}
        ),

        html.Label("Select Domain:", style={"marginRight": "10px", "fontFamily": "Times New Roman"}),
        dcc.Dropdown(
            id="domain-dropdown",
            options=[
                {"label": "Junctions", "value": "junctions"},
                {"label": "Conduits", "value": "conduits"}
            ],
            value="junctions",
            style={"width": "180px", "marginRight": "40px", "fontFamily": "Times New Roman"}
        ),

        html.Label("Select Variable:", style={"marginRight": "10px", "fontFamily": "Times New Roman"}),
        dcc.Dropdown(
            id="variable-dropdown",
            options=[],
            value=None,
            style={"width": "180px", "marginRight": "40px", "fontFamily": "Times New Roman"}
        ),

        html.Label("Select Error Index:", style={"marginRight": "10px", "fontFamily": "Times New Roman"}),
        dcc.Dropdown(
            id="error-index-dropdown",
            options=[
                {"label": "None", "value": "none"},
                {"label": "RMSE", "value": "rmse"},
                {"label": "NRMSE (mean)", "value": "nrmse"},
                {"label": "Relative MAE", "value": "relmae"},
                {"label": "Correlation Coefficient (r)", "value": "r"},
                {"label": "Nash–Sutcliffe Efficiency (NSE)", "value": "nse"}
            ],
            value="none",
            style={"width": "230px", "fontFamily": "Times New Roman", "marginRight": "20px"}
        ),

        ### HELP MODAL ### Add the help button
        dbc.Button(
            "Help",
            id="help-button",
            className="ms-auto",  # "Margin-Start: Auto" pushes it to the right
            n_clicks=0,
            color="info",
        ),

    ], style={"display": "flex", "alignItems": "center", "marginBottom": "20px"}),

    ### HELP MODAL
    dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Dashboard Help & Information")),
            dbc.ModalBody(
                dbc.Tabs(
                    [
                        ### TAB 1: ABOUT THE FRAMEWORK ###
                        dbc.Tab(
                            [
                                html.H4("Paper Title:", className="mt-3"),
                                html.P(
                                    "End-to-End Graph Neural Networks for Rainfall-Driven Real-Time Hydraulic Prediction in Stormwater Systems"),

                                html.H4("Authors:"),
                                html.P(
                                    "Zanko Zandsalimi, Mehdi Taghizadeh, Savannah Lee Lynn, Jonathan L. Goodall, Majid Shafiee-Jood, Negin Alemazkoor"),

                                html.H4("Affiliation:"),
                                html.P("Department of Civil and Environmental Engineering, University of Virginia"),

                                html.H4("Abstract:"),
                                dcc.Markdown("""
                                    Urban stormwater systems (SWS) play a critical role in protecting communities from pluvial flooding, ensuring public safety, and supporting resilient infrastructure planning. As climate variability intensifies and urbanization accelerates, there is a growing need for timely and accurate hydraulic predictions to support real-time control and flood mitigation strategies. While physics-based models such as SWMM provide detailed simulations of rainfall-runoff and flow routing processes, their computational demands often limit their feasibility for real-time applications. Surrogate models based on machine learning offer faster alternatives, but most rely on fully connected or grid-based architectures that struggle to capture the irregular spatial structure of drainage networks, often requiring precomputed runoff inputs and focusing only on node-level predictions. To address these limitations, we present GNN-SWS, a novel end-to-end graph neural network (GNN) surrogate model that emulates rainfall-driven hydraulic behavior across stormwater systems. The model predicts hydraulic states at both junctions and conduits directly from rainfall inputs, capturing the coupled dynamics of runoff generation and flow routing. It incorporates a spatiotemporal encoder–processor–decoder architecture with tailored message passing, autoregressive forecasting, and physics-guided constraints to improve predictive accuracy and physical consistency. Additionally, a training strategy based on the pushforward trick enhances model stability over extended prediction horizons. Applied to a real-world urban watershed, GNN-SWS demonstrates strong potential as a fast, scalable, and data-efficient alternative to traditional solvers. This framework supports key applications in urban flood risk assessment, real-time stormwater control, and the optimization of resilient infrastructure systems.
                                """),

                                html.Hr(),
                                html.H4("Graphical Abstract:"),
                                # This now has the correct path with a leading slash
                                html.Img(src="/assets/graphical_abstract.png",
                                         style={'maxWidth': '100%', 'border': '1px solid #ddd', 'borderRadius': '4px',
                                                'padding': '5px'})
                            ],
                            label="About the Framework",
                        ),

                        ### TAB 2: DASHBOARD GUIDE ###
                        dbc.Tab(
                            dcc.Markdown("""
                                The GNN-SWS dashboard is an interactive tool for visualizing and evaluating the performance of the Graph Neural Network surrogate model against the physics-based SWMM. It allows for a deep dive into model predictions across different storm events and at various locations within the stormwater system.

                                ---

                                #### 1. Selecting and Filtering Data
                                The dropdown menus at the top control all the data visualized on the dashboard.

                                * **Dataset**: Choose the data split for model evaluation (**Training**, **Validation**, or **Testing**).
                                * **Event**: Select a specific historical rainfall event to analyze.
                                * **Domain**: Choose which part of the stormwater system to inspect (`Junctions` or `Conduits`).
                                * **Variable**: Select the specific hydraulic variable to view (`Depth`, `Inflow`, or `Flow`).

                                ---

                                #### 2. Visualizing Network-Wide Errors
                                The main graph on the left provides a spatial overview of model performance. When an **Error Index** is selected, each element is colored based on its performance for the entire event.

                                * The color scheme indicates model performance, where **blue suggests low error** (good) and **red suggests high error** (poor).
                                * **NSE** uses a separate categorical legend for its fit quality.
                                * If the index is set to **None**, the graph highlights all elements in the selected domain.

                                ---

                                #### 3. Inspecting a Single Element's Time Series
                                The top-right plot provides a detailed temporal analysis for an element selected from the main network graph.

                                * **SWMM Model (Solid Black Line)**: This is the "ground truth" from the physics-based simulation.
                                * **GNN-SWS (Dashed Red Line)**: This is the surrogate model's prediction.
                                * **Rainfall (Gray Bars)**: For junctions, this secondary axis shows the corresponding rainfall intensity.
                                * **Metrics Display**: The middle panel shows the precise error values for the selected element.

                                ---

                                #### 4. Animating Network-Averaged Error
                                The bottom-right plot visualizes how the model's average performance evolves throughout the storm event. The slider and animation button allow for exploring these error dynamics over time.

                                * **Blue Line**: Represents the average error at each time step across all active network elements.
                                * **Shaded Area**: Represents the standard deviation of the error. A wider band indicates more performance variability across the system.
                            """, className="mt-3"),
                            label="Dashboard Guide",
                        ),

                        ### TAB 3: ERROR METRICS ###
                        dbc.Tab(
                            dcc.Markdown(r'''
                            #### Root Mean Squared Error (RMSE)
                            Measures the average magnitude of the errors. It's in the same units as the predicted variable. Lower values are better.

                            $$
                            RMSE = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}
                            $$

                            ---

                            #### Normalized RMSE (NRMSE)
                            Normalizes the RMSE by the mean of the observed values, making it easier to compare across variables with different scales. Lower values are better.

                            $$
                            NRMSE = \frac{RMSE}{\bar{y}}
                            $$

                            ---
                            
                            ### Relative MAE
                            Normalizes the Mean Absolute Error (MAE) relative to the magnitude of the observed values, making it a scale-independent percentage error. Lower values indicate a better fit.

                            $$
                            \text{Relative MAE} = \frac{\sum_{i=1}^{N} |y_i - \hat{y}_i|}{\sum_{i=1}^{N} |y_i|}
                            $$
                        
                            ---

                            #### Nash–Sutcliffe Efficiency (NSE)
                            Determines the relative magnitude of the residual variance compared to the measured data variance. 

                            - NSE = 1 corresponds to a perfect match  
                            - NSE = 0 means the model is only as good as the mean of the observations  
                            - NSE < 0 indicates that the model performs worse than the mean

                            $$
                            NSE = 1 - \frac{\sum_{i=1}^{N} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{N} (y_i - \bar{y})^2}
                            $$

                            ---

                            #### Correlation Coefficient (r)
                            Measures the linear relationship between observed and predicted values. A value of +1 is a perfect positive linear relationship.

                            $$
                            r = \frac{\sum_{i=1}^{N} (y_i - \bar{y})(\hat{y}_i - \bar{\hat{y}})}{
                            \sqrt{\sum_{i=1}^{N} (y_i - \bar{y})^2} \cdot \sqrt{\sum_{i=1}^{N} (\hat{y}_i - \bar{\hat{y}})^2}}
                            $$

                            ---

                            **Where:**

                            - $N$: Total number of time steps  
                            - $y_i$: Observed value at time $i$
                            - $\hat{y}_i$: Predicted value at time $i$
                            - $\bar{y}$: Mean of the observed values
                            - $\bar{\hat{y}}$: Mean of the predicted values
                            ''', className="mt-3", mathjax=True),
                            label="Error Metrics Explained",
                        ),
                    ]
                )
            ),
            dbc.ModalFooter(
                dbc.Button("Close", id="close-help-button", className="ms-auto", n_clicks=0)
            ),
        ],
        id="help-modal",
        is_open=False,
        size="xl",
        centered=True,
    ),

    # ---------- (B) Main Row: Left = Network, Middle = Metrics, Right = Timeseries + Animation ----------
    html.Div([
        # Left Column: Network Graph
        dcc.Graph(
            id="network-graph",
            style={"width": "40%", "height": "100%", "marginRight": "10px"},  # Reduced width to 40%
            figure=go.Figure().update_layout(
                autosize=False,
                height=None,  # Let CSS style control height
                margin=dict(l=20, r=20, t=20, b=50),  # Increased bottom margin for X-axis label
                showlegend=False,
                font=dict(family="Times New Roman"),
                xaxis=dict(title="X Coordinate", tickfont=dict(family="Times New Roman")),  # Explicit X-axis title
                yaxis=dict(title="Y Coordinate", tickfont=dict(family="Times New Roman"))
            )
        ),

        # Middle Column: Error Metrics
        html.Div(
            id="error-metrics-div",
            style={
                "width": "15%",  # Fixed width at 15%
                "height": "100%",  # Fill parent height
                "padding": "5px",
                "fontWeight": "bold",
                "color": "#333",
                "fontFamily": "Times New Roman",
                "flexShrink": 0,  # Prevent it from shrinking
                "marginRight": "10px",  # Added margin to the right
                "border": "1px solid #eee"  # Optional: for visual separation
            }
        ),

        # Rightmost Column: Timeseries Plot + Animation Graph + Controls
        html.Div([
            dcc.Graph(
                id="timeseries-plot",
                style={"width": "100%", "flexGrow": 2, "minHeight": "150px"},  # FlexGrow 2 (reduced share), minHeight
                figure=go.Figure().update_layout(
                    height=None,  # Let CSS style control height
                    margin=dict(l=40, r=40, t=40, b=50),  # Increased bottom margin for X-axis label
                    showlegend=True,
                    font=dict(family="Times New Roman"),
                    xaxis=dict(title="Time Step", tickfont=dict(family="Times New Roman")),  # Explicit X-axis title
                    yaxis=dict(title="Variable Value", tickfont=dict(family="Times New Roman")),
                    legend=dict(
                        orientation="h",  # Changed to horizontal to save vertical space
                        yanchor="bottom",
                        y=1.08,
                        xanchor="center",
                        x=0.5,
                        font=dict(family="Times New Roman")
                    )
                )
            ),
            # Moved animation graph, slider, and button below the timeseries plot
            dcc.Graph(
                id="network-error-animation-graph",
                style={"width": "100%", "flexGrow": 3, "minHeight": "200px"},  # FlexGrow 3 (increased share), minHeight
                figure=go.Figure().update_layout(
                    height=None,  # Let CSS style control height
                    margin=dict(l=40, r=40, t=40, b=50),  # Increased bottom margin for X-axis label
                    font=dict(family="Times New Roman"),
                    xaxis_title="Time Step",  # Explicit X-axis title
                    legend=dict(
                        orientation="v",
                        yanchor="top",
                        y=1,
                        xanchor="right",
                        x=1,
                        font=dict(family="Times New Roman")
                    )
                )
            ),
            html.Div(
                dcc.Slider(
                    id="error-time-slider",
                    min=0,
                    max=100,
                    step=1,
                    value=0,
                    marks={i: str(i) for i in range(0, 101, 10)}
                ),
                style={"width": "100%", "margin": "10px 0", "flexShrink": 0}  # Smaller margin
            ),
            html.Div([  # This div acts as a wrapper for the button and interval to keep them together
                html.Button("Start Animation", id="toggle-animation", n_clicks=0),
                dcc.Interval(id="animation-interval", interval=1000, n_intervals=0, disabled=True)
            ], style={"textAlign": "center", "flexShrink": 0})  # Prevent it from shrinking
        ], style={
            "display": "flex",
            "flexDirection": "column",  # Stack elements vertically
            "width": "45%",  # Increased width for the rightmost column
            "height": "100%",  # Fill parent height
            "padding": "5px",  # Added padding
            "overflowY": "auto",  # Allow vertical scrolling for THIS column if content overflows (fallback)
            "border": "1px solid #eee"  # Optional: for visual separation
        })
    ], style={
        "display": "flex",
        "flexDirection": "row",
        "justifyContent": "space-between",
        "width": "100%",
        "height": f"calc(100vh - {HEADER_HEIGHT_PX} - {APP_MARGIN_PX})",  # Dynamic height for this main row
        "minHeight": "500px"  # A sensible minimum height to avoid collapsing on very small screens
    })
], style={"margin": "20px", "fontFamily": "Times New Roman", "height": "calc(100vh - 40px)",
          "overflow": "hidden"})  # Overall app container


# ===========================================================
# 7) Callbacks
# ===========================================================

### HELP MODAL ### Callback to open and close the help modal
@app.callback(
    dd.Output("help-modal", "is_open"),
    [dd.Input("help-button", "n_clicks"), dd.Input("close-help-button", "n_clicks")],
    [dd.State("help-modal", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    [dd.Output("event-dropdown", "options"),
     dd.Output("event-dropdown", "value")],
    [dd.Input("dataset-dropdown", "value")]
)
def update_event_dropdown(selected_dataset):
    """Update Event dropdown based on selected dataset."""
    events = available_events[selected_dataset]
    options = [{"label": evt, "value": evt} for evt in events]
    return options, (events[0] if events else None)


@app.callback(
    [dd.Output("variable-dropdown", "options"),
     dd.Output("variable-dropdown", "value")],
    [dd.Input("domain-dropdown", "value")]
)
def update_variable_dropdown(selected_domain):
    """Update Variable dropdown based on domain."""
    if selected_domain == "junctions":
        options = [
            {"label": "Depth", "value": "depth"},
            {"label": "Inflow", "value": "inflow"},
        ]
        default_value = "depth"
    else:
        options = [
            {"label": "Flow", "value": "flow"},
            {"label": "Depth", "value": "depth"},
        ]
        default_value = "flow"
    return options, default_value

# ------------------- Network Graph Update -------------------
@app.callback(
    dd.Output("network-graph", "figure"),
    [
        dd.Input("dataset-dropdown", "value"),
        dd.Input("event-dropdown", "value"),
        dd.Input("domain-dropdown", "value"),
        dd.Input("variable-dropdown", "value"),
        dd.Input("error-index-dropdown", "value"),
        dd.Input("network-graph", "clickData")
    ]
)
def update_network_graph(selected_dataset, selected_event, selected_domain,
                         selected_variable, selected_error_index, clickData):
    """
    Update the main network visualization on the left.
    Colors nodes/edges according to the selected error metric if provided.
    """
    edge_x, edge_y = [], []
    for edge in graph_nx.edges():
        x0, y0 = node_positions[edge[0]]
        x1, y1 = node_positions[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color="#888"),
        mode="lines",
        hoverinfo="none"
    )

    node_x, node_y, node_names_list = [], [], []
    for node, (nx_val, ny_val) in node_positions.items():
        node_x.append(nx_val)
        node_y.append(ny_val)
        node_names_list.append(node)

    is_node_domain = (selected_domain == "junctions")
    error_selected = (selected_error_index != "none")
    clicked_label = None
    if clickData and "points" in clickData:
        clicked_label = clickData["points"][0].get("text", None)

    # --- NSE Specific Settings for Classified Legend ---
    nse_bin_edges = [0.0, 0.25, 0.5, 0.75, 1.0]
    nse_bin_labels = ["< 0", "0–0.25", "0.25–0.5", "0.5–0.75", "0.75–1.0"]
    nse_bin_colors = ["red", "orange", "yellow", "lightgreen", "green"]
    nan_color = "lightgray"  # Color for elements where NSE is NaN

    # === NEW: Define fixed color scale ranges for consistency ===
    error_metric_ranges = {
        "rmse": {"cmin": 0},          # Min is 0, max is dynamic
        "nrmse": {"cmin": 0, "cmax": 1.0}, # Range is typically 0-1
        "relmae": {"cmin": 0, "cmax": 1.0},# Range is typically 0-1
        "r": {"cmin": -1.0, "cmax": 1.0}  # Correlation range
    }

    if is_node_domain:
        if error_selected and (selected_event in error_metrics and
                              selected_domain in error_metrics[selected_event] and
                              selected_variable in error_metrics[selected_event][selected_domain] and
                              selected_error_index in error_metrics[selected_event][selected_domain][
                                  selected_variable]):

            series_metric = error_metrics[selected_event][selected_domain][selected_variable][selected_error_index]
            metric_values = []
            for n in node_names_list:
                metric_values.append(series_metric.loc[n] if n in series_metric.index else np.nan)
            metric_values = np.array(metric_values)

            # Determine colors based on selected error index
            if selected_error_index == "nse":
                colors = []
                for val in metric_values:
                    if np.isnan(val):
                        colors.append(nan_color)
                    elif val < 0:
                        colors.append(nse_bin_colors[0])
                    else:
                        idx = np.digitize(val, bins=nse_bin_edges, right=True)
                        idx = min(idx, len(nse_bin_colors) - 1)
                        colors.append(nse_bin_colors[idx])
            else:  # For other error metrics, use continuous colorscale
                colors = metric_values

            # Get the color range for the selected metric
            range_settings = error_metric_ranges.get(selected_error_index, {})
            cmin = range_settings.get("cmin")
            cmax = range_settings.get("cmax")
            # If cmax is not fixed, calculate it dynamically
            if cmax is None:
                cmax = np.nanmax(metric_values)


            node_colors_display = ["red" if (clicked_label == n) else colors[i] for i, n in enumerate(node_names_list)]

            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode="markers",
                text=node_names_list,
                marker=dict(
                    size=7,
                    color=node_colors_display,
                    colorscale="RdBu_r" if selected_error_index != "nse" else None, # MODIFIED: Reversed scale
                    cmin=cmin if selected_error_index != "nse" else None,          # MODIFIED: Fixed min
                    cmax=cmax if selected_error_index != "nse" else None,          # MODIFIED: Potentially fixed max
                    colorbar=dict(title=selected_error_index.upper()) if selected_error_index != "nse" else None,
                    showscale=True if selected_error_index != "nse" else False
                ),
                name="Junctions"
            )
            # When error metric selected, edges are just gray
            edge_marker_trace = go.Scatter(
                x=edge_marker_x, y=edge_marker_y,
                mode="markers",
                text=edge_names,
                marker=dict(size=8, color="lightgray", symbol="square"),
                name="Conduits"
            )
        else:  # Error selected is "None" AND Node domain is selected
            node_colors = ["red" if (clicked_label == n) else "blue" for n in node_names_list]
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode="markers",
                text=node_names_list,
                marker=dict(size=7, color=node_colors),
                name="Junctions"
            )
            # Ensure edges are explicitly grayed out
            edge_marker_trace = go.Scatter(
                x=edge_marker_x, y=edge_marker_y,
                mode="markers",
                text=edge_names,
                marker=dict(
                    size=8,
                    color="lightgray",  # Gray for non-selected domain elements
                    symbol="square"
                ),
                name="Conduits"
            )
    else:  # Selected domain is conduits
        if error_selected and (selected_event in error_metrics and
                              selected_domain in error_metrics[selected_event] and
                              selected_variable in error_metrics[selected_event][selected_domain] and
                              selected_error_index in error_metrics[selected_event][selected_domain][
                                  selected_variable]):

            series_metric = error_metrics[selected_event][selected_domain][selected_variable][selected_error_index]
            metric_vals = []
            for e in edge_names:
                metric_vals.append(series_metric.loc[e] if e in series_metric.index else np.nan)
            metric_vals = np.array(metric_vals)

            # Determine colors based on selected error index
            if selected_error_index == "nse":
                colors = []
                for val in metric_vals:
                    if np.isnan(val):
                        colors.append(nan_color)
                    elif val < 0:
                        colors.append(nse_bin_colors[0])
                    else:
                        idx = np.digitize(val, bins=nse_bin_edges, right=True)
                        idx = min(idx, len(nse_bin_colors) - 1)
                        colors.append(nse_bin_colors[idx])
            else:  # For other error metrics, use continuous colorscale
                colors = metric_vals

            # Get the color range for the selected metric
            range_settings = error_metric_ranges.get(selected_error_index, {})
            cmin = range_settings.get("cmin")
            cmax = range_settings.get("cmax")
            # If cmax is not fixed, calculate it dynamically
            if cmax is None:
                cmax = np.nanmax(metric_vals)

            edge_colors_display = ["red" if (clicked_label == e) else colors[i] for i, e in enumerate(edge_names)]

            edge_marker_trace = go.Scatter(
                x=edge_marker_x, y=edge_marker_y,
                mode="markers",
                text=edge_names,
                marker=dict(
                    size=8,
                    color=edge_colors_display,
                    colorscale="RdBu_r" if selected_error_index != "nse" else None, # MODIFIED: Reversed scale
                    cmin=cmin if selected_error_index != "nse" else None,          # MODIFIED: Fixed min
                    cmax=cmax if selected_error_index != "nse" else None,          # MODIFIED: Potentially fixed max
                    symbol="square",
                    colorbar=dict(title=selected_error_index.upper()) if selected_error_index != "nse" else None,
                    showscale=True if selected_error_index != "nse" else False
                ),
                name="Conduits"
            )
            # When error metric selected, nodes are just gray
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode="markers",
                text=node_names_list,
                marker=dict(size=7, color="lightgray"),
                name="Junctions"
            )
        else:  # Error selected is "None" AND Conduit domain is selected
            edge_marker_trace = go.Scatter(
                x=edge_marker_x, y=edge_marker_y,
                mode="markers",
                text=edge_names,
                marker=dict(
                    size=8,
                    color=["red" if (clicked_label == e) else "green" for e in edge_names],
                    symbol="square"
                ),
                name="Conduits"
            )
            # Ensure nodes are explicitly grayed out
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode="markers",
                text=node_names_list,
                marker=dict(size=7, color="lightgray"),  # Gray for non-selected domain elements
                name="Junctions"
            )

    data = [edge_trace, node_trace, edge_marker_trace]

    # Add custom legend for NSE if NSE is selected
    if selected_error_index == "nse":
        legend_traces = []
        for i, label in enumerate(nse_bin_labels):
            legend_traces.append(
                go.Scatter(
                    x=[None],  # No actual data points
                    y=[None],
                    mode="markers",
                    marker=dict(size=10, color=nse_bin_colors[i], symbol="circle" if is_node_domain else "square"),
                    # Use circle for nodes, square for edges
                    name=f"NSE: {label}"
                )
            )
        # Add a trace for NaN values if applicable
        legend_traces.append(
            go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=10, color=nan_color, symbol="circle" if is_node_domain else "square"),
                name="NSE: N/A (< Threshold)"
            )
        )
        data.extend(legend_traces)

    fig = go.Figure(data=data)
    fig.update_layout(
        font=dict(family="Times New Roman"),
        xaxis=dict(title="X Coordinate", tickfont=dict(family="Times New Roman")),
        yaxis=dict(title="Y Coordinate", tickfont=dict(family="Times New Roman")),
        showlegend=False if selected_error_index != "nse" else True,
        margin=dict(l=20, r=20, t=20, b=50),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="LightGray",
            borderwidth=1,
            font=dict(family="Times New Roman", size=14),
            title=f"NSE Range ({'Junctions' if is_node_domain else 'Conduits'})"
        )
    )
    return fig

# ------------------- Timeseries Plot Update -------------------
@app.callback(
    dd.Output("timeseries-plot", "figure"),
    [
        dd.Input("network-graph", "clickData"),
        dd.Input("dataset-dropdown", "value"),
        dd.Input("event-dropdown", "value"),
        dd.Input("domain-dropdown", "value"),
        dd.Input("variable-dropdown", "value")
    ]
)
def plot_timeseries(clickData, selected_dataset, selected_event, selected_domain, selected_variable):
    """Display time series for the selected node/edge."""
    if not clickData:
        return go.Figure(layout={"title": "Click a node/edge to display time series"})

    clicked_id = clickData["points"][0]["text"]
    if selected_event not in denormalized_data:
        return go.Figure(layout={"title": f"No observed data for event '{selected_event}'."})
    if selected_domain not in denormalized_data[selected_event]:
        return go.Figure(layout={"title": f"Domain '{selected_domain}' missing in event '{selected_event}'."})
    if selected_variable not in denormalized_data[selected_event][selected_domain]:
        return go.Figure(layout={"title": f"Variable '{selected_variable}' not available in event '{selected_event}'."})

    observed_df = denormalized_data[selected_event][selected_domain][selected_variable]
    if clicked_id not in observed_df.columns:
        return go.Figure(layout={"title": f"'{clicked_id}' not found in observed data."})

    observed_series_full = observed_df[clicked_id].values
    if len(observed_series_full) <= n_time_step:
        return go.Figure(layout={"title": f"Not enough observed data (<= {n_time_step} steps)."})

    # MODIFIED: Remove the warm-up period from observed data for alignment.
    observed_series = observed_series_full[n_time_step:]

    # Load GNN predictions and align lengths with observed series.
    gnn_series = None
    if (selected_event in gnn_predictions and
            selected_domain in gnn_predictions[selected_event] and
            selected_variable in gnn_predictions[selected_event][selected_domain]):
        pred_df = gnn_predictions[selected_event][selected_domain][selected_variable]
        if clicked_id in pred_df.index:
            gnn_series_full = pred_df.loc[clicked_id].values
        elif clicked_id in pred_df.columns:
            gnn_series_full = pred_df[clicked_id].values
        else:
            gnn_series_full = None

        if gnn_series_full is not None:
            # Align both series to the same length
            T = min(len(observed_series), len(gnn_series_full))
            observed_series = observed_series[:T]
            gnn_series = gnn_series_full[:T]

    time_steps = np.arange(len(observed_series))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_steps,
        y=observed_series,
        mode="lines",
        name="SWMM model",
        line=dict(color="black")
    ))
    if gnn_series is not None:
        fig.add_trace(go.Scatter(
            x=np.arange(len(gnn_series)),
            y=gnn_series,
            mode="lines",
            name="GNN-SWS",
            line=dict(color="red", dash="dash")
        ))

    if selected_domain == "junctions":
        if (selected_event in transformed_rainfall and
                "rainfall" in transformed_rainfall[selected_event] and
                clicked_id in transformed_rainfall[selected_event]["rainfall"].columns):
            rainfall_full = transformed_rainfall[selected_event]["rainfall"][clicked_id].values
            if len(rainfall_full) > n_time_step:
                rainfall_series = rainfall_full[n_time_step:]
                inverted_rainfall = -1 * rainfall_series
                fig.add_trace(go.Bar(
                    x=np.arange(len(rainfall_series)),
                    y=inverted_rainfall,
                    name="Rainfall",
                    marker_color="gray",
                    yaxis="y2",
                    opacity=0.5
                ))
                fig.update_layout(
                    yaxis2=dict(title="Rainfall (inverted)", overlaying="y", side="right", showgrid=False)
                )
    fig.update_layout(
        title=f"Time Series at {clicked_id} ({selected_variable})",
        # Removed fixed height to let CSS style control
        # height=850,
        font=dict(family="Times New Roman"),
        xaxis=dict(title="Time Step", tickfont=dict(family="Times New Roman")),  # Explicit X-axis title
        yaxis=dict(title=f"{selected_variable}", tickfont=dict(family="Times New Roman")),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.5)",
            bordercolor="LightGray",
            borderwidth=1,
            font=dict(family="Times New Roman")
        ),
        margin=dict(l=50, r=50, t=50, b=50)  # Increased bottom margin
    )
    return fig


# ------------------- Error Metrics Display -------------------
@app.callback(
    dd.Output("error-metrics-div", "children"),
    [
        dd.Input("network-graph", "clickData"),
        dd.Input("event-dropdown", "value"),
        dd.Input("domain-dropdown", "value"),
        dd.Input("variable-dropdown", "value")
    ]
)
def display_error_metrics(clickData, selected_event, selected_domain, selected_variable):
    """Show aggregated error metrics (per node/edge) for the clicked element."""
    if not clickData:
        return "Click a node or conduit to see its error indices."

    clicked_id = clickData["points"][0]["text"]
    if (selected_event not in error_metrics or
            selected_domain not in error_metrics[selected_event] or
            selected_variable not in error_metrics[selected_event][selected_domain]):
        return f"No error metrics found for {clicked_id}."

    domain_metrics = error_metrics[selected_event][selected_domain][selected_variable]
    results = {}
    for metric_key in ["rmse", "nrmse", "relmae", "r", "nse"]:  # Ensured 'r' and 'nse' are included
        if metric_key in domain_metrics:
            series_vals = domain_metrics[metric_key]
            results[metric_key] = series_vals.loc[clicked_id] if clicked_id in series_vals.index else None
        else:
            results[metric_key] = None

    mean_obs = None
    if (selected_event in denormalized_data and
            selected_domain in denormalized_data[selected_event] and
            selected_variable in denormalized_data[selected_event][selected_domain]):
        obs_df = denormalized_data[selected_event][selected_domain][selected_variable]
        if clicked_id in obs_df.columns:
            observed_series_full = obs_df[clicked_id].values[n_time_step:]
            T_obs = len(observed_series_full)
            gnn_series_full = None
            if (selected_event in gnn_predictions and
                    selected_domain in gnn_predictions[selected_event] and
                    selected_variable in gnn_predictions[selected_event][selected_domain]):
                pred_df = gnn_predictions[selected_event][selected_domain][selected_variable]
                if clicked_id in pred_df.index:
                    gnn_series_full = pred_df.loc[clicked_id].values
                elif clicked_id in pred_df.columns:
                    gnn_series_full = pred_df[clicked_id].values
            if gnn_series_full is not None:
                T_pred = len(gnn_series_full)
                T_matched = min(T_obs, T_pred)
            else:
                T_matched = T_obs
            matched_observed_series = observed_series_full[:T_matched]
            mean_obs = np.nanmean(matched_observed_series)

    def fmt(v):
        if v is None or np.isnan(v):
            return "N/A"
        return f"{v:.4f}"

    return html.Div([
        html.Div(f"Selected {selected_domain[:-1].title()}: {clicked_id}", style={"marginBottom": "10px"}),
        html.Div(f"Mean Observed: {fmt(mean_obs)}"),
        html.Div(f"RMSE: {fmt(results['rmse'])}"),
        html.Div(f"NRMSE:  {fmt(results['nrmse'])}"),
        html.Div(f"Rel_MAE:  {fmt(results['relmae'])}"),
        html.Div(f"r:    {fmt(results['r'])}"),  # Display 'r'
        html.Div(f"NSE:    {fmt(results['nse'])}")
    ])

# ------------------- Network Error Animation (Revised) -------------------
@app.callback(
    dd.Output("network-error-animation-graph", "figure"),
    [
        dd.Input("error-time-slider", "value"),
        dd.Input("dataset-dropdown", "value"),
        dd.Input("event-dropdown", "value"),
        dd.Input("domain-dropdown", "value"),
        dd.Input("variable-dropdown", "value"),
        dd.Input("error-index-dropdown", "value")
    ]
)
def update_network_error_animation(time_step, selected_dataset, selected_event,
                                   selected_domain, selected_variable, selected_error_index):
    """
    Display the network-level error metric (aggregated over active nodes or edges) at each time step.
    """
    domain_label = "Node" if selected_domain == "junctions" else "Edge"

    if (selected_event not in denormalized_data or
            selected_domain not in denormalized_data[selected_event] or
            selected_variable not in denormalized_data[selected_event][selected_domain]):
        return go.Figure(layout={"title": "Data missing for selected event/domain/variable."})

    obs_df = denormalized_data[selected_event][selected_domain][selected_variable]
    observed_array = obs_df.values[n_time_step:, :]

    if (selected_event not in gnn_predictions or
            selected_domain not in gnn_predictions[selected_event] or
            selected_variable not in gnn_predictions[selected_event][selected_domain]):
        return go.Figure(layout={"title": "Predictions not available."})

    pred_df = gnn_predictions[selected_event][selected_domain][selected_variable]
    common_points = list(set(obs_df.columns).intersection(pred_df.index))

    if not common_points:
        return go.Figure(layout={"title": "No common elements between observed and predicted data."})

    # Ensure consistent ordering
    observed_array = obs_df[common_points].values[n_time_step:]
    pred_array_full = pred_df.loc[common_points].values.T

    T = min(observed_array.shape[0], pred_array_full.shape[0])
    observed_array = observed_array[:T, :]
    pred_array = pred_array_full[:T, :]

    if selected_error_index in ["none", "nse"]:
        title = "No error metric selected." if selected_error_index == "none" else "NSE does not have a time-series animation."
        return go.Figure(layout={"title": title})

    if time_step >= T:
        time_step = T - 1

    # === KEY CHANGE: Apply Thresholding ===
    nse_thresholds = {"depth": 0.01, "inflow": 0.001, "flow": 0.001}
    threshold = nse_thresholds.get(selected_variable, 0.0)

    # 1. Identify active elements based on their mean value over the whole event
    mean_abs_y_true = np.mean(np.abs(observed_array), axis=0)
    active_mask = mean_abs_y_true >= threshold

    # If no elements meet the threshold, return a message
    if not np.any(active_mask):
        return go.Figure(layout={"title": f"No {selected_domain} met the activity threshold of {threshold}."})

    avg_errors, std_errors = [], []
    for t in range(T):
        # 2. Filter data at the current time step to include only active elements
        y_true_t = observed_array[t, active_mask]
        y_pred_t = pred_array[t, active_mask]

        diff_t = y_true_t - y_pred_t
        errs = np.array([])  # Initialize as empty array

        if selected_error_index == "rmse":
            errs = np.sqrt((diff_t) ** 2)
        elif selected_error_index == "nrmse":
            eps = 1e-6
            errs = np.abs(diff_t) / (np.abs(y_true_t) + eps)
        elif selected_error_index == "relmae":
            mean_y = np.mean(np.abs(y_true_t))
            eps = 1e-6
            errs = np.abs(diff_t) / (mean_y + eps)

        avg_errors.append(np.nanmean(errs))
        std_errors.append(np.nanstd(errs))

    time_axis = np.arange(T)
    curr_avg = avg_errors[time_step]
    curr_std = std_errors[time_step]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=time_axis, y=avg_errors, mode="lines+markers", name="Average Error", line=dict(color="blue")))

    fig.add_trace(go.Scatter(
        x=np.concatenate([time_axis, time_axis[::-1]]),
        y=np.concatenate(
            [np.array(avg_errors) - np.array(std_errors), (np.array(avg_errors) + np.array(std_errors))[::-1]]),
        fill="toself", fillcolor="rgba(0,100,80,0.2)", line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip", showlegend=False
    ))

    fig.add_vline(x=time_step, line_width=2, line_dash="dash", line_color="red", annotation_text=f"Step {time_step}",
                  annotation_position="top left")

    variable_str = selected_variable.capitalize()
    error_str = selected_error_index.upper()
    title_text = f"Network {error_str} Over Time – {domain_label} {variable_str}"

    fig.update_layout(
        title=dict(text=(
            f"{title_text}<br><span style='font-size:13px'>Time Step: {time_step} (Avg: {curr_avg:.4f}, Std: {curr_std:.4f})</span>"),
                   x=0.01, xanchor="left", yanchor="top"),
        xaxis_title="Time Step", yaxis_title=error_str, font=dict(family="Times New Roman"),
        margin=dict(l=60, r=40, t=80, b=50),
        legend=dict(orientation="v", yanchor="top", y=0.98, xanchor="right", x=0.98,
                    font=dict(family="Times New Roman"), bgcolor="rgba(255,255,255,0.5)", bordercolor="LightGray",
                    borderwidth=1)
    )
    return fig

# ------------------- Start/Stop Animation -------------------
@app.callback(
    [dd.Output("animation-interval", "disabled"),
     dd.Output("toggle-animation", "children")],
    [dd.Input("toggle-animation", "n_clicks")],
    [dd.State("animation-interval", "disabled")]
)
def toggle_animation(n_clicks, interval_disabled):
    """Toggle the animation interval on/off based on button clicks."""
    if n_clicks is None or n_clicks % 2 == 0:
        return True, "Start Animation"
    else:
        return False, "Stop Animation"


@app.callback(
    dd.Output("error-time-slider", "value"),
    [dd.Input("animation-interval", "n_intervals")],
    [dd.State("error-time-slider", "value"),
     dd.State("error-time-slider", "max")]
)
def update_slider(n_intervals, current_value, slider_max):
    """Advance the error time slider automatically when animation is enabled."""
    if current_value >= slider_max:
        return 0
    else:
        return current_value + 1


# ===========================================================
# 8) Run the App
# ===========================================================
if __name__ == "__main__":
    app.run(debug=True)




import os
import numpy as np
from scipy.spatial import KDTree
import torch
from torch_geometric.data import Data
import networkx as nx
import pandas as pd
from pathlib import Path
import pickle
import json
from tqdm import tqdm  # Import tqdm for progress bar

# Utility Functions
def min_max_normalize(data, min_val, max_val):
    return (data - min_val) / (max_val - min_val - 1e-8)


def read_text_file_data_processing(file_path, sep="\t"):
    """
    Reads a text file with headers and converts it to a DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, "r") as f:
        lines = f.readlines()
    headers = lines[0].strip().split(sep)
    data = [line.strip().split(sep) for line in lines[1:]]
    return pd.DataFrame(data, columns=headers)


def process_and_save_conduit_with_area(original_file_path, output_directory):
    """
    Reads the original conduits.txt using the standard data processing function,
    adds an Area column assuming circular cross-section,
    and saves the updated file into the constant_data_dir.

    Parameters:
    - original_file_path (str): Path to the original raw conduits.txt file.
    - output_directory (str): Directory where the updated conduits.txt will be saved.
    """
    if not os.path.exists(original_file_path):
        raise FileNotFoundError(f"{original_file_path} not found.")

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    #  Load conduits.txt using the existing function
    conduits_df = read_text_file_data_processing(original_file_path)

    if "MaxDepth" not in conduits_df.columns:
        raise ValueError("'MaxDepth' column is missing in conduits.txt")

    #  Ensure MaxDepth is numeric before calculation
    conduits_df["MaxDepth"] = pd.to_numeric(conduits_df["MaxDepth"], errors="coerce")

    #  Calculate Area (assuming circular cross-section)
    conduits_df["Area"] = (np.pi * conduits_df["MaxDepth"] ** 2) / 4

    #  Save the updated conduits.txt
    output_file_path = os.path.join(output_directory, "conduits.txt")
    conduits_df.to_csv(output_file_path, sep="\t", index=False)

    print(f" Area column added and saved to '{output_file_path}'")


# Static Data Processing
def load_constant_features(base_dir, config=None, normalize=True):
    """
    Load and optionally normalize static features for subcatchments, junctions, conduits, and outfalls.

    Parameters:
    - base_dir: Path to the directory containing the static feature files.
    - config: Dictionary specifying the desired static features for each component.
    - normalize: Boolean to apply normalization to numeric columns.

    Returns:
    - A dictionary containing normalized DataFrames for subcatchments, junctions, conduits, and outfalls.
    """
    def check_file_exists(file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

    # File paths
    subcatchments_file = os.path.join(base_dir, "subcatchments.txt")
    junctions_file = os.path.join(base_dir, "junctions.txt")
    conduits_file = os.path.join(base_dir, "conduits.txt")
    outfalls_file = os.path.join(base_dir, "outfalls.txt")

    # Check file existence
    for file_path in [subcatchments_file, junctions_file, conduits_file, outfalls_file]:
        check_file_exists(file_path)

    # Read files
    subcatchments = read_text_file_data_processing(subcatchments_file)
    junctions = read_text_file_data_processing(junctions_file)
    conduits = read_text_file_data_processing(conduits_file)
    outfalls = read_text_file_data_processing(outfalls_file)

    # Default columns to load (all numeric columns by default)
    default_config = {
        "subcatchments": ["X", "Y", "Area", "Perc_Imperv", "Slope", "N_Imperv", "N_Perv"],
        "junctions": ["X", "Y", "Elevation", "MaxDepth", "InitDepth"],
        "conduits": ["Length", "Roughness", "MaxDepth", "Area"],
        "outfalls": ["X", "Y", "Elevation"]
    }

    # Use user-defined config or fallback to default
    config = config or default_config


    # Filter features based on config
    subcatchments = subcatchments[["Name", "Outlet"] + config.get("subcatchments", [])]
    junctions = junctions[["Name"] + config.get("junctions", [])]
    conduits = conduits[["Name", "From_Node", "To_Node"] + config.get("conduits", [])]
    outfalls = outfalls[["Name"] + config.get("outfalls", [])]

    # Convert numeric columns
    subcatchments = convert_numeric(subcatchments, config.get("subcatchments", []))
    junctions = convert_numeric(junctions, config.get("junctions", []))
    conduits = convert_numeric(conduits, config.get("conduits", []))
    outfalls = convert_numeric(outfalls, config.get("outfalls", []))
    #
    # # Calculate CrossSectionArea assuming circular cross-section
    # conduits["CrossSectionArea"] = (np.pi * conduits["MaxDepth"] ** 2) / 4
    # print("✅ CrossSectionArea calculated and added to conduits.")

    # store original X and Y coordinates before normalization
    node_coordinates = {}

    # Store junction coordinates
    for _, row in junctions.iterrows():
        node_coordinates[row["Name"]] = (row["X"], row["Y"])

    # Store outfall coordinates
    for _, row in outfalls.iterrows():
        node_coordinates[row["Name"]] = (row["X"], row["Y"])

    # print(f" Stored {len(node_coordinates)} node coordinates.")  # Debugging

    # Extract and store **original subcatchment Area values**
    original_subcatchment_areas = subcatchments.set_index("Name")["Area"].astype(float)

    # Normalize numeric columns
    if normalize:
        for df, component in zip(
            [subcatchments, junctions, conduits, outfalls],
            ["subcatchments", "junctions", "conduits", "outfalls"]
        ):
            numeric_cols = config.get(component,[]) # Include all numerical columns (including X, Y)   #[col for col in config.get(component, []) if col not in ["X", "Y"]]

            for col in numeric_cols:
                if col in df.columns:
                    min_val, max_val = df[col].min(), df[col].max()
                    df[col] = min_max_normalize(df[col], min_val, max_val)

    # Validate critical columns
    if subcatchments["Outlet"].isnull().any():
        raise ValueError("Outlet column in subcatchments contains null values.")
    if "Name" not in outfalls.columns:
        raise KeyError("Outfalls data must include a 'Name' column.")

    # Extract outfall names
    outfall_names = outfalls["Name"].tolist()
    # print("Outfall Names:", outfall_names)

    return {
        "subcatchments": subcatchments,
        "junctions": junctions,
        "conduits": conduits,
        "outfalls": outfalls,
        "outfall_names": outfall_names,
        "node_coordinates": node_coordinates,
    }, original_subcatchment_areas

# {
#     "subcatchments": DataFrame,  # Static features for subcatchments
#     "junctions": DataFrame,      # Static features for junctions
#     "conduits": DataFrame,       # Static features for conduits
#     "outfalls": DataFrame,       # Static features for outfalls
#     "outfall_names": List        # Names of outfalls
# }

def convert_numeric(df, numeric_columns):
    """
    Convert specified columns to numeric, handling non-numeric entries gracefully.

    Parameters:
    - df: DataFrame to convert.
    - numeric_columns: List of column names to convert to numeric.

    Returns:
    - DataFrame with specified columns converted to numeric.
    """
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# Function to create realtive coordinates
def calculate_relative_coordinates(conduits_df, junctions_df, outfalls_df, computed_config):
    """
    Adds relative coordinates (dx, dy) to conduits while preserving original conduit features.
    Only computes 'dx' and 'dy' if they are enabled in `computed_config["conduits"]`.

    Parameters:
    - conduits_df: DataFrame containing original conduit data.
    - junctions_df: DataFrame containing junction data with 'X' and 'Y' coordinates.
    - outfalls_df: DataFrame containing outfall data with 'X' and 'Y' coordinates.
    - computed_config: Dictionary specifying which computed features should be included.

    Returns:
    - Updated `conduits_df` with all original conduit features + computed features (dx, dy, coordinates).
    """

    # Check if we need to compute dx, dy
    compute_dx = "dx" in computed_config.get("conduits", [])
    compute_dy = "dy" in computed_config.get("conduits", [])

    # If neither dx nor dy is required, return the dataframe as is
    if not compute_dx and not compute_dy:
        print("Skipping relative coordinate calculation (dx, dy not in computed_config).")
        return conduits_df

    # Create a dictionary for junction and outfall coordinates
    node_coords = {row["Name"]: (row["X"], row["Y"]) for _, row in junctions_df.iterrows()}
    node_coords.update({row["Name"]: (row["X"], row["Y"]) for _, row in outfalls_df.iterrows()})

    # Map node coordinates to the conduits DataFrame
    conduits_df["From_X"] = conduits_df["From_Node"].map(lambda node: node_coords.get(node, (None, None))[0])
    conduits_df["From_Y"] = conduits_df["From_Node"].map(lambda node: node_coords.get(node, (None, None))[1])
    conduits_df["To_X"] = conduits_df["To_Node"].map(lambda node: node_coords.get(node, (None, None))[0])
    conduits_df["To_Y"] = conduits_df["To_Node"].map(lambda node: node_coords.get(node, (None, None))[1])

    # Compute dx and dy if required
    if compute_dx:
        conduits_df["dx"] = conduits_df["To_X"] - conduits_df["From_X"]
    if compute_dy:
        conduits_df["dy"] = conduits_df["To_Y"] - conduits_df["From_Y"]

    # Normalize only if computed
    if compute_dx:
        dx_min, dx_max = conduits_df["dx"].min(), conduits_df["dx"].max()
        conduits_df["dx"] = min_max_normalize(conduits_df["dx"], dx_min, dx_max)
    if compute_dy:
        dy_min, dy_max = conduits_df["dy"].min(), conduits_df["dy"].max()
        conduits_df["dy"] = min_max_normalize(conduits_df["dy"], dy_min, dy_max)

    # Debugging: Print final dx/dy values
    #print(f"\n📌 Sample dx/dy values: {conduits_df[['dx', 'dy']].dropna().head()}")

    # --- Build a mapping from each edge (by its conduit name) to its endpoints ---
    edge_to_nodes = {}
    for _, row in conduits_df.iterrows():
        edge_to_nodes[row["Name"]] = (row["From_Node"], row["To_Node"])
    # print("Edge-to-nodes mapping created for", len(edge_to_nodes), "edges.")

    return conduits_df, edge_to_nodes



def save_global_min_max(global_min_max, file_path="global_min_max.json"):
    """Save global min-max values to a JSON file."""
    with open(file_path, "w") as f:
        json.dump(global_min_max, f)
    #print(f"Global min-max saved to {file_path}")

def load_global_min_max(file_path="global_min_max.json"):
    """Load global min-max values from a JSON file."""
    with open(file_path, "r") as f:
        global_min_max = json.load(f)
    #print(f"Global min-max loaded from {file_path}")
    return global_min_max



def load_dynamic_data(folder, config=None, outfall_names=None, apply_normalization=True,
                      save_min_max_path="global_min_max.json", use_abs_flow=False):
    """
    Load and optionally normalize dynamic simulation data for all events.

    Args:
        folder (str): Directory containing simulation output files.
        config (dict): Feature configuration per component.
        outfall_names (list): Names of junctions to treat as outfalls.
        apply_normalization (bool): Whether to apply min-max normalization.
        save_min_max_path (str): Path to save global min/max values.
        use_abs_flow (bool): If True, use abs(flow) and return separate flow_direction.

    Returns:
        data (dict): Normalized data [event][component][feature]
        raw_data (dict): Raw unnormalized data
        outfall_data (dict): Outfall portion of normalized junction data
        raw_outfall_data (dict): Raw outfall data
        time_steps_per_event (dict): Time steps per event
        global_min_max (dict): Global min/max per feature
        flow_direction_data (dict): Directional info for conduit flow (-1, 0, +1)
    """
    import re

    data = {}
    raw_data = {}
    outfall_data = {}
    raw_outfall_data = {}
    time_steps_per_event = {}
    global_min_max = {}
    flow_direction_data = {}  # NEW dictionary to store flow directions

    default_config = {
        "subcatchments": ["rainfall", "runoff", "infiltration"],
        "junctions": ["head", "depth", "inflow", "lateralinflow"],
        "conduits": ["flow", "depth", "volume"]
    }
    config = config or default_config

    all_files = os.listdir(folder)
    prefixes = {
        re.match(r"(storm|real|synt)_\d+", f).group(0)
        for f in all_files if re.match(r"(storm|real|synt)_\d+", f)
    }
    sorted_prefixes = sorted(prefixes, key=lambda x: (x.split("_")[0], int(x.split("_")[1])))

    # Step 1: Compute global min/max if needed
    if apply_normalization:
        for event_prefix in sorted_prefixes:
            for component in ["subcatchments", "junctions", "conduits"]:
                for feature in config.get(component, []):
                    file_path = os.path.join(folder, f"{event_prefix}_{component}_{feature}.txt")
                    if os.path.exists(file_path):
                        try:
                            feature_data = pd.read_csv(file_path, sep="\t", index_col=0)
                            if use_abs_flow and component == "conduits" and feature == "flow":
                                feature_data = feature_data.abs()

                            key = f"{component}_{feature}"
                            min_val = feature_data.min().min()
                            max_val = feature_data.max().max()

                            if key in global_min_max:
                                global_min_max[key]["min"] = min(global_min_max[key]["min"], min_val)
                                global_min_max[key]["max"] = max(global_min_max[key]["max"], max_val)
                            else:
                                global_min_max[key] = {"min": min_val, "max": max_val}
                        except Exception as e:
                            print(f"[ERROR] Could not read file {file_path}: {e}")
        save_global_min_max(global_min_max, save_min_max_path)

    # Step 2: Load and process data
    for event_prefix in sorted_prefixes:
        data[event_prefix] = {"subcatchments": {}, "junctions": {}, "conduits": {}}
        raw_data[event_prefix] = {"subcatchments": {}, "junctions": {}, "conduits": {}}
        flow_direction_data[event_prefix] = {"conduits": {}}  # Initialize per event

        if outfall_names:
            outfall_data[event_prefix] = {}
            raw_outfall_data[event_prefix] = {}

        num_time_steps = None

        for component in ["subcatchments", "junctions", "conduits"]:
            for feature in config.get(component, []):
                file_path = os.path.join(folder, f"{event_prefix}_{component}_{feature}.txt")
                if os.path.exists(file_path):
                    try:
                        feature_data = pd.read_csv(file_path, sep="\t", index_col=0)
                        if num_time_steps is None:
                            num_time_steps = feature_data.shape[0]

                        # Store raw data before any normalization
                        raw_data[event_prefix][component][feature] = feature_data.copy()

                        # Handle abs(flow) + flow_direction
                        if use_abs_flow and component == "conduits" and feature == "flow":
                            flow_direction = np.sign(feature_data).astype(int)
                            flow_direction_data[event_prefix]["conduits"]["flow_direction"] = flow_direction
                            feature_data = feature_data.abs()

                        # Normalize if requested
                        if apply_normalization:
                            key = f"{component}_{feature}"
                            if key in global_min_max:
                                min_val = global_min_max[key]["min"]
                                max_val = global_min_max[key]["max"]
                                if max_val != min_val:
                                    feature_data = (feature_data - min_val) / (max_val - min_val)
                                else:
                                    feature_data *= 0

                        data[event_prefix][component][feature] = feature_data
                    except Exception as e:
                        print(f"[ERROR] Could not read file {file_path}: {e}")
                        data[event_prefix][component][feature] = None
                        raw_data[event_prefix][component][feature] = None
                else:
                    data[event_prefix][component][feature] = None
                    raw_data[event_prefix][component][feature] = None

        time_steps_per_event[event_prefix] = num_time_steps

        # Step 3: Handle outfall separation
        if outfall_names and "junctions" in data[event_prefix]:
            outfall_data[event_prefix] = {
                f: df[outfall_names] if df is not None else None
                for f, df in data[event_prefix]["junctions"].items()
            }
            data[event_prefix]["junctions"] = {
                f: df.drop(columns=outfall_names, errors="ignore") if df is not None else None
                for f, df in data[event_prefix]["junctions"].items()
            }

            raw_outfall_data[event_prefix] = {
                f: df[outfall_names] if df is not None else None
                for f, df in raw_data[event_prefix]["junctions"].items()
            }
            raw_data[event_prefix]["junctions"] = {
                f: df.drop(columns=outfall_names, errors="ignore") if df is not None else None
                for f, df in raw_data[event_prefix]["junctions"].items()
            }

    return data, raw_data, outfall_data, raw_outfall_data, time_steps_per_event, global_min_max, flow_direction_data





"""
data = {
    "real_1": {
        "subcatchments": {"rainfall": DataFrame(120, 50)},
        "junctions": {"head": DataFrame(120, 30)},
        "conduits": {"flow": DataFrame(120, 20)}
    }
}

outfall_data = {
    "real_1": {"head": DataFrame(120, 5)}  # Extracted outfall data
}

time_steps_per_event = {
    "real_1": 120,
    "synt_2": 90
}
"""


def denormalize_dynamic_data(normalized_data, outfall_data, global_min_max):
    """
    Denormalizes dynamic data (both SWMM simulation and GNN predictions)
    using global min-max values.

    Parameters:
    - normalized_data (dict): Dictionary containing normalized time-series data.
      Expected structure:
      {
          "event1": {
              "junctions": {
                  "head": DataFrame,
                  "flow": DataFrame,
              },
              "conduits": {
                  "flow": DataFrame,
                  "depth": DataFrame,
              }
          },
          "event2": { ... }
      }
    - outfall_data (dict): Dictionary containing normalized time-series data for outfalls.
    - global_min_max (dict): Dictionary storing min-max values for dynamic variables.
      Example:
      {
          "head": {"min": min_value, "max": max_value},
          "flow": {"min": min_value, "max": max_value},
          "rainfall": {"min": min_value, "max": max_value},
      }

    Returns:
    - denormalized_data (dict): Same structure as `normalized_data`, but with denormalized values.
    """
    denormalized_data = {}

    for event, event_data in normalized_data.items():
        denormalized_data[event] = {}

        # Process junction data (nodes)
        if "junctions" in event_data:
            denormalized_data[event]["junctions"] = {}
            for feature, df in event_data["junctions"].items():
                key = f"junctions_{feature}"
                if key not in global_min_max and feature in global_min_max:
                    key = feature  # fallback if composite key not available
                if key in global_min_max:
                    min_val = global_min_max[key]["min"]
                    max_val = global_min_max[key]["max"]
                    denormalized_data[event]["junctions"][feature] = (df * (max_val - min_val)) + min_val
                else:
                    denormalized_data[event]["junctions"][feature] = df.copy()

        # Process conduit data (edges) if available.
        if "conduits" in event_data:
            denormalized_data[event]["conduits"] = {}
            for feature, df in event_data["conduits"].items():
                key = f"conduits_{feature}"
                if key not in global_min_max and feature in global_min_max:
                    key = feature
                if key in global_min_max:
                    min_val = global_min_max[key]["min"]
                    max_val = global_min_max[key]["max"]
                    denormalized_data[event]["conduits"][feature] = (df * (max_val - min_val)) + min_val
                else:
                    denormalized_data[event]["conduits"][feature] = df.copy()

        # Process outfall data and merge into junctions if provided.
        if event in outfall_data:
            for feature, df in outfall_data[event].items():
                key = f"junctions_{feature}"
                if key not in global_min_max and feature in global_min_max:
                    key = feature
                if key in global_min_max:
                    min_val = global_min_max[key]["min"]
                    max_val = global_min_max[key]["max"]
                    denorm_df = (df * (max_val - min_val)) + min_val
                else:
                    denorm_df = df.copy()
                if "junctions" in denormalized_data[event]:
                    # Concatenate along columns.
                    denormalized_data[event]["junctions"][feature] = pd.concat(
                        [denormalized_data[event]["junctions"].get(feature, pd.DataFrame()), denorm_df],
                        axis=1
                    )
                else:
                    denormalized_data[event]["junctions"][feature] = denorm_df

    return denormalized_data


# Function to transform subcatchment dynamic features to match junction dynamic format
def transform_subcatchment_dynamic_data(static_data, dynamic_data, original_subcatchment_areas):
    """
    Convert subcatchment dynamic features into junction + outfall-based dynamic features.
    - Uses area-weighted averages for nodes with subcatchments.
    - For nodes without subcatchments, appends default dynamic values from the first subcatchment.

    Returns:
        dict: Transformed dynamic features for each event with consistent node coverage.
    """
    subcatchments = static_data["subcatchments"]
    junctions = static_data["junctions"]
    outfalls = static_data["outfalls"]

    # Map subcatchment -> outlet node (junction or outfall)
    subcatchment_to_outlet = dict(zip(subcatchments["Name"], subcatchments["Outlet"]))
    subcatchment_areas = original_subcatchment_areas.to_dict()

    all_nodes = list(junctions["Name"]) + list(outfalls["Name"])
    transformed_dynamic_data = {}

    for event, event_dynamic in dynamic_data.items():
        transformed_event_data = {}

        for feature_name, feature_data in event_dynamic.get("subcatchments", {}).items():
            node_feature_data = {}

            # Get a default time series from the **first subcatchment**
            default_series = feature_data.iloc[:, 0].copy()

            for node in all_nodes:
                # Find subcatchments draining into this node
                draining_subs = [
                    sub for sub, outlet in subcatchment_to_outlet.items()
                    if outlet == node
                ]

                if draining_subs:
                    weights = [subcatchment_areas[sub] for sub in draining_subs]
                    total = sum(weights)
                    normalized_weights = [w / total for w in weights]

                    sub_data = feature_data[draining_subs]
                    aggregated = (sub_data * normalized_weights).sum(axis=1)

                    node_feature_data[node] = aggregated
                else:
                    # Use the first subcatchment’s time series as default
                    node_feature_data[node] = default_series.copy()

            # Store result for this feature
            transformed_event_data[feature_name] = pd.DataFrame(node_feature_data)

        transformed_dynamic_data[event] = transformed_event_data

    return transformed_dynamic_data


def transform_edge_full_rainfall_features(edge_to_nodes, transformed_dynamic_data):
    """
    Convert full-length dynamic features from subcatchments into edge-based dynamic features.
    For each event, computes the element-wise average of the rainfall and accumulative rainfall time
    series for each edge, based on the two endpoints (nodes) of that edge.

    Uses the transformed dynamic data (e.g., from transform_subcatchment_dynamic_data) which contains,
    for each event, DataFrames for features such as "rainfall" and "acc_rainfall" with nodes as columns
    and time steps as rows.

    Returns:
        dict: Transformed edge-based dynamic features for each event with consistent edge coverage.
              Structure:
              {
                  event: {
                      "rainfall": DataFrame,       # Columns: edge names; Index: time steps; Values: averaged rainfall
                      "acc_rainfall": DataFrame    # Columns: edge names; Index: time steps; Values: averaged accumulative rainfall
                  },
                  ...
              }
    """

    edge_features_full_rainfall = {}

    # Iterate over each event in the transformed dynamic data.
    for event, event_data in transformed_dynamic_data.items():
        # Retrieve full-length node-based time series for rainfall and accumulative rainfall.
        df_rainfall = event_data.get("rainfall")
        df_acc_rainfall = event_data.get("acc_rainfall")

        # Create empty DataFrames for edge features using the same time index as the node DataFrames.
        if df_rainfall is not None:
            edge_rainfall = pd.DataFrame(index=df_rainfall.index)
        else:
            edge_rainfall = pd.DataFrame()

        if df_acc_rainfall is not None:
            edge_acc_rainfall = pd.DataFrame(index=df_acc_rainfall.index)
        else:
            edge_acc_rainfall = pd.DataFrame()

        # Compute averaged features for each edge based on its two endpoints.
        for edge_name, (from_node, to_node) in edge_to_nodes.items():
            # For rainfall: compute element-wise average if both nodes are present.
            if df_rainfall is not None and from_node in df_rainfall.columns and to_node in df_rainfall.columns:
                edge_rainfall[edge_name] = (df_rainfall[from_node] + df_rainfall[to_node]) / 2.0
            else:
                edge_rainfall[edge_name] = None

            # For accumulative rainfall: compute element-wise average if both nodes are present.
            if df_acc_rainfall is not None and from_node in df_acc_rainfall.columns and to_node in df_acc_rainfall.columns:
                edge_acc_rainfall[edge_name] = (df_acc_rainfall[from_node] + df_acc_rainfall[to_node]) / 2.0
            else:
                edge_acc_rainfall[edge_name] = None

        # Store the computed DataFrames in the returned dictionary.
        edge_features_full_rainfall[event] = {
            "rainfall": edge_rainfall,
            "acc_rainfall": edge_acc_rainfall
        }

    return edge_features_full_rainfall


# Function to create constant node features
def create_constant_node_features(static_data, dynamic_data):
    """
    Create constant node features formatted per event.

    Args:
        static_data (dict): Static features for subcatchments, junctions, and outfalls.
        dynamic_data (dict): Used to extract event names.

    Returns:
        dict: Constant node features structured by event.
        dict: Node type labels (type_a or type_b) for use in downstream GNN models.
    """

    def aggregate_subcatchment_features(subcatchments_df, junctions_df, outfalls_df):
        """
        Aggregate subcatchment features for both junctions and outfalls using weighted averages.
        For nodes without any connected subcatchments, zero padding is applied.
        Also returns node type labels (type_a or type_b).
        """
        aggregated_features = {}
        node_type_dict = {}

        # Combine valid outlet node names (junctions + outfalls)
        valid_outlet_names = set(junctions_df["Name"]).union(set(outfalls_df["Name"]))

        outlet_mapping = subcatchments_df.groupby("Outlet")
        sample_aggregated = None

        for outlet, group in outlet_mapping:
            if outlet in valid_outlet_names:
                total_area = group["Area"].sum()
                weights = group["Area"] / total_area

                columns_to_exclude = ["Name", "Outlet", "Area"]
                if "X" in group.columns: columns_to_exclude.append("X")
                if "Y" in group.columns: columns_to_exclude.append("Y")

                aggregated = (
                    group.drop(columns=columns_to_exclude, errors="ignore").astype(float)
                    .multiply(weights.values, axis=0)
                    .sum()
                )

                result_vector = [total_area] + aggregated.tolist()
                aggregated_features[outlet] = result_vector
                node_type_dict[outlet] = "type_a"

                if sample_aggregated is None:
                    sample_aggregated = result_vector

        # Determine padding size
        padding_length = len(sample_aggregated) if sample_aggregated else 1
        zero_padded = [0.0] * padding_length

        # Zero-pad remaining nodes
        for node in valid_outlet_names:
            if node not in aggregated_features:
                aggregated_features[node] = zero_padded.copy()
                node_type_dict[node] = "type_b"

        return aggregated_features, node_type_dict

    # --- Step 1: Static Data ---
    subcatchments = static_data["subcatchments"]
    junctions = static_data["junctions"]
    outfalls = static_data["outfalls"]

    # --- Step 2: Aggregate Subcatchment Features ---
    aggregated_features, node_type_dict = aggregate_subcatchment_features(subcatchments, junctions, outfalls)

    # --- Step 3: Process Junctions ---
    node_feature_dict = {}
    junction_static_feature_len = len(junctions.columns.drop("Name"))

    for _, row in junctions.iterrows():
        node_name = row["Name"]
        features = []

        # Add static junction features
        features.extend(row.drop(labels=["Name"]).values.tolist())

        # Add aggregated subcatchment features (or zero padding)
        features.extend(aggregated_features.get(node_name, []))

        node_feature_dict[node_name] = [features]

    # --- Step 4: Process Outfalls ---
    for _, row in outfalls.iterrows():
        node_name = row["Name"]
        features = []

        # Add static outfall features
        outfall_static_features = row.drop(labels=["Name"]).values.tolist()
        features.extend(outfall_static_features)

        # Pad to match junction static feature length
        padding_needed = junction_static_feature_len - len(outfall_static_features)
        features.extend([0.0] * padding_needed)

        # Add aggregated subcatchment features (already padded if needed)
        features.extend(aggregated_features.get(node_name, []))

        node_feature_dict[node_name] = [features]

    # --- Step 5: Format Output Per Event ---
    constant_node_features = {}
    for event in dynamic_data.keys():
        constant_node_features[event] = node_feature_dict.copy()
    return constant_node_features, node_type_dict


# Function to create dynamic node features
def create_dynamic_node_features(transformed_dynamic_data, dynamic_data, outfall_dynamic_data,
                                 n_time_step, max_steps_ahead, num_time_steps_per_event):
    """
    Create dynamic node features using rainfall-related subcatchment features (now transformed to nodes),
    and optionally additional dynamic features for junctions and outfalls.

    Returns:
        dict: Dynamic node features formatted per node with time step entries.
        dict: Dynamic node targets formatted per node.
        dict: Future rainfall values per node (separate from targets). For each future step, this includes
              both [rainfall, acc_rainfall].
    """
    dynamic_node_features = {}
    dynamic_node_targets = {}
    future_rainfall_dict = {}

    for event in dynamic_data.keys():
        print(f"\n Processing event: {event}")

        num_time_steps = num_time_steps_per_event.get(event, 0)
        if num_time_steps < n_time_step + max_steps_ahead:
            print(f"⚠ Skipping event {event}: Not enough time steps.")
            continue

        event_dynamic_features = {}
        event_dynamic_targets = {}
        event_future_rainfall = {}

        # Get all node names from transformed dynamic data (already includes junctions + outfalls)
        node_names = list(transformed_dynamic_data[event][next(iter(transformed_dynamic_data[event]))].columns)

        for node_name in node_names:
            node_feature_sequence = []
            node_target_sequence = []
            future_rainfall_sequence = []

            for t in range(num_time_steps - (n_time_step + max_steps_ahead) + 1):
                timestep_features = []
                timestep_targets = []
                future_rainfall_values = []

                # Add transformed subcatchment-based dynamic features (rainfall, acc_rainfall, etc.)
                for feature_name, feature_data in transformed_dynamic_data[event].items():
                    values = feature_data[node_name].iloc[t:t + n_time_step].values.tolist()
                    timestep_features.extend(values)

                # Add junction-based dynamic features
                if event in dynamic_data:
                    for feature_name, feature_data in dynamic_data[event].get("junctions", {}).items():
                        if node_name in feature_data.columns:
                            values = feature_data[node_name].iloc[t:t + n_time_step].values.tolist()
                            timestep_features.extend(values)

                # Add outfall-based dynamic features
                if event in outfall_dynamic_data:
                    for feature_name, feature_data in outfall_dynamic_data[event].items():
                        if node_name in feature_data.columns:
                            values = feature_data[node_name].iloc[t:t + n_time_step].values.tolist()
                            timestep_features.extend(values)

                # Assign future targets from junction and outfall dynamic data
                for source_data in [dynamic_data[event].get("junctions", {}),
                                    outfall_dynamic_data.get(event, {})]:
                    for feature_name, feature_data in source_data.items():
                        if node_name in feature_data.columns:
                            future_values = feature_data[node_name].iloc[t + n_time_step:t + n_time_step + max_steps_ahead].values
                            timestep_targets.extend(future_values.tolist())

                # Assign future rainfall from transformed subcatchment features
                for feature_name, feature_data in transformed_dynamic_data[event].items():
                    values = feature_data[node_name].iloc[t + n_time_step:t + n_time_step + max_steps_ahead].values.tolist()
                    future_rainfall_values.extend(values)

                # Append everything
                node_feature_sequence.append(timestep_features)
                node_target_sequence.append(timestep_targets)
                future_rainfall_sequence.append(future_rainfall_values)

            # Store per node
            event_dynamic_features[node_name] = node_feature_sequence
            event_dynamic_targets[node_name] = node_target_sequence
            event_future_rainfall[node_name] = future_rainfall_sequence

        dynamic_node_features[event] = event_dynamic_features
        dynamic_node_targets[event] = event_dynamic_targets
        future_rainfall_dict[event] = event_future_rainfall

    return dynamic_node_features, dynamic_node_targets, future_rainfall_dict


def calculate_diff_features_for_edges(
    node_dynamic_features,
    edge_to_nodes,
    num_time_steps_per_event,
    n_time_step,
    max_steps_ahead,
    apply_normalization=True,
    save_min_max_path="global_diff_min_max.json"
):
    """
    Calculates diff_depth and diff_inflow from raw node features, with optional global min-max normalization.
    """
    diff_features = {}
    all_diff_depth_values = []
    all_diff_inflow_values = []

    # Step 1: Calculate diff values over valid time steps
    for event, nodes_data in node_dynamic_features.items():
        diff_features[event] = {}
        num_valid_time_steps = num_time_steps_per_event[event] - (n_time_step + max_steps_ahead) + 1

        for edge_name, (from_node, to_node) in edge_to_nodes.items():
            diff_features[event][edge_name] = {
                "diff_depth": [],
                "diff_inflow": []
            }

            from_node_data = nodes_data.get(from_node)
            to_node_data = nodes_data.get(to_node)

            if from_node_data is None or to_node_data is None:
                continue  # Skip if node data is missing

            n_total_features = len(from_node_data[0])
            n_subcatchment_features = (n_total_features // n_time_step) - 2
            depth_offset = n_subcatchment_features * n_time_step
            inflow_offset = depth_offset + n_time_step

            for t in range(num_valid_time_steps):
                depth_from = from_node_data[t][depth_offset:depth_offset + n_time_step]
                depth_to = to_node_data[t][depth_offset:depth_offset + n_time_step]
                inflow_from = from_node_data[t][inflow_offset:inflow_offset + n_time_step]
                inflow_to = to_node_data[t][inflow_offset:inflow_offset + n_time_step]

                diff_depth = [hf - ht for hf, ht in zip(depth_from, depth_to)]
                diff_inflow = [inf_from - inf_to for inf_from, inf_to in zip(inflow_from, inflow_to)]

                diff_features[event][edge_name]["diff_depth"].append(diff_depth)
                diff_features[event][edge_name]["diff_inflow"].append(diff_inflow)

                if apply_normalization:
                    all_diff_depth_values.extend(diff_depth)
                    all_diff_inflow_values.extend(diff_inflow)

        # print(f"\n Calculated diff features for event '{event}' ({num_valid_time_steps} valid time steps).")

    if apply_normalization:
        # Step 2: Compute global min-max
        diff_depth_min = min(all_diff_depth_values) if all_diff_depth_values else 0.0
        diff_depth_max = max(all_diff_depth_values) if all_diff_depth_values else 1.0
        diff_inflow_min = min(all_diff_inflow_values) if all_diff_inflow_values else 0.0
        diff_inflow_max = max(all_diff_inflow_values) if all_diff_inflow_values else 1.0

        global_min_max = {
            "diff_depth": {"min": diff_depth_min, "max": diff_depth_max},
            "diff_inflow": {"min": diff_inflow_min, "max": diff_inflow_max}
        }

        # Save global min-max
        save_global_min_max(global_min_max, save_min_max_path)

        # Step 3: Normalize diff values
        for event in diff_features:
            for edge_name in diff_features[event]:
                # Normalize diff_depth
                for t in range(len(diff_features[event][edge_name]["diff_depth"])):
                    diff_depth = diff_features[event][edge_name]["diff_depth"][t]
                    normalized_diff_depth = [
                        (v - diff_depth_min) / (diff_depth_max - diff_depth_min) if diff_depth_max != diff_depth_min else 0.0
                        for v in diff_depth
                    ]
                    diff_features[event][edge_name]["diff_depth"][t] = normalized_diff_depth

                # Normalize diff_inflow
                for t in range(len(diff_features[event][edge_name]["diff_inflow"])):
                    diff_inflow = diff_features[event][edge_name]["diff_inflow"][t]
                    normalized_diff_inflow = [
                        (v - diff_inflow_min) / (diff_inflow_max - diff_inflow_min) if diff_inflow_max != diff_inflow_min else 0.0
                        for v in diff_inflow
                    ]
                    diff_features[event][edge_name]["diff_inflow"][t] = normalized_diff_inflow

    else:

        global_min_max = None

    return diff_features, global_min_max



# Function to merge constant and dynamic node features and also return targets
def merge_node_features(constant_node_features, dynamic_node_features, dynamic_node_targets):
    """
    Merge constant and dynamic node features to create a unified feature set and return targets.

    Args:
        constant_node_features (dict): Constant node features (output from `create_constant_node_features`).
        dynamic_node_features (dict): Dynamic node features (output from `create_dynamic_node_features`).
        dynamic_node_targets (dict): Target values (output from `create_dynamic_node_features`).

    Returns:
        tuple:
            dict: Merged node features (constant + dynamic).
            dict: Node targets.
    """

    node_features = {}
    node_targets = {}

    for event in dynamic_node_features.keys():  # Iterate through each event
        print(f"\n🔄 Merging features for event: {event}")

        # Initialize dictionaries for this event
        event_merged_features = {}
        event_merged_targets = {}

        # Get constant, dynamic features, and targets for the event
        static_features = constant_node_features.get(event, {})
        dynamic_features = dynamic_node_features.get(event, {})
        dynamic_targets = dynamic_node_targets.get(event, {})

        # Ensure both constant and dynamic features exist
        all_nodes = set(static_features.keys()).union(set(dynamic_features.keys()))

        for node_name in all_nodes:
            static_f = static_features.get(node_name, [[]])  # Default to empty list of lists if missing
            dynamic_f = dynamic_features.get(node_name, [])  # Default to empty list if missing
            targets_f = dynamic_targets.get(node_name, [])  # Targets

            # Ensure static features are repeated for each time step
            merged_f = [
                static_f[0] + dynamic_t  # Append constant features to each time step
                for dynamic_t in dynamic_f
            ]

            event_merged_features[node_name] = merged_f
            event_merged_targets[node_name] = targets_f

        node_features[event] = event_merged_features
        node_targets[event] = event_merged_targets

    return node_features, node_targets


def calculate_rainfall_features_for_edges(
        node_dynamic_features,
        edge_to_nodes,
        num_time_steps_per_event,
        n_time_step,
        max_steps_ahead
):
    """
    Calculates the averaged rainfall and averaged accumulative rainfall for each edge,
    based on the two endpoints' node dynamic features.

    Assumes that each node's sliding-window vector is organized as follows:
      [rainfall (n_time_step), acc_rainfall (n_time_step), head (n_time_step), inflow (n_time_step), ...]

    For each valid sliding window, the function computes:
      - avg_rainfall: the element-wise average of the rainfall values from the two endpoints.
      - avg_acc_rainfall: the element-wise average of the accumulative rainfall values.

    Args:
        node_dynamic_features (dict): Node dynamic features (per event) with sliding window vectors.
        edge_to_nodes (dict): Mapping from edge names to a tuple (from_node, to_node).
        num_time_steps_per_event (dict): Number of time steps available for each event.
        n_time_step (int): Number of past time steps in the sliding window.
        max_steps_ahead (int): Number of future steps to predict (used here to determine valid windows).

    Returns:
        dict: A dictionary structured as:
            rainfall_features[event][edge_name] = {
                "avg_rainfall": [list of n_time_step values per valid time step],
                "avg_acc_rainfall": [list of n_time_step values per valid time step]
            }
    """
    rainfall_features = {}

    for event, nodes_data in node_dynamic_features.items():
        rainfall_features[event] = {}
        num_valid_time_steps = num_time_steps_per_event[event] - (n_time_step + max_steps_ahead) + 1

        for edge_name, (from_node, to_node) in edge_to_nodes.items():
            rainfall_features[event][edge_name] = {
                "avg_rainfall": [],
                "avg_acc_rainfall": []
            }

            from_node_data = nodes_data.get(from_node)
            to_node_data = nodes_data.get(to_node)
            if from_node_data is None or to_node_data is None:
                continue  # Skip if node data is missing

            # For each valid sliding window, compute averaged values:
            for t in range(num_valid_time_steps):
                avg_rainfall = []
                avg_acc_rainfall = []
                for i in range(n_time_step):
                    # The first n_time_step values are for rainfall.
                    rf_from = float(from_node_data[t][i])
                    rf_to = float(to_node_data[t][i])
                    avg_r = (rf_from + rf_to) / 2.0
                    avg_rainfall.append(avg_r)

                    # The next n_time_step values are for acc_rainfall.
                    arf_from = float(from_node_data[t][n_time_step + i])
                    arf_to = float(to_node_data[t][n_time_step + i])
                    avg_acc = (arf_from + arf_to) / 2.0
                    avg_acc_rainfall.append(avg_acc)

                rainfall_features[event][edge_name]["avg_rainfall"].append(avg_rainfall)
                rainfall_features[event][edge_name]["avg_acc_rainfall"].append(avg_acc_rainfall)

    return rainfall_features


# Function to create constant edge features
def create_constant_edge_features(conduits_df, constant_config, computed_config):
    """
    Create constant edge features including both user-defined static features and computed ones (dx, dy).

    Parameters:
    - conduits_df: DataFrame containing conduit features (including computed dx, dy).
    - constant_config: Dictionary specifying static features for conduits.
    - computed_config: Dictionary specifying computed features (e.g., dx, dy).

    Returns:
    - edge_features: Dictionary with keys as edge names and values as feature vectors.
    """
    constant_edge_features = {}

    # Merge static and computed features inside the function
    selected_features = constant_config.get("conduits", []) + computed_config.get("conduits", [])

    for _, row in conduits_df.iterrows():
        edge_name = row["Name"]
        constant_edge_features[edge_name] = [row.get(feature, 0.0) for feature in selected_features]

    return constant_edge_features


def create_dynamic_edge_features(dynamic_data, dynamic_config, n_time_step, max_steps_ahead,
                                 num_time_steps_per_event, constant_edge_features, diff_features, edge_to_nodes, rainfall_features,
                                 node_future_rainfall_dict):
    """
    Create dynamic edge features by combining:
      - Averaged rainfall and averaged acc_rainfall features (each of length n_time_step),
      - Precomputed diff_head and diff_inflow from diff_features (each of length n_time_step),
      - Sliding window conduit dynamic features (each of length n_time_step).

    The final edge feature vector order is:
      [avg_rainfall (n_time_step), avg_acc_rain (n_time_step),
       diff_head (n_time_step), diff_inflow (n_time_step),
       conduit features (each n_time_step)]

    Args:
        dynamic_data (dict): Dynamic data for all components.
        dynamic_config (dict): Configuration for dynamic features.
        n_time_step (int): Number of past time steps in the sliding window.
        max_steps_ahead (int): Number of future steps to predict.
        num_time_steps_per_event (dict): Number of time steps available per event.
        constant_edge_features (dict): Constant edge features from conduits.
        diff_features (dict): Precomputed diff features.
        edge_to_nodes (dict): Mapping from edge names to (from_node, to_node).
        rainfall_features (dict): Averaged rainfall features for edges computed from node dynamic features.
        node_future_rainfall_dict (dict): Future rainfall dictionary for nodes, as returned by create_dynamic_node_features.
                                         Expected structure:
                                           node_future_rainfall_dict[event][node] = list of future rainfall vectors
                                           (each of length max_steps_ahead*2)
    Returns:
        tuple: (dynamic_edge_features, dynamic_edge_targets, future_edge_rainfall_dict)
          - dynamic_edge_features: Dictionary of edge features per event.
          - dynamic_edge_targets: Dictionary of edge targets per event.
          - future_edge_rainfall_dict: Dictionary of future rainfall targets per edge and event.
              For each valid time step, the vector is the elementwise average of the two endpoints' future rainfall
              vectors (length max_steps_ahead*2).
    """
    dynamic_edge_features = {}
    dynamic_edge_targets = {}
    future_edge_rainfall_dict = {}  # Separate storage for future rainfall features on edges

    for event, event_dynamic_data in dynamic_data.items():
        num_time_steps = num_time_steps_per_event.get(event, 0)
        required_steps = n_time_step + max_steps_ahead
        if num_time_steps < required_steps:
            continue

        valid_time_steps = num_time_steps - required_steps + 1
        event_features = {}
        event_targets = {}
        event_future_rainfall = {}  # For each edge in this event

        for edge_name in constant_edge_features.keys():
            # Check that the edge is defined in edge_to_nodes.
            if edge_name not in edge_to_nodes:
                continue
            from_node, to_node = edge_to_nodes[edge_name]

            edge_feature_list = []
            edge_target_list = []
            edge_future_rain_list = []  # Future rainfall for this edge

            for t in range(valid_time_steps):
                edge_feature_vec = []

                # 1. Append averaged rainfall and averaged acc_rainfall (current features)
                if event in rainfall_features and edge_name in rainfall_features[event]:
                    avg_rain = rainfall_features[event][edge_name]["avg_rainfall"][t]
                    avg_acc_rain = rainfall_features[event][edge_name]["avg_acc_rainfall"][t]
                else:
                    avg_rain = [0.0] * n_time_step
                    avg_acc_rain = [0.0] * n_time_step
                edge_feature_vec.extend(avg_rain)
                edge_feature_vec.extend(avg_acc_rain)

                # 2. Append precomputed diff_depth and diff_inflow from diff_features
                if event in diff_features and edge_name in diff_features[event]:
                    diff_depth = diff_features[event][edge_name]["diff_depth"][t]
                    diff_inflow = diff_features[event][edge_name]["diff_inflow"][t]
                else:
                    diff_depth = [0.0] * n_time_step
                    diff_inflow = [0.0] * n_time_step
                edge_feature_vec.extend(diff_depth)
                edge_feature_vec.extend(diff_inflow)

                # 3. Append sliding window conduit dynamic features (e.g., flow, depth)
                for feature in dynamic_config.get("conduits", []):
                    feature_data = event_dynamic_data["conduits"].get(feature)
                    if feature_data is not None and edge_name in feature_data.columns:
                        sliding_window = feature_data.iloc[t: t + n_time_step,
                                                           feature_data.columns.get_loc(edge_name)].tolist()
                        edge_feature_vec.extend(sliding_window)
                    else:
                        edge_feature_vec.extend([0.0] * n_time_step)

                edge_feature_list.append(edge_feature_vec)

                # 4. Create targets (only conduit-specific features, without rainfall or diff values)
                target_vec = []
                for feature in dynamic_config.get("conduits", []):
                    feature_data = event_dynamic_data["conduits"].get(feature)
                    if feature_data is not None and edge_name in feature_data.columns:
                        future_steps = feature_data.iloc[t + n_time_step: t + n_time_step + max_steps_ahead,
                                                         feature_data.columns.get_loc(edge_name)].tolist()
                        target_vec.extend(future_steps)
                    else:
                        target_vec.extend([0.0] * max_steps_ahead)
                edge_target_list.append(target_vec)

                # 5. Retrieve future rainfall targets for this edge using node_future_rainfall_dict.
                # For each valid time step, get the future rainfall vector from both endpoints (each vector of length max_steps_ahead*2)
                # and compute their elementwise average.
                if (event in node_future_rainfall_dict and
                    from_node in node_future_rainfall_dict[event] and
                    to_node in node_future_rainfall_dict[event]):
                    future_from = node_future_rainfall_dict[event][from_node][t]  # vector of length max_steps_ahead*2
                    future_to = node_future_rainfall_dict[event][to_node][t]      # vector of length max_steps_ahead*2
                    future_rain = [(a + b) / 2.0 for a, b in zip(future_from, future_to)]
                else:
                    future_rain = [0.0] * (max_steps_ahead * 2)
                edge_future_rain_list.append(future_rain)

            event_features[edge_name] = edge_feature_list
            event_targets[edge_name] = edge_target_list
            event_future_rainfall[edge_name] = edge_future_rain_list

        dynamic_edge_features[event] = event_features
        dynamic_edge_targets[event] = event_targets
        future_edge_rainfall_dict[event] = event_future_rainfall

    return dynamic_edge_features, dynamic_edge_targets, future_edge_rainfall_dict


def merge_edge_features(constant_edge_features, dynamic_edge_features, dynamic_edge_targets, num_time_steps_per_event):
    """
    Merge constant and dynamic edge features to create a unified feature set and return edge targets.

    Parameters:
    - constant_edge_features: Dictionary {edge_name: constant feature list}
    - dynamic_edge_features: Dictionary {event: {edge_name: [dynamic features per time step]} }
    - dynamic_edge_targets: Dictionary {event: {edge_name: [targets per time step]} }
    - num_time_steps_per_event: Dictionary {event: total number of time steps}

    Returns:
    - edge_features: {event: {edge_name: merged feature list per time step} }
    - edge_targets: {event: {edge_name: target list per time step} }
    """
    edge_features = {}
    edge_targets = {}

    for event in dynamic_edge_features.keys():  # Iterate through each event

        # Initialize dictionaries for this event
        event_merged_features = {}
        event_merged_targets = {}

        # Get constant, dynamic features, and targets for the event
        static_features = constant_edge_features
        dynamic_features = dynamic_edge_features.get(event, {})
        dynamic_targets = dynamic_edge_targets.get(event, {})

        # Get number of time steps for this event
        num_time_steps = num_time_steps_per_event.get(event, 1)  # Default to 1 if missing

        # Ensure both constant and dynamic features exist
        all_edges = set(static_features.keys()).union(set(dynamic_features.keys()))

        for edge_name in all_edges:
            static_f = static_features.get(edge_name, [])  # Fixed static feature vector
            dynamic_f = dynamic_features.get(edge_name, [])  # Time-dependent dynamic features
            targets_f = dynamic_targets.get(edge_name, [])  # Time-dependent targets

            if not dynamic_f:
                # If no dynamic data, repeat constant features for all time steps
                event_merged_features[edge_name] = [static_f] * num_time_steps
            else:
                # Merge constant and dynamic features at each time step
                event_merged_features[edge_name] = [static_f + dynamic_t for dynamic_t in dynamic_f]

            # Ensure targets exist for each time step
            if not targets_f:
                event_merged_targets[edge_name] = [[0.0] * len(static_f)] * num_time_steps  # Zero padding for missing targets
            else:
                event_merged_targets[edge_name] = targets_f  # Use available targets

        edge_features[event] = event_merged_features
        edge_targets[event] = event_merged_targets

    return edge_features, edge_targets


def convert_conduits_to_graph(updated_conduits_df, node_coordinates, config=None, computed_config=None):

    # Set default configurations if not provided
    config = config or {"conduits": ["Length", "Roughness", "MaxDepth"]}
    computed_config = computed_config or {"conduits": []}

    # Build the list of conduit keys (static + computed)
    conduit_keys = config.get("conduits", [])
    for feature in computed_config.get("conduits", []):
        if feature in updated_conduits_df.columns and feature not in conduit_keys:
            conduit_keys.append(feature)

    # Check that required columns exist
    required_columns = ["From_Node", "To_Node", "Name"]
    for col in required_columns:
        if col not in updated_conduits_df.columns:
            raise KeyError(f"Required column `{col}` is missing from updated_conduits_df.")

    # Create a mapping from node names to unique indices (sorted for consistency)
    node_mapping = {
        node: idx
        for idx, node in enumerate(
            sorted(set(updated_conduits_df["From_Node"]).union(set(updated_conduits_df["To_Node"])))
        )
    }

    # Prepare lists for edge connectivity, attributes, and an ordered list for edge info.
    edge_list = []
    edge_attributes = []
    edge_info_list = []  # Each element is a dict: { "from_node": ..., "to_node": ..., "conduit_name": ... }
    edge_names = []

    for _, row in updated_conduits_df.iterrows():
        from_node_name = row["From_Node"]
        to_node_name = row["To_Node"]
        conduit_name = row["Name"]

        # Skip rows where nodes are missing in the mapping.
        if from_node_name not in node_mapping or to_node_name not in node_mapping:
            continue

        # Get integer indices from node_mapping
        from_node_idx = node_mapping[from_node_name]
        to_node_idx = node_mapping[to_node_name]

        # Append the edge connectivity based on node indices.
        edge_list.append([from_node_idx, to_node_idx])

        # Build the edge attribute vector using the selected conduit keys.
        edge_attr = [row[key] for key in conduit_keys if key in row]
        edge_attributes.append(edge_attr)

        # Append edge info dict to the ordered list.
        # We store both the string node names and the integer indices.
        edge_info_list.append({
            "from_node": from_node_name,
            "to_node": to_node_name,
            "from_node_idx": from_node_idx,
            "to_node_idx": to_node_idx,
            "conduit_name": conduit_name
        })

        # Store the conduit (edge) name
        edge_names.append(conduit_name)

    # Create the edge_index tensor (shape [2, num_edges]).
    edge_index = torch.tensor(edge_list, dtype=torch.long).T

    # Create the edge attribute tensor.
    edge_attr_tensor = torch.tensor(edge_attributes, dtype=torch.float) if edge_attributes else None

    # Build node positions tensor.
    # Use the sorted order of node names to match the node_mapping.
    sorted_nodes = sorted(node_mapping.keys())
    node_pos = torch.tensor(
        [list(node_coordinates.get(node, (0.0, 0.0))) for node in sorted_nodes],
        dtype=torch.float
    )

    # Create dummy node features (using ones).
    x = torch.ones((len(node_mapping), 1), dtype=torch.float)

    # Construct the PyTorch Geometric Data object.
    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr_tensor, pos=node_pos)

    # Add node and edge names as attributes.
    graph.node_names = sorted_nodes  # In the same order as x
    graph.edge_names = edge_names  # Matches the order in edge_index and edge_attr
    graph.edge_info_list = edge_info_list  # For full reference if needed


    # Return the graph, edge_index, node_mapping, and the ordered edge info list.
    return graph, edge_index, node_mapping, edge_info_list


def build_graph_per_event(
        merged_node_features, merged_edge_features, merged_node_targets, merged_edge_targets,
        edge_index, node_mapping, ordered_edge_info_list,
        n_time_step, max_steps_ahead, num_time_steps_per_event, future_rainfall_dict, future_edge_rainfall_dict, node_type_dict
):
    """
    Create PyTorch Geometric graphs for each time step per event, including multi-step node and edge targets.

    Parameters:
      - merged_node_features: {event: {node_name: feature_list_per_time_step}}
      - merged_edge_features: {event: {edge_name: feature_list_per_time_step}}
      - merged_node_targets: {event: {node_name: target_list_per_time_step}}
          (flattened; e.g. for max_steps_ahead=2, order: [head(t+1), head(t+2), inflow(t+1), inflow(t+2)])
      - merged_edge_targets: {event: {edge_name: target_list_per_time_step}}
      - edge_index: PyTorch tensor (constant across all events)
      - node_mapping: Dictionary mapping node names to indices (constant across all events)
      - ordered_edge_info_list: Ordered list of edge info dictionaries corresponding to edge_index.
      - n_time_step: Number of past time steps used in input.
      - max_steps_ahead: Number of future steps to predict.
      - num_time_steps_per_event: {event: total number of time steps available}
      - future_rainfall_dict: {event: {node: list of future rainfall vectors (flattened, length max_steps_ahead*2)}}
           (for nodes)
      - future_edge_rainfall_dict: {event: {edge_name: list of future rainfall vectors (flattened, length max_steps_ahead*2)}}
           (for edges)

    Returns:
      - graphs_per_event: {event: [Graph1, Graph2, ..., GraphT]}
         Each graph has:
             x: [num_nodes, num_node_features]
             y: node targets, reshaped to [num_nodes, max_steps_ahead, num_target_features]
             edge_attr: [num_edges, num_edge_features]
             y_edges: edge targets, reshaped to [num_edges, max_steps_ahead, num_edge_target_features]
             future_rainfall: node future rainfall, reshaped to [num_nodes, max_steps_ahead, 2]
             future_edge_rainfall: edge future rainfall, reshaped to [num_edges, max_steps_ahead, 2]
    """
    from torch_geometric.data import Data
    import torch

    graphs_per_event = {}

    for event in merged_node_features.keys():
        print(f"\n🔄 Creating graphs for event: {event}")

        num_time_steps = num_time_steps_per_event.get(event, 0)
        valid_time_steps = num_time_steps - (n_time_step + max_steps_ahead) + 1

        if valid_time_steps < 1:
            print(f"⚠ Skipping event '{event}': Not enough valid time steps ({valid_time_steps}).")
            continue

        event_graphs = []

        for t in range(valid_time_steps):
            # Extract node features at time t.
            node_features = [merged_node_features[event][node][t] for node in node_mapping.keys()]
            # Extract node targets (flattened).
            node_targets_flat = [merged_node_targets[event][node][t] for node in node_mapping.keys()]
            # Extract edge features at time t.
            edge_features = [merged_edge_features[event][edge_info["conduit_name"]][t]
                             for edge_info in ordered_edge_info_list]
            # Extract edge targets (flattened).
            edge_targets_flat = [merged_edge_targets[event][edge_info["conduit_name"]][t]
                                 for edge_info in ordered_edge_info_list]

            # Retrieve future rainfall for nodes (flattened vector, length max_steps_ahead*2).
            future_node_rainfall_flat = [future_rainfall_dict[event].get(node, [0.0]*(max_steps_ahead*2))[t]
                                          for node in node_mapping.keys()]

            # Retrieve future rainfall for edges (flattened vector, length max_steps_ahead*2).
            future_edge_rainfall_flat = [future_edge_rainfall_dict[event].get(edge_info["conduit_name"],
                                                                               [[0.0]*(max_steps_ahead*2)])[t]
                                          for edge_info in ordered_edge_info_list]

            # Convert lists to PyTorch tensors.
            x = torch.tensor(node_features, dtype=torch.float)  # [num_nodes, num_node_features]

            # Reshape node targets:
            y_nodes_flat = torch.tensor(node_targets_flat, dtype=torch.float)  # shape: [num_nodes, flat_dim]
            if y_nodes_flat.dim() == 2:
                num_nodes, flat_dim = y_nodes_flat.shape
                num_target_features = flat_dim // max_steps_ahead
                # Reshape to [num_nodes, num_target_features, max_steps_ahead] then permute.
                y_nodes = y_nodes_flat.view(num_nodes, num_target_features, max_steps_ahead).permute(0, 2, 1)
            else:
                y_nodes = y_nodes_flat

            # Edge features.
            edge_attr = torch.tensor(edge_features, dtype=torch.float)  # [num_edges, num_edge_features]

            # Reshape edge targets similarly.
            y_edges_flat = torch.tensor(edge_targets_flat, dtype=torch.float)
            if y_edges_flat.dim() == 2:
                num_edges, flat_dim = y_edges_flat.shape
                num_edge_target_features = flat_dim // max_steps_ahead
                y_edges = y_edges_flat.view(num_edges, num_edge_target_features, max_steps_ahead).permute(0, 2, 1)
            else:
                y_edges = y_edges_flat

            # Process future node rainfall:
            future_node_rainfall_tensor = torch.tensor(future_node_rainfall_flat, dtype=torch.float)  # [num_nodes, max_steps_ahead*2]
            if future_node_rainfall_tensor.dim() == 2:
                num_nodes, flat_dim = future_node_rainfall_tensor.shape
                num_rain_features = flat_dim // max_steps_ahead  # should be 2.
                # Reshape to [num_nodes, num_rain_features, max_steps_ahead] and permute.
                future_node_rainfall_tensor = future_node_rainfall_tensor.view(num_nodes, num_rain_features, max_steps_ahead).permute(0, 2, 1)
                # Now shape: [num_nodes, max_steps_ahead, 2]

            # Process future edge rainfall:
            future_edge_rainfall_tensor = torch.tensor(future_edge_rainfall_flat, dtype=torch.float)  # [num_edges, max_steps_ahead*2]
            if future_edge_rainfall_tensor.dim() == 2:
                num_edges, flat_dim = future_edge_rainfall_tensor.shape
                num_rain_features = flat_dim // max_steps_ahead  # should be 2.
                future_edge_rainfall_tensor = future_edge_rainfall_tensor.view(num_edges, num_rain_features, max_steps_ahead).permute(0, 2, 1)
                # Now shape: [num_edges, max_steps_ahead, 2]

            # Node type tensor (0 = type_a, 1 = type_b)
            node_type_tensor = torch.tensor(
                [0 if node_type_dict.get(node, "type_a") == "type_a" else 1 for node in node_mapping.keys()],
                dtype=torch.long
            )

            # Construct the PyG graph object.
            graph = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y_nodes,
                future_rainfall=future_node_rainfall_tensor  # Node-level future rainfall.
            )
            graph.y_edges = y_edges  # Edge targets.
            graph.future_edge_rainfall = future_edge_rainfall_tensor  # Edge-level future rainfall.
            graph.node_type = node_type_tensor  # 0 → type_a (has subcatchments) & 1 → type_b (no subcatchments)


            # Add additional attributes.
            graph.time_index = t  # Each graph remembers its starting time step.
            graph.event_name = event  # Store the event name in each graph.
            graph.node_names = list(node_mapping.keys())  # Matches node order in x.
            graph.edge_names = [info["conduit_name"] for info in ordered_edge_info_list]  # Matches edge_index order.

            event_graphs.append(graph)

        graphs_per_event[event] = event_graphs
        print(f"Completed {len(event_graphs)} graphs for event '{event}'.")

    return graphs_per_event


# Create initial graphs from test graphs
def extract_initial_graphs_and_attach_rainfall(graph_data_per_event,
                                                 augmented_rainfall_data,
                                                 augmented_edge_rainfall_data,
                                                 node_mapping,
                                                 edge_info_list,
                                                 n_time_steps):
    """
    Extracts the initial graph (t=0) for each event and attaches the full-length rainfall data
    for both nodes and edges.

    Parameters:
    - graph_data_per_event (dict): Dictionary containing test graphs per event.
    - augmented_rainfall_data (dict): Dictionary containing full-length rainfall (including outfalls) for nodes.
         Expected to be a dict with event keys and values being dicts with keys "rainfall" and "acc_rainfall"
         (both as DataFrames with node names as columns).
    - augmented_edge_rainfall_data (dict): Dictionary containing full-length rainfall for edges.
         Expected to be a dict with event keys and values being dicts with keys "rainfall" and "acc_rainfall"
         (both as DataFrames with edge (conduit) names as columns).
    - node_mapping (dict): Mapping between real node names and numerical indices in the graph.
    - edge_info_list (list): Ordered list of edge info dictionaries. Each dict contains keys:
         "from_node", "to_node", "from_node_idx", "to_node_idx", and "conduit_name".
    - n_time_steps (int): Number of past time steps already included in node features.

    Returns:
    - initial_graphs_per_event (dict): Dictionary with modified initial graphs. Each graph now has the following additional attributes:
         - full_rainfall: Tensor of full-length node rainfall (after removing the first n_time_steps).
         - full_acc_rainfall: Tensor of full-length node accumulative rainfall (after removing the first n_time_steps).
         - full_edge_rainfall: Tensor of full-length edge rainfall (after removing the first n_time_steps).
         - full_edge_acc_rainfall: Tensor of full-length edge accumulative rainfall (after removing the first n_time_steps).
    """
    import torch

    initial_graphs_per_event = {}

    # Create an edge mapping from conduit name to edge index using the edge_info_list.
    edge_mapping = {edge_info["conduit_name"]: idx for idx, edge_info in enumerate(edge_info_list)}

    for event, event_graphs in graph_data_per_event.items():
        if not event_graphs:
            print(f"Warning: No graphs found for event '{event}'")
            continue

        # Extract the initial graph (t=0)
        initial_graph = event_graphs[0]

        # ===== Node-Level Full Rainfall =====
        if event in augmented_rainfall_data:
            event_rainfall = augmented_rainfall_data[event].get("rainfall")
            event_acc_rainfall = augmented_rainfall_data[event].get("acc_rainfall")

            if event_rainfall is None or event_acc_rainfall is None:
                print(f"Warning: Missing node rainfall or acc_rainfall data for event '{event}'")
                continue

            num_nodes = len(node_mapping)
            num_timesteps = len(event_rainfall)  # total time steps
            full_rainfall = torch.zeros((num_nodes, num_timesteps), dtype=torch.float)
            full_acc_rainfall = torch.zeros((num_nodes, num_timesteps), dtype=torch.float)

            # Map each node's full rainfall based on node_mapping.
            for node_name, node_idx in node_mapping.items():
                if node_name in event_rainfall.columns:
                    full_rainfall[node_idx, :] = torch.tensor(event_rainfall[node_name].values, dtype=torch.float)
                if node_name in event_acc_rainfall.columns:
                    full_acc_rainfall[node_idx, :] = torch.tensor(event_acc_rainfall[node_name].values, dtype=torch.float)

            # Remove the first n_time_steps as these are already part of the node features.
            if full_rainfall.shape[1] > n_time_steps:
                full_rainfall = full_rainfall[:, n_time_steps:]
            if full_acc_rainfall.shape[1] > n_time_steps:
                full_acc_rainfall = full_acc_rainfall[:, n_time_steps:]
        else:
            print(f"Warning: Missing node rainfall data for event '{event}'")
            continue

        # Attach node-level full rainfall data to the graph.
        initial_graph.full_rainfall = full_rainfall
        initial_graph.full_acc_rainfall = full_acc_rainfall

        # ===== Edge-Level Full Rainfall =====
        if event in augmented_edge_rainfall_data:
            event_edge_rainfall = augmented_edge_rainfall_data[event].get("rainfall")
            event_edge_acc_rainfall = augmented_edge_rainfall_data[event].get("acc_rainfall")

            if event_edge_rainfall is None or event_edge_acc_rainfall is None:
                print(f"Warning: Missing edge rainfall or acc_rainfall data for event '{event}'")
                num_edges = len(edge_mapping)
                full_edge_rainfall = torch.empty((num_edges, 0))
                full_edge_acc_rainfall = torch.empty((num_edges, 0))
            else:
                num_edges = len(edge_mapping)
                num_edge_timesteps = len(event_edge_rainfall)  # total time steps in edge data
                full_edge_rainfall = torch.zeros((num_edges, num_edge_timesteps), dtype=torch.float)
                full_edge_acc_rainfall = torch.zeros((num_edges, num_edge_timesteps), dtype=torch.float)

                # Use the edge_mapping (like node_mapping) to attach edge rainfall.
                for conduit_name, idx in edge_mapping.items():
                    if conduit_name in event_edge_rainfall.columns:
                        full_edge_rainfall[idx, :] = torch.tensor(event_edge_rainfall[conduit_name].values, dtype=torch.float)
                    if conduit_name in event_edge_acc_rainfall.columns:
                        full_edge_acc_rainfall[idx, :] = torch.tensor(event_edge_acc_rainfall[conduit_name].values, dtype=torch.float)

                if full_edge_rainfall.shape[1] > n_time_steps:
                    full_edge_rainfall = full_edge_rainfall[:, n_time_steps:]
                if full_edge_acc_rainfall.shape[1] > n_time_steps:
                    full_edge_acc_rainfall = full_edge_acc_rainfall[:, n_time_steps:]
        else:
            print(f"Warning: Missing edge rainfall data for event '{event}'")
            num_edges = len(edge_mapping)
            full_edge_rainfall = torch.empty((num_edges, 0))
            full_edge_acc_rainfall = torch.empty((num_edges, 0))

        # Attach edge-level full rainfall data to the graph.
        initial_graph.full_edge_rainfall = full_edge_rainfall
        initial_graph.full_edge_acc_rainfall = full_edge_acc_rainfall

        # Save the modified initial graph.
        initial_graphs_per_event[event] = [initial_graph]

        print(f" Event: {event} | Graph Shape: {initial_graph.x.shape} | "
              f"Node Rainfall: {full_rainfall.shape} | Node Acc Rainfall: {full_acc_rainfall.shape} | "
              f"Edge Rainfall: {full_edge_rainfall.shape} | Edge Acc Rainfall: {full_edge_acc_rainfall.shape}")

    print(f"Extracted and modified {len(initial_graphs_per_event)} initial graphs with full-length node and edge rainfall data.")
    return initial_graphs_per_event


# Function to save graph
def save_graphs(graph_data_per_event, save_dir):
    """
    Save precomputed graphs for each event.

    Parameters:
    - graph_data_per_event: Dictionary of graph data per event.
    - save_dir: Directory where graphs will be saved.
    """
    os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist

    for event, graphs in graph_data_per_event.items():
        file_path = os.path.join(save_dir, f"{event}_graphs.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(graphs, f)
        print(f"Saved graphs for event '{event}' to '{file_path}'")


# Function to load graph
def load_graphs(save_dir):
    """
    Load precomputed graphs from disk.

    Parameters:
    - save_dir: Directory where graphs are saved.

    Returns:
    - graph_data_per_event: Dictionary of loaded graph data.
    """
    graph_data_per_event = {}
    for file_name in os.listdir(save_dir):
        if file_name.endswith("_graphs.pkl"):
            event = file_name.split("_graphs.pkl")[0]
            file_path = os.path.join(save_dir, file_name)
            with open(file_path, "rb") as f:
                graph_data_per_event[event] = pickle.load(f)
            print(f"Loaded graphs for event '{event}' from '{file_path}'")
    return graph_data_per_event



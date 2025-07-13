import torch
import yaml
from data_processing_conduits import ( # Note: Changed 'data_processing' to 'data_processing_conduits'
    load_constant_features, load_dynamic_data, transform_subcatchment_dynamic_data, convert_conduits_to_graph,
    create_dynamic_node_features, load_graphs, save_graphs, extract_initial_graphs_and_attach_rainfall,
    calculate_relative_coordinates, calculate_rainfall_features_for_edges, transform_edge_full_rainfall_features
)

# Load config file
CONFIG_PATH = "config.yml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Extract configurations
data_config = config["data"]
dynamic_features = config["dynamic_features"]
constant_features = config["constant_features"]
n_time_steps = config["model"]["n_time_step"]
max_steps_ahead = config["model"]["max_steps_ahead"]
computed_features = config["computed_features"]

print("--- Preparing Initial Test Graphs for Rollout ---")

# Load static and dynamic data
print("Loading constant features...")
static_data, original_subcatchment_areas = load_constant_features(
    data_config["constant_data_dir"], config=constant_features, normalize=data_config["normalize_features"]
)
node_coordinates = static_data["node_coordinates"]

print("Loading dynamic data...")
dynamic_data, raw_dynamic_data, outfall_dynamic_data, raw_outfall_dynamic_data, time_steps_per_event, global_min_max, flow_direction_data = load_dynamic_data(
    data_config["dynamic_data_dir"],
    config=dynamic_features,
    outfall_names=static_data["outfall_names"],
    apply_normalization=data_config["normalize_features"],
    save_min_max_path=data_config["global_min_max_dir"],
    use_abs_flow=True
)

print("Transforming subcatchment dynamic data to node-level...")
transformed_dynamic_data = transform_subcatchment_dynamic_data(static_data, dynamic_data, original_subcatchment_areas)

# Create future rainfall dictionary for nodes (including both rainfall and acc_rainfall)
_dynamic_node_features, _dynamic_node_targets, future_rainfall_dict = create_dynamic_node_features(
    transformed_dynamic_data, dynamic_data, outfall_dynamic_data, n_time_steps, max_steps_ahead, time_steps_per_event
)

# Calculate relative coordinates for conduits
updated_conduits_df, edge_to_nodes = calculate_relative_coordinates(
    static_data["conduits"], static_data["junctions"], static_data["outfalls"], computed_config=computed_features
)

# Transform full-length rainfall features for edges
edge_features_full_rainfall = transform_edge_full_rainfall_features(edge_to_nodes, transformed_dynamic_data)

# Calculate rainfall features for edges (for the sliding window input, not the full future)
# This part might be redundant if only future_rainfall_dict and edge_features_full_rainfall are needed for extract_initial_graphs_and_attach_rainfall.
# Keeping it for now as it was in original, but could be reviewed if it's not strictly necessary here.
_rainfall_features = calculate_rainfall_features_for_edges(node_dynamic_features=_dynamic_node_features,
                                                           edge_to_nodes=edge_to_nodes,
                                                           num_time_steps_per_event=time_steps_per_event,
                                                           n_time_step=n_time_steps,
                                                           max_steps_ahead=max_steps_ahead)


# Convert conduits to graph to get node_mapping and edge_info_list
_graph, _edge_index, node_mapping, edge_info_list = convert_conduits_to_graph(updated_conduits_df, node_coordinates)

# Load test graphs (these are the full event graphs, from data_processing)
print("Loading pre-split test graphs...")
test_graph_data = load_graphs(config["data"]["test_graph_dir"])
print(f"Loaded {len(test_graph_data)} full test events.")

# Attach full-length dynamic rainfall (nodes and edges) to initial test graphs
print("Attaching full-length dynamic rainfall to initial test graphs for rollout...")
initial_test_graphs = extract_initial_graphs_and_attach_rainfall(
    test_graph_data,
    augmented_rainfall_data=transformed_dynamic_data, # For nodes, using computed node rainfall features
    augmented_edge_rainfall_data=edge_features_full_rainfall,  # For edges, using computed edge rainfall features
    node_mapping=node_mapping,
    edge_info_list=edge_info_list,
    n_time_steps=n_time_steps
)
print(f"Processed {len(initial_test_graphs)} initial test graphs.")

# Save the updated graphs to the designated directory for rollout
save_graphs(initial_test_graphs, config["data"]["initial_test_graph_dir"])
print("--- Initial test graphs with full dynamic rainfall (nodes and edges) saved successfully! ---")
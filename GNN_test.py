import os
import torch
import pickle
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
import random

from GNN_models import EncodeProcessDecodeDual as StandardGNN
from rollout_prediction import rollout_prediction, prepare_node_input, prepare_edge_input, rollout_update_node, rollout_update_node
from data_processing import (
    load_graphs, denormalize_dynamic_data, load_global_min_max,
    calculate_relative_coordinates, convert_conduits_to_graph, load_constant_features
)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def convert_rollout_results_all(raw_results, domain, feature_names):
    converted = {}
    for event, event_data in raw_results.items():
        if domain not in event_data:
            continue
        all_graphs = event_data[domain]
        rollout_list = []
        for graph_rollout in all_graphs:
            # If graph_rollout is already a tensor, append directly
            rollout_list.append(graph_rollout)

        if len(rollout_list) == 0:
            continue
        full_rollout = torch.cat(rollout_list, dim=0)  # (total_time_steps, num_entities, 2)
        full_rollout = full_rollout.transpose(0, 1)  # (num_entities, total_time_steps, 2)

        feat_dict = {}
        for i, feat in enumerate(feature_names):
            feat_array = full_rollout[:, :, i].cpu().numpy()
            df = pd.DataFrame(feat_array)
            feat_dict[feat] = df
        converted[event] = feat_dict
    return converted
#########################################
# Main Routine
#########################################
if __name__ == "__main__":
    set_random_seed(42)

    CONFIG_PATH = "config.yml"
    print(f"Loading configuration from {CONFIG_PATH}...")
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    print("Configuration loaded.\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    def load_graphs_grouped(graph_dir):
        graphs_dict = {}
        print(f"Loading test graphs from directory: {graph_dir}")
        for file in os.listdir(graph_dir):
            if file.endswith("_graphs.pkl"):
                event_name = file.split("_graphs.pkl")[0]
                file_path = os.path.join(graph_dir, file)
                try:
                    with open(file_path, "rb") as f:
                        graphs = pickle.load(f)
                    graphs_dict[event_name] = graphs
                    print(f"  Loaded {len(graphs)} graphs for event: {event_name}")
                except Exception as e:
                    print(f"Skipping file {file} due to error: {e}")
        print(f"Total events loaded: {len(graphs_dict)}\n")
        return graphs_dict

    test_graphs = load_graphs_grouped(config["data"]["initial_test_graph_dir"])
    if len(test_graphs) == 0:
        raise ValueError("No test graphs found!")

    # Determine effective feature dimensions from a sample graph.
    first_event = list(test_graphs.keys())[0]
    sample_graph = test_graphs[first_event][0]
    total_node_features = sample_graph.x.shape[1]
    total_edge_features = sample_graph.edge_attr.shape[1]

    # Here we assume that dynamic node features include [rainfall, acc_rainfall, head, inflow].
    num_dyn_node_features = 4
    constant_dim_node = total_node_features - (config["model"]["n_time_step"] * num_dyn_node_features)
    nnode_in_features = constant_dim_node + ((config["model"]["n_time_step"] - 1) * num_dyn_node_features)

    # For edges, assume dynamic edge features include [rainfall, acc_rainfall, diff_head, diff_inflow, flow, depth].
    num_dyn_edge_features = 6
    constant_dim_edge = total_edge_features - (config["model"]["n_time_step"] * num_dyn_edge_features)
    nedge_in_features = constant_dim_edge + ((config["model"]["n_time_step"] - 1) * num_dyn_edge_features)

    print(f"Effective input dimensions: nnode_in_features={nnode_in_features}, nedge_in_features={nedge_in_features}")

    # Calculate removal Indices for Nodes type_b
    junction_features = config["constant_features"]["junctions"]
    subcatchment_features = config["constant_features"]["subcatchments"]

    # Find lengths
    junction_len = len(junction_features)
    subcatchment_len = len(subcatchment_features)

    # Calculate removal indices (subcatchment feature positions)
    removal_indices = list(range(junction_len, junction_len + subcatchment_len))

    # Removal_indices_ Nodes type_b
    config['model']['removal_indices'] = removal_indices
    print("removal_indices:", removal_indices)

    if config["data"]["normalize_features"]:
        print("Loading global min-max values for normalization...")
        global_min_max = load_global_min_max(config["data"]["global_min_max_dir"])
    else:
        global_min_max = None
    print("Global min-max loaded.\n")

    from data_processing_conduits import calculate_relative_coordinates, convert_conduits_to_graph, load_constant_features

    static_data, _ = load_constant_features(
        config["data"]["constant_data_dir"],
        config=config["constant_features"],
        normalize=config["data"]["normalize_features"]
    )
    node_coordinates = static_data["node_coordinates"]
    updated_conduits_df, edge_to_nodes = calculate_relative_coordinates(
        static_data["conduits"],
        static_data["junctions"],
        static_data["outfalls"],
        computed_config=config["computed_features"]
    )
    full_graph, _, node_mapping, edge_info_list = convert_conduits_to_graph(updated_conduits_df, node_coordinates)
    print(f"Computed edge_info_list with {len(edge_info_list)} entries.\n")


    def load_trained_model(config, device, nnode_in_features, nedge_in_features):
        """
        Load the best trained model (Standard GNN, GNN-KAN, or FourierKAN-GNN) based on the configuration.
        """
        print("Loading best trained model...")

        model_type = config['model']['model_type']


        if model_type == "StandardGNN":
            print("Using Standard GNN model for testing")
            model = StandardGNN(
                nnode_in_features=nnode_in_features,
                nnode_out_features=2,
                nedge_in_features=nedge_in_features,
                nedge_out_features=2,
                latent_dim=config['model']['latent_dim'],
                nmessage_passing_steps=config['model']['nmessage_passing_steps'],
                nmlp_layers=config['model']['nmlp_layers'],
                mlp_hidden_dim=config['model']['mlp_hidden_dim'],
                residual=config['model']['residual'],
                n_time_steps=config['model']['n_time_step'],
                removal_indices=config['model']['removal_indices']
            ).to(device)

        else:
            # Placeholder for other models if they are added in the future
            raise ValueError(f"Unknown model type: {model_type}")

        checkpoint_path = os.path.join(config["checkpoint"]["save_dir"], "best_swmm_gnn.pth")
        print(f"Loading checkpoint from: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        print("Model loaded and set to evaluation mode.\n")

        return model

    model = load_trained_model(config, device, nnode_in_features, nedge_in_features)

    print("Starting rollout predictions...\n")
    rollout_results = {}
    n_time_steps = config["model"]["n_time_step"]

    for event, graph_list in test_graphs.items():
        print(f"Rolling out event: {event} with {len(graph_list)} graph(s).")
        event_node_rollouts = []
        event_edge_rollouts = []
        for idx, graph in enumerate(graph_list):
            if len(graph_list) > 1:
                print(f"  Rolling out graph {idx + 1}/{len(graph_list)} for event {event}...")
            graph = graph.to(device)
            # Add this print to check rainfall shape
            print(f"    - Full Rainfall Shape for event '{event}': {graph.full_rainfall.shape}")
            rollout_steps = graph.full_rainfall.shape[1] - 1
            print(
                f"    - Full Rainfall Steps: {graph.full_rainfall.shape[1]}, Rollout Steps (adjusted): {rollout_steps}")

            node_roll, edge_roll, updated_graph = rollout_prediction(
                model, graph, rollout_steps, config, edge_info_list, device, n_time_steps
            )
            event_node_rollouts.append(node_roll)
            event_edge_rollouts.append(edge_roll)
        rollout_results[event] = {
            "junctions": event_node_rollouts,
            "conduits": event_edge_rollouts
        }
    print("Rollout predictions completed.\n")

    junction_results = convert_rollout_results_all(rollout_results, "junctions", ["depth", "inflow"])
    conduit_results = convert_rollout_results_all(rollout_results, "conduits", ["flow", "depth"])

    rollout_converted = {}
    for event in rollout_results.keys():
        rollout_converted[event] = {
            "junctions": junction_results.get(event, {}),
            "conduits": conduit_results.get(event, {})
        }

    if config["data"]["normalize_features"]:
        print("Denormalizing rollout predictions...")
        rollout_converted = denormalize_dynamic_data(rollout_converted, {}, global_min_max)
        print("Denormalization complete.\n")

    SAVE_DIR = config["data"]["gnn_predictions_dir"]
    os.makedirs(SAVE_DIR, exist_ok=True)
    rollout_save_path = os.path.join(SAVE_DIR, "gnn_rollout_predictions.pkl")
    with open(rollout_save_path, "wb") as f:
        pickle.dump(rollout_converted, f)
    print(f"✅ Rollout predictions saved at: {rollout_save_path}")


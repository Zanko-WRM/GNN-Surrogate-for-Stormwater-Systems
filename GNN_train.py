import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch_geometric.loader import DataLoader
import wandb
import os
import yaml
import shutil
import random
import pickle

from data_processing import (
    load_constant_features, process_and_save_conduit_with_area, load_dynamic_data, transform_subcatchment_dynamic_data,
    create_constant_node_features, create_dynamic_node_features, merge_node_features,
    create_constant_edge_features, create_dynamic_edge_features, merge_edge_features, calculate_rainfall_features_for_edges,
    convert_conduits_to_graph, build_graph_per_event, save_graphs, load_graphs, calculate_relative_coordinates, calculate_diff_features_for_edges,
    extract_initial_graphs_and_attach_rainfall
)

from GNN_models import EncodeProcessDecodeDual as StandardGNN
from GNN_utils import train_pushforward, validate_pushforward

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(directory):
    """Creates a directory if it does not exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

# Load config file
CONFIG_PATH = "config.yml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

data_config = config["data"]
dynamic_features = config["dynamic_features"]
constant_features = config["constant_features"]
computed_features = config["computed_features"]
n_time_steps = config["model"]["n_time_step"]
max_steps_ahead = config["model"]["max_steps_ahead"]

# Preprocess conduits.txt to add Area column
original_conduits_file = data_config["original_conduits_file"]
constant_data_dir = data_config["constant_data_dir"]
process_and_save_conduit_with_area(original_conduits_file, constant_data_dir)

# Ensure necessary directories exist
for key in ["precomputed_graph_dir", "train_graph_dir", "val_graph_dir", "test_graph_dir"]:
    os.makedirs(data_config[key], exist_ok=True)

# Step 1: Check if we should use precomputed graphs
if data_config["use_precomputed_graphs"] and os.listdir(data_config["precomputed_graph_dir"]):
    print("Using precomputed graphs.")
else:
    print("Generating new graphs...")


    # Load static and dynamic data
    static_data, original_subcatchment_areas = load_constant_features(data_config["constant_data_dir"], config=constant_features,
                                         normalize=data_config["normalize_features"])
    node_coordinates = static_data["node_coordinates"]

    dynamic_data, raw_dynamic_data, outfall_dynamic_data, raw_outfall_dynamic_data, time_steps_per_event, global_min_max, flow_direction_data = load_dynamic_data(
        data_config["dynamic_data_dir"],
        config=dynamic_features,
        outfall_names=static_data["outfall_names"],
        apply_normalization=data_config["normalize_features"],
        save_min_max_path=data_config["global_min_max_dir"],
        use_abs_flow=True
    )
    transformed_dynamic_data = transform_subcatchment_dynamic_data(static_data, dynamic_data, original_subcatchment_areas)

    constant_node_features, node_type_dict = create_constant_node_features(static_data, dynamic_data)
    dynamic_node_features, dynamic_node_targets, future_rainfall_dict = create_dynamic_node_features(
        transformed_dynamic_data, dynamic_data, outfall_dynamic_data, n_time_steps, max_steps_ahead, time_steps_per_event
    )

    merged_node_features, merged_node_targets = merge_node_features(
        constant_node_features, dynamic_node_features, dynamic_node_targets
    )

    updated_conduits_df, edge_to_nodes = calculate_relative_coordinates(
        static_data["conduits"], static_data["junctions"], static_data["outfalls"], computed_config=computed_features
    )

    rainfall_features = calculate_rainfall_features_for_edges(node_dynamic_features=dynamic_node_features,
                                                              edge_to_nodes=edge_to_nodes,
                                                              num_time_steps_per_event=time_steps_per_event,
                                                              n_time_step=n_time_steps,
                                                              max_steps_ahead=max_steps_ahead)

    dynamic_node_features_raw, dynamic_node_targets_raw, future_rainfall_dict_raw = create_dynamic_node_features(
        transformed_dynamic_data, raw_dynamic_data, raw_outfall_dynamic_data, n_time_steps, max_steps_ahead,
        time_steps_per_event
    )

    diff_features, dif_global_min_max = calculate_diff_features_for_edges(
        node_dynamic_features=dynamic_node_features_raw,
        edge_to_nodes=edge_to_nodes,
        num_time_steps_per_event=time_steps_per_event,
        n_time_step=n_time_steps,
        max_steps_ahead=max_steps_ahead,
        apply_normalization=data_config["normalize_features"],
        save_min_max_path=data_config["global_dif_min_max_dir"]
    )

    constant_edge_features = create_constant_edge_features(updated_conduits_df, constant_features, computed_features)
    dynamic_edge_features, dynamic_edge_targets, future_edge_rainfall_dict = create_dynamic_edge_features(dynamic_data,
                                                                                                          dynamic_features,
                                                                                                          n_time_steps,
                                                                                                          max_steps_ahead,
                                                                                                          time_steps_per_event,
                                                                                                          constant_edge_features,
                                                                                                          diff_features,
                                                                                                          edge_to_nodes,
                                                                                                          rainfall_features,
                                                                                                          future_rainfall_dict)

    merged_edge_features, merged_edge_targets = merge_edge_features(
                 constant_edge_features, dynamic_edge_features, dynamic_edge_targets, time_steps_per_event
             )

    graph, edge_index, node_mapping, edge_info_list = convert_conduits_to_graph(updated_conduits_df, node_coordinates)
    print(f"Edge Index Shape: {edge_index.shape}")
    print(f"First 5 Edges:\n{edge_index[:, :5] if edge_index.numel() > 0 else 'No edges!'}")

    graph_data_per_event = build_graph_per_event(
                            merged_node_features, merged_edge_features, merged_node_targets, merged_edge_targets,
                            edge_index, node_mapping, edge_info_list, n_time_steps, max_steps_ahead, time_steps_per_event,
                            future_rainfall_dict, future_edge_rainfall_dict, node_type_dict
    )

    # Save graphs
    save_graphs(graph_data_per_event, data_config["precomputed_graph_dir"])
    print("Graphs generated and saved successfully.")

# Step 2: Ensure graphs are available before splitting
all_graphs = [f for f in os.listdir(data_config["precomputed_graph_dir"]) if f.endswith(".pkl")]
if not all_graphs:
    raise FileNotFoundError("No graphs found in precomputed_graph_dir. Ensure graphs are generated before splitting.")

# Build a dictionary that groups event files by the number of time steps
event_groups = {}
for file in all_graphs:
    file_path = os.path.join(data_config["precomputed_graph_dir"], file)
    try:
        with open(file_path, "rb") as f:
            event = pickle.load(f)  # each event is a list of graphs
        num_time_steps = len(event)
        # Group by the number of time steps
        if num_time_steps not in event_groups:
            event_groups[num_time_steps] = []
        event_groups[num_time_steps].append(file)
    except Exception as e:
        print(f"Error reading {file}: {e}")

# From each group, randomly pick 2 events for the test set.
set_random_seed(42)  # For reproducibility
test_event_files = []
remaining_event_files = []
for ts, files in event_groups.items():
    if len(files) >= 2:
        test_samples = random.sample(files, 2)
        test_event_files.extend(test_samples)
        remaining = [f for f in files if f not in test_samples]
        remaining_event_files.extend(remaining)
    else:
        # If a group has fewer than 2 events, assign them to remaining events.
        remaining_event_files.extend(files)

# ----------------------------------------------------------------
# Copy test events as whole files into the test directory.
# ----------------------------------------------------------------
def copy_files(file_list, destination):
    for file in file_list:
        src = os.path.join(data_config["precomputed_graph_dir"], file)
        dst = os.path.join(destination, file)
        shutil.copy(src, dst)

copy_files(test_event_files, data_config["test_graph_dir"])
print(f"\nCopied {len(test_event_files)} test event files to {data_config['test_graph_dir']}")

# ----------------------------------------------------------------
# Now, for the remaining events, we perform graph-based splitting:
# 1. Load each remaining event and flatten into individual graphs.
# 2. Shuffle and split these graphs into train and validation sets.
# 3. Save each individual graph as a separate .pkl file.
# ----------------------------------------------------------------
remaining_graphs = []  # Each element will be a tuple: (graph, source_event_file, graph_index)
for file in remaining_event_files:
    file_path = os.path.join(data_config["precomputed_graph_dir"], file)
    try:
        with open(file_path, "rb") as f:
            event = pickle.load(f)
        for i, graph in enumerate(event):
            remaining_graphs.append((graph, file, i))
    except Exception as e:
        print(f"Error loading remaining event {file}: {e}")

print(f"\nTotal number of individual graphs from remaining events: {len(remaining_graphs)}")

# Graph-based splitting for train and validation sets:
set_random_seed(42)
random.shuffle(remaining_graphs)
num_remaining = len(remaining_graphs)
train_split = int(0.85 * num_remaining) # 85% for training

train_graphs = remaining_graphs[:train_split]
val_graphs = remaining_graphs[train_split:] # Remaining 15% for validation

print(f"\nSplit Summary:")
print(f"Train graphs: {len(train_graphs)}")
print(f"Validation graphs: {len(val_graphs)}")
print(f"Test events (as full events): {len(test_event_files)}")

# Function to save individual graphs as separate .pkl files.
def save_graph_list(graph_list, destination, prefix):
    ensure_dir(destination)
    for idx, (graph, source_file, graph_idx) in enumerate(graph_list):
        # Construct a filename using a prefix, the source event file name (without extension) and the graph index.
        filename = f"{prefix}_{source_file.replace('.pkl','')}_graph{graph_idx}.pkl"
        file_path = os.path.join(destination, filename)
        with open(file_path, "wb") as f:
            pickle.dump(graph, f)

# Save the split graphs to the corresponding directories.
save_graph_list(train_graphs, data_config["train_graph_dir"], "train")
save_graph_list(val_graphs, data_config["val_graph_dir"], "val")

print(f"\nSaved train graphs to: {data_config['train_graph_dir']}")
print(f"Saved validation graphs to: {data_config['val_graph_dir']}")

# Recompute edge_info_list for training
# (This is necessary because edge_info_list is needed in train/validate functions)
static_data, _ = load_constant_features(data_config["constant_data_dir"], config=constant_features, normalize=data_config["normalize_features"])
node_coordinates = static_data["node_coordinates"]
updated_conduits_df, edge_to_nodes = calculate_relative_coordinates(
    static_data["conduits"], static_data["junctions"], static_data["outfalls"], computed_config=computed_features)
_, edge_index, node_mapping, edge_info_list = convert_conduits_to_graph(updated_conduits_df, node_coordinates)
print(f"(Recomputed) Edge Info List length: {len(edge_info_list)}")


# Save model checkpoint
def save_checkpoint(epoch, model, optimizer, scheduler, best_val_loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss
    }, path)

# Load model checkpoint
def load_checkpoint(path, model, optimizer, scheduler):
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        return epoch, best_val_loss
    else:
        return 0, float('inf')

# Count model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Load precomputed graphs
def load_graphs(graph_dir):
    graphs = []
    for file in os.listdir(graph_dir):
        if file.endswith(".pkl"):
            file_path = os.path.join(graph_dir, file)
            try:
                with open(file_path, "rb") as f:
                    graphs.append(pickle.load(f))
                    #graphs.extend(pickle.load(f))
            except Exception as e:
                print(f"Skipping corrupt file {file}: {e}")
    return graphs

#####
sample_graph = load_graphs(data_config["train_graph_dir"])[0]
total_node_features = sample_graph.x.shape[1]
total_edge_features = sample_graph.edge_attr.shape[1]

# node dimention
num_dyn_node_features = 4 # rainfall, acc_rainfall, inflow, depth
constant_dim = total_node_features - (n_time_steps * num_dyn_node_features)
nnode_in_features = constant_dim + ((n_time_steps-1) * num_dyn_node_features)

# dge_dimention
num_dyn_edge_features = 6 # rainfall, acc_rainfall, dif_depth, dif_inflow, flow, depth
constant_dim = total_edge_features - (n_time_steps * num_dyn_edge_features)
nedge_in_features = constant_dim + ((n_time_steps-1) * num_dyn_edge_features)

print(f"Automatically determined: nnode_in_features={nnode_in_features}, nedge_in_features={nedge_in_features}")

# Update config dynamically
config['model']['nnode_in_features'] = nnode_in_features
config['model']['nedge_in_features'] = nedge_in_features


#Calculate removal Indices for Nodes type_b
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

def train_swmm_gnn(config, device):
    """
    Train the SWMM-GNN model using pushforward training.
    This version supports three model types based on the config:

      - GNN-KAN: Uses a standard KAN (e.g., spline-based) in the message passing.
      - FurierKAN-GNN: Uses a Fourier-based KAN in the message passing.
      - StandardGNN: A conventional message passing GNN.
    """
    # Load training and validation graphs.
    train_graphs = load_graphs(config['data']['train_graph_dir'])
    val_graphs = load_graphs(config['data']['val_graph_dir'])

    train_loader = DataLoader(train_graphs, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=config['training']['batch_size'], shuffle=False)

    model_type = config['model']['model_type']

    if model_type == "StandardGNN":
        print("Using Standard GNN model")
        model = StandardGNN(
            nnode_in_features=config['model']['nnode_in_features'],
            nnode_out_features=2,
            nedge_in_features=config['model']['nedge_in_features'],
            nedge_out_features=2,
            latent_dim=config['model']['latent_dim'],
            nmessage_passing_steps=config['model']['nmessage_passing_steps'],
            nmlp_layers=config['model']['nmlp_layers'],
            mlp_hidden_dim=config['model']['mlp_hidden_dim'],
            residual=config['model']['residual'],
            n_time_steps=config['model']['n_time_step'],
            removal_indices =config['model']['removal_indices']
        ).to(device)

    else:
        # Placeholder for other models if they are added in the future
        raise ValueError("Unknown model type: {}".format(model_type))

    print(f"Number of trainable parameters: {count_parameters(model)}")

    # Define optimizer and scheduler
    weight_decay = float(config["training"]["weight_decay"])
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'], weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['training']['scheduler_step'], gamma=config['training']['scheduler_gamma'])

    ensure_dir(config['checkpoint']['save_dir'])
    print(f"Initial Learning Rate: {optimizer.param_groups[0]['lr']}")
    print(f"Scheduler Step Size: {scheduler.step_size}")
    print(f"Scheduler Gamma: {scheduler.gamma}")

    start_epoch, best_val_loss = load_checkpoint(
        os.path.join(config['checkpoint']['save_dir'], 'swmm_checkpoint.pth'),
        model, optimizer, scheduler
    )
    print(f"Loaded Learning Rate: {scheduler.get_last_lr()[0]:.6e}")

    n_time_steps = config['model']['n_time_step']
    max_steps_ahead = config['model']['max_steps_ahead']
    loss_config = config['loss']

    for epoch in range(start_epoch, config['training']['epochs']):
        print(f"\n **Epoch {epoch + 1}/{config['training']['epochs']}**")

        # Train one epoch using pushforward training
        train_results = train_pushforward(
            model, train_loader, optimizer, scheduler, device, n_time_steps, max_steps_ahead, loss_config, edge_info_list
        )
        # Extract loss values from dictionary
        avg_train_loss = train_results["avg_loss"]
        avg_train_node_loss = train_results["avg_loss_node"]
        avg_train_edge_loss = train_results["avg_loss_edge"]
        avg_train_node_head = train_results["avg_loss_node_head"]
        avg_train_node_inflow = train_results["avg_loss_node_inflow"]
        avg_train_edge_flow = train_results["avg_loss_edge_flow"]
        avg_train_edge_depth = train_results["avg_loss_edge_depth"]
        avg_perc_error_node_head = train_results["avg_perc_error_node_head"]
        avg_perc_error_node_inflow = train_results["avg_perc_error_node_inflow"]
        avg_perc_error_edge_flow = train_results["avg_perc_error_edge_flow"]
        avg_perc_error_edge_depth = train_results["avg_perc_error_edge_depth"]
        avg_perc_error_total = train_results["avg_perc_error_total"]


        print(f"After PF Learning Rate: {scheduler.get_last_lr()[0]:.6e}")

        # Validate one epoch using pushforward training
        val_results = validate_pushforward(
            model, val_loader, device, n_time_steps, max_steps_ahead, loss_config, edge_info_list
        )

        # Extract validation loss values from dictionary
        avg_val_loss = val_results["avg_loss"]
        avg_val_node_loss = val_results["avg_loss_node"]
        avg_val_edge_loss = val_results["avg_loss_edge"]
        avg_val_node_head = val_results["avg_loss_node_head"]
        avg_val_node_inflow = val_results["avg_loss_node_inflow"]
        avg_val_edge_flow = val_results["avg_loss_edge_flow"]
        avg_val_edge_depth = val_results["avg_loss_edge_depth"]
        avg_val_perc_error_node_head = val_results["avg_perc_error_node_head"]
        avg_val_perc_error_node_inflow = val_results["avg_perc_error_node_inflow"]
        avg_val_perc_error_edge_flow = val_results["avg_perc_error_edge_flow"]
        avg_val_perc_error_edge_depth = val_results["avg_perc_error_edge_depth"]
        avg_val_perc_error_total = val_results["avg_perc_error_total"]

        #scheduler.step()

        print(f"Epoch {epoch + 1}/{config['training']['epochs']}, "
              f"Train Total Loss: {avg_train_loss:.4e}, Node Loss: {avg_train_node_loss:.4e}, Edge Loss: {avg_train_edge_loss:.4e}, "
              f"Validation Total Loss: {avg_val_loss:.4e}, Node Loss: {avg_val_node_loss:.4e}, Edge Loss: {avg_val_edge_loss:.4e}, "
              f"LR: {(scheduler.get_last_lr()[0]):.6e}")

        wandb.log({
            'epoch': epoch + 1,
            'train_loss_total': avg_train_loss,
            'train_loss_node': avg_train_node_loss,
            'train_loss_edge': avg_train_edge_loss,
            'train_loss_node_head': avg_train_node_head,
            'train_loss_node_inflow': avg_train_node_inflow,
            'train_loss_edge_flow': avg_train_edge_flow,
            'train_loss_edge_depth': avg_train_edge_depth,
            #'perc_error_node_head': avg_perc_error_node_head,
            #'perc_error_node_inflow': avg_perc_error_node_inflow,
            #'perc_error_edge_flow': avg_perc_error_edge_flow,
            #'perc_error_edge_depth': avg_perc_error_edge_depth,
            'train_perc_error_total': avg_perc_error_total,
            'val_loss_total': avg_val_loss,
            'val_loss_node': avg_val_node_loss,
            'val_loss_edge': avg_val_edge_loss,
            'val_loss_node_head': avg_val_node_head,
            'val_loss_node_inflow': avg_val_node_inflow,
            'val_loss_edge_flow': avg_val_edge_flow,
            'val_loss_edge_depth': avg_val_edge_depth,
            #'val_perc_error_node_head': avg_val_perc_error_node_head,
            #'val_perc_error_node_inflow': avg_val_perc_error_node_inflow,
            #'val_perc_error_edge_flow': avg_val_perc_error_edge_flow,
            #'val_perc_error_edge_depth': avg_val_perc_error_edge_depth,
            'val_perc_error_total': avg_val_perc_error_total,
            'learning_rate': scheduler.get_last_lr()[0]
        })

        save_checkpoint(epoch + 1, model, optimizer, scheduler, best_val_loss,
                        os.path.join(config['checkpoint']['save_dir'], 'swmm_checkpoint.pth'))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(config['checkpoint']['save_dir'], 'best_swmm_gnn.pth'))
            print(f"🌟 Best model saved with val loss: {avg_val_loss:.4e}")

    model.load_state_dict(torch.load(os.path.join(config['checkpoint']['save_dir'], 'best_swmm_gnn.pth')))
    print("✅ Best SWMM-GNN Model Loaded")
    return model

def main():
    seed = 42
    set_random_seed(seed)

    wandb.init(project=config['wandb']['project'],
               entity=config['wandb'].get('entity', None),
               config=config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_swmm_gnn(config, device)

if __name__ == "__main__":
    main()


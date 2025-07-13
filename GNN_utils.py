import torch
from tqdm import tqdm
import torch.nn.functional as F
from scipy.spatial import KDTree
import random
import numpy as np
from torch_scatter import scatter


#########################################
# Set Random Seed for Reproducibility
#########################################
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


#########################################
# Loss Function for SWMM GNN
#########################################
def swmm_gnn_loss(preds_node, preds_edge, targets_node, targets_edge, loss_type="hybrid", config=None, edge_info_list=None, apply_penalties=False, use_diff_loss=False):
    """
    Compute loss for node and edge predictions.
    In training, additional loss components (penalties, diff loss) are applied (apply_penalties=True).
    In validation, we set apply_penalties=False so that only the base relative MSE (or MAE/MSE) is used.
    """
    if config is not None:
        loss_type = config.get("loss_type", loss_type)
    num_steps = preds_node.shape[0]  # Number of prediction steps

    # --- Node Loss Calculation ---
    pred_head = preds_node[..., 0]  # predicted head
    pred_inflow = preds_node[..., 1]  # predicted inflow
    target_head = targets_node[..., 0]  # ground truth head
    target_inflow = targets_node[..., 1]  # ground truth inflow

    if loss_type == "mse":
        loss_node_head = F.mse_loss(pred_head, target_head, reduction="mean")
        loss_node_inflow = F.mse_loss(pred_inflow, target_inflow, reduction="mean")
    elif loss_type == "mae":
        loss_node_head = F.l1_loss(pred_head, target_head, reduction="mean")
        loss_node_inflow = F.l1_loss(pred_inflow, target_inflow, reduction="mean")
    elif loss_type == "relative_mse":
        loss_node_head = torch.sum((pred_head - target_head) ** 2) / (torch.sum(target_head ** 2) + 1e-8)
        loss_node_inflow = torch.sum((pred_inflow - target_inflow) ** 2) / (torch.sum(target_inflow ** 2) + 1e-8)
    elif loss_type == "hybrid":
        # Get weights from config, or set default values (e.g., 0.5 each).
        lambda_rel = config.get("lambda_relative", 0.5)
        lambda_abs = config.get("lambda_abs", 0.5)
        # Compute relative MSE component.
        rel_loss_head = torch.sum((pred_head - target_head) ** 2) / (torch.sum(target_head ** 2) + 1e-8)
        rel_loss_inflow = torch.sum((pred_inflow - target_inflow) ** 2) / (torch.sum(target_inflow ** 2) + 1e-8)
        # Compute MAE component.
        abs_loss_head = F.l1_loss(pred_head, target_head, reduction="mean")
        abs_loss_inflow = F.l1_loss(pred_inflow, target_inflow, reduction="mean")
        # Combine them.
        loss_node_head = lambda_rel * rel_loss_head + lambda_abs * abs_loss_head
        loss_node_inflow = lambda_rel * rel_loss_inflow + lambda_abs * abs_loss_inflow
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    # Add the negative penalty for node head only if we apply penalties.
    if apply_penalties:
        node_head_negative_penalty = torch.mean(F.relu(-pred_head))
        lambda_negative_head_node = config.get("lambda_negative_depth_node", 1.0)
        loss_node_head = loss_node_head + lambda_negative_head_node * node_head_negative_penalty

    loss_node = (loss_node_head + loss_node_inflow) / num_steps

    # --- Edge Loss Calculation ---
    pred_edge_flow = preds_edge[..., 0]  # predicted edge flow
    pred_edge_depth = preds_edge[..., 1]  # predicted edge depth
    target_edge_flow = targets_edge[..., 0]  # ground truth flow
    target_edge_depth = targets_edge[..., 1]  # ground truth depth
    if loss_type == "mse":
        loss_edge_flow = F.mse_loss(pred_edge_flow, target_edge_flow, reduction="mean")
        loss_edge_depth = F.mse_loss(pred_edge_depth, target_edge_depth, reduction="mean")
    elif loss_type == "mae":
        loss_edge_flow = F.l1_loss(pred_edge_flow, target_edge_flow, reduction="mean")
        loss_edge_depth = F.l1_loss(pred_edge_depth, target_edge_depth, reduction="mean")
    elif loss_type == "relative_mse":
        loss_edge_flow = torch.sum((pred_edge_flow - target_edge_flow) ** 2) / (torch.sum(target_edge_flow ** 2) + 1e-8)
        loss_edge_depth = torch.sum((pred_edge_depth - target_edge_depth) ** 2) / (
                    torch.sum(target_edge_depth ** 2) + 1e-8)
    elif loss_type == "hybrid":
        lambda_rel = config.get("lambda_relative", 0.5)
        lambda_abs = config.get("lambda_abs", 0.5)
        rel_loss_edge_flow = torch.sum((pred_edge_flow - target_edge_flow) ** 2) / (
                    torch.sum(target_edge_flow ** 2) + 1e-8)
        rel_loss_edge_depth = torch.sum((pred_edge_depth - target_edge_depth) ** 2) / (
                    torch.sum(target_edge_depth ** 2) + 1e-8)
        abs_loss_edge_flow = F.l1_loss(pred_edge_flow, target_edge_flow, reduction="mean")
        abs_loss_edge_depth = F.l1_loss(pred_edge_depth, target_edge_depth, reduction="mean")
        loss_edge_flow = lambda_rel * rel_loss_edge_flow + lambda_abs * abs_loss_edge_flow
        loss_edge_depth = lambda_rel * rel_loss_edge_depth + lambda_abs * abs_loss_edge_depth

    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    # Add negative penalty for edge depth if using penalties.
    if apply_penalties:
        negative_depth_penalty = torch.mean(F.relu(-pred_edge_depth))
        lambda_negative = config.get("lambda_negative_depth", 1.0)
        loss_edge_depth = loss_edge_depth + lambda_negative * negative_depth_penalty

    loss_edge = (loss_edge_flow + loss_edge_depth) / num_steps

    # --- Diff Loss Calculation (derived from node values) ---
    diff_loss = 0.0
    if edge_info_list is not None and use_diff_loss:
        # Build index tensors for source and target nodes for all edges.
        from_idx = torch.tensor([edge["from_node_idx"] for edge in edge_info_list], device=preds_node.device)
        to_idx = torch.tensor([edge["to_node_idx"] for edge in edge_info_list], device=preds_node.device)
        # Gather predictions and targets for the corresponding nodes.
        # New shapes: [num_steps, num_edges, 2]
        pred_from = preds_node[:, from_idx, :]
        pred_to = preds_node[:, to_idx, :]
        target_from = targets_node[:, from_idx, :]
        target_to = targets_node[:, to_idx, :]

        # Compute difference for each channel.
        diff_pred = pred_from - pred_to
        diff_target = target_from - target_to
        # Compute squared error and then mean.
        # Note: torch.mean here divides by (num_steps*num_edges*2) (2 channels),
        # but the original loop divided by (num_steps*num_edges) after summing both channels.
        # Multiply by 2 to cancel the extra division.
        diff_loss = torch.mean((diff_pred - diff_target) ** 2) * 2.0

        lambda_diff = config.get("diff_loss_weight", 1.0) if config is not None else 1.0
        diff_loss = lambda_diff * diff_loss

    # Use the loss weights alpha and beta (for training they might be used to balance the components).
    alpha = config.get("loss_alpha", 1.0) if config is not None else 1.0
    beta = config.get("loss_beta", 1.0) if config is not None else 1.0
    total_loss = (alpha * loss_node) + (beta * loss_edge) + diff_loss


    # --- Compute RMSE and Percentage Error Metrics ---
    # These are computed regardless of loss_type.
    rmse_node_head = torch.sqrt(F.mse_loss(pred_head, target_head, reduction="mean"))
    rmse_node_inflow = torch.sqrt(F.mse_loss(pred_inflow, target_inflow, reduction="mean"))
    rmse_edge_flow = torch.sqrt(F.mse_loss(pred_edge_flow, target_edge_flow, reduction="mean"))
    rmse_edge_depth = torch.sqrt(F.mse_loss(pred_edge_depth, target_edge_depth, reduction="mean"))

    rmse_node = torch.sqrt(F.mse_loss(preds_node, targets_node, reduction="mean"))
    rmse_edge = torch.sqrt(F.mse_loss(preds_edge, targets_edge, reduction="mean"))
    rmse_total = (rmse_node + rmse_edge) / 2.0

    # Convert RMSE to percentage error.
    perc_error_node_head = rmse_node_head * 100.0
    perc_error_node_inflow = rmse_node_inflow * 100.0
    perc_error_edge_flow = rmse_edge_flow * 100.0
    perc_error_edge_depth = rmse_edge_depth * 100.0
    perc_error_node = rmse_node * 100.0
    perc_error_edge = rmse_edge * 100.0
    perc_error_total = rmse_total * 100.0

    # --- Assemble Result Dictionary ---
    loss_dict = {
        "total_loss": total_loss,
        "loss_node": loss_node,
        "loss_node_head": loss_node_head / num_steps,
        "loss_node_inflow": loss_node_inflow / num_steps,
        "loss_edge": loss_edge,
        "loss_edge_flow": loss_edge_flow / num_steps,
        "loss_edge_depth": loss_edge_depth / num_steps,
        "diff_loss": diff_loss,
        "rmse_node_head": rmse_node_head,
        "rmse_node_inflow": rmse_node_inflow,
        "rmse_edge_flow": rmse_edge_flow,
        "rmse_edge_depth": rmse_edge_depth,
        "rmse_node": rmse_node,
        "rmse_edge": rmse_edge,
        "rmse_total": rmse_total,
        "perc_error_node_head": perc_error_node_head,
        "perc_error_node_inflow": perc_error_node_inflow,
        "perc_error_edge_flow": perc_error_edge_flow,
        "perc_error_edge_depth": perc_error_edge_depth,
        "perc_error_node": perc_error_node,
        "perc_error_edge": perc_error_edge,
        "perc_error_total": perc_error_total,
    }
    return loss_dict

#########################################
# shift full graph left # Node version
#########################################
def shift_full_graph_left(x, head_inflow_target, future_node_rainfall, n_time_steps, step_idx=0):
    """
    Shift left the dynamic portion of the node features.
    New ordering:
      [constant, rainfall, acc_rainfall, head, inflow]
    Args:
      x: [num_nodes, total_features] full node graph.
      head_inflow_target: [num_nodes, 2] ground truth (head, inflow) for the next time step.
      future_node_rainfall: [num_nodes, max_steps_ahead, 2] future rainfall values.
                           We'll extract the next step (t+1) from here.
      n_time_steps: Number of dynamic time steps.
    Returns:
      x_new: Updated node feature tensor.
    """
    num_constant = x.shape[1] - (4 * n_time_steps)

    # Extract current dynamic groups.
    rainfall = x[:, num_constant:num_constant+n_time_steps]
    acc_rainfall = x[:, num_constant+n_time_steps:num_constant+2*n_time_steps]
    head = x[:, num_constant+2*n_time_steps:num_constant+3*n_time_steps]
    inflow = x[:, num_constant+3*n_time_steps:num_constant+4*n_time_steps]

    # Extract new ground truth values from the 3D future tensor.
    new_rainfall_value = future_node_rainfall[:, step_idx, 0:1]    # rainfall(t+1)
    new_acc_rainfall_value = future_node_rainfall[:, step_idx, 1:2]  # acc_rainfall(t+1)

    # Shift left and append new values.
    new_rainfall = torch.cat([rainfall[:, 1:], new_rainfall_value], dim=1)
    new_acc_rainfall = torch.cat([acc_rainfall[:, 1:], new_acc_rainfall_value], dim=1)
    new_head = torch.cat([head[:, 1:], head_inflow_target[:, 0:1]], dim=1)
    new_inflow = torch.cat([inflow[:, 1:], head_inflow_target[:, 1:2]], dim=1)

    # Reconstruct the full node feature tensor.
    x_new = torch.cat([x[:, :num_constant], new_rainfall, new_acc_rainfall, new_head, new_inflow], dim=1)
    return x_new

#########################################
# shift full graph left # Edge version
#########################################
def shift_full_graph_left_edge(edge_attr, edge_target, node_target, future_edge_rainfall, n_time_steps, edge_info_list, step_idx=0):
    """
    Shift left the dynamic portion of the edge features.
    New ordering:
      [constant, rainfall, acc_rainfall, diff_head, diff_inflow, flow, depth]
    Args:
      edge_attr: [num_edges, total_features] full edge features.
      edge_target: [num_edges, 2] ground truth (flow, depth) for the next time step.
      node_target: [num_nodes, 2] ground truth (head, inflow) for the next time step (used to update diffs).
      future_edge_rainfall: [num_edges, max_steps_ahead, 2] future edge rainfall values.
      n_time_steps: Number of dynamic time steps.
      edge_info_list: List of dictionaries with connectivity info.
    Returns:
      edge_attr_new: Updated edge feature tensor.
    """
    num_constant_edge = edge_attr.shape[1] - (6 * n_time_steps)

    # Extract dynamic groups.
    rainfall = edge_attr[:, num_constant_edge:num_constant_edge+n_time_steps]
    acc_rainfall = edge_attr[:, num_constant_edge+n_time_steps:num_constant_edge+2*n_time_steps]
    diff_head = edge_attr[:, num_constant_edge+2*n_time_steps:num_constant_edge+3*n_time_steps]
    diff_inflow = edge_attr[:, num_constant_edge+3*n_time_steps:num_constant_edge+4*n_time_steps]
    flow = edge_attr[:, num_constant_edge+4*n_time_steps:num_constant_edge+5*n_time_steps]
    depth = edge_attr[:, num_constant_edge+5*n_time_steps:num_constant_edge+6*n_time_steps]

    # Shift rainfall and acc_rainfall using future ground truth.
    new_rainfall_value = future_edge_rainfall[:, step_idx, 0:1]    # rainfall for t+1
    new_acc_rainfall_value = future_edge_rainfall[:, step_idx, 1:2]  # acc_rainfall for t+1
    new_rainfall = torch.cat([rainfall[:, 1:], new_rainfall_value], dim=1)
    new_acc_rainfall = torch.cat([acc_rainfall[:, 1:], new_acc_rainfall_value], dim=1)

    # Shift flow and depth and append ground truth.
    new_flow = torch.cat([flow[:, 1:], edge_target[:, 0:1]], dim=1)
    new_depth = torch.cat([depth[:, 1:], edge_target[:, 1:2]], dim=1)

    # For diff_head and diff_inflow, compute new differences using node_target.
    num_edges = edge_attr.shape[0]
    new_diff_head_col = torch.empty((num_edges, 1), device=edge_attr.device)
    new_diff_inflow_col = torch.empty((num_edges, 1), device=edge_attr.device)

    debug_edge_idx = 42  # change this to the edge you want to inspect

    for i in range(num_edges):
        info = edge_info_list[i]
        from_idx = info["from_node_idx"]
        to_idx = info["to_node_idx"]

        from_name = info["from_node"]
        to_name = info["to_node"]
        conduit_name = info["conduit_name"]

        new_diff_head_col[i, 0] = node_target[from_idx, 0] - node_target[to_idx, 0]
        new_diff_inflow_col[i, 0] = node_target[from_idx, 1] - node_target[to_idx, 1]

    new_diff_head = torch.cat([diff_head[:, 1:], new_diff_head_col], dim=1)
    new_diff_inflow = torch.cat([diff_inflow[:, 1:], new_diff_inflow_col], dim=1)

    # Reconstruct updated edge features.
    edge_attr_new = torch.cat([
        edge_attr[:, :num_constant_edge],
        new_rainfall,
        new_acc_rainfall,
        new_diff_head,
        new_diff_inflow,
        new_flow,
        new_depth
    ], dim=1)
    return edge_attr_new

#########################################
# Prepare Node and edge input
#########################################
def prepare_node_input(x_pred, n_time_steps):
    """
    Prepares node input by keeping constant features and selecting the most recent dynamic steps.
    With n_time_steps dynamic steps in x_pred, this function drops the oldest (index 0)
    and keeps the remaining n_time_steps - 1 steps.
    For instance:
      - If n_time_steps = 3, keep 2 most recent steps.
      - If n_time_steps = 4, keep 3 most recent steps.
      - If n_time_steps = 5, keep 4 most recent steps.
    """
    num_dyn_features = 4  # for nodes: rainfall, acc_rainfall, head, inflow.
    num_constant = x_pred.shape[1] - (num_dyn_features * n_time_steps)

    rainfall_start = num_constant
    acc_rainfall_start = rainfall_start + n_time_steps
    head_start = acc_rainfall_start + n_time_steps
    inflow_start = head_start + n_time_steps

    # Instead of using (n_time_steps - 2), start at index 1 to always drop the oldest.
    x_input_nodes = torch.cat([
        x_pred[:, :num_constant],  # constant features remain unchanged.
        x_pred[:, rainfall_start + 1:rainfall_start + n_time_steps],      # keeps indices 1 to n_time_steps-1
        x_pred[:, acc_rainfall_start + 1:acc_rainfall_start + n_time_steps],
        x_pred[:, head_start + 1:head_start + n_time_steps],
        x_pred[:, inflow_start + 1:inflow_start + n_time_steps],
    ], dim=1)
    return x_input_nodes

def prepare_edge_input(edge_attr_pred, n_time_steps):
    """
    Prepares edge input by keeping constant edge features and selecting the most recent dynamic steps.
    For example, if n_time_steps = 4, the function keeps the last 3 steps (ignoring the oldest).
    """
    num_dyn_edge_features = 6  # for edges: rainfall, acc_rainfall, diff_head, diff_inflow, flow, depth.
    num_constant_edge = edge_attr_pred.shape[1] - (num_dyn_edge_features * n_time_steps)

    rainfall_start = num_constant_edge
    acc_rainfall_start = rainfall_start + n_time_steps
    diff_head_start = acc_rainfall_start + n_time_steps
    diff_inflow_start = diff_head_start + n_time_steps
    flow_start = diff_inflow_start + n_time_steps
    depth_start = flow_start + n_time_steps

    edge_input = torch.cat([
        edge_attr_pred[:, :num_constant_edge],  # constant edge features.
        edge_attr_pred[:, rainfall_start + 1:rainfall_start + n_time_steps],
        edge_attr_pred[:, acc_rainfall_start + 1:acc_rainfall_start + n_time_steps],
        edge_attr_pred[:, diff_head_start + 1:diff_head_start + n_time_steps],
        edge_attr_pred[:, diff_inflow_start + 1:diff_inflow_start + n_time_steps],
        edge_attr_pred[:, flow_start + 1:flow_start + n_time_steps],
        edge_attr_pred[:, depth_start + 1:depth_start + n_time_steps],
    ], dim=1)
    return edge_input

##############################################
# 1. Helper Functions for Stability Branch
##############################################
def prepare_stability_first_step_input(x_pred, n_time_steps):
    """
    Prepares node input for the first stability prediction.
    For nodes, dynamic features are now: [rainfall, acc_rainfall, head, inflow].
    This function removes the most current time step (i.e. column index n_time_steps - 1)
    so that the input contains only (n_time_steps - 1) columns for each dynamic group.
    """
    num_dyn_features = 4  # rainfall, acc_rainfall, head, inflow
    num_constant = x_pred.shape[1] - (num_dyn_features * n_time_steps)

    # Compute starting indices for each dynamic group.
    rainfall_start = num_constant
    acc_rainfall_start = rainfall_start + n_time_steps
    head_start = acc_rainfall_start + n_time_steps
    inflow_start = head_start + n_time_steps

    # Keep only the first (n_time_steps - 1) columns from each group.
    x_input_nodes = torch.cat([
        x_pred[:, :num_constant],
        x_pred[:, rainfall_start:rainfall_start + (n_time_steps - 1)],
        x_pred[:, acc_rainfall_start:acc_rainfall_start + (n_time_steps - 1)],
        x_pred[:, head_start:head_start + (n_time_steps - 1)],
        x_pred[:, inflow_start:inflow_start + (n_time_steps - 1)],
    ], dim=1)
    return x_input_nodes


def prepare_stability_first_step_edge(edge_attr_pred, n_time_steps):
    """
    Prepares edge input for the first stability prediction.
    For edges, dynamic features are now:
    [rainfall, acc_rainfall, diff_head, diff_inflow, flow, depth].
    This function removes the most current time step from each dynamic group.
    """
    num_dyn_edge_features = 6
    num_constant_edge = edge_attr_pred.shape[1] - (num_dyn_edge_features * n_time_steps)

    # Starting indices:
    rainfall_start = num_constant_edge
    acc_rainfall_start = rainfall_start + n_time_steps
    diff_head_start = acc_rainfall_start + n_time_steps
    diff_inflow_start = diff_head_start + n_time_steps
    flow_start = diff_inflow_start + n_time_steps
    depth_start = flow_start + n_time_steps

    edge_input = torch.cat([
        edge_attr_pred[:, :num_constant_edge],
        edge_attr_pred[:, rainfall_start:rainfall_start + (n_time_steps - 1)],
        edge_attr_pred[:, acc_rainfall_start:acc_rainfall_start + (n_time_steps - 1)],
        edge_attr_pred[:, diff_head_start:diff_head_start + (n_time_steps - 1)],
        edge_attr_pred[:, diff_inflow_start:diff_inflow_start + (n_time_steps - 1)],
        edge_attr_pred[:, flow_start:flow_start + (n_time_steps - 1)],
        edge_attr_pred[:, depth_start:depth_start + (n_time_steps - 1)],
    ], dim=1)
    return edge_input

###########################################
# Update current graph with prediction (t)
###########################################
def update_current_graph_with_prediction(x, pred, n_time_steps):
    """
    Update full node graph by replacing the most current dynamic column for head and inflow
    with the predicted values. Here x has ordering:
      [constant, rainfall, acc_rainfall, head, inflow]
    """
    num_constant = x.shape[1] - (4 * n_time_steps)
    rainfall_start = num_constant
    acc_rainfall_start = rainfall_start + n_time_steps
    head_start = acc_rainfall_start + n_time_steps
    inflow_start = head_start + n_time_steps

    rainfall = x[:, rainfall_start:rainfall_start+n_time_steps]
    acc_rainfall = x[:, acc_rainfall_start:acc_rainfall_start+n_time_steps]
    head = x[:, head_start:head_start+n_time_steps].clone()
    inflow = x[:, inflow_start:inflow_start+n_time_steps].clone()

    # Replace the last column with predicted head and inflow.
    head[:, -1] = pred[:, 0]
    inflow[:, -1] = pred[:, 1]

    x_updated = torch.cat([x[:, :num_constant], rainfall, acc_rainfall, head, inflow], dim=1)
    return x_updated


def update_current_edge_with_prediction(edge_attr, edge_pred, node_pred, n_time_steps, edge_info_list):
    """
    Update full edge graph by replacing the last dynamic columns for flow and depth with predictions,
    and recompute diff_head and diff_inflow using predicted node values.
    New ordering: [constant, rainfall, acc_rainfall, diff_head, diff_inflow, flow, depth]
    """
    num_constant_edge = edge_attr.shape[1] - (6 * n_time_steps)
    rainfall_start = num_constant_edge
    acc_rainfall_start = rainfall_start + n_time_steps
    diff_head_start = acc_rainfall_start + n_time_steps
    diff_inflow_start = diff_head_start + n_time_steps
    flow_start = diff_inflow_start + n_time_steps
    depth_start = flow_start + n_time_steps

    rainfall = edge_attr[:, rainfall_start:rainfall_start+n_time_steps]
    acc_rainfall = edge_attr[:, acc_rainfall_start:acc_rainfall_start+n_time_steps]
    diff_head = edge_attr[:, diff_head_start:diff_head_start+n_time_steps].clone()
    diff_inflow = edge_attr[:, diff_inflow_start:diff_inflow_start+n_time_steps].clone()
    flow = edge_attr[:, flow_start:flow_start+n_time_steps].clone()
    depth = edge_attr[:, depth_start:depth_start+n_time_steps].clone()

    # Replace last column for flow and depth with predictions.
    flow[:, -1] = edge_pred[:, 0]
    depth[:, -1] = edge_pred[:, 1]

    # Recompute last column for diff_head and diff_inflow using predicted node values.
    num_edges = edge_attr.shape[0]
    new_diff_head_col = torch.empty((num_edges,), device=edge_attr.device)
    new_diff_inflow_col = torch.empty((num_edges,), device=edge_attr.device)
    for i in range(num_edges):
        info = edge_info_list[i]
        from_idx = info["from_node_idx"]
        to_idx = info["to_node_idx"]
        new_diff_head_col[i] = node_pred[from_idx, 0] - node_pred[to_idx, 0]
        new_diff_inflow_col[i] = node_pred[from_idx, 1] - node_pred[to_idx, 1]
    diff_head[:, -1] = new_diff_head_col
    diff_inflow[:, -1] = new_diff_inflow_col

    edge_attr_updated = torch.cat([
        edge_attr[:, :num_constant_edge],
        rainfall,
        acc_rainfall,
        diff_head,
        diff_inflow,
        flow,
        depth
    ], dim=1)
    return edge_attr_updated

#########################################
# Pushforward update # Node and Edge
#########################################
def use_prediction_pushforward(x, pred, future_rainfall, n_time_steps, max_steps_ahead, step_idx=0):
    """
    Update node features using model predictions (pushforward).
    x is arranged as [constant, rainfall, acc_rainfall, head, inflow].
    We shift left (drop the oldest) and append:
      - For rainfall and acc_rainfall: new values from future_rainfall (a 3D tensor)
      - For head/inflow: the predicted values.
    """
    num_constant = x.shape[1] - (4 * n_time_steps)
    rainfall = x[:, num_constant:num_constant+n_time_steps]
    acc_rainfall = x[:, num_constant+n_time_steps:num_constant+2*n_time_steps]
    head = x[:, num_constant+2*n_time_steps:num_constant+3*n_time_steps]
    inflow = x[:, num_constant+3*n_time_steps:num_constant+4*n_time_steps]

    # Extract new values from future_rainfall, which is [num_nodes, max_steps_ahead, 2]
    new_rainfall_value = future_rainfall[:, step_idx, 0:1]
    new_acc_rainfall_value = future_rainfall[:, step_idx, 1:2]

    new_rainfall = torch.cat([rainfall[:, 1:], new_rainfall_value], dim=1)
    new_acc_rainfall = torch.cat([acc_rainfall[:, 1:], new_acc_rainfall_value], dim=1)
    new_head = torch.cat([head[:, 1:], pred[:, 0:1]], dim=1)
    new_inflow = torch.cat([inflow[:, 1:], pred[:, 1:2]], dim=1)

    updated_x = torch.cat([x[:, :num_constant], new_rainfall, new_acc_rainfall, new_head, new_inflow], dim=1)
    return updated_x


def use_edge_prediction_pushforward(edge_features, edge_pred, node_pred, future_edge_rainfall, n_time_steps_edge,
                                    edge_info_list, step_idx=0):
    """
    Update edge features using model predictions (pushforward) while updating the rainfall channels with ground truth.
    For edges with ordering:
        [constant, rainfall, acc_rainfall, diff_head, diff_inflow, flow, depth]
    The function shifts left (drops the oldest column) for each dynamic group and appends:
      - For rainfall and acc_rainfall: new values from future_edge_rainfall.
      - For flow and depth: the predicted values (from edge_pred).
      - For diff_head and diff_inflow: newly computed differences using node_pred.

    Parameters:
      - edge_features (torch.Tensor): Current edge features. Shape: [num_edges, total_features].
      - edge_pred (torch.Tensor): Predicted edge values for [flow, depth]. Shape: [num_edges, 2].
      - node_pred (torch.Tensor): Latest predicted node values for [head, inflow]. Shape: [num_nodes, 2].
      - future_edge_rainfall (torch.Tensor): Future edge rainfall values. Shape: [num_edges, max_steps_ahead, 2].
      - n_time_steps_edge (int): Number of dynamic time steps stored for each dynamic edge feature.
      - edge_info_list (list): List of dictionaries for each edge with keys "from_node_idx" and "to_node_idx".

    Returns:
      - updated_edge_features (torch.Tensor): Updated edge features.
    """
    # Compute the constant portion.
    num_constant = edge_features.shape[1] - (6 * n_time_steps_edge)

    # Extract dynamic groups.
    # Rainfall and acc_rainfall.
    rainfall = edge_features[:, num_constant: num_constant + n_time_steps_edge]
    acc_rainfall = edge_features[:, num_constant + n_time_steps_edge: num_constant + 2 * n_time_steps_edge]
    # diff_head and diff_inflow.
    diff_head = edge_features[:, num_constant + 2 * n_time_steps_edge: num_constant + 3 * n_time_steps_edge].clone()
    diff_inflow = edge_features[:, num_constant + 3 * n_time_steps_edge: num_constant + 4 * n_time_steps_edge].clone()
    # flow and depth.
    flow = edge_features[:, num_constant + 4 * n_time_steps_edge: num_constant + 5 * n_time_steps_edge].clone()
    depth = edge_features[:, num_constant + 5 * n_time_steps_edge: num_constant + 6 * n_time_steps_edge].clone()

    # --- Shift rainfall and acc_rainfall ---
    # Extract new rainfall values from future_edge_rainfall.
    # Here, for each edge, we take the first future step.
    new_rainfall_value = future_edge_rainfall[:, step_idx, 0:1]  # rainfall (t+1)
    new_acc_rainfall_value = future_edge_rainfall[:, step_idx, 1:2]  # acc_rainfall (t+1)
    new_rainfall = torch.cat([rainfall[:, 1:], new_rainfall_value], dim=1)
    new_acc_rainfall = torch.cat([acc_rainfall[:, 1:], new_acc_rainfall_value], dim=1)

    # --- Shift flow and depth ---
    new_flow = torch.cat([flow[:, 1:], edge_pred[:, 0:1]], dim=1)
    new_depth = torch.cat([depth[:, 1:], edge_pred[:, 1:2]], dim=1)

    # --- Shift diff_head and diff_inflow and compute new differences ---
    # First, shift left the existing columns.
    new_diff_head = diff_head[:, 1:].clone()
    new_diff_inflow = diff_inflow[:, 1:].clone()
    # For each edge, compute the new differences using node_pred.
    num_edges = edge_features.shape[0]
    new_diff_head_col = torch.empty((num_edges, 1), device=edge_features.device)
    new_diff_inflow_col = torch.empty((num_edges, 1), device=edge_features.device)
    for i in range(num_edges):
        info = edge_info_list[i]
        from_idx = info["from_node_idx"]
        to_idx = info["to_node_idx"]
        new_diff_head_col[i, 0] = node_pred[from_idx, 0] - node_pred[to_idx, 0]
        new_diff_inflow_col[i, 0] = node_pred[from_idx, 1] - node_pred[to_idx, 1]
    new_diff_head = torch.cat([new_diff_head, new_diff_head_col], dim=1)
    new_diff_inflow = torch.cat([new_diff_inflow, new_diff_inflow_col], dim=1)

    # Reassemble updated edge_features.
    updated_edge_features = torch.cat([
        edge_features[:, :num_constant],
        new_rainfall,
        new_acc_rainfall,
        new_diff_head,
        new_diff_inflow,
        new_flow,
        new_depth
    ], dim=1)
    return updated_edge_features


#########################################
# Real Loss Training Loop (Structure)
#########################################
def train_real_loss(model, dataloader, optimizer, device, n_time_steps, max_steps_ahead, loss_config, edge_info_list):
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    for data in dataloader:
        data = data.to(device)
        batch_size = data.num_graphs
        # Repeat edge_info_list for the batch.
        batch_edge_info_list = edge_info_list * batch_size

        # Full graphs (with all dynamic time steps).
        x_full = data.x.clone()         # shape: [num_nodes, constant + 4*n_time_steps]
        edge_full = data.edge_attr.clone()  # shape: [num_edges, constant + 6*n_time_steps]

        # Permuted targets: shape [max_steps_ahead, num_nodes, num_target_features] (for nodes)
        y_tensor = data.y.permute(1, 0, 2)
        y_edges_tensor = data.y_edges.permute(1, 0, 2)

        predictions_node, predictions_edge = [], []

        for step in range(max_steps_ahead):
            # Step A: Prepare model inputs.
            node_input = prepare_node_input(x_full, n_time_steps)
            edge_input = prepare_edge_input(edge_full, n_time_steps)
            node_type = getattr(data, "node_type", None)

            # Step B: Make prediction.
            node_pred, edge_pred = model(node_input, data.edge_index, edge_input, node_type=node_type)
            predictions_node.append(node_pred)
            predictions_edge.append(edge_pred)

            # Step C: For subsequent steps, update the full graph using ground truth.
            if step < max_steps_ahead - 1:
                # For nodes: y_tensor[step] contains ground truth head/inflow, and data.future_rainfall is [num_nodes, max_steps_ahead, 2]
                x_full = shift_full_graph_left(x_full, y_tensor[step], data.future_rainfall, n_time_steps, step_idx=step)
                # For edges: assume data.future_edge_rainfall is provided with shape [num_edges, max_steps_ahead, 2]
                # Also, pass the ground truth node values (here using y_tensor[step] as a proxy) for diff updates.
                edge_full = shift_full_graph_left_edge(edge_full, y_edges_tensor[step], y_tensor[step],
                                                       data.future_edge_rainfall, n_time_steps,
                                                       edge_info_list=batch_edge_info_list, step_idx=step)

        # Stack predictions across time steps. Resulting shapes:
        #   nodes: [max_steps_ahead, num_nodes, num_target_features]
        #   edges: [max_steps_ahead, num_edges, num_edge_target_features]
        predictions_node = torch.stack(predictions_node, dim=0)
        predictions_edge = torch.stack(predictions_edge, dim=0)

        # Compute loss over all future steps.
        loss_dict = swmm_gnn_loss(
            preds_node=predictions_node,
            preds_edge=predictions_edge,
            targets_node=y_tensor,
            targets_edge=y_edges_tensor,
            loss_type=loss_config.get("loss_type", "relative_mse"),
            config=loss_config,
            edge_info_list=batch_edge_info_list,
            apply_penalties=loss_config.get("apply_penalties", False),
            use_diff_loss=loss_config.get("use_diff_loss", False)
        )
        loss = loss_dict["total_loss"]

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    return avg_loss

##############################################
# Training Loop for Stability Loss
##############################################
def train_stability_loss(model, dataloader, optimizer, device, n_time_steps, max_steps_ahead, loss_config, edge_info_list):
    """
    Training function for the stability loss branch (recursive pushforward).
    For the first step, the full graph is trimmed by dropping the most current time step.
    Then, predictions are made and the full graph is updated using those predictions.
    """
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    for data in dataloader:
        data = data.to(device)
        batch_size = data.num_graphs
        batch_edge_info_list = edge_info_list * batch_size

        # Full graphs with all dynamic values.
        x_full = data.x.clone()         # Shape: [num_nodes, constant + 4*n_time_steps]
        edge_full = data.edge_attr.clone()  # Shape: [num_edges, constant + 6*n_time_steps]

        # Permuted targets (3D): [max_steps_ahead, num_nodes, target_features] for nodes, similarly for edges.
        y_tensor = data.y.permute(1, 0, 2)
        y_edges_tensor = data.y_edges.permute(1, 0, 2)

        stability_node_preds = []
        stability_edge_preds = []

        # ----- FIRST STABILITY STEP -----
        # Trim the full graph by dropping the most current time step from each dynamic group.
        x_stability = prepare_stability_first_step_input(x_full, n_time_steps)
        edge_stability = prepare_stability_first_step_edge(edge_full, n_time_steps)

        node_type = getattr(data, "node_type", None)
        # Predict using the trimmed graph.
        node_input = x_stability
        edge_input = edge_stability
        node_pred_initial, edge_pred_initial = model(node_input, data.edge_index, edge_input, node_type=node_type)
        node_pred_initial = node_pred_initial.detach()
        edge_pred_initial = edge_pred_initial.detach()

        # Update the full graph by inserting the predicted values for t.
        x_stability_full = update_current_graph_with_prediction(x_full, node_pred_initial, n_time_steps)
        edge_stability_full = update_current_edge_with_prediction(edge_full, edge_pred_initial, node_pred_initial, n_time_steps, edge_info_list=batch_edge_info_list)

        # # future_rainfall is now 3D: [num_nodes, max_steps_ahead, 2]
        # future_rainfall = data.future_rainfall

        # ----- SUBSEQUENT STABILITY STEPS -----
        for step in range(0, max_steps_ahead):
            # Prepare input by dropping the oldest dynamic column.
            node_input = prepare_node_input(x_stability_full, n_time_steps)
            edge_input = prepare_edge_input(edge_stability_full, n_time_steps)

            # Predict next time step.
            node_pred, edge_pred = model(node_input, data.edge_index, edge_input, node_type=node_type)
            stability_node_preds.append(node_pred)
            stability_edge_preds.append(edge_pred)

            # Update full graphs with pushforward using predictions.
            if step < max_steps_ahead - 1:
                x_stability_full = use_prediction_pushforward(x_stability_full, node_pred.detach(), data.future_rainfall, n_time_steps, max_steps_ahead, step_idx=step)
                edge_stability_full = use_edge_prediction_pushforward(edge_stability_full, edge_pred.detach(), node_pred.detach(), data.future_edge_rainfall, n_time_steps, edge_info_list=batch_edge_info_list, step_idx=step)

        stability_node_preds = torch.stack(stability_node_preds, dim=0)  # [max_steps_ahead, num_nodes, ...]
        stability_edge_preds = torch.stack(stability_edge_preds, dim=0)  # [max_steps_ahead, num_edges, ...]

        # Compute stability loss over all future steps.
        loss_dict_stability = swmm_gnn_loss(
            preds_node=stability_node_preds,
            preds_edge=stability_edge_preds,
            targets_node=y_tensor,
            targets_edge=y_edges_tensor,
            loss_type=loss_config.get("loss_type", "relative_mse"),
            config=loss_config,
            edge_info_list=batch_edge_info_list,
            apply_penalties=loss_config.get("apply_penalties", False),
            use_diff_loss=loss_config.get("use_diff_loss", False)
        )
        loss_stability = loss_dict_stability["total_loss"]
        loss_stability.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss_stability.item()

    avg_loss = total_loss / num_batches
    return avg_loss

############################################
# Modified Pushforward Training Function
############################################
def train_pushforward(model, dataloader, optimizer, scheduler, device, n_time_steps, max_steps_ahead, loss_config, edge_info_list):
    """
    Combined training function that computes both:
      (1) Real Loss: Uses ground-truth updates.
      (2) Stability Loss: Uses recursive pushforward updates.
    The total loss is the sum of these two branches.
    """
    model.train()

    # Overall accumulators
    total_loss = 0.0
    total_loss_node = 0.0
    total_loss_edge = 0.0

    # Accumulators for individual node losses
    total_loss_node_head = 0.0
    total_loss_node_inflow = 0.0
    # Accumulators for individual edge losses
    total_loss_edge_flow = 0.0
    total_loss_edge_depth = 0.0

    # Accumulators for percentage error metrics (when using relative_mse)
    total_perc_error_node_head = 0.0
    total_perc_error_node_inflow = 0.0
    total_perc_error_edge_flow = 0.0
    total_perc_error_edge_depth = 0.0
    total_perc_error_total = 0.0

    num_batches = len(dataloader)

    for batch_idx, data in enumerate(tqdm(dataloader, desc="Training", leave=False)):
        data = data.to(device)
        batch_size = data.num_graphs

        # Repeat edge_info_list for the batch.
        batch_edge_info_list = edge_info_list * batch_size

        # Full graphs: contain all n_time_steps dynamic values.
        x_full = data.x.clone()        # Node features full graph.
        edge_full = data.edge_attr.clone()  # Edge features full graph.

        # Permuted targets: shape [max_steps_ahead, num_nodes, target_features]
        y_tensor = data.y.permute(1, 0, 2)
        y_edges_tensor = data.y_edges.permute(1, 0, 2)

        #### Real Loss Branch ####
        predictions_node, predictions_edge = [], []
        x_real = x_full.clone()
        edge_real = edge_full.clone()
        node_type = getattr(data, "node_type", None)

        for step in range(max_steps_ahead):
            # Prepare input by dropping the oldest dynamic time step.
            node_input = prepare_node_input(x_real, n_time_steps)
            edge_input = prepare_edge_input(edge_real, n_time_steps)
            node_pred, edge_pred = model(node_input, data.edge_index, edge_input, node_type=node_type)
            predictions_node.append(node_pred)
            predictions_edge.append(edge_pred)

            # For steps before the last, update full graphs using ground truth.
            if step < max_steps_ahead - 1:
                # Note the updated function call passing data.future_rainfall and data.future_edge_rainfall.
                x_real = shift_full_graph_left(x_real, y_tensor[step], data.future_rainfall, n_time_steps, step_idx=step)
                edge_real = shift_full_graph_left_edge(edge_real, y_edges_tensor[step], y_tensor[step],
                                                       data.future_edge_rainfall, n_time_steps,
                                                       edge_info_list=batch_edge_info_list, step_idx=step)

        predictions_node = torch.stack(predictions_node, dim=0)
        predictions_edge = torch.stack(predictions_edge, dim=0)
        loss_dict_real = swmm_gnn_loss(
            preds_node=predictions_node,
            preds_edge=predictions_edge,
            targets_node=y_tensor,
            targets_edge=y_edges_tensor,
            loss_type=loss_config.get("loss_type", "relative_mse"),
            config=loss_config,
            edge_info_list=batch_edge_info_list,
            apply_penalties=loss_config.get("apply_penalties", False),
            use_diff_loss=loss_config.get("use_diff_loss", False)
        )
        loss_real = loss_dict_real["total_loss"]

        #### Stability Loss Branch ####
        stability_node_preds, stability_edge_preds = [], []
        # Trim the full graph by dropping the most current time step.
        x_stability = prepare_stability_first_step_input(x_full, n_time_steps)
        edge_stability = prepare_stability_first_step_edge(edge_full, n_time_steps)

        # Make the initial prediction using the trimmed graph.
        node_input = x_stability
        edge_input = edge_stability
        node_pred_initial, edge_pred_initial = model(node_input, data.edge_index, edge_input, node_type=node_type)
        node_pred_initial = node_pred_initial.detach()
        edge_pred_initial = edge_pred_initial.detach()

        # Update full graph with predicted t.
        x_stability_full = update_current_graph_with_prediction(x_full, node_pred_initial, n_time_steps)
        edge_stability_full = update_current_edge_with_prediction(edge_full, edge_pred_initial, node_pred_initial,
                                                                  n_time_steps, edge_info_list=batch_edge_info_list)

        # For each subsequent step, generate predictions and update using pushforward.
        for step in range(0, max_steps_ahead):
            node_input = prepare_node_input(x_stability_full, n_time_steps)
            edge_input = prepare_edge_input(edge_stability_full, n_time_steps)
            node_pred, edge_pred = model(node_input, data.edge_index, edge_input, node_type=node_type)
            stability_node_preds.append(node_pred)
            stability_edge_preds.append(edge_pred)

            # Only update the full graph if there will be a next prediction.
            if step < max_steps_ahead - 1:
                x_stability_full = use_prediction_pushforward(x_stability_full, node_pred.detach(),
                                                              data.future_rainfall, n_time_steps, max_steps_ahead, step_idx=step)
                edge_stability_full = use_edge_prediction_pushforward(edge_stability_full, edge_pred.detach(),
                                                                      node_pred.detach(), data.future_edge_rainfall,
                                                                      n_time_steps, edge_info_list=batch_edge_info_list, step_idx=step)

        stability_node_preds = torch.stack(stability_node_preds, dim=0)
        stability_edge_preds = torch.stack(stability_edge_preds, dim=0)
        loss_dict_stability = swmm_gnn_loss(
            preds_node=stability_node_preds,
            preds_edge=stability_edge_preds,
            targets_node=y_tensor,
            targets_edge=y_edges_tensor,
            loss_type=loss_config.get("loss_type", "relative_mse"),
            config=loss_config,
            edge_info_list=batch_edge_info_list,
            apply_penalties=loss_config.get("apply_penalties", False),
            use_diff_loss=loss_config.get("use_diff_loss", False)
        )
        loss_stability = loss_dict_stability["total_loss"]

        # Combined loss (sum of real and stability losses)
        total_loss_batch = loss_real + loss_stability
        total_loss_batch.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Step the scheduler after each batch update.
        scheduler.step()

        # Accumulate losses.
        total_loss += total_loss_batch.item()
        total_loss_node += (loss_dict_real.get("loss_node", 0.0) + loss_dict_stability.get("loss_node", 0.0)).item()
        total_loss_edge += (loss_dict_real.get("loss_edge", 0.0) + loss_dict_stability.get("loss_edge", 0.0)).item()

        total_loss_node_head += (loss_dict_real.get("loss_node_head", 0.0) + loss_dict_stability.get("loss_node_head", 0.0)).item()
        total_loss_node_inflow += (loss_dict_real.get("loss_node_inflow", 0.0) + loss_dict_stability.get("loss_node_inflow", 0.0)).item()
        total_loss_edge_flow += (loss_dict_real.get("loss_edge_flow", 0.0) + loss_dict_stability.get("loss_edge_flow", 0.0)).item()
        total_loss_edge_depth += (loss_dict_real.get("loss_edge_depth", 0.0) + loss_dict_stability.get("loss_edge_depth", 0.0)).item()

        total_perc_error_node_head += (loss_dict_real.get("perc_error_node_head", 0.0) + loss_dict_stability.get("perc_error_node_head", 0.0)).item()
        total_perc_error_node_inflow += (loss_dict_real.get("perc_error_node_inflow", 0.0) + loss_dict_stability.get("perc_error_node_inflow", 0.0)).item()
        total_perc_error_edge_flow += (loss_dict_real.get("perc_error_edge_flow", 0.0) + loss_dict_stability.get("perc_error_edge_flow", 0.0)).item()
        total_perc_error_edge_depth += (loss_dict_real.get("perc_error_edge_depth", 0.0) + loss_dict_stability.get("perc_error_edge_depth", 0.0)).item()
        total_perc_error_total += (loss_dict_real.get("perc_error_total", 0.0) + loss_dict_stability.get("perc_error_total", 0.0)).item()

    # Average the accumulated values over the number of batches.
    avg_loss = total_loss / num_batches
    avg_loss_node = total_loss_node / num_batches
    avg_loss_edge = total_loss_edge / num_batches

    avg_loss_node_head = total_loss_node_head / num_batches
    avg_loss_node_inflow = total_loss_node_inflow / num_batches
    avg_loss_edge_flow = total_loss_edge_flow / num_batches
    avg_loss_edge_depth = total_loss_edge_depth / num_batches

    avg_perc_error_node_head = total_perc_error_node_head / num_batches
    avg_perc_error_node_inflow = total_perc_error_node_inflow / num_batches
    avg_perc_error_edge_flow = total_perc_error_edge_flow / num_batches
    avg_perc_error_edge_depth = total_perc_error_edge_depth / num_batches
    avg_perc_error_total = total_perc_error_total / num_batches

    return {
        "avg_loss": avg_loss,
        "avg_loss_node": avg_loss_node,
        "avg_loss_edge": avg_loss_edge,
        "avg_loss_node_head": avg_loss_node_head,
        "avg_loss_node_inflow": avg_loss_node_inflow,
        "avg_loss_edge_flow": avg_loss_edge_flow,
        "avg_loss_edge_depth": avg_loss_edge_depth,
        "avg_perc_error_node_head": avg_perc_error_node_head,
        "avg_perc_error_node_inflow": avg_perc_error_node_inflow,
        "avg_perc_error_edge_flow": avg_perc_error_edge_flow,
        "avg_perc_error_edge_depth": avg_perc_error_edge_depth,
        "avg_perc_error_total": avg_perc_error_total,
    }

##############################################
# Validation Loop for Real Loss
##############################################
@torch.no_grad()
def validate_real_loss(model, dataloader, device, n_time_steps, max_steps_ahead, loss_config, edge_info_list):
    """
    Validation function for the real loss branch.
    Uses ground-truth updates (i.e. shifting the full graph with ground truth).
    """
    model.eval()
    total_loss = 0.0
    num_batches = len(dataloader)
    pbar = tqdm(dataloader, desc="Validating", leave=False)

    for batch_idx, data in enumerate(pbar):
        data = data.to(device)
        batch_size = data.num_graphs
        # Repeat edge_info_list for the batch.
        batch_edge_info_list = edge_info_list * batch_size

        # Full graphs: contain all n_time_steps dynamic values.
        x_full = data.x.clone()
        edge_full = data.edge_attr.clone()

        # Permuted targets: shape [max_steps_ahead, num_nodes, target_features]
        y_tensor = data.y.permute(1, 0, 2)
        y_edges_tensor = data.y_edges.permute(1, 0, 2)

        predictions_node, predictions_edge = [], []
        x_real = x_full.clone()
        edge_real = edge_full.clone()
        node_type = getattr(data, "node_type", None)

        for step in range(max_steps_ahead):
            # Prepare input by removing the oldest dynamic column.
            node_input = prepare_node_input(x_real, n_time_steps)
            edge_input = prepare_edge_input(edge_real, n_time_steps)
            node_pred, edge_pred = model(node_input, data.edge_index, edge_input, node_type=node_type)
            predictions_node.append(node_pred)
            predictions_edge.append(edge_pred)

            # For steps before the last, update the full graph using ground truth.
            if step < max_steps_ahead - 1:
                x_real = shift_full_graph_left(x_real, y_tensor[step], data.future_rainfall, n_time_steps, step_idx=step)
                edge_real = shift_full_graph_left_edge(
                    edge_real,
                    y_edges_tensor[step],
                    y_tensor[step],
                    data.future_edge_rainfall,
                    n_time_steps,
                    edge_info_list=batch_edge_info_list,
                    step_idx = step
                )
        predictions_node = torch.stack(predictions_node, dim=0)
        predictions_edge = torch.stack(predictions_edge, dim=0)
        loss_dict = swmm_gnn_loss(
            preds_node=predictions_node,
            preds_edge=predictions_edge,
            targets_node=y_tensor,
            targets_edge=y_edges_tensor,
            loss_type="relative_mse" , #config.get("loss_type", "relative_mse"),
            config=loss_config,
            edge_info_list=batch_edge_info_list,
            apply_penalties=False,
            use_diff_loss = False
        )
        loss = loss_dict["total_loss"]
        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())
    avg_loss = total_loss / num_batches
    return avg_loss

##############################################
# Validation Loop for Stability Loss
##############################################
@torch.no_grad()
def validate_stability_loss(model, dataloader, device, n_time_steps, max_steps_ahead, loss_config, edge_info_list):
    """
    Validation function for the stability loss branch.
    Uses recursive pushforward updates.

    The process is as follows for each batch:
      1. Trim the full graph (removing current t) so that the input contains only (t-2, t-1).
      2. Make an initial prediction using this trimmed input; update the full graph by replacing the current t with the prediction.
      3. For subsequent steps, repeatedly use pushforward updates to predict t+1, t+2, etc.
    """
    model.eval()
    total_loss = 0.0
    num_batches = len(dataloader)
    pbar = tqdm(dataloader, desc="Validating", leave=False)

    for batch_idx, data in enumerate(pbar):
        data = data.to(device)
        batch_size = data.num_graphs
        batch_edge_info_list = edge_info_list * batch_size

        x_full = data.x.clone()
        edge_full = data.edge_attr.clone()
        y_tensor = data.y.permute(1, 0, 2)
        y_edges_tensor = data.y_edges.permute(1, 0, 2)

        stability_node_preds, stability_edge_preds = [], []
        # Trim the full graph to remove current t (keeping t-2 and t-1).
        x_stability = prepare_stability_first_step_input(x_full, n_time_steps)
        edge_stability = prepare_stability_first_step_edge(edge_full, n_time_steps)
        node_type = getattr(data, "node_type", None)

        # Make the initial prediction using the trimmed graph.
        node_input = x_stability
        edge_input = edge_stability
        node_pred_initial, edge_pred_initial = model(node_input, data.edge_index, edge_input, node_type=node_type)

        # Update the full graph by inserting the predicted values for t.
        x_stability_full = update_current_graph_with_prediction(x_full, node_pred_initial.detach(), n_time_steps)
        edge_stability_full = update_current_edge_with_prediction(
            edge_full,
            edge_pred_initial.detach(),
            node_pred_initial.detach(),
            n_time_steps,
            edge_info_list=batch_edge_info_list
        )

        # For subsequent steps, generate predictions and update using pushforward.
        for step in range(0, max_steps_ahead):
            node_input = prepare_node_input(x_stability_full, n_time_steps)
            edge_input = prepare_edge_input(edge_stability_full, n_time_steps)
            node_pred, edge_pred = model(node_input, data.edge_index, edge_input, node_type=node_type)
            stability_node_preds.append(node_pred)
            stability_edge_preds.append(edge_pred)

            # Update the graph only if there will be a next prediction.
            if step < max_steps_ahead - 1:
                x_stability_full = use_prediction_pushforward(
                    x_stability_full,
                    node_pred.detach(),
                    data.future_rainfall,
                    n_time_steps,
                    max_steps_ahead,
                    step_idx=step
                )
                edge_stability_full = use_edge_prediction_pushforward(
                    edge_stability_full,
                    edge_pred.detach(),
                    node_pred.detach(),
                    data.future_edge_rainfall,
                    n_time_steps,
                    edge_info_list=batch_edge_info_list,
                    step_idx=step
                )
        stability_node_preds = torch.stack(stability_node_preds, dim=0)
        stability_edge_preds = torch.stack(stability_edge_preds, dim=0)
        loss_dict = swmm_gnn_loss(
            preds_node=stability_node_preds,
            preds_edge=stability_edge_preds,
            targets_node=y_tensor,
            targets_edge=y_edges_tensor,
            loss_type="relative_mse", #config.get("loss_type", "relative_mse"),
            config=loss_config,
            edge_info_list=batch_edge_info_list,
            apply_penalties=False,
            use_diff_loss=False
        )
        loss = loss_dict["total_loss"]
        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())
    avg_loss = total_loss / num_batches
    return avg_loss

############################################
# Validation Function with Pushforward Trick
############################################
def validate_pushforward(model, dataloader, device, n_time_steps, max_steps_ahead, loss_config, edge_info_list):
    """
    Combined validation function that computes both:
      (1) Real Loss: Uses ground-truth updates.
      (2) Stability Loss: Uses recursive pushforward updates.
    The total loss (and all individual metrics) are averaged over the dataset.
    """
    with torch.no_grad():
        model.eval()

        # Overall accumulators.
        total_loss = 0.0
        total_loss_node = 0.0
        total_loss_edge = 0.0

        # Accumulators for individual node losses.
        total_loss_node_head = 0.0
        total_loss_node_inflow = 0.0
        # Accumulators for individual edge losses.
        total_loss_edge_flow = 0.0
        total_loss_edge_depth = 0.0

        # Accumulators for percentage error metrics (when using relative_mse).
        total_perc_error_node_head = 0.0
        total_perc_error_node_inflow = 0.0
        total_perc_error_edge_flow = 0.0
        total_perc_error_edge_depth = 0.0
        total_perc_error_total = 0.0

        num_batches = len(dataloader)
        pbar = tqdm(dataloader, desc="Validating", leave=False)

        for batch_idx, data in enumerate(pbar):
            data = data.to(device)
            batch_size = data.num_graphs
            batch_edge_info_list = edge_info_list * batch_size

            # --- Prepare Full Graphs and Targets ---
            x_full = data.x.clone()  # Full node features.
            edge_full = data.edge_attr.clone()  # Full edge features.
            # Permute targets: shape [max_steps_ahead, num_nodes, target_features]
            y_tensor = data.y.permute(1, 0, 2)
            y_edges_tensor = data.y_edges.permute(1, 0, 2)

            #### Real Loss Branch ####
            predictions_node, predictions_edge = [], []
            x_real = x_full.clone()
            edge_real = edge_full.clone()
            node_type = getattr(data, "node_type", None)

            for step in range(max_steps_ahead):
                node_input = prepare_node_input(x_real, n_time_steps)
                edge_input = prepare_edge_input(edge_real, n_time_steps)
                node_pred, edge_pred = model(node_input, data.edge_index, edge_input, node_type=node_type)
                predictions_node.append(node_pred)
                predictions_edge.append(edge_pred)

                if step < max_steps_ahead - 1:
                    # Use full ground-truth future rainfall.
                    x_real = shift_full_graph_left(x_real, y_tensor[step], data.future_rainfall, n_time_steps, step_idx=step)
                    edge_real = shift_full_graph_left_edge(
                        edge_real,
                        y_edges_tensor[step],
                        y_tensor[step],
                        data.future_edge_rainfall,  # Updated to pass future edge rainfall.
                        n_time_steps,
                        edge_info_list=batch_edge_info_list,
                        step_idx=step
                    )
            predictions_node = torch.stack(predictions_node, dim=0)
            predictions_edge = torch.stack(predictions_edge, dim=0)
            loss_dict_real = swmm_gnn_loss(
                preds_node=predictions_node,
                preds_edge=predictions_edge,
                targets_node=y_tensor,
                targets_edge=y_edges_tensor,
                loss_type="relative_mse", #config.get("loss_type", "relative_mse"),
                config=loss_config,
                edge_info_list=batch_edge_info_list,
                apply_penalties=False,
                use_diff_loss=False
            )
            loss_real = loss_dict_real["total_loss"]

            #### Stability Loss Branch ####
            stability_node_preds, stability_edge_preds = [], []
            # Trim the full graph to remove current t (keeping t-2 and t-1)
            x_stability = prepare_stability_first_step_input(x_full, n_time_steps)
            edge_stability = prepare_stability_first_step_edge(edge_full, n_time_steps)
            node_type = getattr(data, "node_type", None)

            # Make the initial prediction using the trimmed graph.
            node_input = x_stability
            edge_input = edge_stability
            node_pred_initial, edge_pred_initial = model(node_input, data.edge_index, edge_input, node_type=node_type)
            # This prediction is used only to update the current graph.
            x_stability_full = update_current_graph_with_prediction(x_full, node_pred_initial.detach(), n_time_steps)
            edge_stability_full = update_current_edge_with_prediction(
                edge_full,
                edge_pred_initial.detach(),
                node_pred_initial.detach(),
                n_time_steps,
                edge_info_list=batch_edge_info_list
            )
            # Use full future rainfall directly.
            future_rainfall = data.future_rainfall  # [num_nodes, 1]

            # For subsequent steps, generate predictions.
            for step in range(0, max_steps_ahead):
                node_input = prepare_node_input(x_stability_full, n_time_steps)
                edge_input = prepare_edge_input(edge_stability_full, n_time_steps)
                node_pred, edge_pred = model(node_input, data.edge_index, edge_input, node_type=node_type)
                stability_node_preds.append(node_pred)
                stability_edge_preds.append(edge_pred)

                # Update only if there's another prediction.
                if step < max_steps_ahead - 1:
                    x_stability_full = use_prediction_pushforward(
                        x_stability_full,
                        node_pred.detach(),
                        future_rainfall,
                        n_time_steps,
                        max_steps_ahead,
                        step_idx=step
                    )
                    edge_stability_full = use_edge_prediction_pushforward(
                        edge_stability_full,
                        edge_pred.detach(),
                        node_pred.detach(),
                        data.future_edge_rainfall,  # Updated to pass future edge rainfall.
                        n_time_steps,
                        edge_info_list=batch_edge_info_list,
                        step_idx=step
                    )
            stability_node_preds = torch.stack(stability_node_preds, dim=0)
            stability_edge_preds = torch.stack(stability_edge_preds, dim=0)
            loss_dict_stability = swmm_gnn_loss(
                preds_node=stability_node_preds,
                preds_edge=stability_edge_preds,
                targets_node=y_tensor,
                targets_edge=y_edges_tensor,
                loss_type="relative_mse", #config.get("loss_type", "relative_mse"),
                config=loss_config,
                edge_info_list=batch_edge_info_list,
                apply_penalties=False,
                use_diff_loss=False
            )
            loss_stability = loss_dict_stability["total_loss"]

            # Combined loss for this batch.
            total_loss_batch = loss_real + loss_stability
            total_loss += total_loss_batch.item()
            total_loss_node += (loss_dict_real.get("loss_node", 0.0) + loss_dict_stability.get("loss_node", 0.0)).item()
            total_loss_edge += (loss_dict_real.get("loss_edge", 0.0) + loss_dict_stability.get("loss_edge", 0.0)).item()

            total_loss_node_head += (loss_dict_real.get("loss_node_head", 0.0) +
                                     loss_dict_stability.get("loss_node_head", 0.0)).item()
            total_loss_node_inflow += (loss_dict_real.get("loss_node_inflow", 0.0) +
                                       loss_dict_stability.get("loss_node_inflow", 0.0)).item()
            total_loss_edge_flow += (loss_dict_real.get("loss_edge_flow", 0.0) +
                                     loss_dict_stability.get("loss_edge_flow", 0.0)).item()
            total_loss_edge_depth += (loss_dict_real.get("loss_edge_depth", 0.0) +
                                      loss_dict_stability.get("loss_edge_depth", 0.0)).item()

            total_perc_error_node_head += (loss_dict_real.get("perc_error_node_head", 0.0) +
                                           loss_dict_stability.get("perc_error_node_head", 0.0)).item()
            total_perc_error_node_inflow += (loss_dict_real.get("perc_error_node_inflow", 0.0) +
                                             loss_dict_stability.get("perc_error_node_inflow", 0.0)).item()
            total_perc_error_edge_flow += (loss_dict_real.get("perc_error_edge_flow", 0.0) +
                                           loss_dict_stability.get("perc_error_edge_flow", 0.0)).item()
            total_perc_error_edge_depth += (loss_dict_real.get("perc_error_edge_depth", 0.0) +
                                            loss_dict_stability.get("perc_error_edge_depth", 0.0)).item()
            total_perc_error_total += (loss_dict_real.get("perc_error_total", 0.0) +
                                       loss_dict_stability.get("perc_error_total", 0.0)).item()

            pbar.set_postfix(loss=(loss_real.item() + loss_stability.item()))

        avg_loss = total_loss / num_batches
        avg_loss_node = total_loss_node / num_batches
        avg_loss_edge = total_loss_edge / num_batches
        avg_loss_node_head = total_loss_node_head / num_batches
        avg_loss_node_inflow = total_loss_node_inflow / num_batches
        avg_loss_edge_flow = total_loss_edge_flow / num_batches
        avg_loss_edge_depth = total_loss_edge_depth / num_batches
        avg_perc_error_node_head = total_perc_error_node_head / num_batches
        avg_perc_error_node_inflow = total_perc_error_node_inflow / num_batches
        avg_perc_error_edge_flow = total_perc_error_edge_flow / num_batches
        avg_perc_error_edge_depth = total_perc_error_edge_depth / num_batches
        avg_perc_error_total = total_perc_error_total / num_batches

        return {
            "avg_loss": avg_loss,
            "avg_loss_node": avg_loss_node,
            "avg_loss_edge": avg_loss_edge,
            "avg_loss_node_head": avg_loss_node_head,
            "avg_loss_node_inflow": avg_loss_node_inflow,
            "avg_loss_edge_flow": avg_loss_edge_flow,
            "avg_loss_edge_depth": avg_loss_edge_depth,
            "avg_perc_error_node_head": avg_perc_error_node_head,
            "avg_perc_error_node_inflow": avg_perc_error_node_inflow,
            "avg_perc_error_edge_flow": avg_perc_error_edge_flow,
            "avg_perc_error_edge_depth": avg_perc_error_edge_depth,
            "avg_perc_error_total": avg_perc_error_total,
        }

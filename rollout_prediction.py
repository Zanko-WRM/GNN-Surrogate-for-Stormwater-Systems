import torch

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

########################################
## Rollout update (Node and edge)
########################################
def rollout_update_node(x, pred, new_rainfall, new_acc_rainfall, config, n_time_steps):
    """
    Update the modified node features for rollout.

    Assumes x is arranged as:
      [constant, rainfall (n_mod), acc_rainfall (n_mod), head (n_mod), inflow (n_mod)]
    where n_mod = n_time_steps - 1.

    The function shifts each dynamic group left and appends:
      - new_rainfall (from full node rainfall),
      - new_acc_rainfall (from full node accumulated rainfall),
      - the model's predicted head and inflow.
    """
    n_mod = n_time_steps - 1
    num_constant = x.shape[1] - (4 * n_mod)

    rainfall = x[:, num_constant: num_constant + n_mod]
    acc_rainfall = x[:, num_constant + n_mod: num_constant + 2 * n_mod]
    head = x[:, num_constant + 2 * n_mod: num_constant + 3 * n_mod]
    inflow = x[:, num_constant + 3 * n_mod:]

    new_rainfall_window = torch.cat([rainfall[:, 1:], new_rainfall], dim=1)
    new_acc_rainfall_window = torch.cat([acc_rainfall[:, 1:], new_acc_rainfall], dim=1)
    new_head_window = torch.cat([head[:, 1:], pred[:, 0:1]], dim=1)
    new_inflow_window = torch.cat([inflow[:, 1:], pred[:, 1:2]], dim=1)

    updated_x = torch.cat([x[:, :num_constant],
                           new_rainfall_window,
                           new_acc_rainfall_window,
                           new_head_window,
                           new_inflow_window], dim=1)
    return updated_x


def rollout_update_edge(edge_attr, edge_pred, node_pred, new_rainfall, new_acc_rainfall, config, edge_info_list,
                        n_time_steps):
    """
    Update the modified edge features for rollout.

    Assumes edge_attr is arranged as:
      [constant, rainfall (n_mod), acc_rainfall (n_mod), diff_head (n_mod), diff_inflow (n_mod), flow (n_mod), depth (n_mod)]
    where n_mod = n_time_steps - 1.

    The update shifts each dynamic group left and appends:
      - new_rainfall and new_acc_rainfall from full edge rainfall,
      - new flow and depth from the predicted edge values,
      - new diff_head and diff_inflow computed from the predicted node values.
    """
    n_mod = n_time_steps - 1
    num_constant_edge = edge_attr.shape[1] - (6 * n_mod)

    rainfall = edge_attr[:, num_constant_edge: num_constant_edge + n_mod]
    acc_rainfall = edge_attr[:, num_constant_edge + n_mod: num_constant_edge + 2 * n_mod]
    diff_head = edge_attr[:, num_constant_edge + 2 * n_mod: num_constant_edge + 3 * n_mod]
    diff_inflow = edge_attr[:, num_constant_edge + 3 * n_mod: num_constant_edge + 4 * n_mod]
    flow = edge_attr[:, num_constant_edge + 4 * n_mod: num_constant_edge + 5 * n_mod]
    depth = edge_attr[:, num_constant_edge + 5 * n_mod:]

    new_rainfall_window = torch.cat([rainfall[:, 1:], new_rainfall], dim=1)
    new_acc_rainfall_window = torch.cat([acc_rainfall[:, 1:], new_acc_rainfall], dim=1)
    new_flow = torch.cat([flow[:, 1:], edge_pred[:, 0:1]], dim=1)
    new_depth = torch.cat([depth[:, 1:], edge_pred[:, 1:2]], dim=1)

    # For diff values, shift left and then append newly computed differences.
    new_diff_head = diff_head[:, 1:].clone()
    new_diff_inflow = diff_inflow[:, 1:].clone()
    num_edges = edge_attr.shape[0]
    new_diff_head_col = torch.empty((num_edges, 1), device=edge_attr.device)
    new_diff_inflow_col = torch.empty((num_edges, 1), device=edge_attr.device)
    for i in range(num_edges):
        info = edge_info_list[i]
        from_idx = info["from_node_idx"]
        to_idx = info["to_node_idx"]
        new_diff_head_col[i, 0] = node_pred[from_idx, 0] - node_pred[to_idx, 0]
        new_diff_inflow_col[i, 0] = node_pred[from_idx, 1] - node_pred[to_idx, 1]
    new_diff_head = torch.cat([new_diff_head, new_diff_head_col], dim=1)
    new_diff_inflow = torch.cat([new_diff_inflow, new_diff_inflow_col], dim=1)

    updated_edge_attr = torch.cat([
        edge_attr[:, :num_constant_edge],
        new_rainfall_window,
        new_acc_rainfall_window,
        new_diff_head,
        new_diff_inflow,
        new_flow,
        new_depth
    ], dim=1)
    return updated_edge_attr

#########################################
#### Rollout prediction
#########################################
def rollout_prediction(model, graph, rollout_steps, config, edge_info_list, device, n_time_steps):
    """
    Perform a rollout prediction using a modified initial graph.

    The modified graph is assumed to have node features arranged as:
       [constant, rainfall (t-1), rainfall (t), acc_rainfall (t-1), acc_rainfall (t),
        head (t-1), head (t), inflow (t-1), inflow (t)]
    and edge features as:
       [constant, rainfall (t-1), rainfall (t), acc_rainfall (t-1), acc_rainfall (t),
        diff_head (t-1), diff_head (t), diff_inflow (t-1), diff_inflow (t),
        flow (t-1), flow (t), depth (t-1), depth (t)]

    The graph has attributes:
       graph.full_rainfall and graph.full_acc_rainfall (for nodes)
       graph.full_edge_rainfall and graph.full_edge_acc_rainfall (for edges)
    which contain the full-length future rainfall series.

    At each rollout step, the function:
      1. Prepares the current input using prepare_node_input/prepare_edge_input (only once at start).
      2. Predicts the next time step.
      3. Extracts the next rainfall and acc_rainfall values from the full-length data.
      4. Updates the modified graph (with dynamic groups of length n_time_steps - 1) using the rollout update functions.
      5. Uses the updated graph as input for the next step.

    Returns:
      rollout_node_preds: Tensor of shape [rollout_steps, num_nodes, target_features] with node predictions.
      rollout_edge_preds: Tensor of shape [rollout_steps, num_edges, target_features] with edge predictions.
      graph: The updated graph after the rollout.
    """
    model.eval()
    graph = graph.to(device)
    rollout_node_preds = []
    rollout_edge_preds = []

    # Prepare initial inputs from the graph (only once).
    node_input = prepare_node_input(graph.x, n_time_steps)
    edge_input = prepare_edge_input(graph.edge_attr, n_time_steps)

    # We'll use a common index for both nodes and edges.
    current_rainfall_index = 0

    with torch.no_grad():
        for step in range(rollout_steps):
            node_type = getattr(graph, "node_type", None)
            node_pred, edge_pred = model(node_input, graph.edge_index, edge_input, node_type=node_type)
            rollout_node_preds.append(node_pred)
            rollout_edge_preds.append(edge_pred)

            # For nodes: extract the next rainfall and acc_rainfall values.
            if graph.full_rainfall.shape[1] > current_rainfall_index:
                new_node_rainfall = graph.full_rainfall[:, current_rainfall_index].unsqueeze(1)
            else:
                new_node_rainfall = torch.zeros((graph.x.shape[0], 1), device=device)
            if graph.full_acc_rainfall.shape[1] > current_rainfall_index:
                new_node_acc_rainfall = graph.full_acc_rainfall[:, current_rainfall_index].unsqueeze(1)
            else:
                new_node_acc_rainfall = torch.zeros((graph.x.shape[0], 1), device=device)

            # For edges: extract the next rainfall and acc_rainfall values.
            if graph.full_edge_rainfall.shape[1] > current_rainfall_index:
                new_edge_rainfall = graph.full_edge_rainfall[:, current_rainfall_index].unsqueeze(1)
            else:
                new_edge_rainfall = torch.zeros((graph.edge_attr.shape[0], 1), device=device)
            if graph.full_edge_acc_rainfall.shape[1] > current_rainfall_index:
                new_edge_acc_rainfall = graph.full_edge_acc_rainfall[:, current_rainfall_index].unsqueeze(1)
            else:
                new_edge_acc_rainfall = torch.zeros((graph.edge_attr.shape[0], 1), device=device)

            current_rainfall_index += 1

            # Update the modified node and edge inputs.
            node_input = rollout_update_node(node_input, node_pred.detach(),
                                             new_node_rainfall, new_node_acc_rainfall,
                                             config, n_time_steps)
            edge_input = rollout_update_edge(edge_input, edge_pred.detach(), node_pred.detach(),
                                             new_edge_rainfall, new_edge_acc_rainfall,
                                             config, edge_info_list, n_time_steps)

        rollout_node_preds = torch.stack(rollout_node_preds, dim=0)
        rollout_edge_preds = torch.stack(rollout_edge_preds, dim=0)

    return rollout_node_preds, rollout_edge_preds, graph

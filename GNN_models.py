import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from typing import List
import torch.nn.functional as F


# Function to initialize weights using Xavier normal initialization
def init_weights(layer):
    if isinstance(layer, nn.Linear):
        torch.nn.init.xavier_normal_(layer.weight)
        if layer.bias is not None:
            torch.nn.init.normal_(layer.bias)


# Function to build an MLP model with the specified structure
def build_mlp(input_size: int, hidden_layer_sizes: List[int], output_size: int = None,
              output_activation: nn.Module = nn.Identity, activation: nn.Module = nn.ReLU) -> nn.Module:
    layer_sizes = [input_size] + hidden_layer_sizes
    if output_size:
        layer_sizes.append(output_size)
    nlayers = len(layer_sizes) - 1
    act = [activation for _ in range(nlayers)]
    act[-1] = output_activation
    mlp = nn.Sequential()
    for i in range(nlayers):
        mlp.add_module("NN-" + str(i), nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        mlp.add_module("Act-" + str(i), act[i]())
    return mlp

##########################################################
# Modified Encoder: Separate branches for nodes and edges.
##########################################################
class Encoder(nn.Module):
    def __init__(self,
                 nnode_in_features: int,
                 nnode_out_features: int,
                 nedge_in_features: int,
                 nedge_out_features: int,
                 nmlp_layers: int,
                 mlp_hidden_dim: int,
                 removal_indices: List[int] = None  # List of column indices to remove for type_b nodes
                 ):
        super(Encoder, self).__init__()
        self.nnode_out_features = nnode_out_features
        self.removal_indices = removal_indices

        # Branch for type_a nodes: use full input.
        self.node_fn = nn.Sequential(
            build_mlp(nnode_in_features, [mlp_hidden_dim] * nmlp_layers, nnode_out_features),
            nn.LayerNorm(nnode_out_features)
        )

        # Branch for type_b nodes: if removal_indices is provided, the reduced input dimension is:
        if removal_indices is not None:
            self.nnode_in_features_reduced = nnode_in_features - len(removal_indices)
        else:
            self.nnode_in_features_reduced = nnode_in_features
        self.node_fn_b = nn.Sequential(
            build_mlp(self.nnode_in_features_reduced, [mlp_hidden_dim] * nmlp_layers, nnode_out_features),
            nn.LayerNorm(nnode_out_features)
        )

        # Edge encoder: Projects raw edge features to latent space.
        self.edge_fn = nn.Sequential(
            build_mlp(nedge_in_features, [mlp_hidden_dim] * nmlp_layers, nedge_out_features),
            nn.LayerNorm(nedge_out_features)
        )

    def forward(self, x: torch.Tensor, edge_features: torch.Tensor, node_type: torch.Tensor = None):
        """
        Args:
            x: Tensor of node features with shape [num_nodes, nnode_in_features].
            edge_features: Tensor of edge features.
            node_type: Optional LongTensor of shape [num_nodes] where 0 indicates type_a and 1 indicates type_b.
        Returns:
            node_latent: Tensor of shape [num_nodes, nnode_out_features].
            edge_latent: Tensor of edge latent features.
        """
        if node_type is not None and self.removal_indices is not None:
            num_nodes = x.size(0)
            node_latent = torch.empty(num_nodes, self.nnode_out_features, device=x.device)
            idx_a = (node_type == 0)
            idx_b = (node_type == 1)
            if idx_a.sum() > 0:
                node_latent[idx_a] = self.node_fn(x[idx_a])
            if idx_b.sum() > 0:
                # Create a boolean mask for the columns to keep.
                full_dim = x.size(1)
                keep_mask = torch.ones(full_dim, dtype=torch.bool, device=x.device)
                for idx in self.removal_indices:
                    keep_mask[idx] = False
                x_reduced = x[idx_b][:, keep_mask]
                node_latent[idx_b] = self.node_fn_b(x_reduced)
        else:
            node_latent = self.node_fn(x)

        edge_latent = self.edge_fn(edge_features)
        return node_latent, edge_latent


#########################################
# InteractionNetwork: Message Passing for Nodes & Updated Edge Aggregation.
#########################################
class InteractionNetwork(MessagePassing):
    def __init__(self, nnode_in: int, nnode_out: int, nedge_in: int, nedge_out: int,
                 nmlp_layers: int, mlp_hidden_dim: int):
        # Use 'add' aggregation for messages.
        super(InteractionNetwork, self).__init__(aggr='add')

        # MLP to compute messages from concatenated [x_i, x_j, edge_features].
        self.edge_fn = nn.Sequential(
            build_mlp(nnode_in * 2 + nedge_in, [mlp_hidden_dim] * nmlp_layers, nedge_out),
            nn.LayerNorm(nedge_out)
        )
        # MLP to update node features given aggregated messages and original node features.
        self.node_fn = nn.Sequential(
            build_mlp(nnode_in + nedge_out, [mlp_hidden_dim] * nmlp_layers, nnode_out),
            nn.LayerNorm(nnode_out)
        )
        # NEW: MLP to update edge features using updated node states.
        self.edge_update_fn = nn.Sequential(
            build_mlp(nnode_out * 2 + nedge_in, [mlp_hidden_dim] * nmlp_layers, nedge_out),
            nn.LayerNorm(nedge_out)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_features: torch.Tensor):
        # Save original node and edge features for residual connections.
        x_residual = x.clone()
        edge_residual = edge_features.clone()

        # ---- Node Message Passing ----
        # Propagate messages and aggregate for nodes.
        aggr_messages = self.propagate(edge_index=edge_index, x=x, edge_features=edge_features)
        # Update node features: concatenate aggregated messages with original node features and process.
        x_updated = self.node_fn(torch.cat([aggr_messages, x], dim=-1))

        # ---- Edge Update via Aggregation ----
        # For each edge, get updated node features from its endpoints.
        src, tgt = edge_index[0], edge_index[1]
        # Here, we use the updated node features from the processor.
        # Concatenate source and target node features with the original edge features.
        edge_input = torch.cat([x_updated[src], x_updated[tgt], edge_features], dim=-1)
        edge_updated = self.edge_update_fn(edge_input)

        # Apply residual connections.
        x_out = x_updated + x_residual
        edge_out = edge_updated + edge_residual

        return x_out, edge_out

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_features: torch.Tensor) -> torch.Tensor:
        # Compute messages for nodes by concatenating source, target, and edge features.
        m = torch.cat([x_i, x_j, edge_features], dim=-1)
        return self.edge_fn(m)



#########################################
# Processor: Stacks multiple InteractionNetwork layers.
#########################################
class Processor(nn.Module):
    def __init__(self, nnode_in: int, nnode_out: int, nedge_in: int, nedge_out: int,
                 nmessage_passing_steps: int, nmlp_layers: int, mlp_hidden_dim: int):
        super(Processor, self).__init__()
        self.message_passing_steps = nmessage_passing_steps
        self.gnn_stacks = nn.ModuleList([
            InteractionNetwork(
                nnode_in=nnode_in, nnode_out=nnode_out, nedge_in=nedge_in, nedge_out=nedge_out,
                nmlp_layers=nmlp_layers, mlp_hidden_dim=mlp_hidden_dim
            ) for _ in range(nmessage_passing_steps)
        ])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_features: torch.Tensor):
        # Sequentially apply multiple message-passing layers.
        for gnn in self.gnn_stacks:
            x, edge_features = gnn(x, edge_index, edge_features)
        return x, edge_features


#########################################
# Decoder for Nodes and Edges
#########################################
class Decoder(nn.Module):
    def __init__(self, nnode_in: int, nnode_out: int, nmlp_layers: int, mlp_hidden_dim: int):
        super(Decoder, self).__init__()
        self.node_fn = build_mlp(nnode_in, [mlp_hidden_dim for _ in range(nmlp_layers)], nnode_out)

    def forward(self, x: torch.Tensor):
        # Decode latent node representations into target outputs (e.g., head and inflow).
        return self.node_fn(x)


class EdgeDecoder(nn.Module):
    def __init__(self, nedge_in: int, nedge_out: int, nmlp_layers: int, mlp_hidden_dim: int):
        super(EdgeDecoder, self).__init__()
        self.edge_fn = build_mlp(nedge_in, [mlp_hidden_dim] * nmlp_layers, nedge_out)

    def forward(self, edge_features: torch.Tensor):
        # Decode latent edge representations into target outputs (e.g., flow and depth).
        return self.edge_fn(edge_features)


#########################################
# Dual Encode-Process-Decode Model
#########################################
class EncodeProcessDecodeDual(nn.Module):
    def __init__(self, nnode_in_features: int, nnode_out_features: int, nedge_in_features: int,
                 nedge_out_features: int, latent_dim: int, nmessage_passing_steps: int,
                 nmlp_layers: int, mlp_hidden_dim: int, residual: bool, n_time_steps: int, removal_indices: List[int] = None):
        super(EncodeProcessDecodeDual, self).__init__()
        # Encoder: Projects raw node and edge features to latent space.
        self.encoder = Encoder(nnode_in_features, latent_dim, nedge_in_features, latent_dim,
                               nmlp_layers, mlp_hidden_dim, removal_indices=removal_indices)
        # Processor: Applies several rounds of message passing to update both node and edge latent features.
        self.processor = Processor(latent_dim, latent_dim, latent_dim, latent_dim,
                                   nmessage_passing_steps, nmlp_layers, mlp_hidden_dim)
        # Decoders: Separate decoders for nodes and edges to produce final predictions.
        self.node_decoder = Decoder(latent_dim, nnode_out_features, nmlp_layers, mlp_hidden_dim)
        self.edge_decoder = EdgeDecoder(latent_dim, nedge_out_features, nmlp_layers, mlp_hidden_dim)
        self.residual = residual
        self.n_time_steps = n_time_steps

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, node_type: torch.Tensor = None):
        # === Compute Residuals for Stability ===
        if self.residual:

            # Effective number of dynamic steps after removal:
            n_eff = self.n_time_steps - 1

            dynamic_vars = n_eff * 4 # rainfall, acc_rainfall, depth, inflow
            input_size = x.shape[1]
            static_vars_size = input_size - dynamic_vars

            head_end = static_vars_size + 3 * n_eff  - 1
            flow_end = static_vars_size + 4 * n_eff  - 1

            node_residual = torch.zeros(x.size(0), 2, device=x.device)
            node_residual[:, 0] = x[:, head_end]  # Last head value
            node_residual[:, 1] = x[:, flow_end]  # Last inflow value

            # For edges:
            n_eff_edge = self.n_time_steps - 1
            dynamic_vars_edge = n_eff_edge * 6  # Here we have 6 dynamic blocks.
            input_size_edge = edge_attr.shape[1]
            static_vars_size_edge = input_size_edge - dynamic_vars_edge

            flow_end_edge = static_vars_size_edge + 5 * n_eff_edge - 1
            depth_end_edge = static_vars_size_edge + 6 * n_eff_edge - 1

            edge_residual = torch.zeros(edge_attr.shape[0], 2, device=edge_attr.device)
            edge_residual[:, 0] = edge_attr[:, flow_end_edge]  # Last flow value
            edge_residual[:, 1] = edge_attr[:, depth_end_edge]  # Last depth value
        else:
            node_residual = 0
            edge_residual = 0

        # === Encoding: Map raw inputs to latent space ===
        node_latent, edge_latent = self.encoder(x, edge_attr, node_type=node_type)
        # === Processing: Update both node and edge latent representations via message passing ===
        node_latent, edge_latent = self.processor(node_latent, edge_index, edge_latent)
        # === Decoding: Produce final predictions from latent representations ===
        node_pred = self.node_decoder(node_latent)
        edge_pred = self.edge_decoder(edge_latent)
        # === Apply Residual Connections (if enabled) ===
        if self.residual:
            node_pred = node_pred + node_residual
            edge_pred = edge_pred + edge_residual

        return node_pred, edge_pred

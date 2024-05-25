import pytorch_lightning as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.utils import dense_to_sparse, to_dense_adj, softmax
from torch_geometric.nn.conv import MessagePassing
from torchmetrics import Metric
import math
from motionnet.models.base_model.base_model import BaseModel
from typing import Optional, Union, Tuple
import json


### ----------------------------------- Transform input data ----------------------------------- ### 
def get_heading(cos):
    heading = np.arccos(cos)
    return heading
def get_centerline(lines, agent):
    """
    in a specific scenario, at a specific time step
    lines: map_polylines[scenario]
    agent: obj_trajs[scenario][0][timestep]
    """
    des_x = agent[0] - 1.3 * agent[34]
    des_y = agent[1] + 1.3 * agent[33]
    
    centerline=[]
    best_dist_x = np.inf
    best_dist_y = np.inf
    for i in range(len(lines)):
        for j in range(len(lines[i])):
            line_x = lines[i][j][0]
            line_y = lines[i][j][1]
            if (abs(line_x-des_x)<best_dist_x):
                best_dist_x=abs(line_x-des_x)
    #getcenterline
    return centerline
def get_features(data):
    feat_dict={}
    #center_dict={"length":}
    #new_dict['centerline']=
    return feat_dict


### ----------------------------------- Original Methods ----------------------------------- ### 
def drop_edge_between_samples(valid_mask: torch.Tensor, batch: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    if isinstance(batch, torch.Tensor):
        batch_matrix = batch.unsqueeze(-1) == batch.unsqueeze(-2)
    else:
        batch_src, batch_dst = batch
        batch_matrix = batch_src.unsqueeze(-1) == batch_dst.unsqueeze(-2)
    valid_mask = valid_mask * batch_matrix.unsqueeze(0)
    return valid_mask

def generate_reachable_matrix(edge_index: torch.Tensor, num_hops: int, max_nodes: int) -> list:
    values = torch.ones(edge_index.size(1), device=edge_index.device)
    sparse_mat = torch.sparse_coo_tensor(edge_index, values, torch.Size([max_nodes, max_nodes]))

    reach_matrices = []
    current_matrix = sparse_mat.clone()
    for _ in range(num_hops):
        current_matrix = current_matrix.coalesce()
        current_matrix = torch.sparse_coo_tensor(current_matrix.indices(), torch.ones_like(current_matrix.values()), current_matrix.size())

        edge_index_now = current_matrix.coalesce().indices()
        reach_matrices.append(edge_index_now)

        next_matrix = torch.sparse.mm(current_matrix, sparse_mat)

        current_matrix = next_matrix
    return reach_matrices

def generate_target(position: torch.Tensor, mask: torch.Tensor, num_historical_steps: int, num_future_steps: int) -> Tuple[torch.Tensor, torch.Tensor]:
    target_traj = [position[:, i+1:i+1+num_future_steps] for i in range(num_historical_steps)]
    target_traj = torch.stack(target_traj, dim=1)
    target_mask = [mask[:,i+1:i+1+num_future_steps] for i in range(num_historical_steps)]
    target_mask = torch.stack(target_mask, dim=1)
    return target_traj, target_mask

def generate_predict_mask(visible_mask: torch.Tensor, num_visible_steps: int) -> torch.Tensor:
    
    window = torch.ones((1, num_visible_steps), dtype=torch.float32, device=visible_mask.device)

    conv_result = torch.nn.functional.conv2d(visible_mask.float().unsqueeze(0).unsqueeze(0), window.unsqueeze(0).unsqueeze(0))
    conv_result = conv_result.squeeze(0).squeeze(0)
    
    predict_mask = conv_result == num_visible_steps
    predict_mask = torch.cat([torch.zeros((visible_mask.size(0), num_visible_steps-1), dtype=torch.bool, device=visible_mask.device), predict_mask], dim=1)
    
    return predict_mask

def wrap_angle(angle: torch.Tensor, min_val: float = -math.pi, max_val: float = math.pi) -> torch.Tensor:
    return min_val + (angle + max_val) % (max_val - min_val)

def compute_angles_lengths_2D(vectors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    length = torch.norm(vectors, dim=-1)
    theta = torch.atan2(vectors[..., 1], vectors[..., 0])
    return length, theta

def generate_clockwise_rotation_matrix(angle: torch.Tensor) -> torch.Tensor:
    matrix = torch.zeros_like(angle).unsqueeze(-1).repeat_interleave(2,-1).unsqueeze(-1).repeat_interleave(2,-1)
    matrix[..., 0, 0] = torch.cos(angle)
    matrix[..., 0, 1] = torch.sin(angle)
    matrix[..., 1, 0] = -torch.sin(angle)
    matrix[..., 1, 1] = torch.cos(angle)
    return matrix

def generate_counterclockwise_rotation_matrix(angle: torch.Tensor) -> torch.Tensor:
    matrix = torch.zeros_like(angle).unsqueeze(-1).repeat_interleave(2,-1).unsqueeze(-1).repeat_interleave(2,-1)
    matrix[..., 0, 0] = torch.cos(angle)
    matrix[..., 0, 1] = -torch.sin(angle)
    matrix[..., 1, 0] = torch.sin(angle)
    matrix[..., 1, 1] = torch.cos(angle)
    return matrix

def transform_point_to_local_coordinate(point: torch.Tensor, position: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
    point = point - position
    rotation_matrix = generate_clockwise_rotation_matrix(heading)
    point = torch.matmul(rotation_matrix, point.unsqueeze(-1)).squeeze(-1)
    return point

def transform_traj_to_local_coordinate(traj: torch.Tensor, position: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
    traj = traj - position.unsqueeze(-2)
    rotation_matrix = generate_clockwise_rotation_matrix(heading)
    traj = torch.matmul(rotation_matrix.unsqueeze(-3), traj.unsqueeze(-1)).squeeze(-1)
    return traj

def transform_traj_to_global_coordinate(traj: torch.Tensor, position: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
    rotation_matrix = generate_counterclockwise_rotation_matrix(heading)
    traj = torch.matmul(rotation_matrix.unsqueeze(-3), traj.unsqueeze(-1)).squeeze(-1)
    traj = traj + position.unsqueeze(-2)
    return traj

def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        fan_in = m.in_channels / m.groups
        fan_out = m.out_channels / m.groups
        bound = (6.0 / (fan_in + fan_out)) ** 0.5
        nn.init.uniform_(m.weight, -bound, bound)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.MultiheadAttention):
        if m.in_proj_weight is not None:
            fan_in = m.embed_dim 
            fan_out = m.embed_dim
            bound = (6.0 / (fan_in + fan_out)) ** 0.5
            nn.init.uniform_(m.in_proj_weight, -bound, bound)
        else:
            nn.init.xavier_uniform_(m.q_proj_weight)
            nn.init.xavier_uniform_(m.k_proj_weight)
            nn.init.xavier_uniform_(m.v_proj_weight)
        if m.in_proj_bias is not None:
            nn.init.zeros_(m.in_proj_bias)
        nn.init.xavier_uniform_(m.out_proj.weight)
        if m.out_proj.bias is not None:
            nn.init.zeros_(m.out_proj.bias)
        if m.bias_k is not None: 
            nn.init.normal_(m.bias_k, mean=0.0, std=0.02)
        if m.bias_v is not None:
            nn.init.normal_(m.bias_v, mean=0.0, std=0.02)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(4, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(4, 0):
                    nn.init.orthogonal_(hh)
            elif 'weight_hr' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)
                nn.init.ones_(param.chunk(4, 0)[1])
    elif isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(3, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(3, 0):
                    nn.init.orthogonal_(hh)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)

class GraphAttention(MessagePassing):

    def __init__(self,
                 hidden_dim: int,
                 num_heads: int,
                 dropout: float,
                 has_edge_attr: bool,
                 if_self_attention: bool,
                 **kwargs) -> None:
        super(GraphAttention, self).__init__(aggr='add', node_dim=0, **kwargs)
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.has_edge_attr = has_edge_attr
        self.if_self_attention = if_self_attention

        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        if has_edge_attr:
            self.edge_k = nn.Linear(hidden_dim, hidden_dim)
            self.edge_v = nn.Linear(hidden_dim, hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.attn_drop = nn.Dropout(dropout)
        if if_self_attention:
            self.mha_prenorm_src = nn.LayerNorm(hidden_dim)
        else:
            self.mha_prenorm_src = nn.LayerNorm(hidden_dim)
            self.mha_prenorm_dst = nn.LayerNorm(hidden_dim)
        if has_edge_attr:
            self.mha_prenorm_edge = nn.LayerNorm(hidden_dim)
        self.ffn_prenorm = nn.LayerNorm(hidden_dim)
        self.apply(init_weights)

    def forward(self,
                x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.if_self_attention:
            x_src = x_dst = self.mha_prenorm_src(x)
        else:
            x_src, x_dst = x
            x_src = self.mha_prenorm_src(x_src)
            x_dst = self.mha_prenorm_dst(x_dst)
        if self.has_edge_attr:
            edge_attr = self.mha_prenorm_edge(edge_attr)
        x_dst = x_dst + self._mha_layer(x_src, x_dst, edge_index, edge_attr)
        x_dst = x_dst + self._ffn_layer(self.ffn_prenorm(x_dst))
        return x_dst

    def message(self,
                x_dst_i: torch.Tensor,
                x_src_j: torch.Tensor,
                edge_attr: Optional[torch.Tensor],
                index: torch.Tensor,
                ptr: Optional[torch.Tensor]) -> torch.Tensor:
        query_i = self.q(x_dst_i).view(-1, self.num_heads, self.head_dim)
        key_j = self.k(x_src_j).view(-1, self.num_heads, self.head_dim)
        value_j = self.v(x_src_j).view(-1, self.num_heads, self.head_dim)
        if self.has_edge_attr:
            key_j = key_j + self.edge_k(edge_attr).view(-1, self.num_heads, self.head_dim)
            value_j = value_j + self.edge_v(edge_attr).view(-1, self.num_heads, self.head_dim)
        scale = self.head_dim ** 0.5
        weight = (query_i * key_j).sum(dim=-1) / scale
        weight = softmax(weight, index, ptr)
        weight = self.attn_drop(weight)
        return (value_j * weight.unsqueeze(-1)).view(-1, self.num_heads*self.head_dim)

    def _mha_layer(self,
                   x_src: torch.Tensor,
                   x_dst: torch.Tensor,
                   edge_index: torch.Tensor,
                   edge_attr: Optional[torch.Tensor]=None) -> torch.Tensor:
        return self.propagate(edge_index=edge_index, edge_attr=edge_attr, x_dst=x_dst, x_src=x_src)

    def _ffn_layer(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)

class TwoLayerMLP(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int) -> None:
        super(TwoLayerMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
        self.apply(init_weights)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.mlp(input)

class Backbone(nn.Module):

    def __init__(self,
                 hidden_dim: int,
                 num_historical_steps: int,
                 num_future_steps: int,
                 pos_duration: int,
                 pred_duration: int,
                 a2a_radius: float,
                 l2a_radius: float,
                 num_attn_layers: int, 
                 num_modes: int,
                 num_heads: int,
                 dropout: float) -> None:
        super(Backbone, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.pos_duration = pos_duration
        self.pred_duration = pred_duration
        self.a2a_radius = a2a_radius
        self.l2a_radius = l2a_radius
        self.num_attn_layers = num_attn_layers
        self.num_modes = num_modes
        self.num_heads = num_heads
        self.dropout = dropout

        self.mode_tokens = nn.Embedding(num_modes, hidden_dim)     #[K,D]

        self.a_emb_layer = TwoLayerMLP(input_dim=1, hidden_dim=hidden_dim, output_dim=hidden_dim)
        
        self.l2m_emb_layer = TwoLayerMLP(input_dim=3, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.t2m_emb_layer = TwoLayerMLP(input_dim=4, hidden_dim=hidden_dim, output_dim=hidden_dim)

        self.m2m_h_emb_layer = TwoLayerMLP(input_dim=4, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.m2m_a_emb_layer = TwoLayerMLP(input_dim=3, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.m2m_s_emb_layer = TwoLayerMLP(input_dim=3, hidden_dim=hidden_dim, output_dim=hidden_dim)

        self.l2m_attn_layer = GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=False)
        self.t2m_attn_layer = GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=False)

        self.m2m_h_attn_layers = nn.ModuleList([GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=True) for _ in range(num_attn_layers)])
        self.m2m_a_attn_layers = nn.ModuleList([GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=True) for _ in range(num_attn_layers)])
        self.m2m_s_attn_layers = nn.ModuleList([GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=False, if_self_attention=True) for _ in range(num_attn_layers)])

        self.traj_propose = TwoLayerMLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=self.num_future_steps*2)

        self.proposal_to_anchor = TwoLayerMLP(input_dim=self.num_future_steps*2, hidden_dim=hidden_dim, output_dim=hidden_dim)

        self.l2n_attn_layer = GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=False)
        self.t2n_attn_layer = GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=False)

        self.n2n_h_attn_layers = nn.ModuleList([GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=True) for _ in range(num_attn_layers)])
        self.n2n_a_attn_layers = nn.ModuleList([GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=True) for _ in range(num_attn_layers)])
        self.n2n_s_attn_layers = nn.ModuleList([GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=True) for _ in range(num_attn_layers)])

        self.traj_refine = TwoLayerMLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=self.num_future_steps*2)

        self.prob_decoder = TwoLayerMLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=1)    
        self.prob_norm = nn.Softmax(dim=-1)

        self.apply(init_weights)

    def forward(self, data, l_embs: torch.Tensor) -> torch.Tensor:
        # initialization
        a_length = data['agent']['length']                          #[(N1,...,Nb),H]
        a_embs = self.a_emb_layer(input=a_length.unsqueeze(-1))    #[(N1,...,Nb),H,D]
        
        num_all_agent = a_length.size(0)                # N1+...+Nb
        m_embs = self.mode_tokens.weight.unsqueeze(0).repeat_interleave(self.num_historical_steps,0)            #[H,K,D]
        m_embs = m_embs.unsqueeze(0).repeat_interleave(num_all_agent,0).reshape(-1, self.hidden_dim)            #[(N1,...,Nb)*H*K,D]

        m_batch = data['agent']['batch'].unsqueeze(1).repeat_interleave(self.num_modes,1)                       # [(N1,...,Nb),K]
        m_position = data['agent']['position'][:,:self.num_historical_steps].unsqueeze(2).repeat_interleave(self.num_modes,2)  #[(N1,...,Nb),H,K,2]
        m_heading = data['agent']['heading'].unsqueeze(2).repeat_interleave(self.num_modes,2)                   #[(N1,...,Nb),H,K]
        m_valid_mask = data['agent']['visible_mask'][:,:self.num_historical_steps].unsqueeze(2).repeat_interleave(self.num_modes,2)  #[(N1,...,Nb),H,K]

        #ALL EDGE
        #t2m edge
        t2m_position_t = data['agent']['position'][:,:self.num_historical_steps].reshape(-1,2)      #[(N1,...,Nb)*H,2]
        t2m_position_m = m_position.reshape(-1,2)                                                   #[(N1,...,Nb)*H*K,2]
        t2m_heading_t = data['agent']['heading'].reshape(-1)                                        #[(N1,...,Nb)]
        t2m_heading_m = m_heading.reshape(-1)                                                       #[(N1,...,Nb)*H*K]
        t2m_valid_mask_t = data['agent']['visible_mask'][:,:self.num_historical_steps]              #[(N1,...,Nb),H]
        t2m_valid_mask_m = m_valid_mask.reshape(num_all_agent,-1)                                   #[(N1,...,Nb),H*K]
        t2m_valid_mask = t2m_valid_mask_t.unsqueeze(2) & t2m_valid_mask_m.unsqueeze(1)              #[(N1,...,Nb),H,H*K]
        t2m_edge_index = dense_to_sparse(t2m_valid_mask)[0]
        t2m_edge_index = t2m_edge_index[:, torch.floor(t2m_edge_index[1]/self.num_modes) >= t2m_edge_index[0]]
        t2m_edge_index = t2m_edge_index[:, torch.floor(t2m_edge_index[1]/self.num_modes) - t2m_edge_index[0] <= self.pos_duration]
        t2m_edge_vector = transform_point_to_local_coordinate(t2m_position_t[t2m_edge_index[0]], t2m_position_m[t2m_edge_index[1]], t2m_heading_m[t2m_edge_index[1]])
        t2m_edge_attr_length, t2m_edge_attr_theta = compute_angles_lengths_2D(t2m_edge_vector)
        t2m_edge_attr_heading = wrap_angle(t2m_heading_t[t2m_edge_index[0]] - t2m_heading_m[t2m_edge_index[1]])
        t2m_edge_attr_interval = t2m_edge_index[0] - torch.floor(t2m_edge_index[1]/self.num_modes)
        t2m_edge_attr_input = torch.stack([t2m_edge_attr_length, t2m_edge_attr_theta, t2m_edge_attr_heading, t2m_edge_attr_interval], dim=-1)
        t2m_edge_attr_embs = self.t2m_emb_layer(input=t2m_edge_attr_input)

        #l2m edge
        l2m_position_l = data['lane']['position']                       #[(M1,...,Mb),2]
        l2m_position_m = m_position.reshape(-1,2)                       #[(N1,...,Nb)*H*K,2]
        l2m_heading_l = data['lane']['heading']                         #[(M1,...,Mb)]
        l2m_heading_m = m_heading.reshape(-1)                           #[(N1,...,Nb)]
        l2m_batch_l = data['lane']['batch']                             #[(M1,...,Mb)]
        l2m_batch_m = m_batch.unsqueeze(1).repeat_interleave(self.num_historical_steps,1).reshape(-1)       #[(N1,...,Nb)*H*K]
        l2m_valid_mask_l = data['lane']['visible_mask']                                                     #[(M1,...,Mb)]
        l2m_valid_mask_m = m_valid_mask.reshape(-1)                                                         #[(N1,...,Nb)*H*K]
        l2m_valid_mask = l2m_valid_mask_l.unsqueeze(1)&l2m_valid_mask_m.unsqueeze(0)                        #[(M1,...,Mb),(N1,...,Nb)*H*K]
        l2m_valid_mask = drop_edge_between_samples(l2m_valid_mask, batch=(l2m_batch_l, l2m_batch_m))
        l2m_edge_index = dense_to_sparse(l2m_valid_mask)[0]
        l2m_edge_index = l2m_edge_index[:, torch.norm(l2m_position_l[l2m_edge_index[0]] - l2m_position_m[l2m_edge_index[1]], p=2, dim=-1) < self.l2a_radius]
        l2m_edge_vector = transform_point_to_local_coordinate(l2m_position_l[l2m_edge_index[0]], l2m_position_m[l2m_edge_index[1]], l2m_heading_m[l2m_edge_index[1]])
        l2m_edge_attr_length, l2m_edge_attr_theta = compute_angles_lengths_2D(l2m_edge_vector)
        l2m_edge_attr_heading = wrap_angle(l2m_heading_l[l2m_edge_index[0]] - l2m_heading_m[l2m_edge_index[1]])
        l2m_edge_attr_input = torch.stack([l2m_edge_attr_length, l2m_edge_attr_theta, l2m_edge_attr_heading], dim=-1)
        l2m_edge_attr_embs = self.l2m_emb_layer(input=l2m_edge_attr_input)

        #mode edge
        #m2m_a_edge
        m2m_a_position = m_position.permute(1,2,0,3).reshape(-1, 2)    #[H*K*(N1,...,Nb),2]
        m2m_a_heading = m_heading.permute(1,2,0).reshape(-1)           #[H*K*(N1,...,Nb)]
        m2m_a_batch = data['agent']['batch']                           #[(N1,...,Nb)]
        m2m_a_valid_mask = m_valid_mask.permute(1,2,0).reshape(self.num_historical_steps * self.num_modes, -1)  #[H*K,(N1,...,Nb)]
        m2m_a_valid_mask = m2m_a_valid_mask.unsqueeze(2) & m2m_a_valid_mask.unsqueeze(1)                        #[H*K,(N1,...,Nb),(N1,...,Nb)]
        m2m_a_valid_mask = drop_edge_between_samples(m2m_a_valid_mask, m2m_a_batch)
        m2m_a_edge_index = dense_to_sparse(m2m_a_valid_mask)[0]
        m2m_a_edge_index = m2m_a_edge_index[:, m2m_a_edge_index[1] != m2m_a_edge_index[0]]
        m2m_a_edge_index = m2m_a_edge_index[:, torch.norm(m2m_a_position[m2m_a_edge_index[1]] - m2m_a_position[m2m_a_edge_index[0]],p=2,dim=-1) < self.a2a_radius]
        m2m_a_edge_vector = transform_point_to_local_coordinate(m2m_a_position[m2m_a_edge_index[0]], m2m_a_position[m2m_a_edge_index[1]], m2m_a_heading[m2m_a_edge_index[1]])
        m2m_a_edge_attr_length, m2m_a_edge_attr_theta = compute_angles_lengths_2D(m2m_a_edge_vector)
        m2m_a_edge_attr_heading = wrap_angle(m2m_a_heading[m2m_a_edge_index[0]] - m2m_a_heading[m2m_a_edge_index[1]])
        m2m_a_edge_attr_input = torch.stack([m2m_a_edge_attr_length, m2m_a_edge_attr_theta, m2m_a_edge_attr_heading], dim=-1)
        m2m_a_edge_attr_embs = self.m2m_a_emb_layer(input=m2m_a_edge_attr_input)

        #m2m_h                        
        m2m_h_position = m_position.permute(2,0,1,3).reshape(-1, 2)    #[K*(N1,...,Nb)*H,2]
        m2m_h_heading = m_heading.permute(2,0,1).reshape(-1)           #[K*(N1,...,Nb)*H]
        m2m_h_valid_mask = m_valid_mask.permute(2,0,1).reshape(-1, self.num_historical_steps)   #[K*(N1,...,Nb),H]
        m2m_h_valid_mask = m2m_h_valid_mask.unsqueeze(2) & m2m_h_valid_mask.unsqueeze(1)        #[K*(N1,...,Nb),H,H]     
        m2m_h_edge_index = dense_to_sparse(m2m_h_valid_mask)[0]
        m2m_h_edge_index = m2m_h_edge_index[:, m2m_h_edge_index[1] > m2m_h_edge_index[0]]
        m2m_h_edge_index = m2m_h_edge_index[:, m2m_h_edge_index[1] - m2m_h_edge_index[0] <= self.pred_duration]
        m2m_h_edge_vector = transform_point_to_local_coordinate(m2m_h_position[m2m_h_edge_index[0]], m2m_h_position[m2m_h_edge_index[1]], m2m_h_heading[m2m_h_edge_index[1]])
        m2m_h_edge_attr_length, m2m_h_edge_attr_theta = compute_angles_lengths_2D(m2m_h_edge_vector)
        m2m_h_edge_attr_heading = wrap_angle(m2m_h_heading[m2m_h_edge_index[0]] - m2m_h_heading[m2m_h_edge_index[1]])
        m2m_h_edge_attr_interval = m2m_h_edge_index[0] - m2m_h_edge_index[1]
        m2m_h_edge_attr_input = torch.stack([m2m_h_edge_attr_length, m2m_h_edge_attr_theta, m2m_h_edge_attr_heading, m2m_h_edge_attr_interval], dim=-1)
        m2m_h_edge_attr_embs = self.m2m_h_emb_layer(input=m2m_h_edge_attr_input)

        #m2m_s edge
        m2m_s_valid_mask = m_valid_mask.transpose(0,1).reshape(-1, self.num_modes)              #[H*(N1,...,Nb),K]
        m2m_s_valid_mask = m2m_s_valid_mask.unsqueeze(2) & m2m_s_valid_mask.unsqueeze(1)        #[H*(N1,...,Nb),K,K]
        m2m_s_edge_index = dense_to_sparse(m2m_s_valid_mask)[0]
        m2m_s_edge_index = m2m_s_edge_index[:, m2m_s_edge_index[0] != m2m_s_edge_index[1]]

        #ALL ATTENTION
        #t2m attention
        t_embs = a_embs.reshape(-1, self.hidden_dim)  #[(N1,...,Nb)*H,D]
        m_embs_t = self.t2m_attn_layer(x = [t_embs, m_embs], edge_index = t2m_edge_index, edge_attr = t2m_edge_attr_embs)         #[(N1,...,Nb)*H*K,D]

        #l2m attention
        m_embs_l = self.l2m_attn_layer(x = [l_embs, m_embs], edge_index = l2m_edge_index, edge_attr = l2m_edge_attr_embs)         #[(N1,...,Nb)*H*K,D]
        
        m_embs = m_embs_t + m_embs_l
        m_embs = m_embs.reshape(num_all_agent, self.num_historical_steps, self.num_modes, self.hidden_dim).transpose(0,1).reshape(-1,self.hidden_dim)       #[H*(N1,...,Nb)*K,D]
        #moda attention  
        for i in range(self.num_attn_layers):
            #m2m_a
            m_embs = m_embs.reshape(self.num_historical_steps, num_all_agent, self.num_modes, self.hidden_dim).transpose(1,2).reshape(-1, self.hidden_dim)  #[H*K*(N1,...,Nb),D]
            m_embs = self.m2m_a_attn_layers[i](x = m_embs, edge_index = m2m_a_edge_index, edge_attr = m2m_a_edge_attr_embs)
            #m2m_h
            m_embs = m_embs.reshape(self.num_historical_steps, self.num_modes, num_all_agent, self.hidden_dim).permute(1,2,0,3).reshape(-1, self.hidden_dim)  #[K*(N1,...,Nb)*H,D]
            m_embs = self.m2m_h_attn_layers[i](x = m_embs, edge_index = m2m_h_edge_index, edge_attr = m2m_h_edge_attr_embs)
            #m2m_s
            m_embs = m_embs.reshape(self.num_modes, num_all_agent, self.num_historical_steps, self.hidden_dim).transpose(0,2).reshape(-1, self.hidden_dim)  #[H*(N1,...,Nb)*K,D]
            m_embs = self.m2m_s_attn_layers[i](x = m_embs, edge_index = m2m_s_edge_index)
        m_embs = m_embs.reshape(self.num_historical_steps, num_all_agent, self.num_modes, self.hidden_dim).transpose(0,1).reshape(-1, self.hidden_dim)      #[(N1,...,Nb)*H*K,D]

        #generate traj
        traj_propose = self.traj_propose(m_embs).reshape(num_all_agent, self.num_historical_steps, self.num_modes, self.num_future_steps, 2)         #[(N1,...,Nb),H,K,F,2]
        traj_propose = transform_traj_to_global_coordinate(traj_propose, m_position, m_heading)        #[(N1,...,Nb),H,K,F,2]

        #generate anchor
        proposal = traj_propose.detach()        #[(N1,...,Nb),H,K,F,2]
        
        n_batch = m_batch                                                                                                                                             #[(N1,...,Nb),K]
        n_position = proposal[:,:,:, self.num_future_steps // 2,:]                                                                                                    #[(N1,...,Nb),H,K,2]
        _, n_heading = compute_angles_lengths_2D(proposal[:,:,:, self.num_future_steps // 2,:] - proposal[:,:,:, (self.num_future_steps // 2) - 1,:])                 #[(N1,...,Nb),H,K]
        n_valid_mask = m_valid_mask                                                                                                                                   #[(N1,...,Nb),H,K]
        
        proposal = transform_traj_to_local_coordinate(proposal, n_position, n_heading)                      #[(N1,...,Nb),H,K,F,2]
        anchor = self.proposal_to_anchor(proposal.reshape(-1, self.num_future_steps*2))                     #[(N1,...,Nb)*H*K,D]
        n_embs = anchor                                                                                                                                               #[(N1,...,Nb)*H*K,D]

        #t2n edge
        t2n_position_t = data['agent']['position'][:,:self.num_historical_steps].reshape(-1,2)      #[(N1,...,Nb)*H,2]
        t2n_position_n = n_position.reshape(-1,2)                                                   #[(N1,...,Nb)*H*K,2]
        t2n_heading_t = data['agent']['heading'].reshape(-1)                                        #[(N1,...,Nb)]
        t2n_heading_n = n_heading.reshape(-1)                                                       #[(N1,...,Nb)*H*K]
        t2n_valid_mask_t = data['agent']['visible_mask'][:,:self.num_historical_steps]              #[(N1,...,Nb),H]
        t2n_valid_mask_n = n_valid_mask.reshape(num_all_agent,-1)                                   #[(N1,...,Nb),H*K]
        t2n_valid_mask = t2n_valid_mask_t.unsqueeze(2) & t2n_valid_mask_n.unsqueeze(1)              #[(N1,...,Nb),H,H*K]
        t2n_edge_index = dense_to_sparse(t2n_valid_mask)[0]
        t2n_edge_index = t2n_edge_index[:, torch.floor(t2n_edge_index[1]/self.num_modes) >= t2n_edge_index[0]]
        t2n_edge_index = t2n_edge_index[:, torch.floor(t2n_edge_index[1]/self.num_modes) - t2n_edge_index[0] <= self.pos_duration]
        t2n_edge_vector = transform_point_to_local_coordinate(t2n_position_t[t2n_edge_index[0]], t2n_position_n[t2n_edge_index[1]], t2n_heading_n[t2n_edge_index[1]])
        t2n_edge_attr_length, t2n_edge_attr_theta = compute_angles_lengths_2D(t2n_edge_vector)
        t2n_edge_attr_heading = wrap_angle(t2n_heading_t[t2n_edge_index[0]] - t2n_heading_n[t2n_edge_index[1]])
        t2n_edge_attr_interval = t2n_edge_index[0] - torch.floor(t2n_edge_index[1]/self.num_modes) - self.num_future_steps//2
        t2n_edge_attr_input = torch.stack([t2n_edge_attr_length, t2n_edge_attr_theta, t2n_edge_attr_heading, t2n_edge_attr_interval], dim=-1)
        t2n_edge_attr_embs = self.t2m_emb_layer(input=t2n_edge_attr_input)

        #l2n edge
        l2n_position_l = data['lane']['position']                       #[(M1,...,Mb),2]
        l2n_position_n = n_position.reshape(-1,2)                       #[(N1,...,Nb)*H*K,2]
        l2n_heading_l = data['lane']['heading']                         #[(M1,...,Mb)]
        l2n_heading_n = n_heading.reshape(-1)                           #[(N1,...,Nb)*H*K]
        l2n_batch_l = data['lane']['batch']                             #[(M1,...,Mb)]
        l2n_batch_n = n_batch.unsqueeze(1).repeat_interleave(self.num_historical_steps,1).reshape(-1)       #[(N1,...,Nb)*H*K]
        l2n_valid_mask_l = data['lane']['visible_mask']                                                     #[(M1,...,Mb)]
        l2n_valid_mask_n = n_valid_mask.reshape(-1)                                                         #[(N1,...,Nb)*H*K]
        l2n_valid_mask = l2n_valid_mask_l.unsqueeze(1) & l2n_valid_mask_n.unsqueeze(0)                      #[(M1,...,Mb),(N1,...,Nb)*H*K]
        l2n_valid_mask = drop_edge_between_samples(l2n_valid_mask, batch=(l2n_batch_l, l2n_batch_n))
        l2n_edge_index = dense_to_sparse(l2n_valid_mask)[0]
        l2n_edge_index = l2n_edge_index[:, torch.norm(l2n_position_l[l2n_edge_index[0]] - l2n_position_n[l2n_edge_index[1]], p=2, dim=-1) < self.l2a_radius]
        l2n_edge_vector = transform_point_to_local_coordinate(l2n_position_l[l2n_edge_index[0]], l2n_position_n[l2n_edge_index[1]], l2n_heading_n[l2n_edge_index[1]])
        l2n_edge_attr_length, l2n_edge_attr_theta = compute_angles_lengths_2D(l2n_edge_vector)
        l2n_edge_attr_heading = wrap_angle(l2n_heading_l[l2n_edge_index[0]] - l2n_heading_n[l2n_edge_index[1]])
        l2n_edge_attr_input = torch.stack([l2n_edge_attr_length, l2n_edge_attr_theta, l2n_edge_attr_heading], dim=-1)
        l2n_edge_attr_embs = self.l2m_emb_layer(input = l2n_edge_attr_input)

        #mode edge
        #n2n_a_edge
        n2n_a_position = n_position.permute(1,2,0,3).reshape(-1, 2)    #[H*K*(N1,...,Nb),2]
        n2n_a_heading = n_heading.permute(1,2,0).reshape(-1)           #[H*K*(N1,...,Nb)]
        n2n_a_batch = data['agent']['batch']                           #[(N1,...,Nb)]
        n2n_a_valid_mask = n_valid_mask.permute(1,2,0).reshape(self.num_historical_steps * self.num_modes, -1)   #[H*K,(N1,...,Nb)]
        n2n_a_valid_mask = n2n_a_valid_mask.unsqueeze(2) & n2n_a_valid_mask.unsqueeze(1)        #[H*K,(N1,...,Nb),(N1,...,Nb)]
        n2n_a_valid_mask = drop_edge_between_samples(n2n_a_valid_mask, n2n_a_batch)
        n2n_a_edge_index = dense_to_sparse(n2n_a_valid_mask)[0]
        n2n_a_edge_index = n2n_a_edge_index[:, n2n_a_edge_index[1] != n2n_a_edge_index[0]]
        n2n_a_edge_index = n2n_a_edge_index[:, torch.norm(n2n_a_position[n2n_a_edge_index[1]] - n2n_a_position[n2n_a_edge_index[0]],p=2,dim=-1) < self.a2a_radius]
        n2n_a_edge_vector = transform_point_to_local_coordinate(n2n_a_position[n2n_a_edge_index[0]], n2n_a_position[n2n_a_edge_index[1]], n2n_a_heading[n2n_a_edge_index[1]])
        n2n_a_edge_attr_length, n2n_a_edge_attr_theta = compute_angles_lengths_2D(n2n_a_edge_vector)
        n2n_a_edge_attr_heading = wrap_angle(n2n_a_heading[n2n_a_edge_index[0]] - n2n_a_heading[n2n_a_edge_index[1]])
        n2n_a_edge_attr_input = torch.stack([n2n_a_edge_attr_length, n2n_a_edge_attr_theta, n2n_a_edge_attr_heading], dim=-1)
        n2n_a_edge_attr_embs = self.m2m_a_emb_layer(input=n2n_a_edge_attr_input)

        #n2n_h edge                        
        n2n_h_position = n_position.permute(2,0,1,3).reshape(-1, 2)    #[K*(N1,...,Nb)*H,2]
        n2n_h_heading = n_heading.permute(2,0,1).reshape(-1)           #[K*(N1,...,Nb)*H]
        n2n_h_valid_mask = n_valid_mask.permute(2,0,1).reshape(-1, self.num_historical_steps)   #[K*(N1,...,Nb),H]
        n2n_h_valid_mask = n2n_h_valid_mask.unsqueeze(2) & n2n_h_valid_mask.unsqueeze(1)        #[K*(N1,...,Nb),H,H]        
        n2n_h_edge_index = dense_to_sparse(n2n_h_valid_mask)[0]
        n2n_h_edge_index = n2n_h_edge_index[:, n2n_h_edge_index[1] > n2n_h_edge_index[0]]
        n2n_h_edge_index = n2n_h_edge_index[:, n2n_h_edge_index[1] - n2n_h_edge_index[0] <= self.pred_duration]   
        n2n_h_edge_vector = transform_point_to_local_coordinate(n2n_h_position[n2n_h_edge_index[0]], n2n_h_position[n2n_h_edge_index[1]], n2n_h_heading[n2n_h_edge_index[1]])
        n2n_h_edge_attr_length, n2n_h_edge_attr_theta = compute_angles_lengths_2D(n2n_h_edge_vector)
        n2n_h_edge_attr_heading = wrap_angle(n2n_h_heading[n2n_h_edge_index[0]] - n2n_h_heading[n2n_h_edge_index[1]])
        n2n_h_edge_attr_interval = n2n_h_edge_index[0] - n2n_h_edge_index[1]
        n2n_h_edge_attr_input = torch.stack([n2n_h_edge_attr_length, n2n_h_edge_attr_theta, n2n_h_edge_attr_heading, n2n_h_edge_attr_interval], dim=-1)
        n2n_h_edge_attr_embs = self.m2m_h_emb_layer(input=n2n_h_edge_attr_input)

        #n2n_s edge
        n2n_s_position = n_position.transpose(0,1).reshape(-1,2)                                #[H*(N1,...,Nb)*K,2]
        n2n_s_heading = n_heading.transpose(0,1).reshape(-1)                                    #[H*(N1,...,Nb)*K]
        n2n_s_valid_mask = n_valid_mask.transpose(0,1).reshape(-1, self.num_modes)              #[H*(N1,...,Nb),K]
        n2n_s_valid_mask = n2n_s_valid_mask.unsqueeze(2) & n2n_s_valid_mask.unsqueeze(1)        #[H*(N1,...,Nb),K,K]
        n2n_s_edge_index = dense_to_sparse(n2n_s_valid_mask)[0]
        n2n_s_edge_index = n2n_s_edge_index[:, n2n_s_edge_index[0] != n2n_s_edge_index[1]]
        n2n_s_edge_vector = transform_point_to_local_coordinate(n2n_s_position[n2n_s_edge_index[0]], n2n_s_position[n2n_s_edge_index[1]], n2n_s_heading[n2n_s_edge_index[1]])
        n2n_s_edge_attr_length, n2n_s_edge_attr_theta = compute_angles_lengths_2D(n2n_s_edge_vector)
        n2n_s_edge_attr_heading = wrap_angle(n2n_s_heading[n2n_s_edge_index[0]] - n2n_s_heading[n2n_s_edge_index[1]])
        n2n_s_edge_attr_input = torch.stack([n2n_s_edge_attr_length, n2n_s_edge_attr_theta, n2n_s_edge_attr_heading], dim=-1)
        n2n_s_edge_attr_embs = self.m2m_s_emb_layer(input=n2n_s_edge_attr_input)

        #t2n attention
        t_embs = a_embs.reshape(-1, self.hidden_dim)  #[(N1,...,Nb)*H,D]
        n_embs_t = self.t2n_attn_layer(x = [t_embs, n_embs], edge_index = t2n_edge_index, edge_attr = t2n_edge_attr_embs)         #[(N1,...,Nb)*H*K,D]

        #l2m attention
        n_embs_l = self.l2n_attn_layer(x = [l_embs, n_embs], edge_index = l2n_edge_index, edge_attr = l2n_edge_attr_embs)         #[(N1,...,Nb)*H*K,D]

        n_embs = n_embs_t + n_embs_l
        n_embs = n_embs.reshape(num_all_agent, self.num_historical_steps, self.num_modes, self.hidden_dim).transpose(0,1).reshape(-1,self.hidden_dim)       #[H*(N1,...,Nb)*K,D]
        #moda attention  
        for i in range(self.num_attn_layers):
            #m2m_a
            n_embs = n_embs.reshape(self.num_historical_steps, num_all_agent, self.num_modes, self.hidden_dim).transpose(1,2).reshape(-1, self.hidden_dim)  #[H*K*(N1,...,Nb),D]
            n_embs = self.n2n_a_attn_layers[i](x = n_embs, edge_index = n2n_a_edge_index, edge_attr = n2n_a_edge_attr_embs)
            #m2m_h
            n_embs = n_embs.reshape(self.num_historical_steps, self.num_modes, num_all_agent, self.hidden_dim).permute(1,2,0,3).reshape(-1, self.hidden_dim)  #[K*(N1,...,Nb)*H,D]
            n_embs = self.n2n_h_attn_layers[i](x = n_embs, edge_index = n2n_h_edge_index, edge_attr = n2n_h_edge_attr_embs)
            #m2m_s
            n_embs = n_embs.reshape(self.num_modes, num_all_agent, self.num_historical_steps, self.hidden_dim).transpose(0,2).reshape(-1, self.hidden_dim)  #[H*(N1,...,Nb)*K,D]
            n_embs = self.n2n_s_attn_layers[i](x = n_embs, edge_index = n2n_s_edge_index, edge_attr = n2n_s_edge_attr_embs)
        n_embs = n_embs.reshape(self.num_historical_steps, num_all_agent, self.num_modes, self.hidden_dim).transpose(0,1).reshape(-1, self.hidden_dim)      #[(N1,...,Nb)*H*K,D]

        #generate refinement
        traj_refine = self.traj_refine(n_embs).reshape(num_all_agent, self.num_historical_steps, self.num_modes, self.num_future_steps, 2)       #[(N1,...,Nb),H,K,F,2]         
        traj_output = transform_traj_to_global_coordinate(proposal + traj_refine, n_position, n_heading)                    #[(N1,...,Nb),H,K,F,2]

        #generate prob
        prob_output = self.prob_decoder(n_embs.detach()).reshape(-1, self.num_historical_steps, self.num_modes)       #[(N1,...,Nb),H,K]
        prob_output = self.prob_norm(prob_output)                                       #[(N1,...,Nb),H,K]
        
        return traj_propose, traj_output, prob_output        #[(N1,...,Nb),H,K,F,2],[(N1,...,Nb),H,K,F,2],[(N1,...,Nb),H,K]

class MapEncoder(nn.Module):

    def __init__(self, hidden_dim: int, num_hops:int, num_heads: int, dropout: float) -> None:
        super(MapEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_hops = num_hops
        self.num_heads = num_heads
        self.dropout = dropout

        self._l2l_edge_type = ['adjacent', 'predecessor', 'successor']

        self.c_emb_layer = TwoLayerMLP(input_dim=1, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.l_emb_layer = TwoLayerMLP(input_dim=4, hidden_dim=hidden_dim, output_dim=hidden_dim)

        self.c2l_emb_layer = TwoLayerMLP(input_dim=3, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.l2l_emb_layer = TwoLayerMLP(input_dim=7, hidden_dim=hidden_dim, output_dim=hidden_dim)

        self.c2l_attn_layer = GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=False)
        self.l2l_attn_layer = GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=True)

        self.apply(init_weights)

    def forward(self, data) -> torch.Tensor:
        #embedding
        c_length = data['centerline']['length']
        c_embs = self.c_emb_layer(input=c_length.unsqueeze(-1))        #[(C1,...,Cb),D]

        l_length = data['lane']['length']
        l_is_intersection = data['lane']['is_intersection']
        l_turn_direction = data['lane']['turn_direction']
        l_traffic_control = data['lane']['traffic_control']
        l_input = torch.stack([l_length, l_is_intersection, l_turn_direction, l_traffic_control], dim=-1)        #[(M1,...,Mb),4]
        l_embs = self.l_emb_layer(input=l_input)                      #[(M1,...,Mb),D]

        #edge
        #c2l
        c2l_position_c = data['centerline']['position']             #[(C1,...,Cb),2]
        c2l_position_l = data['lane']['position']                   #[(M1,...,Mb),2]
        c2l_heading_c = data['centerline']['heading']               #[(C1,...,Cb)]
        c2l_heading_l = data['lane']['heading']                     #[(M1,...,Mb)]
        c2l_edge_index = data['centerline', 'lane']['centerline_to_lane_edge_index']    #[2,(C1,...,Cb)]
        c2l_edge_vector = transform_point_to_local_coordinate(c2l_position_c[c2l_edge_index[0]], c2l_position_l[c2l_edge_index[1]], c2l_heading_l[c2l_edge_index[1]])
        c2l_edge_attr_length, c2l_edge_attr_theta = compute_angles_lengths_2D(c2l_edge_vector)
        c2l_edge_attr_heading = wrap_angle(c2l_heading_c[c2l_edge_index[0]] - c2l_heading_l[c2l_edge_index[1]])
        c2l_edge_attr_input = torch.stack([c2l_edge_attr_length, c2l_edge_attr_theta, c2l_edge_attr_heading], dim=-1)
        c2l_edge_attr_embs = self.c2l_emb_layer(input = c2l_edge_attr_input)

        #l2l
        l2l_position = data['lane']['position']                     #[(M1,...,Mb),2]
        l2l_heading = data['lane']['heading']                       #[(M1,...,Mb)]
        l2l_edge_index = []
        l2l_edge_attr_type = []
        l2l_edge_attr_hop = []
        
        l2l_adjacent_edge_index = data['lane', 'lane']['adjacent_edge_index']
        num_adjacent_edges = l2l_adjacent_edge_index.size(1)
        l2l_edge_index.append(l2l_adjacent_edge_index)
        l2l_edge_attr_type.append(F.one_hot(torch.tensor(self._l2l_edge_type.index('adjacent')), num_classes=len(self._l2l_edge_type)).to(l2l_adjacent_edge_index.device).repeat(num_adjacent_edges, 1))
        l2l_edge_attr_hop.append(torch.ones(num_adjacent_edges, device=l2l_adjacent_edge_index.device))

        num_lanes = data['lane']['num_nodes']
        l2l_predecessor_edge_index = data['lane', 'lane']['predecessor_edge_index']
        l2l_predecessor_edge_index_all = generate_reachable_matrix(l2l_predecessor_edge_index, self.num_hops, num_lanes)
        for i in range(self.num_hops):
            num_edges_now = l2l_predecessor_edge_index_all[i].size(1)
            l2l_edge_index.append(l2l_predecessor_edge_index_all[i])
            l2l_edge_attr_type.append(F.one_hot(torch.tensor(self._l2l_edge_type.index('predecessor')), num_classes=len(self._l2l_edge_type)).to(l2l_predecessor_edge_index.device).repeat(num_edges_now, 1))
            l2l_edge_attr_hop.append((i + 1) * torch.ones(num_edges_now, device=l2l_predecessor_edge_index.device))
        
        l2l_successor_edge_index = data['lane', 'lane']['successor_edge_index']
        l2l_successor_edge_index_all = generate_reachable_matrix(l2l_successor_edge_index, self.num_hops, num_lanes)
        for i in range(self.num_hops):
            num_edges_now = l2l_successor_edge_index_all[i].size(1)
            l2l_edge_index.append(l2l_successor_edge_index_all[i])
            l2l_edge_attr_type.append(F.one_hot(torch.tensor(self._l2l_edge_type.index('successor')), num_classes=len(self._l2l_edge_type)).to(l2l_successor_edge_index.device).repeat(num_edges_now, 1))
            l2l_edge_attr_hop.append((i + 1) * torch.ones(num_edges_now, device=l2l_successor_edge_index.device))

        l2l_edge_index = torch.cat(l2l_edge_index, dim=1)
        l2l_edge_attr_type = torch.cat(l2l_edge_attr_type, dim=0)
        l2l_edge_attr_hop = torch.cat(l2l_edge_attr_hop, dim=0)
        l2l_edge_vector = transform_point_to_local_coordinate(l2l_position[l2l_edge_index[0]], l2l_position[l2l_edge_index[1]], l2l_heading[l2l_edge_index[1]])
        l2l_edge_attr_length, l2l_edge_attr_theta = compute_angles_lengths_2D(l2l_edge_vector)
        l2l_edge_attr_heading = wrap_angle(l2l_heading[l2l_edge_index[0]] - l2l_heading[l2l_edge_index[1]])
        l2l_edge_attr_input = torch.cat([l2l_edge_attr_length.unsqueeze(-1), l2l_edge_attr_theta.unsqueeze(-1), l2l_edge_attr_heading.unsqueeze(-1), l2l_edge_attr_hop.unsqueeze(-1), l2l_edge_attr_type], dim=-1)
        l2l_edge_attr_embs = self.l2l_emb_layer(input=l2l_edge_attr_input)

        #attention
        #c2l
        l_embs = self.c2l_attn_layer(x = [c_embs, l_embs], edge_index = c2l_edge_index, edge_attr = c2l_edge_attr_embs)         #[(M1,...,Mb),D]

        #l2l
        l_embs = self.l2l_attn_layer(x = l_embs, edge_index = l2l_edge_index, edge_attr = l2l_edge_attr_embs)                   #[(M1,...,Mb),D]

        return l_embs

class MR(Metric):
    def __init__(self, 
                 threshold: float=2.0,
                 dist_sync_on_step: bool = False) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.threshold = threshold
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self,
               predictions: torch.Tensor,
               targets: torch.Tensor) -> None:
        errors = torch.norm(predictions[:,-1] - targets[:,-1], dim=-1)
        MR_values = errors > self.threshold
        self.sum = self.sum + MR_values.sum()
        self.count = self.count + len(MR_values)
    
    def compute(self) -> torch.Tensor:
        return self.sum / self.count

class BrierMinFDE(Metric):
    def __init__(self, 
                 dist_sync_on_step: bool = False) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self,
               predictions: torch.Tensor,
               targets: torch.Tensor,
               prob_best_forecasts: torch.Tensor) -> None:
        errors = torch.norm(predictions - targets, dim=-1)
        minFDE_values = errors[..., -1]
        prob_term = (1.0 - prob_best_forecasts) ** 2
        brier_minFDE_values = minFDE_values + prob_term
        self.sum = self.sum + brier_minFDE_values.sum()
        self.count = self.count + len(brier_minFDE_values)
    
    def compute(self) -> torch.Tensor:
        return self.sum / self.count

class MinADE(Metric):
    def __init__(self, 
                 dist_sync_on_step: bool = False) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self,
               predictions: torch.Tensor,
               targets: torch.Tensor) -> None:
        errors = torch.norm(predictions - targets, dim=-1)
        minADE_values = errors.mean(dim=-1)
        self.sum = self.sum + minADE_values.sum()
        self.count = self.count + len(minADE_values)
    
    def compute(self) -> torch.Tensor:
        return self.sum / self.count

class MinFDE(Metric):
    def __init__(self, 
                 dist_sync_on_step: bool = False) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self,
               predictions: torch.Tensor,
               targets: torch.Tensor) -> None:
        errors = torch.norm(predictions - targets, dim=-1)
        minFDE_values = errors[..., -1]
        self.sum = self.sum + minFDE_values.sum()
        self.count = self.count + len(minFDE_values)
    
    def compute(self) -> torch.Tensor:
        return self.sum / self.count

class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()

    def forward(self, predictions, targets):
        loss = F.nll_loss(torch.log(predictions), targets)
        return loss

class Huber2DLoss(nn.Module):
    def __init__(self):
        super(Huber2DLoss, self).__init__()
        self.huber = nn.SmoothL1Loss(reduction='none')

    def forward(self, predictions, targets):
        loss = self.huber(predictions, targets).sum(-1)
        return loss.mean()

class HPNet(BaseModel):

    def __init__(self, config):
                 #hidden_dim: int, num_historical_steps: int, num_future_steps: int, pos_duration: int, pred_duration: int, a2a_radius: float, 
                 #l2a_radius: float, num_visible_steps: int, num_modes: int, num_attn_layers: int, num_hops: int, num_heads: int, dropout: float, lr: float,
                 #weight_decay: float, warmup_epochs: int, T_max: int,**kwargs) -> None:
        super(HPNet, self).__init__(config)
        #self.save_hyperparameters()
        self.config = config
        self.hidden_dim = config['hidden_dim']
        self.num_historical_steps = config['num_historical_steps']
        self.num_future_steps = config['num_future_steps']
        self.pos_duration = config['pos_duration']
        self.pred_duration = config['pred_duration']
        self.a2a_radius = config['a2a_radius']
        self.l2a_radius = config['l2a_radius']
        self.num_visible_steps = config['num_visible_steps']
        self.num_modes = config['num_modes']
        self.num_attn_layers = config['num_attn_layers']
        self.num_hops = config['num_hops']
        self.num_heads = config['num_heads']
        self.dropout = config['dropout']
        self.lr = config['learning_rate']
        self.weight_decay = config['weight_decay']
        self.warmup_epochs = config['warmup_epochs']
        self.T_max = config['T_max']

        self.Backbone = Backbone(
            hidden_dim=self.hidden_dim,
            num_historical_steps=self.num_historical_steps,
            num_future_steps=self.num_future_steps,
            pos_duration=self.pos_duration,
            pred_duration=self.pred_duration,
            a2a_radius=self.a2a_radius,
            l2a_radius=self.l2a_radius,
            num_attn_layers=self.num_attn_layers,
            num_modes=self.num_modes,
            num_heads=self.num_heads,
            dropout=self.dropout
        )
        self.EnhancedMapEncoder = MapEncoder(
            hidden_dim=self.hidden_dim,
            num_hops=self.num_hops,
            num_heads=self.num_heads,
            dropout=self.dropout
        )

        self.reg_loss = Huber2DLoss()
        self.prob_loss = CELoss()

        self.brier_minFDE = BrierMinFDE()
        self.minADE = MinADE()
        self.minFDE = MinFDE()
        self.MR = MR()

        self.test_traj_output = dict()
        self.test_prob_output = dict()

    def forward(self, data):
        data=data['input_dict']
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                data[key] = value.numpy().tolist()
            if isinstance(value, np.ndarray):
                data[key] = value.tolist()

        # Save the dictionary to a JSON file
        with open('data.json', 'w') as file:
            json.dump(data, file)

        lane_embs = MapEncoder(data=data)
        pred = self.Backbone(data=data, l_embs=lane_embs)
        return pred

    def training_step(self,data,batch_idx):
        traj_propose, traj_output, prob_output = self(data)               #[(N1,...,Nb),H,K,F,2],[(N1,...,Nb),H,K,F,2],[(N1,...,Nb),H,K]
        target_traj, target_mask = generate_target(position=data['agent']['position'], 
                                                   mask=data['agent']['visible_mask'],
                                                   num_historical_steps=self.num_historical_steps,
                                                   num_future_steps=self.num_future_steps)  #[(N1,...Nb),H,F,2],[(N1,...Nb),H,F]

        errors = (torch.norm(traj_propose[...,:2] - target_traj.unsqueeze(2), p=2, dim=-1) * target_mask.unsqueeze(2)).sum(dim=-1)  #[(N1,...Nb),H,K]
        best_mode_index = errors.argmin(dim=-1)     #[(N1,...Nb),H]
        traj_best_propose = traj_propose[torch.arange(traj_propose.size(0))[:, None], torch.arange(traj_propose.size(1))[None, :], best_mode_index]   #[(N1,...Nb),H,F,2]
        traj_best_output = traj_output[torch.arange(traj_output.size(0))[:, None], torch.arange(traj_output.size(1))[None, :], best_mode_index]   #[(N1,...Nb),H,F,2]

        predict_mask = generate_predict_mask(data['agent']['visible_mask'][:,:self.num_historical_steps], self.num_visible_steps)   #[(N1,...Nb),H]
        targ_mask = target_mask[predict_mask]                             #[Na,F]
        traj_pro = traj_best_propose[predict_mask]                        #[Na,F,2]
        traj_ref = traj_best_output[predict_mask]                         #[Na,F,2]
        prob = prob_output[predict_mask]                                  #[Na,K]
        targ = target_traj[predict_mask]                                  #[Na,F,2]
        label = best_mode_index[predict_mask]                             #[Na]

        reg_loss_propose = self.reg_loss(traj_pro[targ_mask], targ[targ_mask]) 
        reg_loss_refine = self.reg_loss(traj_ref[targ_mask], targ[targ_mask])    
        prob_loss = self.prob_loss(prob, label)
        loss = reg_loss_propose + reg_loss_refine + prob_loss
        self.log('train_reg_loss_propose', reg_loss_propose, prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('train_reg_loss_refine', reg_loss_refine, prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('train_prob_loss', prob_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)

        return loss

    def validation_step(self,data,batch_idx):
        traj_propose, traj_output, prob_output = self(data)               #[(N1,...,Nb),H,K,F,2],[(N1,...,Nb),H,K,F,2],[(N1,...,Nb),H,K]
        target_traj, target_mask = generate_target(position=data['agent']['position'], 
                                                   mask=data['agent']['visible_mask'],
                                                   num_historical_steps=self.num_historical_steps,
                                                   num_future_steps=self.num_future_steps)  #[(N1,...Nb),H,F,2],[(N1,...Nb),H,F]

        errors = (torch.norm(traj_propose[...,:2] - target_traj.unsqueeze(2), p=2, dim=-1) * target_mask.unsqueeze(2)).sum(dim=-1)  #[(N1,...Nb),H,K]
        best_mode_index = errors.argmin(dim=-1)     #[(N1,...Nb),H]
        traj_best_propose = traj_propose[torch.arange(traj_propose.size(0))[:, None], torch.arange(traj_propose.size(1))[None, :], best_mode_index]   #[(N1,...Nb),H,F,2]
        traj_best_output = traj_output[torch.arange(traj_output.size(0))[:, None], torch.arange(traj_output.size(1))[None, :], best_mode_index]   #[(N1,...Nb),H,F,2]

        predict_mask = generate_predict_mask(data['agent']['visible_mask'][:,:self.num_historical_steps], self.num_visible_steps)   #[(N1,...Nb),H]
        targ_mask = target_mask[predict_mask]                             #[Na,F]
        traj_pro = traj_best_propose[predict_mask]                        #[Na,F,2]
        traj_ref = traj_best_output[predict_mask]                         #[Na,F,2]
        prob = prob_output[predict_mask]                                  #[Na,K]
        targ = target_traj[predict_mask]                                  #[Na,F,2]
        label = best_mode_index[predict_mask]                             #[Na]

        reg_loss_propose = self.reg_loss(traj_pro[targ_mask], targ[targ_mask]) 
        reg_loss_refine = self.reg_loss(traj_ref[targ_mask], targ[targ_mask])    
        prob_loss = self.prob_loss(prob, label)
        loss = reg_loss_propose + reg_loss_refine + prob_loss
        self.log('val_reg_loss_propose', reg_loss_propose, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('val_reg_loss_refine', reg_loss_refine, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('val_prob_loss', prob_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)

        agent_index = data['agent']['agent_index'] + data['agent']['ptr'][:-1]
        num_agents = agent_index.size(0)
        agent_traj = traj_output[agent_index, -1]                           #[N,K,F,2]
        agent_prob = prob_output[agent_index, -1]                           #[N,K]
        agent_targ = target_traj[agent_index, -1]                           #[N,F,2]
        fde = torch.norm(agent_traj[:, :, -1, :2] - agent_targ[:,-1, :2].unsqueeze(1), p=2, dim=-1)     #[N,K]
        best_mode_index = fde.argmin(dim=-1)    #[N] 
        agent_traj_best = agent_traj[torch.arange(num_agents), best_mode_index]   #[N,F,2]
        self.brier_minFDE.update(agent_traj_best[..., :2], agent_targ[..., :2], agent_prob[torch.arange(num_agents),best_mode_index])
        self.minADE.update(agent_traj_best[..., :2], agent_targ[..., :2])
        self.minFDE.update(agent_traj_best[..., :2], agent_targ[..., :2])
        self.MR.update(agent_traj_best[..., :2], agent_targ[..., :2])
        self.log('val_minADE', self.minADE, prog_bar=True, on_step=False, on_epoch=True, batch_size=num_agents, sync_dist=True)
        self.log('val_minFDE', self.minFDE, prog_bar=True, on_step=False, on_epoch=True, batch_size=num_agents, sync_dist=True)
        self.log('val_MR', self.MR, prog_bar=True, on_step=False, on_epoch=True, batch_size=num_agents, sync_dist=True)
        self.log('val_brier_minFDE', self.brier_minFDE, prog_bar=True, on_step=False, on_epoch=True, batch_size=num_agents, sync_dist=True)


    def test_step(self,data,batch_idx):
        traj_propose, traj_output, prob_output = self(data)               #[(N1,...,Nb),H,K,F,2],[(N1,...,Nb),H,K,F,2],[(N1,...,Nb),H,K]
        
        prob_output = prob_output**2
        prob_output = prob_output / prob_output.sum(dim=-1, keepdim=True)

        agent_index = data['agent']['agent_index'] + data['agent']['ptr'][:-1]
        num_agents = agent_index.size(0)
        agent_traj = traj_output[agent_index, -1]           #[N,K,F,2]
        agent_prob = prob_output[agent_index, -1]           #[N,K]

        for i in range(num_agents):
            id = int(data['scenario_id'][i])
            traj = agent_traj[i].cpu().numpy()
            prob = agent_prob[i].tolist()

            self.test_traj_output[id] = traj
            self.test_prob_output[id] = prob


    """ def on_test_end(self):
        output_path = './test_output'
        filename = 'submission'
        generate_forecasting_h5(self.test_traj_output, output_path, filename, self.test_prob_output)
 """
    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM,
                                    nn.LSTMCell, nn.GRU, nn.GRUCell)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
             "weight_decay": self.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr, weight_decay=self.weight_decay)
        
        warmup_epochs = self.warmup_epochs
        T_max = self.T_max

        def warmup_cosine_annealing_schedule(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            return 0.5 * (1.0 + math.cos(math.pi * (epoch - warmup_epochs + 1) / (T_max - warmup_epochs + 1)))

        scheduler = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine_annealing_schedule),
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [scheduler]

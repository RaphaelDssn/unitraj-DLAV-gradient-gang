from .base_dataset import BaseDataset
import numpy as np
import torch
from typing import  Dict
import pandas as pd
import math
from typing import Optional, Union, Tuple, List
from torch_geometric.data import Batch, Data, Dataset



class HPNetDataset(BaseDataset):

    def __init__(self, config=None, is_validation=False, is_noisy=None):
        super().__init__(config, is_validation)
        self.config = config

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        features = self.get_features(data=data[0], config=self.config)
        # Assuming features is a dictionary, convert it to a Data object
        return features

    def euc_dist(self, p1, p2):
        return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    
    def merge_past_future_positions(self, obj_trajs_future_state, obj_trajs_pos):
        num_agents = len(obj_trajs_future_state)
        num_past_timesteps = len(obj_trajs_pos[0])
        num_future_timesteps = len(obj_trajs_future_state[0])

        positions = np.empty((num_agents, num_past_timesteps + num_future_timesteps, 2))

        for i in range(num_agents):
            # Concatenate past and future positions along the timestep dimension
            positions[i][:num_past_timesteps] = obj_trajs_pos[i][:,:2]
            positions[i][num_past_timesteps:] = obj_trajs_future_state[i][:, :2]

        return positions
    
    def merge_past_future_masks(self, obj_trajs_masks, obj_trajs_future_mask):
        num_agents = len(obj_trajs_masks)
        num_past_timesteps = len(obj_trajs_masks[0])
        num_future_timesteps = len(obj_trajs_future_mask[0])

        masks = np.empty((num_agents, num_past_timesteps + num_future_timesteps))

        for i in range(num_agents):
            # Concatenate past and future masks along the timestep dimension
            masks[i][:num_past_timesteps] = obj_trajs_masks[i]
            masks[i][num_past_timesteps:] = obj_trajs_future_mask[i]

        return masks

    def get_lanes(self, map_polylines, num_lines):
        counter = 0
        type_list = []
        lane_ids = []

        for i in range(num_lines):
            line_list = []
            for j in range(20):
                line_list.append(map_polylines[i][j][6])
            type_list.append(line_list)
        
        for i in range(num_lines):
            types = np.unique(type_list[i])
            
            if (types[0]==1) or (types[0]==2):
                counter += 1
                lane_ids.append(i)
        return counter, lane_ids

    def get_turn_direction(self, heading, reference_point, target_point): 
        # Convert heading to direction vector
        dx = np.cos(heading)
        dy = np.sin(heading)
        direction_vector = (dx, dy)
        
        # Vector from reference point to target point
        vector_ref_to_target = (target_point[0] - reference_point[0], target_point[1] - reference_point[1])
        
        # Cross product of direction_vector and vector_ref_to_target
        cross_product = direction_vector[0] * vector_ref_to_target[1] - direction_vector[1] * vector_ref_to_target[0]
        
        if abs(cross_product) < 0.5:
            return 0 #"NONE"
        elif cross_product > 0.5:
            return 1 #"LEFT"
        else:
            return 2 #"RIGHT"

    def get_centerline(self, line_pts):
        # Convert line_pts to a PyTorch tensor
        line_pts_tensor = torch.tensor(line_pts)
        centerline = torch.zeros(20,2)
        for i in range(20):
            centerline[i] = line_pts_tensor[i][:2]
        return centerline

    def calculate_angle(self, p1, p2):
        delta_x = p2[0] - p1[0]
        delta_y = p2[1] - p1[1]
        angle = math.atan2(delta_y, delta_x)
        return angle

    def angle_range(self, angle):
        # Normalize angles to be within [0, 2π)
        angle_norm = angle % (2 * math.pi)
        if angle_norm < math.pi/4:
            return 45
        if angle_norm > math.pi/4 and angle_norm < math.pi/2:
            return 90
        if angle_norm > math.pi/2 and angle_norm < 3*math.pi/4:
            return 135
        if angle_norm > 3*math.pi/4 and angle_norm < math.pi:
            return 180
        if angle_norm > math.pi and angle_norm < 5*math.pi/4:
            return 225
        if angle_norm > 5*math.pi/4 and angle_norm < 3*math.pi/2:
            return 270
        if angle_norm > 3*math.pi/2 and angle_norm < 7*math.pi/4:
            return 315
        if angle_norm > 7*math.pi/4:
            return 360

    def get_neighbours(self, polylines, lane_ids):

        neighbors = {}

        for id in lane_ids:
            start, end = polylines[id][0][0:2], polylines[id][-1][0:2]
            mid = polylines[id][10][0:2]
            heading = self.calculate_angle(start, end)
            range = self.angle_range(heading)
            left_neighbor = None
            right_neighbor = None
            min_left_distance = float('inf')
            min_right_distance = float('inf')

            for other_id in lane_ids:
                if id == other_id:
                    continue
                other_start, other_end = polylines[other_id][0][0:2], polylines[other_id][-1][0:2]
                other_mid = polylines[other_id][10][0:2]
                other_heading = self.calculate_angle(other_start, other_end)
                other_range = self.angle_range(other_heading)
                if range==other_range:
                    if range == 45 or range == 360:
                        dist = mid[1]-other_mid[1]
                        if dist < 0 and abs(dist) < min_left_distance:
                            min_left_distance = dist
                            left_neighbor = other_id
                        if dist > 0 and abs(dist) < min_right_distance:
                            min_right_distance = dist
                            right_neighbor = other_id

                    if range == 90 or range == 135:
                        dist = mid[0]-other_mid[0]
                        if dist > 0 and abs(dist) < min_left_distance:
                            min_left_distance = dist
                            left_neighbor = other_id
                        if dist < 0 and abs(dist) < min_right_distance:
                            min_right_distance = dist
                            right_neighbor = other_id

                    if range == 180 or range == 225:
                        dist = mid[1]-other_mid[1]
                        if dist > 0 and abs(dist) < min_left_distance:
                            min_left_distance = dist
                            left_neighbor = other_id
                        if dist < 0 and abs(dist) < min_right_distance:
                            min_right_distance = dist
                            right_neighbor = other_id

                    if range == 225 or range == 315:
                        dist = mid[0]-other_mid[0]
                        if dist < 0 and abs(dist) < min_left_distance:
                            min_left_distance = dist
                            left_neighbor = other_id
                        if dist > 0 and abs(dist) < min_right_distance:
                            min_right_distance = dist
                            right_neighbor = other_id
            if right_neighbor is None and left_neighbor is None:
                neighbors[id]=[]
            elif right_neighbor is None:
                neighbors[id]=[left_neighbor]
            elif left_neighbor is None:
                neighbors[id]=[right_neighbor]
            else:
                neighbors[id] = [right_neighbor, left_neighbor]

        return neighbors

    def get_successors(self, polylines, lane_ids):
        successors={}

        for id in lane_ids:
            start, end = polylines[id][0][0:2], polylines[id][-1][0:2]
            lan_succ = []
            for other_id in lane_ids:
                if id == other_id:
                    continue
                other_start, other_end = polylines[id][0][0:2], polylines[id][-1][0:2]

                if self.euc_dist(end, other_start)<1:
                    lan_succ.append(other_id)
            successors[id] = lan_succ
        
        return successors

    def get_predecessors(self, polylines, lane_ids):
        predecessors={}

        for id in lane_ids:
            start, end = polylines[id][0][0:2], polylines[id][-1][0:2]
            lan_succ = []
            for other_id in lane_ids:
                if id == other_id:
                    continue
                other_start, other_end = polylines[id][0][0:2], polylines[id][-1][0:2]

                if self.euc_dist(start, other_end)<1:
                    lan_succ.append(other_id)
            predecessors[id] = lan_succ
        
        return predecessors

    def compute_angles_lengths_2D(self, vectors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        length = torch.norm(vectors, dim=-1)
        theta = torch.atan2(vectors[..., 1], vectors[..., 0])
        return length, theta

    def get_features(self, data,config):
        feat_data={
                'agent': {},
                'lane': {},
                'centerline': {},
                ('centerline', 'lane'): {},
                ('lane', 'lane'): {}
            }
        num_steps=config['past_len']+config['future_len']
        timestep_ids = list(range(num_steps))
        historical_timestamps = timestep_ids[:config['num_historical_steps']]
        
        historical_data = data['obj_trajs_pos']

        agent_ids = list(range(config['max_num_agents']))
        num_agents = config['max_num_agents']
        pos_data = self.merge_past_future_positions(data['obj_trajs_future_state'],data['obj_trajs_pos'] ) 
        mask_data = self.merge_past_future_masks(data['obj_trajs_mask'], data['obj_trajs_future_mask'])    
        agent_index = agent_ids

        # initialization
        visible_mask = torch.zeros(num_agents, num_steps, dtype=torch.bool)
        length_mask = torch.zeros(num_agents, config['num_historical_steps'], dtype=torch.bool)
        agent_position = torch.zeros(num_agents, num_steps, 2, dtype=torch.float)
        agent_heading = torch.zeros(num_agents, config['num_historical_steps'], dtype=torch.float)
        agent_length = torch.zeros(num_agents, config['num_historical_steps'], dtype=torch.float)
            
        for i in range(num_agents): #for track_id, track_df in df.groupby('TRACK_ID'): #group by agent so for each agent
            agent_idx = i
            agent_steps = timestep_ids

            visible_mask[agent_idx, agent_steps] = True

            length_mask[agent_idx, 0] = False
            length_mask[agent_idx, 1:] = ~(visible_mask[agent_idx, 1:config['num_historical_steps']] & visible_mask[agent_idx, :config['num_historical_steps']-1])

            agent_position[agent_idx, agent_steps] = torch.tensor(pos_data[agent_idx], dtype=torch.float32)
            motion = torch.cat([agent_position.new_zeros(1, 2), agent_position[agent_idx, 1:] - agent_position[agent_idx, :-1]], dim=0) 
            length, heading = self.compute_angles_lengths_2D(motion)
            agent_length[agent_idx] = length[:config['num_historical_steps']]
            agent_heading[agent_idx] = heading[:config['num_historical_steps']]
            agent_length[agent_idx, length_mask[agent_idx]] = 0
            agent_heading[agent_idx, length_mask[agent_idx]] = 0

        feat_data['agent']['num_nodes'] = num_agents
        feat_data['agent']['agent_index'] = agent_index
        feat_data['agent']['visible_mask'] = visible_mask
        feat_data['agent']['position'] = agent_position
        feat_data['agent']['heading'] = agent_heading
        feat_data['agent']['length'] = agent_length
            
        #MAP
        positions = agent_position[:,:config['num_historical_steps']][visible_mask[:,:config['num_historical_steps']]].reshape(-1,2)
             
        num_lanes, lane_ids = self.get_lanes(data['map_polylines'], config['max_num_roads'])
    
        lane_position = torch.zeros(config['max_num_roads'], 2, dtype=torch.float)
        lane_heading = torch.zeros(config['max_num_roads'], dtype=torch.float)
        lane_length = torch.zeros(config['max_num_roads'], dtype=torch.float)
        lane_is_intersection = torch.zeros(config['max_num_roads'], dtype=torch.uint8)
        lane_turn_direction = torch.zeros(config['max_num_roads'], dtype=torch.uint8)
        lane_traffic_control = torch.zeros(config['max_num_roads'], dtype=torch.uint8)

        num_centerlines = torch.zeros(config['max_num_roads'], dtype=torch.long)
        centerline_position: List[Optional[torch.Tensor]] = [torch.zeros(19, 2)] * config['max_num_roads']
        centerline_heading: List[Optional[torch.Tensor]] = [torch.zeros(19)] * config['max_num_roads']
        centerline_length: List[Optional[torch.Tensor]] = [torch.zeros(19)] * config['max_num_roads']

        lane_adjacent_edge_index = []
        lane_predecessor_edge_index = []
        lane_successor_edge_index = []
        for lane_id in lane_ids: 

            centerlines = self.get_centerline(data['map_polylines'][lane_id])
            num_centerlines[lane_id] = 1
            centerline_position[lane_id] = (centerlines[1:] + centerlines[:-1]) / 2
            centerline_vectors = centerlines[1:] - centerlines[:-1]
            centerline_length[lane_id], centerline_heading[lane_id] = self.compute_angles_lengths_2D(centerline_vectors)
            lane_length[lane_id] = centerline_length[lane_id].sum()
            center_index = int(num_centerlines[lane_id]/2)
            lane_position[lane_id] = centerlines[center_index]
            lane_heading[lane_id] = torch.atan2(centerlines[center_index + 1, 1] - centerlines[center_index, 1], centerlines[center_index + 1, 0] - centerlines[center_index, 0])
                
            lane_is_intersection[lane_id] = torch.tensor(0)
            lane_turn_direction[lane_id] = self.get_turn_direction(lane_heading[lane_id],centerlines[0], centerlines[-1])
            lane_traffic_control[lane_id] = torch.tensor(0)

        neighbours = self.get_neighbours(data['map_polylines'], lane_ids)
        successors = self.get_successors(data['map_polylines'], lane_ids)
        predecessors = self.get_predecessors(data['map_polylines'], lane_ids)

        for lane_id in lane_ids:
            lane_adjacent_ids = neighbours[lane_id]
            lane_adjacent_idx = lane_adjacent_ids
            if len(lane_adjacent_idx) != 0:
                edge_index = torch.stack([torch.tensor(lane_adjacent_idx, dtype=torch.long), torch.full((len(lane_adjacent_idx),), lane_id, dtype=torch.long)], dim=0)
                lane_adjacent_edge_index.append(edge_index)
            lane_predecessor_ids = predecessors[lane_id]
            lane_predecessor_idx = lane_predecessor_ids
            if len(lane_predecessor_idx) != 0:
                edge_index = torch.stack([torch.tensor(lane_predecessor_idx, dtype=torch.long), torch.full((len(lane_predecessor_idx),), lane_id, dtype=torch.long)], dim=0)
                lane_predecessor_edge_index.append(edge_index)
            lane_successor_ids = successors[lane_id]
            lane_successor_idx = lane_successor_ids
            if len(lane_successor_idx) != 0:
                edge_index = torch.stack([torch.tensor(lane_successor_idx, dtype=torch.long), torch.full((len(lane_successor_idx),), lane_id, dtype=torch.long)], dim=0)
                lane_successor_edge_index.append(edge_index)

        feat_data['lane']['num_nodes'] = num_lanes
        feat_data['lane']['position'] = lane_position
        feat_data['lane']['length'] = lane_length
        feat_data['lane']['heading'] = lane_heading
        feat_data['lane']['is_intersection'] = lane_is_intersection
        feat_data['lane']['turn_direction'] = lane_turn_direction
        feat_data['lane']['traffic_control'] = lane_traffic_control

        feat_data['centerline']['num_nodes'] = num_centerlines.sum().item()
        feat_data['centerline']['position'] = torch.cat(centerline_position, dim=0)
        feat_data['centerline']['heading'] = torch.cat(centerline_heading, dim=0)
        feat_data['centerline']['length'] = torch.cat(centerline_length, dim=0)

            
        centerline_to_lane_edge_index = torch.stack([torch.arange(num_centerlines.sum(), dtype=torch.long), torch.arange(num_lanes, dtype=torch.long).repeat_interleave(1)], dim=0)
        feat_data['centerline', 'lane']['centerline_to_lane_edge_index'] = centerline_to_lane_edge_index

        if len(lane_adjacent_edge_index) != 0:
            lane_adjacent_edge_index = torch.cat(lane_adjacent_edge_index, dim=1)
        else:
            lane_adjacent_edge_index = torch.tensor([[], []], dtype=torch.long)

        if len(lane_predecessor_edge_index) != 0:
            lane_predecessor_edge_index = torch.cat(lane_predecessor_edge_index, dim=1)
        else:
            lane_predecessor_edge_index = torch.tensor([[], []], dtype=torch.long)
        
        if len(lane_successor_edge_index) != 0:
            lane_successor_edge_index = torch.cat(lane_successor_edge_index, dim=1)
        else:
            lane_successor_edge_index = torch.tensor([[], []], dtype=torch.long)

        feat_data['lane', 'lane']['adjacent_edge_index'] = lane_adjacent_edge_index
        feat_data['lane', 'lane']['predecessor_edge_index'] = lane_predecessor_edge_index
        feat_data['lane', 'lane']['successor_edge_index'] = lane_successor_edge_index
            
        return [feat_data]
        
    def collate_fn(self, batch):
        # Call collate_fn from parent class
        collated_batch = super().collate_fn(batch)
        batch_size = collated_batch['batch_size']
        input_dict = collated_batch['input_dict']
        batch_sample_count = collated_batch['batch_sample_count']

        # Create a Batch instance
        batch = Batch(batch_size=batch_size, **{str(k): v for k, v in input_dict.items()})
        batch.batch_sample_count = batch_sample_count
        return batch


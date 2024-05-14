from .base_dataset import BaseDataset
import numpy as np


class PTRDataset(BaseDataset):

    def __init__(self, config=None, is_validation=False, is_noisy=None):
        super().__init__(config, is_validation)
        self.noise_level = 0.05
        self.noisy = is_noisy if is_noisy is not None else False
        
        
    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        if self.noisy:
            # Apply additive noise augmentation
            noisy_data = self.additive_noise(data)
            return noisy_data
        else:
            return data
    
    def additive_noise(self, data):
        noisy_data = data[0].copy()  # Create a copy of the original data dictionary
        
        # Apply additive noise to 'obj_trajs' key
        obj_trajs = noisy_data['obj_trajs']
        noise = np.random.normal(loc=0, scale=self.noise_level, size=obj_trajs.shape)
        noisy_obj_trajs = obj_trajs + noise
        noisy_obj_trajs = noisy_obj_trajs.astype(obj_trajs.dtype)
        noisy_data['obj_trajs'] = noisy_obj_trajs

        return [noisy_data]



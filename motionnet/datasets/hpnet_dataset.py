from .base_dataset import BaseDataset
import numpy as np
import torch
from typing import  Dict
import pandas as pd




class HPNetDataset(BaseDataset):

    def __init__(self, config=None, is_validation=False, is_noisy=None):
        super().__init__(config, is_validation)
        
    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        return data





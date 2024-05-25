from .ptr_dataset import PTRDataset
from .hpnet_dataset import HPNetDataset

__all__ = {
    'ptr': PTRDataset,
    'hpnet': HPNetDataset,
}

def build_dataset(config,val=False, noise=False):
    dataset = __all__[config.method.model_name](
        config=config, is_validation=val, is_noisy=noise
    )
    return dataset

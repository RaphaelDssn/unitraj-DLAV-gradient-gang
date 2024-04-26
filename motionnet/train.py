import pytorch_lightning as pl
import torch

torch.set_float32_matmul_precision('medium')
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from models import build_model
from datasets import build_dataset
from utils.utils import set_seed
from pytorch_lightning.callbacks import ModelCheckpoint  # Import ModelCheckpoint
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg):

    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = OmegaConf.merge(cfg, cfg.method)

    model = build_model(cfg)

    train_set = build_dataset(cfg)
    val_set = build_dataset(cfg,val=True)

    train_batch_size = max(cfg.method['train_batch_size'] // len(cfg.devices) // train_set.data_chunk_size,1)
    eval_batch_size = max(cfg.method['eval_batch_size'] // len(cfg.devices) // val_set.data_chunk_size,1)

    call_backs = []

    checkpoint_callback = ModelCheckpoint(
        monitor='val/brier_fde',    # Replace with your validation metric
        filename='{epoch}-{val/brier_fde:.2f}',
        save_top_k=1,
        mode='min',            # 'min' for loss/error, 'max' for accuracy
    )

    call_backs.append(checkpoint_callback)

    train_loader = DataLoader(
        train_set, batch_size=train_batch_size, num_workers=cfg.load_num_workers, drop_last=False,
    collate_fn=train_set.collate_fn)

    val_loader = DataLoader(
        val_set, batch_size=eval_batch_size, num_workers=cfg.load_num_workers, shuffle=False, drop_last=False,
    collate_fn=train_set.collate_fn)

    trainer = pl.Trainer(
        max_epochs=cfg.method.max_epochs,
        logger= None if cfg.debug else WandbLogger(project="motionnet", name=cfg.exp_name),
        devices=1 if cfg.debug else cfg.devices,
        gradient_clip_val=cfg.method.grad_clip_norm,
        accelerator= "gpu", #"cpu" if cfg.debug else "gpu",
        profiler="simple",
        strategy="auto" if cfg.debug else "ddp",
        callbacks=call_backs
    )

    if cfg.ckpt_path is not None:
        trainer.validate(model=model, dataloaders=val_loader, ckpt_path=cfg.ckpt_path)

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader,ckpt_path=cfg.ckpt_path)

""" def grid_search(config):
    # Define hyperparameter grid
    param_grid = {
        'method.learning_rate': [0.001, 0.01, 0.1],
        'method.dropout': [0.1, 0.2, 0.3],
        'method.max_epochs': [50, 100, 150],
        'method.train_batch_size': [64, 128, 256],
        'method.eval_batch_size': [128, 256, 512],
        # Add other hyperparameters to tune here
    }

    # Perform grid search
    best_score = float('inf')  # Initialize with high value for minimization problem
    best_params = None

    for params in itertools.product(*[param_grid[key] for key in param_grid]):
        # Update config with current hyperparameters
        cfg = OmegaConf.merge(config, {'method': {key.split('.')[1]: val for key, val in zip(param_grid.keys(), params)}})
        
        # Train and evaluate model
        train(cfg)
        
        # Get validation score (you can use any validation metric here)
        val_score = ...  # Compute validation score
        
        # Track best hyperparameters
        if val_score < best_score:
            best_score = val_score
            best_params = params

    print("Best hyperparameters:", best_params)
    print("Best validation score:", best_score)

    # Train final model with best hyperparameters
    cfg = OmegaConf.merge(config, {'method': {key.split('.')[1]: val for key, val in zip(param_grid.keys(), best_params)}})
    train(cfg)

if __name__ == '__main__':
    # Load config
    cfg = OmegaConf.load(hydra.utils.get_original_cwd() + "/configs/config.yaml")
    
    # Perform grid search
    grid_search(cfg) """

if __name__ == '__main__':
    train()


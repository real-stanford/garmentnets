# %%
# Change CWD
import os
import sys

from pytorch_lightning.core import datamodule
from torch.utils.data import dataloader
project_dir = os.path.expanduser("~/dev/cloth_tracking")
os.chdir(project_dir)
sys.path.append(project_dir)

# %%
# import
import pathlib
import yaml
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl

from datasets.conv_implicit_wnf_dataset import ConvImplicitWNFDataModule
from networks.pointnet2_nocs import PointNet2NOCS
from networks.conv_implicit_wnf import ConvImplicitWNFPipeline

# %%
# main script
@hydra.main(config_path="../config", config_name="train_conv_implicit_iccv_pipeline_default")
def main(cfg: DictConfig) -> None:
    # hydra creates working directory automatically
    print(os.getcwd())
    os.mkdir("checkpoints")

    datamodule = ConvImplicitWNFDataModule(**cfg.datamodule)
    batch_size = datamodule.kwargs['batch_size']

    pointnet2_model = PointNet2NOCS.load_from_checkpoint(
        cfg.pointnet2_model.checkpoint_path)
    pointnet2_model.batch_size = batch_size

    pointnet2_params = dict(pointnet2_model.hparams)
    pipeline_model = ConvImplicitWNFPipeline(
        pointnet2_params=pointnet2_params, 
        batch_size=batch_size, **cfg.conv_implicit_model)
    pipeline_model.pointnet2_nocs = pointnet2_model
    
    category = pathlib.Path(cfg.datamodule.zarr_path).stem
    cfg.logger.tags.append(category)
    logger = pl.loggers.WandbLogger(
        project=os.path.basename(__file__),
        **cfg.logger)
    # logger.watch(pipeline_model, **cfg.logger_watch)
    wandb_run = logger.experiment
    wandb_meta = {
        'run_name': wandb_run.name,
        'run_id': wandb_run.id
    }

    all_config = {
        'config': OmegaConf.to_container(cfg, resolve=True),
        'output_dir': os.getcwd(),
        'wandb': wandb_meta
    }
    yaml.dump(all_config, open('config.yaml', 'w'), default_flow_style=False)
    logger.log_hyperparams(all_config)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="checkpoints",
        filename="{epoch}-{val_loss:.4f}",
        monitor='val_loss', save_top_k=20, mode='min', 
        save_weights_only=False, period=1)
    trainer = pl.Trainer(
        logger=logger, 
        checkpoint_callback=checkpoint_callback,
        check_val_every_n_epoch=1,
        **cfg.trainer)
    trainer.fit(model=pipeline_model, datamodule=datamodule)

# %%
# driver
if __name__ == "__main__":
    main()

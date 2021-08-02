# %%
# import
import os
import sys
import pathlib

import yaml
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl

from datasets.conv_implicit_wnf_dataset import ConvImplicitWNFDataModule
from networks.pointnet2_nocs import PointNet2NOCS

# %%
# main script
@hydra.main(config_path="config", config_name="train_pointnet2_default")
def main(cfg: DictConfig) -> None:
    # hydra creates working directory automatically
    print(os.getcwd())
    os.mkdir("checkpoints")

    datamodule = ConvImplicitWNFDataModule(**cfg.datamodule)
    batch_size = datamodule.kwargs['batch_size']
    model = PointNet2NOCS(batch_size=batch_size, **cfg.model)
    model.batch_size = batch_size

    category = pathlib.Path(cfg.datamodule.zarr_path).stem
    cfg.logger.tags.append(category)
    logger = pl.loggers.WandbLogger(
        project=os.path.basename(__file__),
        **cfg.logger)
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
        monitor='val_loss',
        save_last=True,
        save_top_k=20,
        mode='min', 
        save_weights_only=False, 
        every_n_epochs=1,
        save_on_train_epoch_end=True)
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        checkpoint_callback=True,
        logger=logger, 
        check_val_every_n_epoch=1,
        **cfg.trainer)
    trainer.fit(model=model, datamodule=datamodule)

# %%
# driver
if __name__ == "__main__":
    main()

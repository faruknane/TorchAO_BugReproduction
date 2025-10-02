import numpy as np
import torch
import torch.backends
import torch.backends.cuda
import torch.backends.cudnn
import torch.nn as nn
import torch.nn.functional as F

import json
import time
import shutil
import os
import yaml
import shutil
import lightning as L
import pytorch_lightning as pl
from lightning.pytorch.strategies import DDPStrategy, FSDPStrategy
from torch.distributed.algorithms.ddp_comm_hooks import post_localSGD_hook as post_localSGD
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as default
from lightning.pytorch.callbacks import ModelCheckpoint
import torch._inductor.config as inductor_config
from models import DiffusionModel

inductor_config.inplace_buffers = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# set precision to medium or high (tf32)
torch.set_float32_matmul_precision("high")


if __name__ == "__main__":

    csv_logger = pl.loggers.CSVLogger("logs", name="diffusion")
    
    # trainer
    trainer = L.Trainer(logger=csv_logger, 
                        enable_checkpointing=False,
                        max_epochs=500,
                        log_every_n_steps=30,
                        accelerator="gpu",
                        precision="bf16",
                        devices=[0],
    )

    model = DiffusionModel(image_key="image")

    # create a dummy dataset
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, length=1000, img_size=1024):
            self.length = length
            self.img_size = img_size

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            img = torch.randn(3, self.img_size, self.img_size)
            sample = {
                "x_cond": img,
                "image": img,
                "instruction": "dummy instruction",
            }
            return sample

    train_dataloader = torch.utils.data.DataLoader(DummyDataset(), batch_size=1, shuffle=True, num_workers=2)

    trainer.fit(model, train_dataloader, train_dataloader)






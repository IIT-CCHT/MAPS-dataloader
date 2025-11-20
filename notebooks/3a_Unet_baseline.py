
##### ======================
# U-Net notebook for the Palaeochannels dataset (MAPS)
# Input: Dataset loaded using Dataloader.py through the variables 'full_train_dataset' and 'full_val_dataset'
##### ======================

# %%
# SET THE WORKING DIRECTORY to match path to local code modules
import os
from pathlib import Path
workdir = Path(os.getenv("WORKDIR", Path(__file__).resolve().parents[1]))
import sys
sys.path.append(str(workdir))
print(workdir)

# %%
# Import libraries
from torchinfo import summary

import geopandas as gpd
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset
import kornia.augmentation as K

from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import segmentation_models_pytorch as smp

from lightning.pytorch.loggers import TensorBoardLogger

from lightning.pytorch import LightningModule
from segmentation_models_pytorch.decoders.unet import Unet
from torchgeo.models.resnet import ResNet50_Weights, resnet50
from torchmetrics.classification import BinaryJaccardIndex, BinaryPrecision, BinaryRecall, BinaryPrecisionRecallCurve

from torchvision.utils import make_grid, draw_segmentation_masks

from typing import Sequence
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from lightning.pytorch.callbacks import StochasticWeightAveraging

from lightning import Trainer
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import ModelCheckpoint

# Import modules and variables
from notebooks.dataset.models import UnetForSeg
from notebooks.dataset.config import SCRATCH_DIR

# %%
#### ======
# LOAD DATA
#### ======

train_loader = DataLoader(dataset=full_train_dataset, 
                          batch_size=16, 
                          num_workers=16, 
                          prefetch_factor=16, 
                          pin_memory=True, 
                          persistent_workers=True, 
                          shuffle=True)
val_loader = DataLoader(dataset=full_val_dataset, 
                        batch_size=16, 
                        num_workers=16, 
                        prefetch_factor=16, 
                        pin_memory=True, 
                        persistent_workers=True, 
                        shuffle=True) 

# %%
# Compile the model
lightning_module = UnetForSeg(model_tag='mit_b5')

logs_dir = SCRATCH_DIR / 'run_logs_test'
logs_dir.mkdir(parents=True, exist_ok=True)

logger = TensorBoardLogger(name='dice-headonly-image_clip-imagenet-mit_b5-median', save_dir=logs_dir)

checkpointing = ModelCheckpoint(filename='epoch={epoch}-val_iou={validation/iou:.8f}', 
                                auto_insert_metric_name=False, 
                                monitor='validation/iou', 
                                mode='max', 
                                save_top_k=2, 
                                save_last=True)

trainer = Trainer(accelerator='gpu', devices=[0], log_every_n_steps=5, logger=logger, callbacks=[checkpointing])

# %%
# Train the model
trainer.fit(lightning_module, train_dataloaders=train_loader, val_dataloaders=val_loader)
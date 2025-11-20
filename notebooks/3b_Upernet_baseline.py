
##### ======================
# UPerNet notebook for the Palaeochannels dataset (MAPS)
# Input: Dataset loaded using dataloader.py through the variables 'full_train_dataset' and 'full_val_dataset'
##### ======================

# %%
# SET THE WORKING DIRECTORY to match path to local code modules
import os
from pathlib import Path
workdir = Path(os.getenv("WORKDIR", Path(__file__).resolve().parents[1]))
import sys
sys.path.append(str(workdir))

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

from lightning.pytorch import LightningModule

from torchmetrics.classification import BinaryJaccardIndex, BinaryPrecision, BinaryRecall, BinaryPrecisionRecallCurve

from torchvision.utils import make_grid, draw_segmentation_masks
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F

from transformers import AutoImageProcessor, SegformerForSemanticSegmentation, UperNetForSemanticSegmentation
from transformers import AutoConfig

from typing import Sequence
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lightning.pytorch.callbacks import StochasticWeightAveraging

from lightning import Trainer
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

# import modules and variables
from notebooks.dataset.models import ViTForSeg
from notebooks.dataset.config import SCRATCH_DIR

# %%
#### ======
# LOAD DATA
#### ======

train_loader = DataLoader(dataset=full_train_dataset, 
                          batch_size=16, 
                          num_workers=8, 
                          prefetch_factor=8, 
                          pin_memory=True, 
                          persistent_workers=True, 
                          shuffle=True)
val_loader = DataLoader(dataset=full_val_dataset, 
                        batch_size=16, 
                        num_workers=8, 
                        prefetch_factor=8, 
                        pin_memory=True, 
                        persistent_workers=True, 
                        shuffle=False) 

# %%
# Compile the model
max_epochs = 50

#### ======= 
# MODEL TAG
#- model_tag = 'nvidia/segformer-b0-finetuned-ade-512-512' version 0
#- model_tag = 'peldrak/segformer-b0-ade-512-512-finetuned-coastTrain' version 1
#- model_tag = 'openmmlab/upernet-convnext-tiny' version 2
#### =======

vit_model = ViTForSeg(model_tag='openmmlab/upernet-convnext-tiny')

logs_dir = SCRATCH_DIR / 'run_logs_UPerNet'
logs_dir.mkdir(parents=True, exist_ok=True)

logger = TensorBoardLogger(name='vit_test', save_dir=logs_dir)

checkpointing = ModelCheckpoint(filename='epoch={epoch}-val_iou={validation/iou:.8f}', 
                                auto_insert_metric_name=False, 
                                monitor='validation/iou', 
                                mode='max', 
                                save_top_k=2, 
                                save_last=True)



trainer = Trainer(accelerator='gpu', devices=[0], log_every_n_steps=5, logger=logger, callbacks=[checkpointing],max_epochs=max_epochs)
# %%
# Train the model
trainer.fit(vit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

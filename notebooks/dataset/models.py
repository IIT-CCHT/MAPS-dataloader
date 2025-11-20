#### =================================
# Code for the Unet and UperNet models
#### =================================

# %%
# Import libraries
import rasterio as rio
import matplotlib.pyplot as plt
from rasterio.plot import reshape_as_image
import torch.nn as nn
from torch.masked import masked_tensor
import torch.nn.functional as F
import numpy as np
from rasterio.transform import from_origin
from pathlib import Path

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
import os
from lightning.pytorch.loggers import TensorBoardLogger
from torch.masked import masked_tensor
from lightning.pytorch import LightningModule
from segmentation_models_pytorch.decoders.unet import Unet
from torchgeo.models.resnet import ResNet50_Weights, resnet50
import torchvision.models as models
from torchmetrics.classification import BinaryJaccardIndex, BinaryPrecision, BinaryRecall, BinaryPrecisionRecallCurve

from torchvision.utils import make_grid, draw_segmentation_masks
from PIL import Image

from typing import Sequence
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from lightning.pytorch.callbacks import StochasticWeightAveraging

from transformers import AutoConfig
from typing import Sequence
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from lightning.pytorch.callbacks import StochasticWeightAveraging
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation, UperNetForSemanticSegmentation
from torch.optim.lr_scheduler import ReduceLROnPlateau


# %%
from pathlib import Path
import os
workdir = Path(os.path.abspath(os.path.join(os.getcwd(), '..')))  # Set workdir to the parent directory 
print(workdir)

# %% 
#### ==============
# Unet model
#### ==============

class UnetForSeg(LightningModule):
    def __init__(self, *args: Any, 
                 learning_rate: float = 1.0e-3, 
                 logits_threshold: float = 0.1, 
                 weight_decay: float = 1.0e-3, 
                 clip_stds: float = 2.5, 
                 swa_lrs = 1e-3,
                 swa_epoch_start = 20,
                 tversky_gamma: float = 1.0, 
                 tversky_alpha: float = 0.4, 
                 tversky_beta: float = 0.6,
                 model_tag: str = 'unet-resnet50-sen2-rgb-moco',
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        
        # Model creation and weights loading
        if model_tag == 'unet-resnet50-sen2-rgb-moco':
            self.model = Unet(encoder_name="resnet50", in_channels=3)
            encoder_model = resnet50(weights=ResNet50_Weights.SENTINEL2_RGB_MOCO)
            self.model.encoder.load_state_dict(encoder_model.state_dict())
        else:
            self.model = Unet(encoder_name="mit_b5", decoder_attention_type='scse', in_channels=3)
        
        # Loss function
        # self.loss_criterion = smp.losses.TverskyLoss(mode="binary", 
        #                                              gamma=self.hparams.tversky_gamma, 
        #                                              alpha=self.hparams.tversky_alpha, 
        #                                              beta=self.hparams.tversky_beta,)
            
        self.loss_criterion = smp.losses.DiceLoss(mode='binary')
        # Augmentations
        self.spatial_augmentation_pipeline = K.AugmentationSequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            data_keys=["input", "mask"]  # Apply to both image and mask
        )
        self.color_augmentation_pipeline = K.AugmentationSequential(
            # K.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.2, hue=0.1),
            data_keys=["input"]  # Apply to images only
        )
        # Metrics creation
        self.train_iou = BinaryJaccardIndex(threshold=self.hparams.logits_threshold)
        self.train_precision = BinaryPrecision(threshold=self.hparams.logits_threshold)
        self.train_recall = BinaryRecall(threshold=self.hparams.logits_threshold)
        
        self.validation_iou = BinaryJaccardIndex(threshold=self.hparams.logits_threshold)
        self.validation_precision = BinaryPrecision(threshold=self.hparams.logits_threshold)
        self.validation_recall = BinaryRecall(threshold=self.hparams.logits_threshold)
        
        self.validation_prec_rec_curve = BinaryPrecisionRecallCurve(thresholds=20)
        
        
        
    def configure_callbacks(self) -> Sequence[Callback] | Callback:
        swa = StochasticWeightAveraging(swa_lrs=self.hparams.swa_lrs, 
                                        swa_epoch_start=self.hparams.swa_epoch_start)
        return [swa]
    
    def configure_optimizers(self) -> OptimizerLRScheduler:
        return optim.Adam(self.model.parameters(),
                          lr=self.hparams.learning_rate, 
                          weight_decay=self.hparams.weight_decay)

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        # Fix batch types.
        images = batch['image'].float()
        masks = batch['mask'].float().unsqueeze(1)
        
        # Finish raster to tensor conversion
        images = torch.movedim(images, -1, -3)
        images = images[:, [3, 2, 1], :, :] # Select RGB bands
         
        # Clip each tile bands using local statistics.
        
        masked_images = masked_tensor(images, images > 0.0)
        images_mean = masked_images.mean(dim=(2, 3), keepdim=True).get_data()
        images_std = masked_images.std(dim=(2, 3), keepdim=True).get_data()
        images_max_clip = images_mean + self.hparams.clip_stds * images_std
        images_min_clip = images_mean - self.hparams.clip_stds * images_std
        images = (images - images_min_clip) / (images_max_clip - images_min_clip)
        
        # Data augmentation during training.
        if self.trainer.training: 
            images = self.color_augmentation_pipeline(images)
            images, masks = self.spatial_augmentation_pipeline(images, masks)
        
        batch['image'] = images
        batch['mask'] = masks.squeeze(1).int()
        return super().on_after_batch_transfer(batch, dataloader_idx)
    
    def forward(self, batch):
        x = batch['image']
        # Do not propagate gradient to the encoder network for now.
        with torch.no_grad():
            features = self.model.encoder(x)
        decoder_output = self.model.decoder(*features)

        return self.model.segmentation_head(decoder_output)
    
    def training_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        logits = self.forward(batch).squeeze()
        loss = self.loss_criterion(logits, batch['mask'])
        
        # Update metrics
        self.train_iou(logits, batch['mask'])
        self.train_precision(logits, batch['mask'])
        self.train_recall(logits, batch['mask'])
        
        # Log step loss
        batch_size = batch['image'].shape[0]
        self.log('train/loss', loss, on_epoch=True, on_step=True, batch_size=batch_size)
        
        # Log batch output (images)
        if batch_idx == 0:
            self.log_batch_output(batch, logits)
            
        return loss
    
    def on_train_epoch_end(self) -> None:
        # Log metrics
        self.log('train/iou', self.train_iou)
        self.log('train/precision', self.train_precision)
        self.log('train/recall', self.train_recall)
        return super().on_train_epoch_end()
    
    def log_batch_output(self, batch, logits):
        if isinstance(self.logger, TensorBoardLogger):
            stage = 'none'
            if self.trainer.validating:
                stage = 'validation'
            if self.trainer.training:
                stage = 'train'
            mask = batch['mask'].unsqueeze(1)
            segmentation_mask = logits.unsqueeze(1) > self.hparams.logits_threshold
            summary_writer: SummaryWriter = self.logger.experiment
            summary_writer.add_images(f'{stage}/image', batch['image'], global_step=self.trainer.global_step)
            summary_writer.add_images(f'{stage}/mask', mask * 255, global_step=self.trainer.global_step)
            summary_writer.add_images(f'{stage}/logits', segmentation_mask, global_step=self.trainer.global_step)
            
            
    # The same thing as the training step but on validation objects.
    def validation_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        logits = self.forward(batch).squeeze()
        loss = self.loss_criterion(logits, batch['mask'])
        
        self.validation_iou(logits, batch['mask'])
        self.validation_precision(logits, batch['mask'])
        self.validation_recall(logits, batch['mask'])
        
        self.validation_prec_rec_curve(logits, batch['mask'])
        
        batch_size = batch['image'].shape[0]
        self.log('validation/loss', loss, on_epoch=True, batch_size=batch_size)
        if batch_idx == 0:
            self.log_batch_output(batch, logits)
        
        return loss
        
    def on_validation_epoch_end(self) -> None:
        self.log('validation/iou', self.validation_iou)
        self.log('validation/precision', self.validation_precision)
        self.log('validation/recall', self.validation_recall)
        
        # Log the precision recall curve!
        if isinstance(self.logger, TensorBoardLogger):
            summary_writer: SummaryWriter = self.logger.experiment
            fig_, ax_ = self.validation_prec_rec_curve.plot(score=True)
            summary_writer.add_figure('validation/prec_rec_curve', figure=fig_, global_step=self.trainer.global_step)
            
        return super().on_validation_epoch_end()
    

##### ====================
# Visual Transformer model
#### =====================

class ViTForSeg(LightningModule):
    def __init__(self, 
                 id2label_mapping: dict = None,
                 learning_rate: float = 1.0e-4, 
                 logits_threshold: float = 0.1, 
                 weight_decay: float = 1.0e-3, 
                 clip_stds: float = 2.5, 
                 swa_lrs = [1e-3],
                 swa_epoch_start = 20,
                 tversky_gamma: float = 1.0, 
                 tversky_alpha: float = 0.4, 
                 tversky_beta: float = 0.6,
                 model_tag: str = 'openmmlab/upernet-convnext-tiny',
                 **kwargs: Any) -> None:
        super(ViTForSeg, self).__init__()
        self.save_hyperparameters()
        
        # Model creation & Load the configuration & Upsampling layer for the segmentation output
        if model_tag == 'peldrak/segformer-b0-ade-512-512-finetuned-coastTrain':
            print('model 2 =',model_tag)
            config = AutoConfig.from_pretrained(workdir /"config.json",)
            self.model = SegformerForSemanticSegmentation.from_pretrained("peldrak/segformer-b0-ade-512-512-finetuned-coastTrain",
                                                                      config=config,ignore_mismatched_sizes=True)
            #self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        elif model_tag == 'nvidia/segformer-b0-finetuned-ade-512-512':
            print('model 1 =',model_tag)
            config = AutoConfig.from_pretrained(workdir /"config.json",)
            self.model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512",
                                                                      config=config,ignore_mismatched_sizes=True)
            self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        else:
            print('model 3 =',model_tag)
            config = AutoConfig.from_pretrained(workdir /"config.json",)
            self.model = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-tiny",
                                                                      config=config,ignore_mismatched_sizes=True)
          
        # TverskyLoss Loss function 
        # self.loss_criterion = smp.losses.TverskyLoss(mode="binary", 
        #                                              gamma=self.tversky_gamma, 
        #                                              alpha=self.tversky_alpha, 
        #                                              beta=self.tversky_beta,)
        # DiceLoss Loss function
        self.loss_criterion = smp.losses.DiceLoss(mode='binary')

        # Augmentations
        self.spatial_augmentation_pipeline = K.AugmentationSequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            data_keys=["input", "mask"]  # Apply to both image and mask
        )
        self.color_augmentation_pipeline = K.AugmentationSequential(
            # K.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.2, hue=0.1),
            data_keys=["input"]  # Apply to images only
        )
        # Metrics creation
        self.train_iou = BinaryJaccardIndex(threshold=self.hparams.logits_threshold)
        self.train_precision = BinaryPrecision(threshold=self.hparams.logits_threshold)
        self.train_recall = BinaryRecall(threshold=self.hparams.logits_threshold)
        self.validation_iou = BinaryJaccardIndex(threshold=self.hparams.logits_threshold)
        self.validation_precision = BinaryPrecision(threshold=self.hparams.logits_threshold)
        self.validation_recall = BinaryRecall(threshold=self.hparams.logits_threshold)
        self.validation_prec_rec_curve = BinaryPrecisionRecallCurve(thresholds=20)


    def forward(self, batch):
        x = batch['image']
        segmentation_output = self.model(x)  # Assuming the model returns segmentation logits directly
        segmentation_output_logits = segmentation_output.logits
        # print('forward segmentation_output',segmentation_output_logits.shape)
        
        # Assuming SemanticSegmenterOutput has a logits attribute
        logits = segmentation_output_logits[:,1:2,:,:]
        # print('forward logits',logits.shape)
        
        return logits


    def configure_callbacks(self) -> Sequence[Callback] | Callback:
        swa = StochasticWeightAveraging(swa_lrs=self.hparams.swa_lrs, 
                                        swa_epoch_start=self.hparams.swa_epoch_start)
        return [swa]
    
    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = optim.Adam(self.model.parameters(),
                          lr=self.hparams.learning_rate, 
                          weight_decay=self.hparams.weight_decay)
    
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True),
            'monitor': 'validation/loss',  # Metric to monitor
            'interval': 'epoch',
            'frequency': 1
        }

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        # Fix batch types.
        images = batch['image'].float()
        masks = batch['mask'].float().unsqueeze(1)
        
        # Finish raster to tensor conversion
        images = torch.movedim(images, -1, -3)
        images = images[:, [3, 2, 1], :, :] # Select RGB bands
        # print(images.shape)
        # Clip each tile bands using local statistics.
        masked_images = masked_tensor(images, images > 0.0)
        images_mean = masked_images.mean(dim=(2, 3), keepdim=True).get_data()
        images_std = masked_images.std(dim=(2, 3), keepdim=True).get_data()
        images_max_clip = images_mean + self.hparams.clip_stds * images_std
        images_min_clip = images_mean - self.hparams.clip_stds * images_std
        images = (images - images_min_clip) / (images_max_clip - images_min_clip)
        
        # Data augmentation during training.
        if self.trainer.training: 
            images = self.color_augmentation_pipeline(images)
            images, masks = self.spatial_augmentation_pipeline(images, masks)
        
        batch['image'] = images
        batch['mask'] = masks.squeeze(1).int()
        return super().on_after_batch_transfer(batch, dataloader_idx)
    

    def training_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        # Move batch data to GPU    
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        logits = self.forward(batch).squeeze()
        loss = self.loss_criterion(logits, batch['mask'])
        
        # Update metrics
        self.train_iou(logits, batch['mask'])
        self.train_precision(logits, batch['mask'])
        self.train_recall(logits, batch['mask'])
        
        # Log step loss
        batch_size = batch['image'].shape[0]
        self.log('train/loss', loss, on_epoch=True, on_step=True, batch_size=batch_size)
        
        # Log batch output (images)
        if batch_idx == 0:
            self.log_batch_output(batch, logits)
            
        return loss
    
    def on_train_epoch_end(self) -> None:
        # Log metrics
        self.log('train/iou', self.train_iou)
        self.log('train/precision', self.train_precision)
        self.log('train/recall', self.train_recall)
        return super().on_train_epoch_end()
    
    def log_batch_output(self, batch, logits):
        if isinstance(self.logger, TensorBoardLogger):
            stage = 'none'
            if self.trainer.validating:
                stage = 'validation'
            if self.trainer.training:
                stage = 'train'
            mask = batch['mask'].unsqueeze(1)
            segmentation_mask = logits.unsqueeze(1) > self.hparams.logits_threshold
            summary_writer: SummaryWriter = self.logger.experiment
            summary_writer.add_images(f'{stage}/image', batch['image'], global_step=self.trainer.global_step)
            summary_writer.add_images(f'{stage}/mask', mask * 255, global_step=self.trainer.global_step)
            summary_writer.add_images(f'{stage}/logits', segmentation_mask, global_step=self.trainer.global_step)
            
            
    # The same thing as the training step but on validation objects.
    def validation_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        # Move batch data to GPU
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        logits = self.forward(batch).squeeze()
        # print ('val logits',logits.shape)
        # print('val batch[mask]',batch['mask'].shape)
        loss = self.loss_criterion(logits, batch['mask'])
        self.validation_iou(logits, batch['mask'])
        self.validation_precision(logits, batch['mask'])
        self.validation_recall(logits, batch['mask'])
        self.validation_prec_rec_curve(logits, batch['mask'])
        
        batch_size = batch['image'].shape[0]
        self.log('validation/loss', loss, on_epoch=True, batch_size=batch_size)
        if batch_idx == 0:
            self.log_batch_output(batch, logits)
        
        return loss
        
    def on_validation_epoch_end(self) -> None:
        self.log('validation/iou', self.validation_iou)
        self.log('validation/precision', self.validation_precision)
        self.log('validation/recall', self.validation_recall)
        
        # Log the precision recall curve!
        if isinstance(self.logger, TensorBoardLogger):
            summary_writer: SummaryWriter = self.logger.experiment
            fig_, ax_ = self.validation_prec_rec_curve.plot(score=True)
            summary_writer.add_figure('validation/prec_rec_curve', figure=fig_, global_step=self.trainer.global_step)
            
        return super().on_validation_epoch_end()
# %%

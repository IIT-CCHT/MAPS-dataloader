# %%
#### =====================================================================================
# DATALOADER
# This dataloader creates TRAINING and VALIDATION sets to train the models
# Before running the code, make sure to set the dataset home location in dataset/config.py 
#### =====================================================================================

# import libraries
from torchinfo import summary
import sys
import geopandas as gpd
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data import Dataset
import rasterio as rio
from typing import Any, List
from rasterio.features import rasterize
from rasterio.plot import reshape_as_image
from rasterio.windows import from_bounds, transform as w_transform
from torchvision.utils import make_grid, draw_segmentation_masks
import numpy as np

import kornia.augmentation as K
from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import segmentation_models_pytorch as smp
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import LightningModule
from torchvision.utils import make_grid, draw_segmentation_masks
from dotenv import load_dotenv
load_dotenv()

from math import floor
from itertools import product
from shapely.geometry import box

# %%
# import modules 
import os
from pathlib import Path
workdir = Path(os.getenv("WORKDIR", Path(__file__).resolve().parents[1]))
import sys
sys.path.append(str(workdir))

from notebooks.dataset.config import BASE_DIR, IMAGES_PATH, REF_IMAGE, OVERLAP, TILE_WIDTH, TILE_HEIGHT, AREA_DIR, ANNOTATIONS_DIR
from notebooks.dataset.dataset import SingleRasterPalaeochannelDataset

# %%
#### ==================================
#  Computing tiles geometry on raster data
#### ==================================

source_db = rio.open(REF_IMAGE)
image_crs = source_db.crs
image_crs

x_indices = range(0, source_db.width, floor(TILE_WIDTH * (1 - OVERLAP)))
y_indices = range(0, source_db.height, floor(TILE_HEIGHT * (1 - OVERLAP)))
grid_coordinates = product(y_indices, x_indices) # Not a typo, the raster starts from the top-left coordinate. 

def tile_box_for_coordinates(grid_x, grid_y):
    tile_min = source_db.xy(grid_x, grid_y)
    tile_max = source_db.xy(grid_x + TILE_WIDTH, grid_y + TILE_HEIGHT)
    tile_box = box(tile_min[0], tile_min[1], tile_max[0], tile_max[1])
    return tile_box
grid_boxes = [tile_box_for_coordinates(grid_xy[0], grid_xy[1]) for grid_xy in grid_coordinates]

raster_tiles_df = gpd.GeoDataFrame(geometry=grid_boxes, crs=image_crs)
raster_tiles_df["height"] = TILE_HEIGHT
raster_tiles_df["width"] = TILE_WIDTH

# generate a visual interactive map
# raster_tiles_df.explore(style_kwds=dict(fill=False))

# %%

# Load train and validation sets
train_set_df = gpd.read_file(AREA_DIR / 'Train_Area.geojson')
val_set_df = gpd.read_file(AREA_DIR / 'Validation_Area.geojson')

# generate a visual interactive map
# val_area_map = val_set_df.explore(style_kwds=dict(fill=False, color='red'))

# The `unary_union` method is used bacause the used DataFrame methods expect 
# series of the same length but work with broadcast semantics.
#
# We consider val tiles those that have more than 50% overlap with the val 
# area polygons.  
val_overlap_query = (raster_tiles_df.intersection(val_set_df.geometry.unary_union).area / raster_tiles_df.geometry.area) > 0.5
val_set_tiles = raster_tiles_df[val_overlap_query]

# We consider train tiles those that have more than 50% overlap with the train
# area polygons and do not intersect with the val area.
train_overlap_query = (raster_tiles_df.intersection(train_set_df.geometry.unary_union).area / raster_tiles_df.geometry.area) > 0.5
train_set_tiles = raster_tiles_df[train_overlap_query & ~(raster_tiles_df.contains(val_set_tiles.geometry.unary_union))]

print(f"Got {len(train_set_tiles)} training tiles and {len(val_set_tiles)} val tiles")

# generate a visual interactive map
#val_set_tiles.explore(m=val_area_map, style_kwds=dict(fill=False, color='blue'))
#train_set_tiles.explore(m=val_area_map, style_kwds=dict(fill=False, color='green'))

#  Saving the tiles to file.
val_set_tiles.to_file(BASE_DIR/ "valTiles_CLS_UTM.geojson", driver='GeoJSON', index=False)
train_set_tiles.to_file(BASE_DIR/ "TrainTiles_CLS_UTM.geojson", driver='GeoJSON', index=False)


# %%
#### ======================
#LOADER
#### ======================

# Creating variables of images and corresponfing vector labels 
train_tiles_df = gpd.read_file(BASE_DIR/'TrainTiles_CLS_UTM.geojson')
train_aoi_df = gpd.read_file(AREA_DIR / 'Train_Area.geojson')
val_tiles_df = gpd.read_file(BASE_DIR/'valTiles_CLS_UTM.geojson')
val_aoi_df = gpd.read_file(AREA_DIR / 'Validation_Area.geojson')

april_tif_path = IMAGES_PATH / 'Train_04_Apr.tif'
march_tif_path = IMAGES_PATH / 'Train_03_March.tif'
march_april_features_df = gpd.read_file(ANNOTATIONS_DIR / 'L1_Train_March-April.geojson')

aug_tif_path = IMAGES_PATH / 'Train_08_Aug.tif'
jul_tif_path = IMAGES_PATH / 'Train_07_Jul.tif'
jul_aug_features_df = gpd.read_file(ANNOTATIONS_DIR / 'L2_Train_July-August.geojson')

nov_tif_path = IMAGES_PATH / 'Train_11_Nov.tif'
jan_tif_path = IMAGES_PATH / 'Train_12_Jan23.tif'
nov_jan_features_df = gpd.read_file(ANNOTATIONS_DIR / 'L3_Train_November-January23.geojson')

# %%
##### =======================
#LOADER FOR TR1 SETTING
##### =======================

# Creating dataloader for TR1
TR1_april_train_dataset = SingleRasterPalaeochannelDataset(train_tiles_df, april_tif_path, march_april_features_df, train_aoi_df)
TR1_august_train_dataset = SingleRasterPalaeochannelDataset(train_tiles_df, aug_tif_path, jul_aug_features_df, train_aoi_df)
TR1_nov_train_dataset = SingleRasterPalaeochannelDataset(train_tiles_df, nov_tif_path, nov_jan_features_df, train_aoi_df)

TR1_april_val_dataset = SingleRasterPalaeochannelDataset(val_tiles_df, april_tif_path, march_april_features_df, val_aoi_df)
TR1_august_val_dataset = SingleRasterPalaeochannelDataset(val_tiles_df, aug_tif_path, jul_aug_features_df, val_aoi_df)
TR1_nov_val_dataset = SingleRasterPalaeochannelDataset(val_tiles_df, nov_tif_path, nov_jan_features_df, val_aoi_df)

full_train_dataset = ConcatDataset([TR1_april_train_dataset, TR1_august_train_dataset, TR1_nov_train_dataset])
full_val_dataset = ConcatDataset([TR1_april_val_dataset, TR1_august_val_dataset, TR1_nov_val_dataset])
print(f"Datasets built! {len(full_train_dataset)} training tiles, {len(full_val_dataset)} validation tiles.")

# %%
##### =======================
#LOADER FOR TR2 SETTING. 
# Use either of this settings(TR1 or TR2) to get different outcomes following paper's methodology
##### =======================
"""
TR2_april_train_dataset = SingleRasterPalaeochannelDataset(train_tiles_df, april_tif_path, march_april_features_df, train_aoi_df)
TR2_march_train_dataset = SingleRasterPalaeochannelDataset(train_tiles_df, march_tif_path, march_april_features_df, train_aoi_df)
TR2_august_train_dataset = SingleRasterPalaeochannelDataset(train_tiles_df, aug_tif_path, jul_aug_features_df, train_aoi_df)
TR2_july_train_dataset = SingleRasterPalaeochannelDataset(train_tiles_df, jul_tif_path, jul_aug_features_df, train_aoi_df)
TR2_nov_train_dataset = SingleRasterPalaeochannelDataset(train_tiles_df, nov_tif_path, nov_jan_features_df, train_aoi_df)
TR2_jan_train_dataset = SingleRasterPalaeochannelDataset(train_tiles_df, jan_tif_path, nov_jan_features_df, train_aoi_df)
TR2_april_val_dataset = SingleRasterPalaeochannelDataset(val_tiles_df, april_tif_path, march_april_features_df, val_aoi_df)
TR2_march_val_dataset = SingleRasterPalaeochannelDataset(val_tiles_df, march_tif_path, march_april_features_df, val_aoi_df)
TR2_august_val_dataset = SingleRasterPalaeochannelDataset(val_tiles_df, aug_tif_path, jul_aug_features_df, val_aoi_df)
TR2_july_val_dataset = SingleRasterPalaeochannelDataset(val_tiles_df, jul_tif_path, jul_aug_features_df, val_aoi_df)
TR2_nov_val_dataset = SingleRasterPalaeochannelDataset(val_tiles_df, nov_tif_path, nov_jan_features_df, val_aoi_df)
TR2_jan_val_dataset = SingleRasterPalaeochannelDataset(val_tiles_df, jan_tif_path, nov_jan_features_df, val_aoi_df)
full_train_dataset = ConcatDataset([TR2_april_train_dataset, TR2_march_train_dataset, TR2_august_train_dataset, TR2_july_train_dataset, TR2_nov_train_dataset, TR2_jan_train_dataset])
full_val_dataset = ConcatDataset([TR2_april_val_dataset, TR2_march_val_dataset, TR2_august_val_dataset, TR2_july_val_dataset, TR2_nov_val_dataset, TR2_jan_val_dataset])
print(f"Datasets built! {len(full_train_dataset)} training tiles, {len(full_val_dataset)} validation tiles.")

"""


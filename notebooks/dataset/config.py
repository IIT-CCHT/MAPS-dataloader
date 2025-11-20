# %%
#####==============================
# CONFIGURATION FILE
# This config file defines local variables to run the MAPS code
#####=============================
# %%
import os
from pathlib import Path

# %%
### =====
# SET PATH TO MAPS FOLDER
### ====

# Set base directory of the MAPS dataset
BASE_DIR = Path("/HDD1/data_paleo/MAPS_dataset/Palaeochannels")  

## =====
#  TILES SPECIFICATION
## ====
OVERLAP = 0.5  ## Overlap among tiles in training data
OVERLAP_TEST = 0  ## No overlap used in test tiles
TILE_WIDTH = 256  ## Tiles pixels size
TILE_HEIGHT = 256

### =====
# DEFAULT VARIABLES
# Do not change if the dataset structure is unaltered" 
### =====

# Set scratch directory for checkpoint and logger savings (create automatically when training models)
SCRATCH_DIR = Path( BASE_DIR / "scratch")

# Path_to_folder_images for training the models
IMAGES_PATH = Path (BASE_DIR / "Imagery/Train_Val" ) ## adjust if necessary

#  Finding one of the train images to get CRS to compute appropriately the grid geometry
REF_IMAGE = IMAGES_PATH /'Train_04_Apr.tif'  ## adjust if necessary

# Settings for testing the model
AREA_DIR = BASE_DIR / "Area Train_Val_Test"
IMAGERY_DIR = BASE_DIR / "Imagery/Test"
ANNOTATIONS_DIR = BASE_DIR / "annotations"
AREA_FILE = AREA_DIR / "Test_Area.geojson"









##### ========================
#  Inference: U-Net notebook for Palaeochannels dataset
#__Input of this notebook:__
#- image_folder path
#- mask_folder path
#- checkpoint path
#### ========================
# %%
import os
workdir = Path(os.getenv("WORKDIR", Path(__file__).resolve().parents[1]))  # Set workdir to the parent directory of the code
import sys
sys.path.append(str(workdir))

# %%
# Import libraries

from pathlib import Path
import rasterio as rio
import matplotlib.pyplot as plt
from rasterio.plot import reshape_as_image
from torch.masked import masked_tensor
import torch.nn.functional as F
import numpy as np
from rasterio.transform import from_origin
from pathlib import Path

from segmentation_models_pytorch.decoders.unet import Unet
from torchinfo import summary

from PIL import Image

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

from lightning.pytorch import LightningModule
from segmentation_models_pytorch.decoders.unet import Unet
from torchgeo.models.resnet import ResNet50_Weights, resnet50
import torchvision.models as models
from torchmetrics.classification import BinaryJaccardIndex, BinaryPrecision, BinaryRecall, BinaryPrecisionRecallCurve

from torchvision.utils import make_grid, draw_segmentation_masks

from typing import Sequence
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from lightning.pytorch.callbacks import StochasticWeightAveraging

# %%
# Import modules
from notebooks.dataset.models import UnetForSeg
from notebooks.dataset.config import IMAGERY_DIR, SCRATCH_DIR
from notebooks.dataset.functions import save_binary_image, calculate_metrics, show_image_mask

# %%
# SET IMAGE FOR TEST

month= "apr" ## change 

test_img_folder = Path(IMAGERY_DIR / month / "output_raster_tiles")
test_mask_folder = Path(IMAGERY_DIR/ month / "output_mask_tiles")

print(test_img_folder)
print(test_mask_folder)

# set output path 
save_output = True

output_path = Path(IMAGERY_DIR / month / "Unet_inference_tiles")
if not os.path.exists(output_path):
    os.makedirs(output_path)
print(output_path)

# %%
# Load checkpoint 
chkp = SCRATCH_DIR /"run_logs_test/dice-headonly-image_clip-imagenet-mit_b5-median/version_1/checkpoints/epoch=16-val_iou=0.19196837.ckpt"  # Change path to checkpoint

# %%
# Load the model
model = Unet(encoder_name='resnet50', in_channels=3)
summary(model.encoder, input_size=(32, 3, 256, 256))

model= UnetForSeg.load_from_checkpoint(chkp)
model.eval()

# %%
# Function to make inference on test set, compute metrics and visualize results
def print_output (test_img,test_mask):
    with rio.open(test_img) as src:
        # Read the image as a numpy array
        image = src.read()
        # profile = src.profile
        # crs = src.crs
        # transform = src.transform

    with rio.open(test_mask) as src:
    # Read the image as a numpy array
        mask = src.read()
        # profile = src.profile
        # crs = src.crs
        # transform = src.transform

    #normalization
    clip_stds =2.5
    image_mean = image.mean()
    image_std = image.std()
    image_max_clip = image_mean + clip_stds * image_std
    image_min_clip = image_mean - clip_stds * image_std
    test_image = (image - image_min_clip) / (image_max_clip - image_min_clip) #[0,1]

    # change to tensor
    tile_image_tensor = torch.from_numpy(test_image)
    # set device
    tile_image_test = tile_image_tensor.to(0)
    tile_image_test = tile_image_test.float()

    # get prediction
    output= model(dict(image=tile_image_test.unsqueeze(0)))
    
    # activate function
    output = F.sigmoid(output)
    # Convert the tensor to a NumPy array and remove the batch dimension
    image_array = output.squeeze().cpu().detach().numpy()
    binary_image = (image_array > 0.5).astype(int)
    # Plot the binary image using matplotlib
    # plt.imshow(binary_image, cmap='binary')  # 'binary' colormap for binary images
    # plt.show()
    show_image_mask(image,mask,binary_image)

   # Calculate metrics
    iou, recall, precision, f1 = calculate_metrics(binary_image, mask)


    print("IoU:", iou)
    print("Recall:", recall)
    print("Precision:", precision)
    print("f1:", f1)


    return iou, recall, precision,f1, binary_image

# %%
#### ==========
#Execute model inference
#### ==========

if __name__ == "__main__":
        # Get a list of all files in the image folder
    img_file_names = sorted(os.listdir(test_img_folder))

    # Get a list of all files in the mask folder
    mask_file_names = sorted(os.listdir(test_mask_folder))

    iou_total, recall_total, precision_total, f1_total = 0.0, 0.0, 0.0, 0.0

    # Use zip to iterate over both lists simultaneously
    for img_file_name, mask_file_name in zip(img_file_names, mask_file_names):
        img_file_path = os.path.join(test_img_folder, img_file_name)
        mask_file_path = os.path.join(test_mask_folder, mask_file_name)

        # Calculate metrics for the current pair of images
        iou, recall, precision, f1, binary_image = print_output(img_file_path, mask_file_path)

        # Accumulate metrics
        iou_total += iou
        recall_total += recall
        precision_total += precision
        f1_total+=f1

        print("Image File Path:", img_file_path)
        print("Mask File Path:", mask_file_path)
        print('iou', iou)
        print('recall', recall)
        print('precision', precision)
        print('f1', f1)

        # Save output using georeferencing from the mask
        if save_output:
            filename = 'result_' + img_file_name
            save_binary_image(binary_image, output_path, filename, mask_file_path)



    # Calculate averages
    num_files = len(img_file_names)
    average_iou = iou_total / num_files
    average_recall = recall_total / num_files
    average_precision = precision_total / num_files
    average_f1= f1_total/num_files

    print('Average IoU:', average_iou)
    print('Average Recall:', average_recall)
    print('Average Precision:', average_precision)
    print('Average F1:', average_f1)

    # Format the results into a string
    results_content = f"""Results Summary:
    ---------------------
    Average IoU: {average_iou}
    Average Recall: {average_recall}
    Average Precision: {average_precision}
    Average F1:{average_f1}
    """

    # Save to a text file
    results_file_path = Path(IMAGERY_DIR / month / "Unet_results_summary.txt")

    with open(results_file_path, "w") as results_file:
        results_file.write(results_content)

    print(f"Results saved to {results_file_path}")
# %% 
### ================
# Functions required in the inference notebooks 
### ================
# %%
# Import libraries
import rasterio as rio
import matplotlib.pyplot as plt
from rasterio.plot import reshape_as_image
from torch.masked import masked_tensor
import torch.nn.functional as F
import numpy as np
from rasterio.transform import from_origin
from pathlib import Path

# %%
# Print test data, predicted masks and groundtruth masks

def show_image_mask (image1,image2,image3):
    # Create a figure and axes objects with 1 row and 3 columns
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))

    # Plot the first RGB image on the first subplot
    image1 = reshape_as_image(image1)
    axes[0].imshow(image1)
    axes[0].set_title('Image')

    # Plot the second binary image on the second subplot
    image2 = reshape_as_image(image2)
    axes[1].imshow(image2, cmap='gray')
    axes[1].set_title('Mask')

    # Plot the third binary image on the third subplot
    axes[2].imshow(image3, cmap='gray')
    axes[2].set_title('result')

    image2=image2.squeeze()
    
    overlay_img= np.zeros((image2.shape[0], image2.shape[1], 3), dtype=np.uint8)
    print('overlay',overlay_img.shape)
    overlay_img[:,:,0]=image3*255 # put the result on red
    overlay_img[:,:,1]=image2*255 # put the mask on green
    print('min,max',overlay_img.min(),overlay_img.max())

    # Plot the overlay_img
    axes[3].imshow(overlay_img)
    axes[3].set_title('overlay')

    # Hide the axis ticks for all subplots
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    # Display the plot
    plt.show()

# Metrics definitions

def calculate_metrics(binary_image, mask): 
    # Calculate True Positives, False Positives, False Negatives
    true_positives = np.logical_and(binary_image, mask).sum()
    false_positives = np.logical_and(binary_image, np.logical_not(mask)).sum()
    false_negatives = np.logical_and(np.logical_not(binary_image), mask).sum()

    # Calculate Intersection over Union (IoU)
    intersection = true_positives
    union = true_positives + false_positives + false_negatives
    iou = intersection / union if union > 0 else 0.0

    # Calculate Recall
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    
    # Calculate Precision
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    print('fp', false_positives,'fn',false_negatives)

    # Calculate F1
    f1 = 2*(precision*recall)/(precision+recall) if recall > 0 else 0.0
    return iou, recall, precision, f1

# Export predicted binary masks

def save_binary_image(binary_image, output_path, filename, mask_file_path):
    """
    Saves a binary image as a georeferenced raster, taking georeferencing information from the mask file.
    
    Args:
        binary_image (np.array): Binary image array (0 and 1 values).
        output_path (str): Directory to save the output raster.
        filename (str): Name of the output file.
        mask_file_path (str): Path to the mask file used for georeferencing.
    """
    # Convert the binary image array to uint8
    binary_image_uint8 = (binary_image * 255).astype(np.uint8)

    # Ensure the output directory exists
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Construct the full file path
    file_path = output_dir / filename

    # Open the original georeferenced image using Rasterio
    with rio.open(mask_file_path) as mask_dataset:
        if mask_dataset is None:
            raise ValueError(f"Could not open {mask_file_path} for georeferencing information.")

        # Retrieve geotransform and projection from the mask image
        geotransform = mask_dataset.transform
        projection = mask_dataset.crs

    # Get image dimensions from the binary image
    height, width = binary_image.shape

    # Create a new georeferenced raster file
    with rio.open(str(file_path), 'w', driver='GTiff', count=1, dtype='uint8', 
                       width=width, height=height, crs=projection, transform=geotransform, nodata=0) as output_dataset:
        # Write the binary image to the new raster
        output_dataset.write(binary_image_uint8, 1)
        
    print(f"Saved georeferenced binary image: {file_path}")



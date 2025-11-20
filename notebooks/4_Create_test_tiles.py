
# %%
import os
from pathlib import Path
from itertools import product
from math import floor
import numpy as np
import geopandas as gpd
import rasterio
from rasterio import features
from shapely.geometry import box, mapping
from tqdm import tqdm
import fiona

# %%
# Import variables from dataset/config.py

import os
from pathlib import Path
workdir = Path(os.getenv("WORKDIR", Path(__file__).resolve().parents[1]))  ### modules path are referred to the main code folder 
print(workdir)
import sys
sys.path.append(str(workdir))

from notebooks.dataset.config import AREA_FILE, IMAGERY_DIR, ANNOTATIONS_DIR, OVERLAP_TEST, TILE_WIDTH, TILE_HEIGHT

# %%
# ============================================================
# CONFIGURATION for each period
# Select a test image and a corresponding label set
# ============================================================

RASTER_FILE = IMAGERY_DIR / "TEST_04_Apr.tif"
LABEL_LAYER = ANNOTATIONS_DIR / "L1_Test_April.geojson" 
MONTH_SUFFIX = "apr" # set a suffix for the selected image 
MULTISPECTRAL = True  
R_band = 4
G_band = 3
B_band = 2
NO_DATA = 0

# Output dirs (automatically set)

TILES_GEOJSON = IMAGERY_DIR / "TestTiles.geojson"
OUTPUT_RASTER_DIR = IMAGERY_DIR / MONTH_SUFFIX/ "output_raster_tiles"
OUTPUT_MASK_DIR = IMAGERY_DIR / MONTH_SUFFIX/ "output_mask_tiles"
os.makedirs(OUTPUT_RASTER_DIR, exist_ok=True)
os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)
# %%
# ============================================================
# Create tile grid for test area
# ============================================================

def generate_tile_grid(raster_path, tile_width, tile_height, overlap=0):
    with rasterio.open(raster_path) as src:
        image_crs = src.crs
        transform = src.transform
        res_x, res_y = src.res
        image_crs = src.crs
    x_indices = range(0, src.width, floor(tile_width * (1 - overlap)))
    y_indices = range(0, src.height, floor(tile_height * (1 - overlap)))
    grid_coordinates = product(y_indices, x_indices)

        # Function to get bounding box for each tile
    def tile_box_for_coordinates(grid_x, grid_y, tile_width, tile_height, source_img):
        # Get the center of the pixel coordinates
        tile_min = source_img.xy(grid_x, grid_y)  # top-left (center of grid_x, grid_y)
        tile_max = source_img.xy(grid_x + tile_width, grid_y + tile_height)  # bottom-right (center of grid_x + tile_width, grid_y + tile_height)

        # Apply correction by half the pixel resolution to shift to top-left corner
        tile_min_x, tile_max_y = tile_min
        tile_max_x, tile_min_y = tile_max

        # Correct the shift by half the pixel resolution (if the raster resolution is 1 unit per pixel)
        tile_min_x -= 0.5 * res_x  # Shift left by half of pixel width
        tile_max_y += 0.5 * res_y  # Shift up by half of pixel height
        tile_max_x -= 0.5 * res_x  # Shift right by half of pixel width
        tile_min_y += 0.5 * res_y  # Shift down by half of pixel height

        # Create a rectangular bounding box for the tile
        #tile_box = box(tile_min_x, tile_min_y, tile_max_x, tile_max_y)

        # Ensure that the tile size is exactly 256x256 pixels
        # Adjust for any pixel misalignment
        corrected_tile_box = box(tile_min_x, tile_min_y, tile_min_x + tile_width * res_x, tile_min_y + tile_height * res_y)
        
        return corrected_tile_box
    grid_boxes = [tile_box_for_coordinates(grid_xy[0], grid_xy[1], tile_width, tile_height, src) for grid_xy in grid_coordinates]
    gdf = gpd.GeoDataFrame(geometry=grid_boxes, crs=image_crs)
    return gdf
# ============================================================
# Create raster tiles and tile masks of ground truth
# ============================================================

def clip_and_save_tiles(tile_gdf, area_path, raster_path, label_path, output_raster_dir, output_mask_dir):
    area_gdf = gpd.read_file(area_path)
    area_gdf = area_gdf.to_crs(tile_gdf.crs)
    aoi_polygon = area_gdf.unary_union
    
    with rasterio.open(raster_path) as src:
        img_box = box(*src.bounds)
        print(f"Raster bands: {src.count}, CRS: {src.crs}")

        feature_gdf = gpd.read_file(label_path)
        feature_gdf = feature_gdf.to_crs(src.crs)

        for i, row in tqdm(tile_gdf.iterrows(), total=len(tile_gdf)):
            if not img_box.covers(row.geometry):
                continue

            tile_bounds = row.geometry.bounds
            window = src.window(*tile_bounds)
            tile_transform = src.window_transform(window)

            if MULTISPECTRAL:
                tile_data = src.read([R_band, G_band, B_band], window=window)  # Extract R(4), G(3), B(2) from MULTISPECTRAL
            else:
                tile_data = src.read(window=window)

            tile_mask = features.geometry_mask(
                [row.geometry], transform=tile_transform, invert=True, out_shape=tile_data.shape[1:]
            )
            tile_data = np.where(tile_mask, tile_data, NO_DATA)

            # AOI intersection
            tile_intersection = row.geometry.intersection(aoi_polygon)
            if tile_intersection.is_empty:
                continue

            aoi_mask = features.geometry_mask(
                [tile_intersection], transform=tile_transform, invert=True, out_shape=tile_data.shape[1:]
            )
            aoi_mask_3d = np.repeat(aoi_mask[None, :, :], tile_data.shape[0], axis=0)
            final_tile = np.where(aoi_mask_3d, tile_data, NO_DATA)

            # Save to 8uint image
            min_val = np.nanmin(final_tile)
            max_val = np.nanmax(final_tile)
            if max_val > min_val:
                final_tile = ((final_tile - min_val) / (max_val - min_val) * 254) + 1
                final_tile = final_tile.astype(np.uint8)
            else:
                final_tile = np.zeros_like(final_tile, dtype=np.uint8)

            # Save raster files
            out_path = output_raster_dir / f"tile_{i}.tif"
            profile = src.profile
            profile.update(
                driver="GTiff",
                width=final_tile.shape[2],
                height=final_tile.shape[1],
                count=final_tile.shape[0],
                transform=tile_transform,
                dtype=final_tile.dtype,
                nodata=NO_DATA,
            )
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(final_tile)

            # Mask labels
            mask_img = np.zeros((final_tile.shape[1], final_tile.shape[2]), dtype=np.uint8)
            intersecting = feature_gdf[feature_gdf.intersects(row.geometry)]
            for feat in intersecting.geometry:
                feat_mask = features.geometry_mask([mapping(feat)], transform=tile_transform, invert=True, out_shape=mask_img.shape)
                mask_img[feat_mask] = 1

            mask_path = output_mask_dir / f"mask_{i}.tif"
            with rasterio.open(
                mask_path, "w", driver="GTiff", count=1, dtype="uint8", crs=src.crs,
                transform=tile_transform, width=mask_img.shape[1], height=mask_img.shape[0]
            ) as dst:
                dst.write(mask_img, 1)

    print("Raster and mask generated.")
# %%
# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("Generating tiles...")
    tiles_gdf = generate_tile_grid(RASTER_FILE, TILE_WIDTH, TILE_HEIGHT, OVERLAP_TEST)
    tiles_gdf.to_file(TILES_GEOJSON, driver="GeoJSON", index=False)
    clip_and_save_tiles(tiles_gdf, AREA_FILE, RASTER_FILE, LABEL_LAYER, OUTPUT_RASTER_DIR, OUTPUT_MASK_DIR)
    print("Completed")

# %%

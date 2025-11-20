#### =====================
# Class to load the dataset
#### =====================

# %%
from torch.utils.data import Dataset
import rasterio as rio
import geopandas as gpd
from typing import Any, List
from rasterio.features import rasterize
from rasterio.plot import reshape_as_image
from rasterio.windows import from_bounds, transform as w_transform
import numpy as np
from pathlib import Path

# %%
class SingleRasterPalaeochannelDataset(Dataset):
    def __init__(self, 
                 tiles_df: gpd.GeoDataFrame, 
                 source_path: Path, 
                 features_df: gpd.GeoDataFrame,
                 aoi_df: gpd.GeoDataFrame):
        self.tiles_df = tiles_df
        self.source_path = source_path
        self.features_df = features_df
        self.aoi_df = aoi_df
    
    def __len__(self) -> int:
        return len(self.tiles_df)
    
    def __getitem__(self, index: int) -> Any:
        # Get the vectorial tile from the tiles GeoDataFrame.
        tile = self.tiles_df.loc[index]
        
        with rio.open(self.source_path) as source_db:
            # Get the raster window from the source dataset.
            tile_window = from_bounds(*tile.geometry.bounds, transform=source_db.transform)
            tile_raster: np.ndarray = source_db.read(window=tile_window)
            tile_raster[np.isnan(tile_raster)] = 0
            if tile_raster.dtype == np.uint16:
                tile_raster = tile_raster.astype(np.int32)
            
            # Compute the window's Affine transform for features and aoi rasterization
            window_transform = w_transform(tile_window, source_db.transform)
            mask_shape = (tile_raster.shape[1], tile_raster.shape[2])
            mask = rasterize(self.features_df.geometry, out_shape=mask_shape, transform=window_transform)
            aoi_mask = rasterize(self.aoi_df.geometry, out_shape=mask_shape, transform=window_transform)
            
            # Mask out data and features falling outside of aoi.
            tile_raster = tile_raster * aoi_mask
            mask = mask * aoi_mask
            
            item = dict(
                image = reshape_as_image(tile_raster),
                mask = mask,
                # add geometry and geografical informations for plotting.
                tile_geometry = tile.geometry.wkt,
                tile_crs = str(self.tiles_df.crs)
            )
            return item



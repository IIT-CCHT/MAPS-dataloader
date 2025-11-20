# MAPS - Multitemporal Multispectral Dataset for Palaeochannels Segmentation

## Project structure

- **notebooks/** - Main project folder containing all Python submodules:
    ```   
    1_Download_dataset.py   : Downloads the MAPS dataset from the HuggingFace repository
    2_Dataloader.py         : Creates the train and validation sets from Sentinel-2 imagery and corresponding annotations
    3a_Unet_baseline.py     : Compiles and trains the Unet model
    3b_UperNet_baseline.py  : Compiles and trains the UperNet model
    4_Create_test_tiles.py  : Creates the test set
    5a_Unet_inference.py    : Performs inference on the test dataset using the U-Net model
    5b_UPerNet_inference.py : Performs inference on the test dataset using the UperNet model
    ```
- **notebooks/dataset/** - contains the modules imported by the main scripts:
    ```
    config.py      : Defines the main dataset directory and global variables
    dataset.py     : Loads the training and validation sets
    functions.py   : Utility functions used across the project
    models.py      : Models architectures and hyperameters
    ```

## Environment Variables

- **WORKDIR** - Automatically set by the code to the root directory of this repository.

Bofore running the code, set the following variable in `config.py`:
- **BASE_DIR**: Path to the MAPS dataset


## Installation
Follow these steps to set up the environment:

1. **Create and activate a Conda environment**  
The code was developed with Python 3.11.10
```bash
conda create -n <env_name> python=3.11.10 pip
conda activate <env_name>
```

2. **Install dependencies**
Use either `requirements.txt` or `einvironment.yml`. 
Ensure you select the correct PyTorch index for your CUDA version.
```bash
# Example for CUDA 12.8
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```
## Links
- [Hugginface Dataset Repository](https://https://huggingface.co/datasets/CCHT-IIT/Palaeochannels)

## Cite

If you use this dataset or code, please cite:
```
@article{poggi2025maps,
  title={Multitemporal Multispectral Dataset for Palaeochannels Segmentation (MAPS)},
  author={Giulio Poggi, Andaleeb Yaseen, Raveerat Jaturapitpornchai, Sara Ferro, Gregory Sech, Peter Naylor, Maria Cristina Salvi, Sebastiano Vascon, Bertrand Le Saux, Marco Fiorucci, Arianna Traviglia},
  journal={IEEE Access},
  year={2025}
  doi={10.1109/ACCESS.2025.3626678}
  volume={},
  number={},
  pages={}
}
```
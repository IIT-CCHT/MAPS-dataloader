#### ===================================
# Download the MAPS dataset from HuggingFace
#### ===================================

# %%
import os

os.environ["HF_HOME"] = "<PATH_TO_FOLDER>" # Specify cache folder (optional)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1" # Avoid timeout in downloading large files

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="CCHT-IIT/Palaeochannels",
    repo_type="dataset",
    local_dir="<PATH_TO_FOLDER>",  # Set folder location for the dataset
    #use_auth_token="XXXXX",  # Only if special permissions are needed
    resume_download=True,
    local_dir_use_symlinks=False, # Copy large files from cache folder to local_dir folder after download
    max_workers=1  # Force serialisation in case of multiple files
)


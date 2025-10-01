from .config import DOWNLOADED_IMAGES_PATHS
from typing import List
import os 

def ensure_directories():
    os.makedirs(DOWNLOADED_IMAGES_PATHS, exist_ok=True)

def delete_files(filespath:List[str]):
    for file in filespath:
        if os.path.exists(file):
            os.remove(file)

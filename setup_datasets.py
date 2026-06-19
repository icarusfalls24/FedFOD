#!/usr/bin/env python3
import os
import zipfile
import shutil
from pathlib import Path

def extract_and_setup(zip_name, dest_dir_name):
    project_root = Path(__file__).parent
    zip_path = project_root / zip_name
    dest_dir = project_root / "data" / dest_dir_name
    
    if not zip_path.exists():
        print(f"Error: Zip file {zip_name} not found in project root.")
        return False
        
    print(f"[*] Extracting {zip_name} to {dest_dir}...")
    
    # Clean destination folder first
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_dir)
        
    # Check for 'valid' folder and rename to 'val'
    valid_folder = dest_dir / "valid"
    val_folder = dest_dir / "val"
    
    if valid_folder.exists() and not val_folder.exists():
        print(f"[*] Renaming '{valid_folder.name}' to 'val' in {dest_dir_name}...")
        valid_folder.rename(val_folder)
        
    print(f"[+] Successfully set up {dest_dir_name}!")
    return True

if __name__ == "__main__":
    # Setup Airport A
    extract_and_setup("Airport runway debris detection.v5i.yolov8.zip", "airport_A")
    
    # Setup Airport B
    extract_and_setup("Airport runway debris detection.v5i.yolov8.zip", "airport_B")
    
    print("\n[+] All datasets extracted and set up successfully for Airport A (Client 0) and Airport B (Client 1)!")

#!/usr/bin/env python3
"""
Clean up merged LoRA models that contain problematic 'merged_from' parameter in adapter_config.json
"""

import os
import json
import shutil
import glob
import argparse
import logging

def clean_adapter_config(config_path):
    """
    Remove problematic parameters from adapter_config.json
    
    Args:
        config_path: Path to adapter_config.json file
        
    Returns:
        bool: True if file was cleaned, False if no changes needed
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Parameters that should not be in LoraConfig
        problematic_params = [
            'merged_from', 'merge_type', 'failed_paths', 
            'successful_count', 'total_paths', 'merge_timestamp',
            'merge_description'
        ]
        
        original_config = config.copy()
        cleaned = False
        
        for param in problematic_params:
            if param in config:
                config.pop(param)
                cleaned = True
                logging.info(f"Removed parameter '{param}' from {config_path}")
        
        if cleaned:
            # Create backup
            backup_path = config_path + '.backup'
            with open(backup_path, 'w') as f:
                json.dump(original_config, f, indent=2)
            logging.info(f"Created backup at {backup_path}")
            
            # Save cleaned config
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            logging.info(f"Cleaned {config_path}")
            
        return cleaned
        
    except Exception as e:
        logging.error(f"Error processing {config_path}: {e}")
        return False

def find_merged_model_dirs(base_path):
    """
    Find directories that contain merged LoRA models
    
    Args:
        base_path: Base path to search for merged models
        
    Returns:
        list: List of directories containing merged models
    """
    merged_dirs = []
    
    # Look for directories with "merged" in the name
    pattern = os.path.join(base_path, "**/merged*")
    for path in glob.glob(pattern, recursive=True):
        if os.path.isdir(path):
            config_file = os.path.join(path, "adapter_config.json")
            if os.path.exists(config_file):
                merged_dirs.append(path)
    
    # Also look for specific pattern used in the code
    pattern = os.path.join(base_path, "**/merged_lora_model*")
    for path in glob.glob(pattern, recursive=True):
        if os.path.isdir(path):
            config_file = os.path.join(path, "adapter_config.json")
            if os.path.exists(config_file):
                merged_dirs.append(path)
    
    return list(set(merged_dirs))  # Remove duplicates

def main():
    parser = argparse.ArgumentParser(description='Clean merged LoRA models with problematic configs')
    parser.add_argument('--base_path', type=str, default='/datanfs2/chenrongyi/models/loraadaptors/',
                      help='Base path to search for merged models')
    parser.add_argument('--remove_dirs', action='store_true',
                      help='Remove entire merged model directories instead of cleaning configs')
    parser.add_argument('--dry_run', action='store_true',
                      help='Show what would be done without actually doing it')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logging.info(f"Searching for merged LoRA models in: {args.base_path}")
    
    merged_dirs = find_merged_model_dirs(args.base_path)
    
    if not merged_dirs:
        logging.info("No merged LoRA model directories found")
        return
    
    logging.info(f"Found {len(merged_dirs)} merged model directories:")
    for dir_path in merged_dirs:
        logging.info(f"  {dir_path}")
    
    if args.dry_run:
        logging.info("DRY RUN MODE - No changes will be made")
    
    for dir_path in merged_dirs:
        config_path = os.path.join(dir_path, "adapter_config.json")
        
        if args.remove_dirs:
            if args.dry_run:
                logging.info(f"Would remove directory: {dir_path}")
            else:
                try:
                    shutil.rmtree(dir_path)
                    logging.info(f"Removed directory: {dir_path}")
                except Exception as e:
                    logging.error(f"Error removing {dir_path}: {e}")
        else:
            if args.dry_run:
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    
                    problematic_params = [
                        'merged_from', 'merge_type', 'failed_paths', 
                        'successful_count', 'total_paths', 'merge_timestamp',
                        'merge_description'
                    ]
                    
                    found_params = [p for p in problematic_params if p in config]
                    if found_params:
                        logging.info(f"Would clean {config_path}, removing: {found_params}")
                    else:
                        logging.info(f"Config {config_path} is already clean")
                        
                except Exception as e:
                    logging.error(f"Error checking {config_path}: {e}")
            else:
                clean_adapter_config(config_path)

if __name__ == "__main__":
    main() 
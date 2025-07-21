import os
import torch
from safetensors.torch import load_file, save_file
import json
import glob
import argparse
import time

def find_lora_files(directory):
    """
    Find all LoRA weight files (safetensors or pt/bin) in a directory.
    Returns empty list if no files found or if directory doesn't exist.
    """
    if not os.path.exists(directory):
        return []
        
    try:
        safetensors_files = glob.glob(os.path.join(directory, "*.safetensors"))
        pt_files = glob.glob(os.path.join(directory, "*.pt"))
        bin_files = glob.glob(os.path.join(directory, "*.bin"))
        
        all_files = safetensors_files + pt_files + bin_files
        
        if len(safetensors_files) > 1:
            print(f"Warning: Multiple safetensors files found in {directory}, using the first one")
            return [safetensors_files[0]]
        elif len(safetensors_files) == 1:
            return safetensors_files
        elif all_files:
            return all_files
        else:
            return []
    except Exception as e:
        print(f"Error accessing directory {directory}: {str(e)}")
        return []

def load_lora_weights(path, device):
    """
    Load LoRA weights from either safetensors or pt/bin file.
    """
    if path.endswith('.safetensors'):
        return load_file(path, device=device)
    else:
        return torch.load(path, map_location=device)

def save_lora_weights(weights, output_path, use_safetensors=True):
    """
    Save LoRA weights in either safetensors or pt format.
    """
    os.makedirs(output_path, exist_ok=True)
    if use_safetensors:
        save_file(weights, os.path.join(output_path, "adapter_model.safetensors"))
    else:
        torch.save(weights, os.path.join(output_path, "adapter_model.bin"))

def uniform_lora_merging(lora_paths: list[str], output_path: str, device: str = "cpu", use_safetensors: bool = True):
    """
    Uniformly merges the parameters of multiple LoRA adapters.所有lora_paths必须都被加载，否则报错
    Args:
        lora_paths: A list of directories containing LoRA adapter weights.
        output_path: The directory path to save the merged LoRA adapter weights.
        device: The device to load the LoRA weights onto ('cpu' or 'cuda').
        use_safetensors: Whether to save the merged weights in safetensors format.
        
    Returns:
        int: Number of successfully merged adapters
    """
    if not lora_paths:
        print("No LoRA adapter paths provided.")
        return 0

    # Load all adapters with better error handling
    adapters = []
    failed_paths = []
    successful_paths = []
    
    for path in lora_paths:
        try:
            lora_files = find_lora_files(path)
            if not lora_files:
                print(f"Warning: No LoRA weight files found in {path}")
                failed_paths.append(path)
                continue
            # Use the first found file in each directory
            adapter = load_lora_weights(lora_files[0], device)
            adapters.append(adapter)
            successful_paths.append(path)
            print(f"Successfully loaded LoRA adapter from {path}")
        except Exception as e:
            print(f"Warning: Failed to load LoRA adapter from {path}: {str(e)}")
            failed_paths.append(path)
            continue

    # Log loading results
    total_paths = len(lora_paths)
    successful_count = len(successful_paths)
    failed_count = len(failed_paths)
    
    print(f"LoRA loading summary: {successful_count}/{total_paths} adapters loaded successfully")
    if failed_paths:
        print(f"Failed to load adapters from: {failed_paths}")

    if not adapters:
        print("No valid LoRA adapters found.")
        return 0

    # Add a unique identifier to output path to avoid concurrent access issues
    import threading
    thread_id = threading.get_ident()
    timestamp = int(time.time() * 1000) % 10000  # Last 4 digits of timestamp
    unique_suffix = f"_{thread_id}_{timestamp}"
    temp_output_path = f"{output_path}{unique_suffix}"
    
    try:
        num_adapters = len(adapters)
        merged_lora = {}

        # Iterate through the keys of the first adapter to find LoRA parameters
        for key in adapters[0]:
            if 'lora_A.weight' in key or 'lora_B.weight' in key:
                # Collect all tensors for the current LoRA parameter across all adapters
                all_tensors = [adapter[key] for adapter in adapters]

                # Stack the tensors and take the mean along the first dimension (number of adapters)
                merged_tensor = torch.stack(all_tensors).mean(dim=0)
                merged_lora[key] = merged_tensor
            else:
                # For non-LoRA parameters, just take the weights from the first adapter
                # merged_lora[key] = adapters[0][key]
                print(f"Warning: Non-LoRA parameter {key} found in {successful_paths[0]}, skipping")
                continue

        # Create temporary output directory
        os.makedirs(temp_output_path, exist_ok=True)

        # Save the merged LoRA weights to temporary location
        save_lora_weights(merged_lora, temp_output_path, use_safetensors)

        # Copy adapter_config.json from the first LoRA path
        first_lora_path = successful_paths[0]
        config_path = os.path.join(first_lora_path, "adapter_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Create a separate metadata file for merge information to avoid conflicts with LoRA config
            merge_metadata = {
                'merged_from': successful_paths,
                'merge_type': 'uniform',
                'failed_paths': failed_paths,
                'successful_count': successful_count,
                'total_paths': total_paths,
                'merge_timestamp': time.time(),
                'merge_description': f'Uniformly merged {successful_count} LoRA adapters'
            }
            
            # Save the original config without merge-specific parameters
            with open(os.path.join(temp_output_path, "adapter_config.json"), 'w') as f:
                json.dump(config, f, indent=2)
            
            # Save merge metadata separately
            with open(os.path.join(temp_output_path, "merge_metadata.json"), 'w') as f:
                json.dump(merge_metadata, f, indent=2)
                
            print(f"Saved merge metadata to merge_metadata.json")
        else:
            print(f"Warning: adapter_config.json not found in {first_lora_path}")

        # Atomically move to final location to avoid concurrent access issues
        if os.path.exists(output_path):
            import shutil
            shutil.rmtree(output_path)
        os.rename(temp_output_path, output_path)
        
        # 现在要求必须完全加载
        assert num_adapters==len(lora_paths)
        print(f"Uniformly merged {num_adapters} LoRA adapters and saved to {output_path}")
        return num_adapters
        
    except Exception as e:
        # Clean up temporary directory on error
        if os.path.exists(temp_output_path):
            import shutil
            shutil.rmtree(temp_output_path)
        print(f"Error during merging: {str(e)}")
        return 0

def parse_args():
    parser = argparse.ArgumentParser(description='Merge multiple LoRA adapters uniformly')
    parser.add_argument('--lora_paths', nargs='+', required=True,
                      help='List of paths to LoRA adapter directories')
    parser.add_argument('--output_path', required=True,
                      help='Output directory path for merged LoRA adapter')
    parser.add_argument('--gpu_id', type=int, default=None,
                      help='GPU ID to use (if not specified, will use CPU)')
    parser.add_argument('--use_safetensors', action='store_true', default=True,
                      help='Whether to save the merged weights in safetensors format')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # lora_paths = [
    #     "/datanfs2/chenrongyi/models/loraadaptors/codegemma-7b-it/udg_docstringSIFT_pandas/",
    #     "/datanfs2/chenrongyi/models/loraadaptors/codegemma-7b-it/udg_docstringSIFT_matplotlib/",
    # ]
    # output_path = "/datanfs2/chenrongyi/models/loraadaptors/codegemma-7b-it/udg_docstringSIFT_merged/"

    # Set device based on GPU ID
    if args.gpu_id is not None:
        if not torch.cuda.is_available():
            print("CUDA is not available. Falling back to CPU.")
            device = "cpu"
        else:
            if args.gpu_id >= torch.cuda.device_count():
                print(f"GPU {args.gpu_id} is not available. Available GPUs: {torch.cuda.device_count()}")
                print("Falling back to CPU.")
                device = "cpu"
            else:
                device = f"cuda:{args.gpu_id}"
                torch.cuda.set_device(args.gpu_id)
                print(f"Using GPU {args.gpu_id}")
    else:
        device = "cpu"
        print("Using CPU")

    uniform_lora_merging(args.lora_paths, args.output_path, device=device, use_safetensors=args.use_safetensors)
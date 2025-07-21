import yaml
import os
from typing import Dict, Any
LORA_CONFIG_PATH = 'benchmark/config/code/config_lora.yaml'
MODEL_SOURCE = "HUGGINGFACE"
def load_config(config_path: str = LORA_CONFIG_PATH) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str, optional): Path to the configuration file.
            If None, uses the default path.
    
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    if config_path is None:
        # Use default path if not specified
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "config", "code", "config_lora.yaml"
        )
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration: {str(e)}")

def save_config(config: Dict[str, Any], config_path: str = None) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary to save
        config_path (str, optional): Path to save the configuration file.
            If None, uses the default path.
    """
    if config_path is None:
        # Use default path if not specified
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "config", "code", "config_lora.yaml"
        )
    
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    except Exception as e:
        raise IOError(f"Error saving configuration: {str(e)}")

if __name__ == "__main__":
    # Test loading the configuration
    try:
        config = load_config()
        print("Configuration loaded successfully:")
        print(f"Model name: {config['model_name']}")
        print(f"Dataset: {config['dataset']}")
        print(f"Task: {config['task']}")
    except Exception as e:
        print(f"Error: {str(e)}") 
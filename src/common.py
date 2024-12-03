import sys, yaml
from pathlib import Path

def load_config():
    """Load configuration parameters from params.yaml."""
    script_dir = Path(__file__).resolve().parent
    config_path = script_dir / 'params.yaml'
    
    if not config_path.is_file():
        print(f"Configuration file '{config_path}' does not exist.")
        sys.exit(1)
        
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file '{config_path}': {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error reading '{config_path}': {e}")
        sys.exit(1)


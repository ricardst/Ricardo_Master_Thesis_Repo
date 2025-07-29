# src/config_loader.py

import yaml
import os
import logging

# Define a default config structure (optional, for validation/defaults)
DEFAULT_CONFIG = {
    'seed_number': 42,
    'base_log_filename': 'pipeline.log',
    'processed_data_input_dir': 'processed_subjects',
    'intermediate_feature_dir': 'features',
    'results_dir': 'results',
    # Add other defaults if desired
}

def load_config(config_path='config.yaml'):
    """Loads configuration from a YAML file."""
    if not os.path.exists(config_path):
        logging.error(f"Configuration file not found at: {config_path}")
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Configuration loaded successfully from {config_path}")

        # Optional: Merge with defaults or validate structure here
        # Example: Overwrite defaults with loaded config
        # merged_config = DEFAULT_CONFIG.copy()
        # merged_config.update(config)
        # return merged_config
        return config

    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML configuration file: {e}", exc_info=True)
        raise yaml.YAMLError(f"Error parsing YAML configuration file: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred loading config: {e}", exc_info=True)
        raise e

if __name__ == '__main__':
    # Example usage if run directly
    logging.basicConfig(level=logging.INFO) # Basic logging for direct run
    try:
        # Assumes config.yaml is in the parent directory relative to src/
        script_dir = os.path.dirname(__file__)
        config_file_path = os.path.join(script_dir, '..', 'config.yaml')
        cfg = load_config(config_file_path)
        print("Config loaded:")
        # Pretty print the config
        import json
        print(json.dumps(cfg, indent=2))
    except Exception as e:
        print(f"Failed to load config: {e}")
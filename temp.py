import torch
import look2hear.models  # adjust import according to your project structure
import yaml
import argparse

def load_config(conf_path):
    with open(conf_path, "r") as f:
        return yaml.safe_load(f)

def print_keys(state_dict, name):
    print(f"\n{name} keys ({len(state_dict)} keys):")
    for key in state_dict.keys():
        print(f"  {key}")

def main(conf_path):
    # Load config
    config = load_config(conf_path)

    # Instantiate model class from config
    audionet_name = config["audionet"]["audionet_name"]
    model_class = getattr(look2hear.models, audionet_name)

    # Create model instance (you might need to pass args from config here)
    model = model_class(config)

    # Load checkpoint
    checkpoint_path = config.get("checkpoint_path", config.get("checkpoint", None))
    if checkpoint_path is None:
        print("Checkpoint path not found in config.")
        return

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Get state_dict inside checkpoint if present
    if 'state_dict' in checkpoint:
        checkpoint_state_dict = checkpoint['state_dict']
    else:
        checkpoint_state_dict = checkpoint

    # Print keys in checkpoint
    print_keys(checkpoint_state_dict, "Checkpoint")

    # Print keys in model's current state_dict
    model_state_dict = model.state_dict()
    print_keys(model_state_dict, "Model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check checkpoint and model keys.")
    parser.add_argument("--conf_dir", type=str, required=True, help="Path to config YAML file.")
    args = parser.parse_args()
    main(args.conf_dir)

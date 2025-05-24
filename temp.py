import yaml

# Define configuration data
config_data = {
    "batch_size": 32,
    "epochs": 100,
    "lr": 0.0002,
    "noise_dim": 100,
    "text_embed_dim": 384,
    "text_latent_dim": 128,
    "num_classes": 102  # Oxford-102 flowers
}

# Save to config.yaml
config_file = "config.yaml"
with open(config_file, "w") as file:
    yaml.dump(config_data, file, default_flow_style=False)

print(f"✅ Configuration file '{config_file}' created successfully!")

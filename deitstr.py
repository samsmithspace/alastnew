import torch
from transformers import DeiTModel, DeiTConfig


def print_model_structure():
    # Load the DeiT model
    model = DeiTModel.from_pretrained("facebook/deit-base-patch16-224")

    # Print the model structure
    print(model)

    # Get number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Print key configuration details
    config = model.config
    print("\nModel Configuration:")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Number of attention heads: {config.num_attention_heads}")
    print(f"Number of hidden layers: {config.num_hidden_layers}")
    print(f"Patch size: {config.patch_size}")
    print(f"Image size: {config.image_size}")


if __name__ == "__main__":
    print_model_structure()
'''
Contains various utility function for model training and saving'''

import torch
from pathlib import Path

def save_model(model: torch.nn.Module, target_dir: str,
               model_name: str):
    '''Saves a PyTorch model to a target directory.
    Args:
        model: model to save.
        target_dir: directory for saving the model.
        model_name: filename to be given to the model with extension.'''

    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    if not model_name.endswith(".pth"):
        return "Not .pth extension. change extension"
    else:
        model_save_path = target_dir_path / model_name

        print("Saving model ...")
        torch.save(obj=model.state_dict(), f= model_save_path)

import os
import pickle

def export_model(model, filepath):
    """
    Exports a trained machine learning model (scikit-learn, diffprivlib, or PyTorch)
    to the specified filepath. Creates directories if they do not exist.
    """
    # Ensure target directory exists
    dir_name = os.path.dirname(filepath)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
        
    is_pytorch = False
    try:
        import torch
        if isinstance(model, torch.nn.Module):
            is_pytorch = True
    except ImportError:
        pass
        
    if is_pytorch:
        import torch
        # Save the full model object
        torch.save(model, filepath)
        print(f"PyTorch model successfully exported to {filepath}")
    else:
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model successfully exported to {filepath}")

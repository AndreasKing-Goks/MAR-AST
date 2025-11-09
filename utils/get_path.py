from pathlib import Path
import re
import datetime
import uuid

'''
Helper function for getting the data path
'''

def get_ship_route_path(root, route_filename):
    return str(root / "data" / "route" / route_filename)

def get_ship_route_path_for_training(root, route_filename=None, pattern="*.txt"):
    """
    If route_filename is None -> return the folder Path.
    If route_filename is "*"... use pattern to list files.
    Otherwise return the full path to a single file.
    """
    base = Path(root) / "data" / "route" / "for_training"
    if route_filename is None:
        return base
    if route_filename == "*":
        # list all matching files (natural sort by number in name)
        files = list(base.glob(pattern))
        def nkey(p: Path):
            parts = re.findall(r"\d+|\D+", p.stem)
            return [int(x) if x.isdigit() else x.lower() for x in parts]
        return sorted(files, key=nkey)
    return base / route_filename

def get_map_path(root, map_filename):
    return str(root / "data" / "map" / map_filename)

def get_saved_model_path(root, saved_model_filename):
    return str(root / "saved_model" / saved_model_filename)

def get_trained_model_path(root, 
                           model_name :str):
    model_path = str(root / "trained_model" / model_name / "model")
    log_path   = str(root / "trained_model" / model_name / "log")
    
    return model_path, log_path

def get_trained_model_and_log_path(root: Path, model_name: str, unique: bool = True):
    """
    Generate model and log paths with unique timestamp (and short UUID) suffix.
    
    Example:
        model_path, log_path = get_trained_model_path(ROOT, "AST-train")
        
        # Returns something like:
        # model_path = ".../trained_model/AST-train_2025-11-09_18-25-03_ab12/model"
        # log_path   = ".../trained_model/AST-train_2025-11-09_18-25-03_ab12/log"
        # tb_path   = ".../trained_model/AST-train_2025-11-09_18-25-03_ab12/tb"
    """
    # Base directory
    base_dir = Path(root) / "trained_model"

    # Create unique suffix (timestamp + short UUID)
    if unique:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        short_id = uuid.uuid4().hex[:4]  # short unique string
        model_name_unique = f"{model_name}_{timestamp}_{short_id}"
    else:
        model_name_unique = model_name

    # Full paths
    model_path = str(base_dir / model_name_unique / "model")
    log_path   = str(base_dir / model_name_unique / "log")
    tb_path   = str(base_dir / model_name_unique / "tb")

    return model_path, log_path, tb_path
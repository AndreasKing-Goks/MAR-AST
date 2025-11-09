from pathlib import Path
import re

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
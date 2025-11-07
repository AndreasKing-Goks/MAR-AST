'''
Helper function for getting the data path
'''

def get_ship_route_path(root, route_filename):
    return str(root / "data" / "route" / route_filename)

def get_map_path(root, map_filename):
    return str(root / "data" / "map" / map_filename)

def get_saved_model_path(root, saved_model_filename):
    return str(root / "saved_model" / saved_model_filename)

def get_trained_model_path(root, 
                           model_name :str):
    model_path = str(root / "trained_model" / model_name / "model")
    log_path   = str(root / "trained_model" / model_name / "log")
    
    return model_path, log_path
'''
Helper function for getting the data path
'''

def get_ship_route_path(root, route_filename):
    return str(root / "data" / "route" / route_filename)

def get_map_path(root, map_filename):
    return str(root / "data" / "map" / map_filename)
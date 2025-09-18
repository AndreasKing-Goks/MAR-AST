import numpy as np

def get_distance(pos1, pos2):
    '''
    Compute the euclidean distance between two points
    '''
    # Unpack position
    n_pos1, e_pos1 = pos1
    n_pos2, e_pos2 = pos2
    
    # Compute the euclidean distance
    eucliden_dist = np.sqrt((n_pos2 - n_pos1)**2 + (e_pos2 - e_pos1)**2)
    
    return eucliden_dist

def get_distance_and_encounter_type(pos_sut, heading_sut, pos_os, heading_os):
    '''
    Computing the euclidean distance between two ships.
    And get the encounter types.
    - head-on     : the absolute value of the encounter angle is < 15 degrees
    - overtaking  : the absolute value of the encounter angle is > 165 degrees
    - crossing    : the absolute value of the encounter angle is between 15 and 165 degrees
    '''
    dx = pos_os[0] - pos_sut[0]
    dy = pos_os[1] - pos_sut[1]
    distance = np.sqrt(dx**2 + dy**2)

    phi = np.arctan2(dy, dx)
    beta = phi - heading_sut
    beta = (beta + np.pi) % (2 * np.pi) - np.pi  # normalize to [-pi, pi]

    # Encounter classification
    if abs(beta) < np.deg2rad(15):
        encounter_type = "head-on"
    elif abs(beta) > np.deg2rad(165):
        encounter_type = "overtaking"
    else:
        encounter_type = "crossing"

    return distance, encounter_type

def get_distance_and_true_encounter_type(pos_sut, heading_sut, pos_os, heading_os):
    '''
    Computing the euclidean distance between two ships.
    And get the encounter types.
    - head-on     : the absolute value of the encounter angle is < 15 degrees
    - overtaking  : the absolute value of the encounter angle is > 165 degrees
    - crossing    : the absolute value of the encounter angle is between 15 and 165 degrees
    '''
    dx = pos_os[0] - pos_sut[0]
    dy = pos_os[1] - pos_sut[1]
    distance = np.sqrt(dx**2 + dy**2)

    relative_bearing = (heading_os - heading_sut + np.pi) % (2 * np.pi) - np.pi

    if abs(relative_bearing) < np.deg2rad(15):
        encounter_type = "overtaking"
    elif abs(relative_bearing) > np.deg2rad(165):
        encounter_type = "head-on"
    else:
        encounter_type = "crossing"
    
    return distance, encounter_type

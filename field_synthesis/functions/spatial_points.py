import numpy as np
from scipy.spatial import cKDTree

def spatial_points(points_k, points_m, min_distance):
    tree = cKDTree(points_m)
    result = []

    __, idx = tree.query(points_k, k=5)

    for i in range(len(points_k)):
        current_neighbor_indices = idx[i]
        current_neighbor_points = points_m[current_neighbor_indices]
        
        valid_indices = []
        for __, p_idx in enumerate(current_neighbor_indices):
            p_coord = points_m[p_idx]
            
            diffs = current_neighbor_points - p_coord
            dist_to_others = np.linalg.norm(diffs, axis=1)
            
            if np.all(dist_to_others <= min_distance):
                valid_indices.append(p_idx)
        
        result.append(valid_indices)

    return result
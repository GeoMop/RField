import numpy as np

def assign_source_fields(anchor_points, num_source_fields):

    if len(anchor_points) == 0 or num_source_fields <= 0:
        return np.array([], dtype=int)
    
    rng = np.random.default_rng()
    field_indices = rng.integers(0, num_source_fields, size=len(anchor_points))
    
    return field_indices
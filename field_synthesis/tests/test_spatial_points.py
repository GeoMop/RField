import numpy as np
from field_synthesis.functions.spatial_points import spatial_points


def test_spatial_points_filtering():
    # Vytvoříme body K (cíle) a M (zdroje)
    points_k = np.array([[10, 10]]) # Jeden bod uprostřed
    
    # M body: vytvoříme shluk 5 bodů velmi blízko sebe
    points_m = np.array([
        [10.1, 10.1],
        [10.2, 10.2],
        [10.1, 10.2],
        [10.2, 10.1],
        [10.15, 10.15],
        [50.0, 50.0] # Jeden bod hodně daleko
    ])
    
    # Pokud nastavíme min_distance dostatečně velkou, 
    # mělo by nám to vrátit indexy těch 5 blízkých bodů.
    # Vzdálenost mezi nimi je cca 0.14, takže limit 1.0 je v pohodě.
    result = spatial_points(points_k, points_m, min_distance=1.0)
    
    assert len(result) == 1
    assert len(result[0]) == 5 # Našlo to všech 5 v clusteru
    assert 5 not in result[0] # Bod na [50, 50] tam nesmí být

def test_spatial_points_too_far():
    points_k = np.array([[0, 0]])
    points_m = np.array([[10, 10], [11, 11], [12, 12], [13, 13], [14, 14]])
    
    # Pokud je limit 0.1, ale body jsou od sebe 1.4, nikdo neprojde filtrem
    result = spatial_points(points_k, points_m, min_distance=0.1)
    assert result == [[]]
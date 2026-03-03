import numpy as np
from field_synthesis.functions.generate_points import generate_points

def test_generate_points_shape():
    """Testuje, zda funkce vrací 2D pole se správným tvarem."""
    points = generate_points(count_points=20, min_distance=2.0, area_size=50.0)
    
    assert isinstance(points, np.ndarray)
    assert points.shape[1] == 2  # Musí mít x a y souřadnice
    assert len(points) <= 20     # PoissonDisk nemusí vždy vygenerovat maximum bodů

def test_generate_points_invalid_inputs():
    """Testuje pojistky proti nulovým a záporným hodnotám."""
    # Očekáváme prázdné 2D pole: tvar (0, 2)
    empty_result = np.zeros((0, 2))
    
    result_zero_points = generate_points(0, 5.0, 50.0)
    result_zero_area = generate_points(10, 5.0, 0.0)
    
    np.testing.assert_array_equal(result_zero_points, empty_result)
    np.testing.assert_array_equal(result_zero_area, empty_result)

def test_generate_points_bounds():
    """Testuje, zda všechny vygenerované body leží uvnitř zadané oblasti."""
    area = 50.0
    points = generate_points(10, 2.0, area)
    
    # Žádná souřadnice nesmí být menší než 0 nebo větší než area_size
    assert np.all(points >= 0.0)
    assert np.all(points <= area)
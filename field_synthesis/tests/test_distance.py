import pytest
import math
from field_synthesis.functions.calc_distance import calc_distance

def test_estimate_distance_standard():
    """Testuje standardní výpočet se 40% volným prostorem v 2D."""
    # count_points=50, area_size=100.0, prostor=2, free_space_ratio=0.4
    # Výpočet: (100^2 * 0.6) / 50 = 120 -> sqrt(120) = 10.95445
    result = calc_distance(50, 100.0, 2, 0.4) 
    assert math.isclose(result, 10.95445, rel_tol=1e-4)

@pytest.mark.parametrize("points, area, prostor, free_space, expected", [
    (10, 50.0, 2, 0.5, 11.1803), # (2500 * 0.5) / 10 = 125 -> sqrt(125) = 11.1803
    (100, 10.0, 2, 0.1, 0.9486)  # (100 * 0.9) / 100 = 0.9 -> sqrt(0.9) = 0.9486
])
def test_estimate_distance_parametrized(points, area, prostor, free_space, expected):
    """Testuje různé poměry a velikosti v n-rozměrném prostoru."""
    result = calc_distance(points, area, prostor, free_space)
    assert math.isclose(result, expected, rel_tol=1e-4)

def test_estimate_distance_invalid_inputs():
    """Testuje chování při neplatných vstupech."""
    # Přidán parametr '2' pro prostor, aby sedělo pořadí argumentů
    assert calc_distance(0, 100.0, 2, 0.4) == 0.0
    assert calc_distance(10, 0.0, 2, 0.4) == 0.0
    assert calc_distance(-5, 100.0, 2, 0.4) == 0.0
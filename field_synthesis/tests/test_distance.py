import pytest
import math
from field_synthesis.functions.calc_distance import calc_distance

def test_estimate_distance_standard():
    """Testuje standardní výpočet se 40% volným prostorem."""
    # 50 bodů, plocha 100x100 (10000), 60% obsazeno = 6000. 6000/50 = 120. Odmocnina z 120 je cca 10.954
    result = calc_distance(50, 100.0, 0.4)
    assert math.isclose(result, 10.95445, rel_tol=1e-4)

@pytest.mark.parametrize("points, area, free_space, expected", [
    (10, 50.0, 0.5, 11.1803),
    (100, 10.0, 0.1, 0.9486)
])
def test_estimate_distance_parametrized(points, area, free_space, expected):
    """Testuje různé poměry a velikosti."""
    result = calc_distance(points, area, free_space)
    assert math.isclose(result, expected, rel_tol=1e-4)

def test_estimate_distance_invalid_inputs():
    """Testuje chování při neplatných vstupech (nula nebo záporná čísla)."""
    assert calc_distance(0, 100.0, 0.4) == 0.0
    assert calc_distance(10, 0.0, 0.4) == 0.0
    assert calc_distance(-5, 100.0, 0.4) == 0.0
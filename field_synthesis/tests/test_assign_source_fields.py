import pytest
import numpy as np
from field_synthesis.functions.assign_source_fields import assign_source_fields

def test_assign_source_fields_valid():
    """Testuje správné hranice náhodných indexů a délku výstupu."""
    anchor_points = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
    num_fields = 3
    
    result = assign_source_fields(anchor_points, num_fields)
    
    assert len(result) == len(anchor_points)
    assert np.issubdtype(result.dtype, np.integer)
    # Indexy musí být 0, 1 nebo 2 (menší než num_fields)
    assert np.all(result >= 0)
    assert np.all(result < num_fields)

@pytest.mark.parametrize("anchors, fields", [
    (np.array([]), 5),        # Prázdné pole bodů
    (np.array([[1, 1]]), 0),  # Nula zdrojových polí
    (np.array([[1, 1]]), -1)  # Záporný počet polí
])
def test_assign_source_fields_invalid(anchors, fields):
    """Testuje okrajové případy a neplatné vstupy."""
    result = assign_source_fields(anchors, fields)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 0
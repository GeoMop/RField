import pytest
import numpy as np
from field_synthesis import FieldSynthesis

@pytest.fixture
def fs_instance():
    """Vytvoří základní testovací instanci třídy."""
    return FieldSynthesis(
        area_size=100.0, 
        count_points=50, 
        num_source=5, 
        dimension=2, 
        seed=42
    )

## --- Testy Inicializace a Vlastností ---

def test_initialization(fs_instance):
    """Ověří, že parametry jsou správně nastaveny."""
    assert fs_instance.area_size == 100.0
    assert fs_instance.count_points == 50
    assert fs_instance.num_source == 5

def test_cached_distance(fs_instance):
    """Ověří výpočet minimální vzdálenosti."""
    dist = fs_instance.min_distance
    assert isinstance(dist, float)
    assert dist > 0
    # Ověření, že se hodnota nemění (cache)
    assert fs_instance.min_distance == dist

## --- Testy Generování Bodů ---

def test_generate_points(fs_instance):
    """Ověří tvar a rozsah vygenerovaných kotevních bodů."""
    points = fs_instance.anchor_points
    assert isinstance(points, np.ndarray)
    # Tvar by měl odpovídat (count_points, dimension)
    assert points.shape == (50, 2)
    # Body musí být v mezích oblasti
    assert np.all(points >= 0)
    assert np.all(points <= 100.0)

def test_assign_source_fields(fs_instance):
    """Ověří přiřazení indexů polí k bodům."""
    indices = fs_instance.fields_indices
    assert len(indices) == 50
    # Indexy musí být v rozsahu [0, num_source - 1]
    assert np.all(indices >= 0)
    assert np.all(indices < 5)
    assert indices.dtype.kind in 'iu' # integer nebo unsigned integer

## --- Testy Prostorové Logiky ---

def test_spatial_points_return_type(fs_instance):
    """Ověří, že vyhledávání sousedů vrací seznam indexů."""
    target = np.array([[50.0, 50.0]])
    neighbors = fs_instance.spatial_points(target)
    assert isinstance(neighbors, list)
    assert len(neighbors) == 1
    assert isinstance(neighbors[0], np.ndarray)

def test_mix_fields_logic(fs_instance):
    """Ověří, že míchání polí vrací správný počet výsledků."""
    target_points = np.array([[10, 10], [50, 50], [90, 90]])
    results = fs_instance.mix_fields(target_points)
    
    assert isinstance(results, np.ndarray)
    assert len(results) == 3
    # Výsledek by měl být buď číslo (průměr) nebo NaN, pokud v okolí nic není
    assert results.dtype == np.float64

## --- Testy Robustnosti ---

def test_reproducibility():
    """Ověří, že stejný seed generuje identické výsledky."""
    fs1 = FieldSynthesis(100, 20, 3, seed=10)
    fs2 = FieldSynthesis(100, 20, 3, seed=10)
    
    np.testing.assert_array_equal(fs1.anchor_points, fs2.anchor_points)
    np.testing.assert_array_equal(fs1.fields_indices, fs2.fields_indices)

def test_empty_area_handling():
    """Ověří chování při nulovém počtu bodů."""
    fs = FieldSynthesis(area_size=100, count_points=0, num_source=2)
    assert fs.min_distance == 0.0
    assert fs.anchor_points.shape == (0, 2)
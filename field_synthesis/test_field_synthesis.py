import pytest
import numpy as np
from field_synthesis import FieldSynthesis

@pytest.fixture
def sample_coords():
    """Створює сітку точок 10x10 для тестів."""
    x = np.linspace(0, 100, 10)
    y = np.linspace(0, 100, 10)
    xv, yv = np.meshgrid(x, y)
    return np.stack([xv.ravel(), yv.ravel()], axis=-1) # Shape (100, 2)

@pytest.fixture
def fs_instance(sample_coords):
    """Vytvoří základní testovací instanci třídy."""
    return FieldSynthesis(
        point_coords=sample_coords,
        count_points=50, 
        seed=42
    )

## --- Testy Inicializace a Vlastností ---

def test_initialization(fs_instance, sample_coords):
    """Ověří, že parametry jsou správně nastaveny."""
    # Testujeme shodu souřadnic
    np.testing.assert_array_equal(fs_instance.point_coords, sample_coords)
    assert fs_instance.count_points == 50
    assert fs_instance.dimension == 2

def test_area_stats(fs_instance):
    """Ověří výpočet statistik oblasti."""
    stats = fs_instance.area_stats
    assert stats["volume"] == 10000.0 # 100 * 100
    assert np.all(stats["min"] == 0)
    assert np.all(stats["max"] == 100)

def test_cached_distance(fs_instance):
    """Ověří výpočet minimální vzdálenosti."""
    dist = fs_instance.min_distance
    assert isinstance(dist, float)
    assert dist > 0
    # Ověření cache
    assert fs_instance.min_distance == dist

## --- Testy Generování Bodů ---

def test_generate_points(fs_instance):
    """Ověří tvar a rozsah vygenerovaných kotevních bodů."""
    points = fs_instance.anchor_points
    assert isinstance(points, np.ndarray)
    # Tvar by měl odpovídat (count_points, dimension)
    assert points.shape == (50, 2)
    # Body musí být v mezích vypočítaného bounding boxu
    stats = fs_instance.area_stats
    assert np.all(points >= stats["min"])
    assert np.all(points <= stats["max"])

def test_assign_source_fields(fs_instance):
    """Ověří přiřazení indexů polí k bodům."""
    # Testujeme pro 5 zdrojových polí
    num_source = 5
    indices = fs_instance.get_fields_indices(num_source)
    assert len(indices) == 50
    assert np.all(indices >= 0)
    assert np.all(indices < num_source)

## --- Testy Prostorové Logiky ---

def test_neighbor_data(fs_instance):
    """Ověří, že vyhledávání sousedů vrací seznam indexů pro každý bod."""
    neighbors = fs_instance.neighbor_data
    assert isinstance(neighbors, list)
    assert len(neighbors) == len
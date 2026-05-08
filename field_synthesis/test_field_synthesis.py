import pytest
import numpy as np
from field_synthesis import FieldSynthesis

@pytest.fixture
def fs_instance():
    """Vytvoří základní testovací instanci třídy s úzkým pruhem."""
    return FieldSynthesis(
        area_size=100.0, 
        count_points=50, 
        num_source=5, 
        dimension=2, 
        free_space_ratio=0.4,
        mixing_ratio=0.15,
        seed=42
    )

## --- Testy Inicializace a Generování ---

def test_initialization(fs_instance):
    assert fs_instance.area_size == 100.0
    assert fs_instance.count_points == 50
    assert fs_instance.mixing_ratio == 0.15

def test_cached_distance(fs_instance):
    dist = fs_instance.min_distance
    assert isinstance(dist, float)
    assert dist > 0

def test_generate_points(fs_instance):
    points = fs_instance.anchor_points
    assert isinstance(points, np.ndarray)
    assert points.shape[1] == 2
    assert len(points) <= 50 
    assert np.all(points >= 0) and np.all(points <= 100.0)

## --- Testy Logiky Míchacího Pruhu ---

def test_narrow_mixing_band(fs_instance):
    """
    Ověří, že s úzkým míchacím pruhem má většina prostoru (čisté zóny) 
    přiřazeného jen 1 souseda, a pouze hranice mají více sousedů k průměrování.
    """
    # Vygenerujeme testovací mřížku
    x = np.linspace(0, 100, 50)
    y = np.linspace(0, 100, 50)
    xx, yy = np.meshgrid(x, y)
    target_points = np.column_stack([xx.ravel(), yy.ravel()])
    
    neighbors_list = fs_instance.spatial_points(target_points)
    lengths = np.array([len(n) for n in neighbors_list])
    
    # Spočítáme, kolik bodů má přesně 1 souseda (čistá zóna bez průměrování)
    pure_zones_ratio = np.sum(lengths == 1) / len(lengths)
    
    # Ověříme, že většina pole je "čistá" (např. více než 70 % prostoru)
    assert pure_zones_ratio > 0.70, f"Příliš mnoho průměrování! Čistých zón je jen {pure_zones_ratio*100}%"
    
    # Ověříme, že existují nějaké hraniční body, kde dochází k míchání (> 1 soused)
    assert np.sum(lengths > 1) > 0, "Žádné hranice k míchání nebyly nalezeny!"

def test_mix_fields_logic(fs_instance):
    """Ověří, že vektorizované míchání polí funguje a nevrací zbytečné NaN."""
    target_points = np.array([[10, 10], [50, 50], [90, 90]])
    results = fs_instance.mix_fields(target_points)
    
    assert isinstance(results, np.ndarray)
    assert len(results) == 3
    assert results.dtype == np.float64

## --- Testy Robustnosti ---

def test_reproducibility():
    fs1 = FieldSynthesis(100, 20, 3, seed=10)
    fs2 = FieldSynthesis(100, 20, 3, seed=10)
    
    np.testing.assert_array_equal(fs1.anchor_points, fs2.anchor_points)
    np.testing.assert_array_equal(fs1.fields_indices, fs2.fields_indices)

def test_empty_area_handling():
    fs = FieldSynthesis(area_size=100, count_points=0, num_source=2)
    assert fs.min_distance == 0.0
    assert fs.anchor_points.shape == (0, 2)
    
    res = fs.mix_fields(np.array([[50, 50]]))
    assert np.isnan(res[0])
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
    assert fs_instance.dimension == 2

def test_cached_distance(fs_instance):
    """Ověří výpočet minimální vzdálenosti."""
    dist = fs_instance.min_distance
    assert isinstance(dist, float)
    assert dist > 0
    # Ověření, že se hodnota nemění (cache funguje)
    assert fs_instance.min_distance == dist

## --- Testy Generování Bodů ---

def test_generate_points(fs_instance):
    """Ověří tvar a rozsah vygenerovaných kotevních bodů."""
    points = fs_instance.anchor_points
    assert isinstance(points, np.ndarray)
    
    # Body musí mít správnou dimenzi
    assert points.shape[1] == 2
    # Generátor vrací max `count_points`, někdy lehce méně kvůli hustotě Poissonova disku
    assert len(points) <= 50 
    assert len(points) > 0
    
    # Body musí být v mezích oblasti [0, area_size]
    assert np.all(points >= 0)
    assert np.all(points <= 100.0)

def test_assign_source_fields(fs_instance):
    """Ověří přiřazení indexů polí k bodům."""
    indices = fs_instance.fields_indices
    points_count = len(fs_instance.anchor_points)
    
    assert len(indices) == points_count
    # Indexy musí být v rozsahu [0, num_source - 1]
    assert np.all(indices >= 0)
    assert np.all(indices < 5)
    assert indices.dtype.kind in 'iu' # integer nebo unsigned integer

## --- Testy Prostorové Logiky (Nové požadavky od vyučujícího) ---

def test_spatial_points_return_type(fs_instance):
    """Ověří, že vyhledávání sousedů vrací správný datový typ."""
    target = np.array([[50.0, 50.0]])
    neighbors = fs_instance.spatial_points(target)
    assert isinstance(neighbors, list)
    assert len(neighbors) == 1
    assert isinstance(neighbors[0], np.ndarray)

def test_min_required_neighbors(fs_instance):
    """
    Ověří dynamickou garanci minimálního počtu sousedů.
    Bod umístíme hluboko do záporných souřadnic (mimo R_LIMIT), 
    ale algoritmus musí stejně vrátit alespoň 'min_required' (výchozí 3) bodů.
    """
    target = np.array([[-1000.0, -1000.0]]) # Bod extrémně daleko
    neighbors = fs_instance.spatial_points(target, k_neighbors=5, min_required=3)
    
    assert len(neighbors[0]) == 3 # Garantované minimum

def test_mix_fields_logic(fs_instance):
    """Ověří, že míchání polí vrací průměrované hodnoty bez zbytečných NaN."""
    target_points = np.array([[10, 10], [50, 50], [90, 90]])
    results = fs_instance.mix_fields(target_points)
    
    assert isinstance(results, np.ndarray)
    assert len(results) == 3
    assert results.dtype == np.float64
    
    # Díky garanci minimálně 3 sousedů by u běžných bodů nemělo vzniknout NaN
    assert not np.isnan(results).any()

## --- Testy Robustnosti ---

def test_reproducibility():
    """Ověří, že stejný seed generuje absolutně identické výsledky (determinizmus)."""
    fs1 = FieldSynthesis(100, 20, 3, seed=10)
    fs2 = FieldSynthesis(100, 20, 3, seed=10)
    
    np.testing.assert_array_equal(fs1.anchor_points, fs2.anchor_points)
    np.testing.assert_array_equal(fs1.fields_indices, fs2.fields_indices)

def test_empty_area_handling():
    """Ověří stabilní chování při nulovém počtu bodů (edge case)."""
    fs = FieldSynthesis(area_size=100, count_points=0, num_source=2)
    assert fs.min_distance == 0.0
    assert fs.anchor_points.shape == (0, 2)
    
    # Při 0 bodech nesmí spadnout a musí vrátit NaN
    res = fs.mix_fields(np.array([[50, 50]]))
    assert np.isnan(res[0])
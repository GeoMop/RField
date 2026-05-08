from dataclasses import dataclass
import numpy as np
import math
from scipy.stats.qmc import PoissonDisk
from scipy.spatial import cKDTree
from functools import cached_property

@dataclass(frozen=True)
class FieldSynthesis:
    """
    Třída pro syntézu prostorových polí pomocí generování kotevních bodů 
    a jejich následného míchání na úzkých hranicích.
    
    Args:
        area_size (float): Délka strany pracovní oblasti.
        count_points (int): Požadovaný počet bodů k vygenerování.
        num_source (int): Počet dostupných zdrojů (typů polí).
        dimension (int): Dimenze prostoru (2D, 3D).
        free_space_ratio (float): Poměr volného prostoru pro odhad vzdálenosti D (výchozí 40 %).
        mixing_ratio (float): Koeficient pro šířku míchacího pruhu (výchozí 0.15 z D).
        seed (int): Seed pro generátor náhodných čísel.
    """
    area_size: float
    count_points: int
    num_source: int
    dimension: int = 2
    free_space_ratio: float = 0.4
    mixing_ratio: float = 0.15
    seed: int = 42 

    @cached_property
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)

    @cached_property
    def min_distance(self) -> float:
        """
        Odhad distance D pro zadaný počet bodů a stranu krychle.
        Uvažuje zadaný poměr volného prostoru (např. 40 %).
        """
        if self.count_points <= 0 or self.area_size <= 0:
            return 0.0
        
        total_vol = self.area_size ** self.dimension
        occupied_ratio = 1.0 - self.free_space_ratio
        vol_per_point = (total_vol * occupied_ratio) / self.count_points
        
        if self.dimension == 2:
            return math.sqrt(vol_per_point)
        else:
            return vol_per_point ** (1 / self.dimension)  

    @cached_property
    def anchor_points(self) -> np.ndarray:
        """Vygeneruje pole náhodných kotevních bodů pomocí modrého šumu."""
        if self.count_points <= 0 or self.area_size <= 0:
            return np.zeros((0, self.dimension))
        
        distance = max(self.min_distance, 0)
        radius = distance / self.area_size
            
        engine = PoissonDisk(d=self.dimension, radius=radius, seed=self.seed)
        return engine.random(self.count_points) * self.area_size

    @cached_property
    def fields_indices(self) -> np.ndarray:
        """Každému bodu přiřadíme náhodně jedno ze zdrojových N polí."""
        if self.anchor_points is None or len(self.anchor_points) == 0:
            return np.array([], dtype=int)
        
        return self.rng.integers(0, self.num_source, size=len(self.anchor_points))

    def spatial_points(self, target_points, k_neighbors=5) -> list:
        """
        Vyhledání sousedů s logikou "úzkého míchacího pruhu".
        Ponechá jen ty sousedy, jejichž vzdálenost se liší max o R_mix od nejbližšího.
        """
        if len(self.anchor_points) == 0:
            return [[] for _ in target_points]
            
        tree = cKDTree(self.anchor_points)
        actual_k = min(k_neighbors, len(self.anchor_points))

        distances, indices = tree.query(target_points, k=actual_k)

        if actual_k == 1:
            distances = distances.reshape(-1, 1)
            indices = indices.reshape(-1, 1)

        # Šířka úzkého míchacího pruhu (např. 0.15 * D)
        R_mix = self.mixing_ratio * self.min_distance
        
        # Maximální dosah pro hledání (pojistka proti prázdným oblastem)
        # R_max = 2 * self.min_distance

        # Vzdálenost k nejbližšímu sousedovi (první sloupec)
        d1 = distances[:, 0:1]

        # Maska: Ponecháme souseda, pokud je uvnitř pruhu (d_i - d1 <= R_mix) 
        # a zároveň neleží dál než R_max (2*D)
        valid_mask = (distances - d1 <= R_mix)

        final_result_indices = []
        for i in range(len(target_points)):
            current_valid = indices[i][valid_mask[i]]
            final_result_indices.append(current_valid)

        return final_result_indices

    def mix_fields(self, target_points) -> np.ndarray:
        """
        Plně vektorizované finální míchání polí (průměrování) pomocí np.bincount.
        Pro většinu bodů zbude jen jeden index, čímž vzniknou čisté homogenní zóny.
        K průměrování dojde pouze na úzkých hranicích.
        """
        neighbor_indices_list = self.spatial_points(target_points)
        
        lengths = np.array([len(neighbors) for neighbors in neighbor_indices_list])
        
        if np.sum(lengths) == 0:
            return np.full(len(target_points), np.nan)
        
        row_idx = np.repeat(np.arange(len(target_points)), lengths)
        flat_neighbors = np.concatenate(neighbor_indices_list)
        values = self.fields_indices[flat_neighbors].astype(float)
        
        n_rows = len(target_points)
        row_sums = np.bincount(row_idx, weights=values, minlength=n_rows)
        row_counts = np.bincount(row_idx, minlength=n_rows)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            row_means = row_sums / row_counts
            
        return row_means
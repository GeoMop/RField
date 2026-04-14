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
    a jejich následného míchání.
    *
    """

    point_coords: np.ndarray  # Shape (K, dimension)
    count_points: int = 100    # Počet kotevních bodů (anchor points)
    safety_factor: float = 1.5 # Pro výpočet min_distance
    seed: int = 42

    @property
    def dimension(self) -> int:
        return self.point_coords.shape[1]

    @cached_property
    def area_stats(self) -> dict:
        """Vypočítá bounding box a objem pracovní oblasti."""
        min_bounds = np.min(self.point_coords, axis=0)
        max_bounds = np.max(self.point_coords, axis=0)
        sides = max_bounds - min_bounds
        volume = np.prod(sides)
        return {
            "min": min_bounds,
            "max": max_bounds,
            "volume": volume,
            "sides": sides
        }

    @cached_property
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)

    @cached_property
    def min_distance(self) -> float:
        """
        Výpočet minimální vzdálenosti na základě objemu a počtu bodů.
        D = (Vol / N)^(1/dim) / safety_factor
        """
        vol = self.area_stats["volume"]
        if self.count_points <= 0 or vol <= 0:
            return 0.0
        
        vol_per_point = vol / self.count_points
        return (vol_per_point ** (1 / self.dimension)) / self.safety_factor

    @cached_property
    def anchor_points(self) -> np.ndarray:
        """Generuje kotevní body pomocí Poisson Disk Sampling v rámci bounding boxu."""
        if self.count_points <= 0:
            return np.zeros((0, self.dimension))

        # Normalizovaný rádius pro PoissonDisk (v jednotkách [0, 1])
        # Používáme průměrnou stranu pro normalizaci
        avg_side = np.mean(self.area_stats["sides"])
        radius = self.min_distance / avg_side if avg_side > 0 else 0.1
        radius = min(max(radius, 0.01), 0.99)

        try:
            engine = PoissonDisk(d=self.dimension, radius=radius, seed=self.seed)
            # Vygenerujeme body a roztáhneme je na rozměry bounding boxu
            points = engine.random(self.count_points)
            return self.area_stats["min"] + points * self.area_stats["sides"]
        except Exception:
            # Fallback na čistě náhodné body v případě chyby engine
            return self.rng.uniform(
                self.area_stats["min"], 
                self.area_stats["max"], 
                (self.count_points, self.dimension)
            )

    def get_fields_indices(self, num_source: int) -> np.ndarray:
        """Přiřadí každému kotevnímu bodu index jednoho ze zdrojových polí."""
        return self.rng.integers(0, num_source, size=len(self.anchor_points))

    @cached_property
    def neighbor_data(self):
        """
        Předvypočítá sousedy pro všechny target_points (point_coords).
        Vrací seznam indexů kotevních bodů pro každý bod v point_coords.
        """
        if len(self.anchor_points) == 0:
            return [np.array([], dtype=int)] * len(self.point_coords)

        tree = cKDTree(self.anchor_points)
        r_limit = 2 * self.min_distance
        # k_neighbors omezíme počtem dostupných kotevních bodů
        k = min(10, len(self.anchor_points)) 
        
        distances, indices = tree.query(self.point_coords, k=k, distance_upper_bound=r_limit)

        final_indices = []
        for i in range(len(self.point_coords)):
            # cKDTree vrací 'inf' a index == len(data) pro nenalezené sousedy
            idx = indices[i]
            valid = idx < len(self.anchor_points)
            final_indices.append(idx[valid])
            
        return final_indices

    def mix_fields(self, field_samples: np.ndarray) -> np.ndarray:
        """
        Míchání polí na základě předvypočítaných sousedů.
        
        Args:
            field_samples (np.ndarray): Shape (N, K), kde N je počet zdrojů 
                                        a K je počet bodů (len(point_coords)).
        """
        num_source, num_points = field_samples.shape
        if num_points != len(self.point_coords):
            raise ValueError("Počet bodů v field_samples neodpovídá point_coords.")

        # Generujeme indexy polí pro kotevní body
        anchor_to_field_map = self.get_fields_indices(num_source)
        
        mixed_result = np.full(num_points, np.nan)
        neighbor_indices_list = self.neighbor_data

        for i_point, neighbors in enumerate(neighbor_indices_list):
            if len(neighbors) == 0:
                continue
            
            # Získáme indexy zdrojových polí, které patří k sousedním kotevním bodům
            source_indices = anchor_to_field_map[neighbors]
            
            # Vybereme hodnoty z příslušných polí pro daný bod i_point
            # field_samples[source_idx, i_point]
            values = field_samples[source_indices, i_point]
            
            mixed_result[i_point] = np.mean(values)

        return mixed_result
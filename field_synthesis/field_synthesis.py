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
    a jejich následného míchání s exportem do formátu Zarr V3.
    
    Args:
        area_size (int): Délka strany pracovní oblasti.
        count_points (int): Požadovaný počet bodů k vygenerování.
        num_source (int): Počet dostupných zdrojů (typů polí).
        dimension (int): Dimenze prostoru (2D, 3D).
        free_space_ratio (float): Poměr volného prostoru pro výpočet Poissonova disku.
        seed (int): Seed pro generátor náhodných čísel.
    """
    area_size: int
    count_points: int
    num_source: int
    dimension: int = 2
    free_space_ratio: float = 0.4
    seed: int = 42 

    @cached_property
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)

    @cached_property
    def min_distance(self) -> float:
        """
        Odhad distance D pro zadaný počet bodů a stranu krychle.
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
        """
        Vygeneruje pole náhodných kotevních bodů pomocí modrého šumu.
        """
        if self.count_points <= 0 or self.area_size <= 0:
            return np.zeros((0, 2))
        
        distance = max(self.min_distance, 0)
        radius = distance / self.area_size

        if radius > 1:
            radius = 0.99
            
        try:
            engine = PoissonDisk(d=self.dimension, radius=radius, seed=self.seed)
            return engine.random(self.count_points) * self.area_size
        except Exception:
            return np.zeros((0, 2))

    @cached_property
    def fields_indices(self) -> np.ndarray:
        """
        Každému bodu přiřadíme náhodně jedno ze zdrojových N polí.
        """
        if self.anchor_points is None or len(self.anchor_points) == 0:
            return np.array([], dtype=int)
        
        return self.rng.integers(0, self.num_source, size=len(self.anchor_points))

    def spatial_points(self, target_points, k_neighbors=5, min_required=3) -> list:
        """
        Vektorizované vyhledání sousedů pomocí cKDTree s garancí minimálního počtu.
        """
        if len(self.anchor_points) == 0:
            return [[] for _ in target_points]
            
        tree = cKDTree(self.anchor_points)
        R_LIMIT = 2 * self.min_distance

        actual_k = min(k_neighbors, len(self.anchor_points))
        actual_min = min(min_required, len(self.anchor_points))

        distances, indices = tree.query(target_points, k=actual_k)

        if actual_k == 1:
            distances = distances.reshape(-1, 1)
            indices = indices.reshape(-1, 1)

        final_result_indices = []
        for i in range(len(target_points)):
            valid_mask = distances[i] <= R_LIMIT
            current_valid = indices[i][valid_mask]
            
            # Garance alespoň 'min_required' sousedů, jak chtěl vyučující
            if len(current_valid) < actual_min:
                current_valid = indices[i][:actual_min]
                
            final_result_indices.append(current_valid)

        return final_result_indices

    def mix_fields(self, target_points) -> np.ndarray:
            """
            Plně vektorizované finální míchání polí (průměrování) pomocí np.bincount.
            Toto řešení eliminuje pomalý Python for-cyklus a přesouvá výpočet do C-backendu NumPy.
            """
            neighbor_indices_list = self.spatial_points(target_points)
            
            # 1. Zjistíme, kolik sousedů má každý cílový bod
            lengths = np.array([len(neighbors) for neighbors in neighbor_indices_list])
            
            # Pojistka: Pokud neexistují vůbec žádní sousedé, vrátíme pole NaN
            if np.sum(lengths) == 0:
                return np.full(len(target_points), np.nan)
            
            # 2. Vytvoříme 'placaté' pole indexů řádků (odpovídá target_points)
            # Magie: pokud mají první tři body 3, 2 a 3 sousedy, 
            # np.repeat vygeneruje row_idx = [0, 0, 0, 1, 1, 2, 2, 2]
            row_idx = np.repeat(np.arange(len(target_points)), lengths)
            
            # 3. Sloučíme všechny indexy sousedů do jednoho 1D pole (flatten)
            flat_neighbors = np.concatenate(neighbor_indices_list)
            
            # 4. Získáme reálné hodnoty zdrojových polí pro tyto sousedy
            values = self.fields_indices[flat_neighbors].astype(float)
            
            # 5. Vektorizované sčítání a počítání (přesně podle návrhu)
            n_rows = len(target_points)
            
            row_sums = np.bincount(row_idx, weights=values, minlength=n_rows)
            row_counts = np.bincount(row_idx, minlength=n_rows)
            
            # 6. Výpočet průměru s ochranou proti dělení nulou
            # np.errstate potlačí varování, pokud má nějaký bod 0 sousedů (výsledek bude automaticky NaN)
            with np.errstate(divide='ignore', invalid='ignore'):
                row_means = row_sums / row_counts
                
            return row_means
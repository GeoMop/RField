import numpy as np
import math
from scipy.stats.qmc import PoissonDisk
from scipy.spatial import cKDTree


class FieldSynthesis():

    def __init__(self, area_size: float, count_points: int, num_source: int, dimension: int = 2, seed: int = 42):
        self.area_size = area_size
        self.count_points = count_points
        self.num_source = num_source
        self.dimension = dimension
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        # Hodnoty, které se dopočítají/vygenerují
        self.min_distance = self.calc_distance(self)
        self.anchor_points = None
        self.field_indices = None

    def calc_distance(self, free_space_ratio: float = 0.4) -> float:
        """
        Odhad distance D pro zadaný počet bodů a stranu krychle.
        """
        if self.count_points <= 0 or self.area_size <= 0:
            return 0.0
            
        total_vol = self.area_size ** self.dimension
        occupied_ratio = 1.0 - free_space_ratio
        vol_per_point = (total_vol * occupied_ratio) / self.count_points
        
        if self.dimension == 2:
            return math.sqrt(vol_per_point)
        else:
            return vol_per_point ** (1/self.dimension)  
        


    def generate_points(self):
        """
        Funkce vygeneruje pole nahodnych bodu.
        
        :param count_points: pocet nahodnych bodu.
        :param min_distance: minimalni vzdalenost mezi bodami.
        :param area_size: velokost matici.
        :return: vrati pole koordinat [[x1, y1], [x2, y2], ...]
        """
        if self.count_points <= 0 or self.area_size <= 0:
            return np.zeros((0, 2))
        
        if self.min_distance < 0:
            self.min_distance = 0
        
        radius = self.min_distance/self.area_size

        if radius > 1:
            radius = 0.99
        try:
            engine = PoissonDisk(d=2, radius=radius, seed=42)

            #ten algorytm je nakladny na cas a resurs pocitace
            # points = engine.fill_space() * area_size

            # if (len(points) > count_points):
            #     return points[:count_points]

            self.anchor_points = engine.random(self.count_points) * self.area_size
            return self.anchor_points
        except Exception:
            self.anchor_points = np.zeros((0, 2))
            return self.anchor_points

    def assign_source_fields(self):
        """
        Každému bodu přiřadíme náhodně jedno ze zdrojových N polí.
        """
        if self.anchor_points is None or len(self.anchor_points) == 0:
            self.field_indices = np.array([], dtype=int)
        else:
            self.field_indices = self.rng.integers(0, self.num_source, size=len(self.anchor_points))
                
        return self.field_indices
    

    def spatial_points(self, target_points, k_neighbors=5):
        """
        Vektorizované vyhledání sousedů pomocí cKDTree.
        """
        if self.anchor_points is None:
            self.generate_points()
            
        tree = cKDTree(self.anchor_points)
        R_LIMIT = 2 * self.min_distance

        # Vektorizovaný dotaz
        distances, indices = tree.query(target_points, k=k_neighbors)

        # Vektorizovaná maska vzdálenosti
        valid_neighbor_mask = distances <= R_LIMIT

        final_result_indices = []
        for i in range(len(target_points)):
            # Výběr platných indexů pro každý cílový bod
            current_valid = indices[i][valid_neighbor_mask[i]]
            final_result_indices.append(current_valid)

        return final_result_indices

     
    def mix_fields(self, target_points):
        """
        Finální míchání polí (průměrování).
        """
        
        neighbor_indices_list = self.spatial_points(target_points)
        
        mixed_results = []
        for neighbors in neighbor_indices_list:
            if len(neighbors) == 0:
                mixed_results.append(np.nan)
            elif len(neighbors) == 1:
                mixed_results.append(self.field_indices[neighbors[0]])
            else:
                # "Provedeme průměr, pokud jich zbyde více
                values = self.field_indices[neighbors]
                mixed_results.append(np.mean(values))
                
        return np.array(mixed_results)
from dataclasses import dataclass
import numpy as np
import math
from scipy.stats.qmc import PoissonDisk
from scipy.spatial import cKDTree
from functools import cached_property


@dataclass(frozen=True)
class FieldSynthesis():
    """
    Třída pro syntézu prostorových polí pomocí generování kotevních bodů 
    a jejich následného míchání.

    Assumption: The set of points where the input and output fields live is the same.
    
        Inicializuje parametry syntézy pole.

        Args:
            area_size (float): Délka strany pracovní oblasti.
            count_points (int): Požadovaný počet bodů k vygenerování.
            num_source (int): Počet dostupných zdrojů (typů polí).
            dimension (int): Dimenze prostoru (2D, 3D).
            seed (int): Seed pro generátor náhodných čísel.

        TODO: add parameter point_coords shape=(K, dimension)  
        
    """

    area_size: int
    # TODO: compute from point_coords as volume of the axis aligned bounding box.
    # must be float.
    # next, minimum distance is vol ** (1/dim) / safety_factor , e.g. saftey_factor = 2.0 
    # should be in interval [1, 2].
    
    count_points: int
    # Suggested number of anchor points, could be refined due to minimal distance.
    
    num_source: int
    # Number of field samples we will mix; 
    # TODO: remove this, we will know that only
    # after passing the field samples to the mixing function.
    
    dimension: int = 2
    # TODO: remove, diven by the shape of point_coords
    
    free_space_ratio: float = 0.4
    # TODO: Remove, given indirectly through `count_points`
    
    seed: int = 42 

        # self.min_distance = self.calc_distance()
        # self.anchor_points = None
        # self.field_indices = None

    @cached_property
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)

    @cached_property
    def min_distance(self) -> float:
        """
        Odhad distance D pro zadaný počet bodů a stranu krychle.
        D se odhaduje na základě poměru volného prostoru a obsazeného prostoru.
        """
        if self.count_points <= 0 or self.area_size <= 0:
            return 0.0
        
        total_vol = self.area_size ** self.dimension
        occupied_ratio = 1.0 - self.free_space_ratio
        vol_per_point = (total_vol * occupied_ratio) / self.count_points
        
        if self.dimension == 2:
            return math.sqrt(vol_per_point)
        else:
            return vol_per_point ** (1/self.dimension)  
        

    @cached_property
    def anchor_points(self) -> np.ndarray:
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
            engine = PoissonDisk(d=self.dimension, radius=radius, seed=42)

            #ten algorytm je nakladny na cas a resurs pocitace
            # points = engine.fill_space() * area_size

            # if (len(points) > count_points):
            #     return points[:count_points]

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
    

    def spatial_points(self, target_points, k_neighbors=5) -> list:
        """
        Vektorizované vyhledání sousedů pomocí cKDTree.
        Pro každý cílový bod získáme k nejbližších kotevních bodů a jejich vzdálenosti.
        Poté aplikujeme vektorizovanou masku pro filtrování sousedů, kteří jsou dále než 2 * D.
        Args:
            target_points (np.ndarray): Pole cílových bodů, pro které chceme naj

        TODO: could be refactored into cached_property since target_points == point_coords are given 
        in constructor.
        """
            
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

     
    def mix_fields(self, target_points) -> np.ndarray:
        """
        Finální míchání polí (průměrování).
        Pro každý cílový bod získáme jeho sousedy pomocí spatial_points.
        Pokud má sousedů více, provedeme průměr jejich hodnot. Pokud žádného
        Args:
            target_points (np.ndarray): Pole cílových bodů, pro které chceme získat smíšené hodnoty.
        TODO: pass in field samples, field_samples shape = (N, K) 
        N .. number of samples
        K ... number of points == len(point_coords)
        """
        
        neighbor_indices_list = self.spatial_points(target_points)
        # List of lists of anchor neighbours of each point.
        
        mixed_results = []
        for i_point, neighbors in enumerate(neighbor_indices_list):
            if len(neighbors) == 0:
                mixed_results.append(np.nan)
            elif len(neighbors) == 1:
                mixed_results.append(self.fields_indices[neighbors[0]])
            else:
                # TODO: use the field samples and compute mean of their values
                # mixed_samples = field_samples[self.fields_indices[neighbors], i_point] # shape (len(neighbours), )
                # value = np.mean( mixed_samples)                
                values = self.fields_indices[neighbors]
                mixed_results.append(np.mean(values))
                
        return np.array(mixed_results)

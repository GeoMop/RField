import numpy as np
import math
from scipy.stats.qmc import PoissonDisk
from scipy.spatial import cKDTree
from functools import cached_property


class FieldSynthesis():
    """
    Třída pro syntézu prostorových polí pomocí generování kotevních bodů 
    a jejich následného míchání.
    """


    def __init__(self, area_size: float, count_points: int, num_source: int, dimension: int = 2, seed: int = 42):
        """
        Inicializuje parametry syntézy pole.

        Args:
            area_size (float): Délka strany pracovní oblasti.
            count_points (int): Požadovaný počet bodů k vygenerování.
            num_source (int): Počet dostupných zdrojů (typů polí).
            dimension (int): Dimenze prostoru (2D, 3D).
            seed (int): Seed pro generátor náhodných čísel.
        """

        self.area_size = area_size
        self.count_points = count_points
        self.num_source = num_source
        self.dimension = dimension
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # self.min_distance = self.calc_distance()
        # self.anchor_points = None
        # self.field_indices = None

    @cached_property
    def min_distance(self) -> float:
        """
        Odhad distance D pro zadaný počet bodů a stranu krychle.
        D se odhaduje na základě poměru volného prostoru a obsazeného prostoru.
        """
        if self.count_points <= 0 or self.area_size <= 0:
            return 0.0
        
        free_space_ratio = 0.4
        total_vol = self.area_size ** self.dimension
        occupied_ratio = 1.0 - free_space_ratio
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
            engine = PoissonDisk(d=2, radius=radius, seed=42)

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
        """
        
        neighbor_indices_list = self.spatial_points(target_points)
        
        mixed_results = []
        for neighbors in neighbor_indices_list:
            if len(neighbors) == 0:
                mixed_results.append(np.nan)
            elif len(neighbors) == 1:
                mixed_results.append(self.fields_indices[neighbors[0]])
            else:
                # "Provedeme průměr, pokud jich zbyde více
                values = self.fields_indices[neighbors]
                mixed_results.append(np.mean(values))
                
        return np.array(mixed_results)
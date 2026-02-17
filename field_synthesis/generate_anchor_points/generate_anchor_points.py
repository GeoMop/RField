from scipy.stats.qmc import PoissonDisk
import numpy as np

def generate_anchor_points(count_points, min_distance, area_size):
    """
    Funkce vygeneruje pole nahodnych bodu.
    
    :param count_points: pocet nahodnych bodu.
    :param min_distance: minimalni vzdalenost mezi bodami.
    :param area_size: velokost matici.
    :return: vrati pole koordinat [[x1, y1], [x2, y2], ...]
    """
    if count_points <= 0 or area_size <= 0:
        return np.zeros((0, 2))
    
    if min_distance < 0:
        min_distance = 0
    
    radius = min_distance/area_size

    if radius > 1:
        radius = 0.99
    try:
        engine = PoissonDisk(d=2, radius=radius, seed=42)

        #ten algorytm je nakladny na cas a resurs pocitace
        # points = engine.fill_space() * area_size

        # if (len(points) > count_points):
        #     return points[:count_points]

        points = engine.random(count_points) * area_size
        return points
    except Exception:
        return np.zeros((0, 2))
import math

def calc_distance(count_points: int, area_size: float, rank: int, free_space_ratio: float = 0.4) -> float:
    """
    Odhadne optimální minimální vzdálenost bodů ve prostoru 
    s ohledem na procento požadovaného volného prostoru.
    """
    if count_points <= 0 or area_size <= 0:
        return 0.0
        
    total_area = area_size ** rank
    occupied_ratio = 1.0 - free_space_ratio
    occupied_area = total_area * occupied_ratio
    
    area_per_point = occupied_area / count_points
    
    min_distance = area_per_point ** (1/rank)
    
    return min_distance
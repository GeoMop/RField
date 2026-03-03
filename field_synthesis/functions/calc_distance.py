import math

def calc_distance(count_points: int, area_size: float, free_space_ratio: float = 0.4) -> float:
    """
    Odhadne optimální minimální vzdálenost bodů ve 2D prostoru 
    s ohledem na procento požadovaného volného prostoru.
    """
    if count_points <= 0 or area_size <= 0:
        return 0.0
        
    total_area = area_size ** 2
    occupied_ratio = 1.0 - free_space_ratio
    occupied_area = total_area * occupied_ratio
    
    area_per_point = occupied_area / count_points
    
    min_distance = math.sqrt(area_per_point)
    
    return min_distance
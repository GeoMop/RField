# 🌐 Balíček Field Synthesis

Tento modul slouží k **prostorovému vzorkování a syntéze polí**. Umožňuje generovat rovnoměrně rozmístěné kotevní body v $N$-dimenzionálním prostoru a následně na jejich základě interpolovat (míchat) hodnoty pro libovolné cílové body.



## 🛠️ Klíčové vlastnosti
* **Poisson Disk Sampling**: Zajišťuje, že kotevní body nejsou příliš blízko u sebe (modré šumy), což vede k přirozenějším výsledkům než čistá náhodnost.
* **Lazy Evaluation**: Díky `@cached_property` se náročné výpočty (generování bodů, hledání sousedů) provádějí až ve chvíli, kdy jsou skutečně potřeba, a výsledek se ukládá do paměti.
* **Vektorizace**: Využívá `scipy.spatial.cKDTree` pro bleskové vyhledávání nejbližších sousedů i ve velkých datasetech.

---

## 🚀 Rychlý start (Použití třídy)

```python
from field_synthesis import FieldSynthesis
import numpy as np

# 1. Inicializace (nastavení scény)
fs = FieldSynthesis(
    area_size=100.0,   # Velikost pracovní plochy
    count_points=500,  # Počet kotevních bodů
    num_source=3,      # Počet typů zdrojů (0, 1, 2)
    dimension=2,       # 2D nebo 3D
    seed=42            # Pro reprodukovatelnost
)

# 2. Definice cílových bodů (kde chceme znát hodnotu pole)
target_points = np.array([
    [10.5, 20.1],
    [55.0, 55.0],
    [90.2, 10.8]
])

# 3. Výpočet (vše se vygeneruje automaticky při prvním volání)
results = fs.mix_fields(target_points)

print(f"Výsledné hodnoty v cílových bodech: {results}")
```

---

## 🏗️ Architektura třídy `FieldSynthesis`

### Atributy (Vlastnosti)
| Atribut | Typ | Popis |
| :--- | :--- | :--- |
| `min_distance` | `float` | **(Cached)** Automaticky vypočítaný poloměr pro Poisson Disk, aby se body vešly do dané plochy. |
| `generate_points` | `np.ndarray` | **(Cached)** Souřadnice kotevních bodů. |
| `field_indices` | `np.ndarray` | **(Cached)** Náhodně přiřazené ID zdroje pro každý kotevní bod. |

### Klíčové Metody
* **`spatial_points(target_points, k_neighbors=5)`**: Pro každý cílový bod najde až `k` sousedů v okruhu `2 * min_distance`. Vrací seznam indexů.
* **`mix_fields(target_points)`**: Hlavní metoda. Provede průměrování hodnot sousedních kotevních bodů pro každý cílový bod.

---

## 🧪 Testování a stabilita
Modul je plně pokryt testy v `pytest`. Před nasazením změn vždy spusťte:
```bash
pytest field_synthesis/test_field_synthesis.py
```

---

### 💡 Doporučení pro kolegy
> Pokud potřebujete resetovat vygenerované body (např. chceš jiný vzorek se stejným nastavením), stačí z objektu smazat cache: `del fs.generate_points`. Při příštím volání se body vygenerují znovu.

---
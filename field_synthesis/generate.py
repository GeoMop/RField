import argparse
import zipfile
import io
import os
import numpy as np
from field_synthesis import FieldSynthesis

def load_data(zip_path):
    """Extrahuje souřadnice a všechny vzorky z jednoho ZIP archivu."""
    coords = None
    all_values = []
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        file_list = z.namelist()
        
        # Načtení souřadnic
        coords_file = [f for f in file_list if 'coords' in f and f.endswith('.npz')][0]
        with z.open(coords_file) as f:
            data = np.load(io.BytesIO(f.read()))
            # Vybereme první pole z npz souboru
            coords = data[data.files[0]]
            
        # Načtení hodnot polí
        value_files = [f for f in file_list if 'values' in f and f.endswith('.npz')]
        print(f"Nalezeno {len(value_files)} zdrojových vzorků v ZIPu.")
        
        for name in sorted(value_files):
            with z.open(name) as f:
                data = np.load(io.BytesIO(f.read()))
                val = data[data.files[0]]
                all_values.append(val)
                
    return coords, np.array(all_values)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", help="Cesta k vstupnímu ZIP archivu (cond_tensors.zip)")
    parser.add_argument("-o", "--output", default="synthesis_results.zip", help="Název výstupního souboru")
    parser.add_argument("--anchors", type=int, default=100, help="Počet kotevních bodů (anchor points)")
    parser.add_argument("--count", type=int, default=200, help="Počet polí k vygenerování")
    parser.add_argument("--seed", type=int, default=42, help="Seed pro generátor náhodných čísel")
    args = parser.parse_args()

    # 1. Načtení dat
    coords, samples_raw = load_data(args.archive)
    
    # Předpokládáme, že samples_raw má tvar (N, K, 1), převedeme na (N, K)
    if samples_raw.ndim == 3:
        samples = samples_raw[:, :, 0]
    else:
        samples = samples_raw

    # 2. Inicializace FieldSynthesis
    # Souřadnice bodů předáváme přímo do konstruktoru
    fs = FieldSynthesis(
        point_coords=coords,
        count_points=args.anchors,
        seed=args.seed
    )

    print(f"Syntetizuji {args.count} polí...")
    
    # 3. Generování nových polí
    with zipfile.ZipFile(args.output, 'w', compression=zipfile.ZIP_DEFLATED) as out_z:
        for i in range(args.count):
            # Každé volání mix_fields vytvoří unikátní pole díky vnitřnímu stavu RNG
            res = fs.mix_fields(samples)
            
            # Uložení výsledku do paměti (io.BytesIO)
            buffer = io.BytesIO()
            np.save(buffer, res)
            
            filename = f"field_{i:04d}.npy"
            out_z.writestr(filename, buffer.getvalue())
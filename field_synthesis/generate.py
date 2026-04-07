import argparse
import zipfile
import io
import os
import numpy as np
from field_synthesis import FieldSynthesis

def load_data(zip_path):
    """Витягує координати та всі сампли з одного ZIP-архіву."""
    coords = None
    all_values = []
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        file_list = z.namelist()
        
        coords_file = [f for f in file_list if 'coords' in f and f.endswith('.npz')][0]
        with z.open(coords_file) as f:
            data = np.load(io.BytesIO(f.read()))
            coords = data[data.files[0]] if isinstance(data, np.lib.npyio.NpzFile) else data

        value_files = [f for f in file_list if 'values' in f and f.endswith('.npz')]
        print(f"Found {len(value_files)} source samples in ZIP.")
        
        for name in sorted(value_files):
            with z.open(name) as f:
                data = np.load(io.BytesIO(f.read()))
                val = data[data.files[0]] if isinstance(data, np.lib.npyio.NpzFile) else data
                all_values.append(val)
                
    return coords, np.array(all_values)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", help="Path to cond_tensors.zip")
    parser.add_argument("-o", "--output", default="synthesis_results.zip")
    parser.add_argument("--anchors", type=int, default=100)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    coords, samples_raw = load_data(args.archive)
    
    samples = samples_raw[:, :, 0]

    max_range = np.max(coords)
    fs = FieldSynthesis(
        area_size=max_range,
        count_points=args.anchors,
        num_source=len(samples),
        dimension=3,
        seed=args.seed
    )

    print(f"Synthesizing {args.count} fields...")
    
    with zipfile.ZipFile(args.output, 'w', compression=zipfile.ZIP_DEFLATED) as out_z:
        for i in range(args.count):
            res = fs.mix_fields(coords)
            
            buffer = io.BytesIO()
            np.save(buffer, res)
            
            filename = f"field_{i:04d}.npy"
            out_z.writestr(filename, buffer.getvalue())
            
            if (i + 1) % 50 == 0:
                print(f"Progress: {i + 1}/{args.count}")

    print(f"Done! Results in {args.output}")

if __name__ == "__main__":
    main()
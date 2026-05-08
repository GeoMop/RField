import argparse
import zipfile
import io
import numpy as np
import xarray as xr
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
    parser.add_argument("-o", "--output", default="synthesis_results.zarr")
    parser.add_argument("--anchors", type=int, default=100)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    # Нові аргументи для керування вузьким кордоном
    parser.add_argument("--mixing-ratio", type=float, default=0.15, help="Width of the mixing band (default: 0.15)")
    parser.add_argument("--free-space", type=float, default=0.4, help="Free space ratio (default: 0.4)")
    args = parser.parse_args()

    coords, samples_raw = load_data(args.archive)
    
    samples = samples_raw[:, :, 0] if samples_raw.ndim == 3 else samples_raw
    max_range = float(np.max(coords))
    
    # Автоматично визначаємо розмірність (2D або 3D) на основі вхідних координат
    dim = coords.shape[1]

    print(f"Synthesizing {args.count} fields (Dimension: {dim}D, Mixing Ratio: {args.mixing_ratio})...")
    all_synthesized_fields = []

    for i in range(args.count):
        fs = FieldSynthesis(
            area_size=max_range,
            count_points=args.anchors,
            num_source=len(samples),
            dimension=dim,
            free_space_ratio=args.free_space,
            mixing_ratio=args.mixing_ratio,
            seed=args.seed + i
        )
        
        res = fs.mix_fields(coords)
        all_synthesized_fields.append(res)
        
        if (i + 1) % 50 == 0:
            print(f"Progress: {i + 1}/{args.count}")

    stacked_results = np.vstack(all_synthesized_fields)

    print("Packing data into xarray Dataset...")
    
    # Динамічне формування data_vars, щоб уникнути помилок з None для 2D даних
    data_vars = {
        "mixed_fields": (["field_idx", "point_idx"], stacked_results),
        "coords_x": (["point_idx"], coords[:, 0]),
        "coords_y": (["point_idx"], coords[:, 1]),
    }
    # Додаємо вісь Z тільки якщо дані дійсно 3D
    if dim > 2:
        data_vars["coords_z"] = (["point_idx"], coords[:, 2])

    ds = xr.Dataset(
        data_vars=data_vars,
        coords=dict(
            field_idx=np.arange(args.count),
            point_idx=np.arange(len(coords))
        ),
        attrs=dict(
            description="Stochasticky generované horninové masivy",
            area_size=max_range,
            anchors_per_field=args.anchors,
            mixing_ratio=args.mixing_ratio,
            free_space_ratio=args.free_space,
            base_seed=args.seed
        )
    )

    print(f"Saving to Zarr V3 format: {args.output}")
    ds.to_zarr(args.output, mode="w", zarr_format=3, consolidated=False)
    
    print("Done! All systems nominal.")

if __name__ == "__main__":
    main()
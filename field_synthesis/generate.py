import argparse
import zipfile
import io
import numpy as np
import xarray as xr
from field_synthesis import FieldSynthesis

def load_data(zip_path):
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
    args = parser.parse_args()

    coords, samples_raw = load_data(args.archive)
    
    samples = samples_raw[:, :, 0] if samples_raw.ndim == 3 else samples_raw
    max_range = float(np.max(coords))

    print(f"Synthesizing {args.count} fields...")
    all_synthesized_fields = []

    for i in range(args.count):
        fs = FieldSynthesis(
            area_size=max_range,
            count_points=args.anchors,
            num_source=len(samples),
            dimension=3,
            seed=args.seed + i
        )
        
        res = fs.mix_fields(coords)
        all_synthesized_fields.append(res)
        
        if (i + 1) % 50 == 0:
            print(f"Progress: {i + 1}/{args.count}")

    stacked_results = np.vstack(all_synthesized_fields)

    print("Packing data into xarray Dataset...")
    
    ds = xr.Dataset(
        data_vars=dict(
            mixed_fields=(["field_idx", "point_idx"], stacked_results),
            coords_x=(["point_idx"], coords[:, 0]),
            coords_y=(["point_idx"], coords[:, 1]),
            coords_z=(["point_idx"], coords[:, 2]) if coords.shape[1] > 2 else None
        ),
        coords=dict(
            field_idx=np.arange(args.count),
            point_idx=np.arange(len(coords))
        ),
        attrs=dict(
            description="Stochasticky generované horninové masivy",
            area_size=max_range,
            anchors_per_field=args.anchors,
            base_seed=args.seed
        )
    )

    print(f"Saving to Zarr V3 format: {args.output}")
    ds.to_zarr(args.output, mode="w", zarr_format=3, consolidated=False)
    
    print("Done! All systems nominal.")

if __name__ == "__main__":
    main()
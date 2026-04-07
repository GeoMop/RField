import numpy as np
import xarray as xr

def vytvor_mock_data(n_points: int = 2500, n_samples_A: int = 10, n_samples_B: int = 20):
    """
    Co to je: Funkce pro vygenerování syntetických testovacích dat.
    Typ výstupu: Tuple (xr.Dataset, np.ndarray, np.ndarray, np.ndarray, np.ndarray).
    S čím pracuje: S knihovnou numpy (náhodná čísla) a xarray (strukturování dat).
    Kde bere data: Generuje je sama od nuly pomocí np.random. Nepoužívá externí soubory.
    """
    n_dim = 2

    X_data = np.random.rand(n_dim, n_points)
    QA_data = 10**(np.random.normal(loc=-10, scale=3, size=(n_points, n_samples_A)))
    QB_data = 10**(np.random.normal(loc=-10, scale=5, size=(n_points, n_samples_B)))

    ds = xr.Dataset(
        data_vars={
            "X": (("i_dim", "i_point"), X_data),
            "QA": (("i_point", "i_sample_A"), QA_data),
            "QB": (("i_point", "i_sample_B"), QB_data),
        },
        coords={
            "i_point": np.arange(n_points),
            "i_sample_A": np.arange(n_samples_A),
            "i_sample_B": np.arange(n_samples_B),
            "i_dim": ["x", "y"]
        }
    )

    ds["QA"].attrs["long_name"] = "Porovnávaná veličina A"
    ds["QB"].attrs["long_name"] = "Porovnávaná veličina B"

    q_a_vals = ds["QA"].values
    q_b_vals = ds["QB"].values
    x_vals = ds["X"].sel(i_dim="x").values
    y_vals = ds["X"].sel(i_dim="y").values

    return ds, q_a_vals, q_b_vals, x_vals, y_vals
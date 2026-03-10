import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Struktura vstupu náhodných polí (xarray)

    Data jsou reprezentována pomocí datové struktury `xarray.Dataset`.

    **Definované dimenze a souřadnice (coords):**
    - `i_point`: Index konkrétního bodu v nepravidelné síti.
    - `i_sample`: Index vzorku (realizace) náhodného pole.
    - `i_dim`: Označení prostorové osy (např. 'x', 'y').

    **Datové proměnné (data_vars):**
    - `X`: Matice prostorových souřadnic bodů. Má tvar `[i_dim, i_point]`.
    - `QA`: První vygenerované náhodné pole. Má tvar `[i_point, i_sample]`.
    - `QB`: Druhé vygenerované náhodné pole. Má tvar `[i_point, i_sample]`.
    """)
    return


@app.cell(hide_code=True)
def _():
    import xarray as xr
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import binned_statistic_2d

    n_points = 1000
    n_samples = 100
    n_dim = 2

    X_data = np.random.rand(n_dim, n_points)
    QA_data = np.random.uniform(0.01, 1.0, size=(n_points, n_samples))
    QB_data = np.random.normal(loc=0.5, scale=0.15, size=(n_points, n_samples))
    QB_data = np.clip(QB_data, 0.01, None)

    ds = xr.Dataset(
        data_vars={
            "X": (("i_dim", "i_point"), X_data),
            "QA": (("i_point", "i_sample"), QA_data),
            "QB": (("i_point", "i_sample"), QB_data),
        },
        coords={
            "i_point": np.arange(n_points),
            "i_sample": np.arange(n_samples),
            "i_dim": ["x", "y"]
        }
    )

    ds["QA"].attrs["long_name"] = "Porovnávaná veličina A"
    ds["QA"].attrs["units"] = "-"
    ds["QB"].attrs["long_name"] = "Porovnávaná veličina B"
    ds["QB"].attrs["units"] = "-"

    stats = xr.Dataset({
        "mean_QA": ds["QA"].mean(dim="i_sample"),
        "var_QA":  ds["QA"].var(dim="i_sample"),
        "mean_QB": ds["QB"].mean(dim="i_sample"),
        "var_QB":  ds["QB"].var(dim="i_sample")
    })

    x = ds["X"].sel(i_dim="x").values
    y = ds["X"].sel(i_dim="y").values
    
    # Pro grafy průměru a rozptylu použít imgshow nebo podobnou funkci
    num_bins = 20
    bins = (num_bins, num_bins)

    binned_mean_QA = binned_statistic_2d(x, y, stats["mean_QA"].values, statistic='mean', bins=bins)
    binned_var_QA = binned_statistic_2d(x, y, stats["var_QA"].values, statistic='mean', bins=bins)
    binned_mean_QB = binned_statistic_2d(x, y, stats["mean_QB"].values, statistic='mean', bins=bins)
    binned_var_QB = binned_statistic_2d(x, y, stats["var_QB"].values, statistic='mean', bins=bins)

    x_edges = binned_mean_QA.x_edge
    y_edges = binned_mean_QA.y_edge

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    im1 = axes[0, 0].pcolormesh(x_edges, y_edges, binned_mean_QA.statistic.T, cmap='viridis', shading='flat')
    axes[0, 0].set_title("QA: Mapa průměru")
    axes[0, 0].set_aspect('equal')
    plt.colorbar(im1, ax=axes[0, 0])

    im2 = axes[0, 1].pcolormesh(x_edges, y_edges, binned_var_QA.statistic.T, cmap='magma', shading='flat')
    axes[0, 1].set_title("QA: Mapa rozptylu")
    axes[0, 1].set_aspect('equal')
    plt.colorbar(im2, ax=axes[0, 1])

    axes[0, 2].hist(ds["QA"].values.flatten(), bins=30, color='skyblue', edgecolor='black')
    axes[0, 2].set_title("QA: Histogram")

    im3 = axes[1, 0].pcolormesh(x_edges, y_edges, binned_mean_QB.statistic.T, cmap='viridis', shading='flat')
    axes[1, 0].set_title("QB: Mapa průměru")
    axes[1, 0].set_aspect('equal')
    plt.colorbar(im3, ax=axes[1, 0])

    im4 = axes[1, 1].pcolormesh(x_edges, y_edges, binned_var_QB.statistic.T, cmap='magma', shading='flat')
    axes[1, 1].set_title("QB: Mapa rozptylu")
    axes[1, 1].set_aspect('equal')
    plt.colorbar(im4, ax=axes[1, 1])

    axes[1, 2].hist(ds["QB"].values.flatten(), bins=30, color='salmon', edgecolor='black')
    axes[1, 2].set_title("QB: Histogram")

    plt.tight_layout()
    plt.show()

    # Vypočtěte různé druhy průměrů (aritmetický, geometrický, harmonický) na podčtvercích (multi-scale) pro různé velikosti oken. Zatím pro jednu velikost okna.
    q_values = stats["mean_QA"].values
    grid_size = 10
    A_arith = np.zeros((grid_size, grid_size))
    A_geom = np.zeros((grid_size, grid_size))
    A_harm = np.zeros((grid_size, grid_size))
    
    dx = 1.0 / grid_size
    dy = 1.0 / grid_size
    
    for i in range(grid_size):
        for j in range(grid_size):
            mask = (x >= i*dx) & (x < (i+1)*dx) & (y >= j*dy) & (y < (j+1)*dy)
            q_in_window = q_values[mask]
            
            if len(q_in_window) > 0:
                A_arith[j, i] = np.mean(q_in_window)
                A_geom[j, i] = np.exp(np.mean(np.log(q_in_window)))
                A_harm[j, i] = 1.0 / np.mean(1.0 / q_in_window)
            else:
                A_arith[j, i] = np.nan
                A_geom[j, i] = np.nan
                A_harm[j, i] = np.nan

    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    
    im5 = axes2[0].imshow(A_arith, origin='lower', extent=[0, 1, 0, 1], cmap='viridis')
    axes2[0].set_title("Aritmetický průměr")
    plt.colorbar(im5, ax=axes2[0])
    
    im6 = axes2[1].imshow(A_geom, origin='lower', extent=[0, 1, 0, 1], cmap='viridis')
    axes2[1].set_title("Geometrický průměr")
    plt.colorbar(im6, ax=axes2[1])
    
    im7 = axes2[2].imshow(A_harm, origin='lower', extent=[0, 1, 0, 1], cmap='viridis')
    axes2[2].set_title("Harmonický průměr")
    plt.colorbar(im7, ax=axes2[2])
    
    plt.tight_layout()
    plt.show()

    return ds, fig, fig2


if __name__ == "__main__":
    app.run()
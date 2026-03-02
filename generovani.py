import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
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


@app.cell
def _():
    import xarray as xr
    import numpy as np
    import matplotlib.pyplot as plt

    n_points = 300
    n_samples = 100
    n_dim = 2

    X_data = np.random.rand(n_dim, n_points)
    QA_data = np.random.rand(n_points, n_samples)
    QB_data = np.random.normal(loc=0.5, scale=0.15, size=(n_points, n_samples))

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

    # Přidání atributů s pojmenováním porovnávaných veličin
    ds["QA"].attrs["long_name"] = "Porovnávaná veličina A (např. propustnost)"
    ds["QA"].attrs["units"] = "-"
    ds["QB"].attrs["long_name"] = "Porovnávaná veličina B (např. hustota puklin)"
    ds["QB"].attrs["units"] = "-"
    ds.attrs["description"] = "Testovací data náhodných polí ze samplů na nepravidelné síti"

    stats = xr.Dataset({
        "mean_QA": ds["QA"].mean(dim="i_sample"),
        "var_QA":  ds["QA"].var(dim="i_sample"),
        "mean_QB": ds["QB"].mean(dim="i_sample"),
        "var_QB":  ds["QB"].var(dim="i_sample")
    })

    x_coords = ds["X"].sel(i_dim="x")
    y_coords = ds["X"].sel(i_dim="y")

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Porovnání polí QA a QB na neregulární síti", fontsize=16)

    sc1 = axes[0, 0].scatter(x_coords, y_coords, c=stats["mean_QA"], cmap='viridis', s=20)
    axes[0, 0].set_title("QA: Mapa průměru")
    plt.colorbar(sc1, ax=axes[0, 0])

    sc2 = axes[0, 1].scatter(x_coords, y_coords, c=stats["var_QA"], cmap='magma', s=20)
    axes[0, 1].set_title("QA: Mapa rozptylu")
    plt.colorbar(sc2, ax=axes[0, 1])

    axes[0, 2].hist(ds["QA"].values.flatten(), bins=30, color='skyblue', edgecolor='black')
    axes[0, 2].set_title("QA: Histogram všech hodnot")

    sc3 = axes[1, 0].scatter(x_coords, y_coords, c=stats["mean_QB"], cmap='viridis', s=20)
    axes[1, 0].set_title("QB: Mapa průměru")
    plt.colorbar(sc3, ax=axes[1, 0])

    sc4 = axes[1, 1].scatter(x_coords, y_coords, c=stats["var_QB"], cmap='magma', s=20)
    axes[1, 1].set_title("QB: Mapa rozptylu")
    plt.colorbar(sc4, ax=axes[1, 1])

    axes[1, 2].hist(ds["QB"].values.flatten(), bins=30, color='salmon', edgecolor='black')
    axes[1, 2].set_title("QB: Histogram všech hodnot")

    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()

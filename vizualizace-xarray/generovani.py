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
    """)
    return


@app.cell(hide_code=True)
def _():
    import xarray as xr
    import numpy as np
    import analyza

    n_points = 1000
    n_samples_A = 10
    n_samples_B = 20
    n_dim = 2

    X_data = np.random.rand(n_dim, n_points)

    # Testovací pole = 10**( np.random.normal(N, loc = -10, scale=3))
    QA_data = 10**(np.random.normal(loc=-10, scale=3, size=(n_points, n_samples_A)))

    # případně zvětšit scale pro větší rozdíl průměrů polí
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
    return analyza, ds


@app.cell(hide_code=True)
def _(ds):
    # Průměry pro každé ze dvou polí. Průměry do samostatné buňky.
    mean_QA = ds["QA"].mean(dim="i_sample_A")
    mean_QB = ds["QB"].mean(dim="i_sample_B")
    return mean_QA, mean_QB


@app.cell(hide_code=True)
def _(analyza, ds, mean_QA, mean_QB):
    grid_mean_A, x_edges, y_edges = analyza.bin_single_field(ds["X"], mean_QA)
    grid_mean_B, _, _ = analyza.bin_single_field(ds["X"], mean_QB)

    fig_means = analyza.plot_means_two_columns(x_edges, y_edges, grid_mean_A, grid_mean_B)
    fig_means
    return x_edges, y_edges


@app.cell(hide_code=True)
def _(analyza, ds):
    grid_A, _, _ = analyza.bin_all_samples(ds["X"], ds["QA"])
    grid_B, _, _ = analyza.bin_all_samples(ds["X"], ds["QB"])

    t_stat, p_value = analyza.perform_ttest(grid_A, grid_B)
    return p_value, t_stat


@app.cell(hide_code=True)
def _(analyza, p_value, t_stat, x_edges, y_edges):
    fig_test = analyza.plot_ttest_results(x_edges, y_edges, t_stat, p_value)
    fig_test
    return


@app.cell(hide_code=True)
def _(analyza, ds):
    x_vals = ds["X"].sel(i_dim="x").values
    y_vals = ds["X"].sel(i_dim="y").values
    q_a_vals = ds["QA"].values
    q_b_vals = ds["QB"].values

    fig_multiscale_maps, fig_multiscale_line = analyza.multiscale_analysis(x_vals, y_vals, q_a_vals, q_b_vals)
    return fig_multiscale_line, fig_multiscale_maps


@app.cell(hide_code=True)
def _(fig_multiscale_maps):
    fig_multiscale_maps
    return


@app.cell(hide_code=True)
def _(fig_multiscale_line):
    fig_multiscale_line
    return


if __name__ == "__main__":
    app.run()

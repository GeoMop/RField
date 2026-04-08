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

@app.cell
def _():
    import xarray as xr
    import numpy as np
    import analyza

    n_points = 10000
    n_samples_A = 10
    n_samples_B = 20
    n_dim = 2

    # Nastudovat xarray, mock data: X náhodné body na čtverci
    X_data = np.random.rand(n_dim, n_points)
    
    # Testovací pole = 10**( np.random.normal(N, loc = -10, scale=3))
    QA_data = 10**(np.random.normal(loc=-10, scale=3, size=(n_points, n_samples_A)))
    
    # případně zvětšit scale pro větší rozdíl průměrů polí
    QB_data = 10**(np.random.normal(loc=-10, scale=5, size=(n_points, n_samples_B)))

    # Definujte vstupní formát náhodných polí (knihovna xarray)
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
    
    # Přidat attribute s pojmenováním porovnávaných veličin.
    ds["QA"].attrs["long_name"] = "Porovnávaná veličina A"
    ds["QB"].attrs["long_name"] = "Porovnávaná veličina B"
    
    return analyza, ds, np, xr

@app.cell
def _(ds):
    # Průměry pro každé ze dvou polí. Průměry do samostatné buňky.
    mean_QA = ds["QA"].mean(dim="i_sample_A")
    mean_QB = ds["QB"].mean(dim="i_sample_B")
    
    var_QA = ds["QA"].var(dim="i_sample_A")
    var_QB = ds["QB"].var(dim="i_sample_B")
    return mean_QA, mean_QB, var_QA, var_QB

@app.cell
def _(analyza, ds, mean_QA, mean_QB):
    # Vytvořte grafy pro porovnání základních statistik dvou polí
    grid_mean_A, x_edges, y_edges = analyza.bin_single_field(ds["X"], mean_QA)
    grid_mean_B, _, _ = analyza.bin_single_field(ds["X"], mean_QB)
    
    fig_means = analyza.plot_means_two_columns(x_edges, y_edges, grid_mean_A, grid_mean_B)
    fig_means
    return fig_means, grid_mean_A, grid_mean_B, x_edges, y_edges

@app.cell
def _(analyza, ds, var_QA, var_QB, x_edges, y_edges):
    grid_var_A, _, _ = analyza.bin_single_field(ds["X"], var_QA)
    grid_var_B, _, _ = analyza.bin_single_field(ds["X"], var_QB)
    
    fig_vars = analyza.plot_variances_two_columns(x_edges, y_edges, grid_var_A, grid_var_B)
    fig_vars
    return fig_vars,

@app.cell
def _(analyza, ds):
    fig_hist = analyza.plot_histograms(ds["QA"].values, ds["QB"].values)
    fig_hist
    return fig_hist,

@app.cell
def _(analyza, ds):
    x_vals = ds["X"].sel(i_dim="x").values
    y_vals = ds["X"].sel(i_dim="y").values
    
    fig_counts = analyza.plot_point_counts(x_vals, y_vals)
    fig_counts
    return fig_counts, x_vals, y_vals

@app.cell
def _(analyza, ds):
    grid_A, _, _ = analyza.bin_all_samples(ds["X"], ds["QA"])
    grid_B, _, _ = analyza.bin_all_samples(ds["X"], ds["QB"])
    
    t_stat, p_value = analyza.perform_ttest(grid_A, grid_B)
    return grid_A, grid_B, p_value, t_stat

@app.cell
def _(analyza, p_value, t_stat, x_edges, y_edges):
    fig_test = analyza.plot_ttest_results(x_edges, y_edges, t_stat, p_value)
    fig_test
    return fig_test,

@app.cell
def _(mo):
    # .. zkusit vyrobit interaktivní přepínání velikosti průměrovacích buňek
    # … přepínání typu průměru
    grid_slider = mo.ui.slider(start=2, stop=30, step=2, value=10, label="Velikost buněk (interaktivní)")
    mean_dropdown = mo.ui.dropdown(options=["Aritmetický", "Geometrický", "Harmonický"], value="Aritmetický", label="Typ průměru")
    return grid_slider, mean_dropdown

@app.cell
def _(grid_slider, mean_dropdown, mo):
    ui_controls = mo.vstack([grid_slider, mean_dropdown])
    ui_controls
    return ui_controls,

@app.cell
def _(analyza, mean_QA, grid_slider, mean_dropdown, x_vals, y_vals):
    fig_interactive_mean = analyza.plot_interactive_mean(x_vals, y_vals, mean_QA.values, mean_type=mean_dropdown.value, grid_size=grid_slider.value)
    fig_interactive_mean
    return fig_interactive_mean,

@app.cell
def _(analyza, mean_QA, grid_slider, x_vals, y_vals):
    fig_geom_log = analyza.plot_geom_and_log(x_vals, y_vals, mean_QA.values, grid_size=grid_slider.value)
    fig_geom_log
    return fig_geom_log,

@app.cell
def _(analyza, ds, grid_slider, x_vals, y_vals):
    q_a_vals = ds["QA"].values
    q_b_vals = ds["QB"].values
    
    fig_wasserstein = analyza.plot_wasserstein_distance(x_vals, y_vals, q_a_vals, q_b_vals, grid_size=grid_slider.value)
    fig_wasserstein
    return fig_wasserstein, q_a_vals, q_b_vals

@app.cell
def _(analyza, q_a_vals, q_b_vals, x_vals, y_vals):
    fig_multi_maps, fig_multi_line = analyza.multiscale_analysis(x_vals, y_vals, q_a_vals, q_b_vals)
    return fig_multi_line, fig_multi_maps

@app.cell
def _(fig_multi_maps):
    fig_multi_maps
    return

@app.cell
def _(fig_multi_line):
    fig_multi_line
    return

if __name__ == "__main__":
    app.run()
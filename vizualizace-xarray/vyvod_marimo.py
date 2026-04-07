import marimo

__generated_with = "0.22.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    # Popsat strukturu vstupu (xarray) pomocí textové buňky v marimo notebooku.
    mo.md("""
    # Analýza a porovnání náhodných polí

    ### Struktura vstupu (xarray)

    Data jsou reprezentována pomocí datové struktury `xarray.Dataset`. 
    Skládají se ze dvou hlavních částí:

    **1. Dimenze a souřadnice (coords):**
    - `i_point`: Index konkrétního bodu v nepravidelné síti (2500 bodů).
    - `i_sample_A`, `i_sample_B`: Indexy vzorků (realizací) náhodného pole (A=10, B=20).
    - `i_dim`: Označení prostorové osy (např. 'x', 'y').

    **2. Datové proměnné (data_vars):**
    - `X`: Prostorové souřadnice bodů (tvar: `i_dim` × `i_point`).
    - `QA`: Hodnoty náhodného pole A (tvar: `i_point` × `i_sample_A`).
    - `QB`: Hodnoty náhodného pole B (tvar: `i_point` × `i_sample_B`).
    """)
    return


@app.cell(hide_code=True)
def _():
    import xarray as xr
    import numpy as np
    import vizualizace

    n_points = 2500 
    n_samples_A = 10
    n_samples_B = 20
    n_dim = 2

    # Nastudovat xarray, mock data: X náhodné body na čtverci, QA - náhodné hodnoty [0, 1]
    X_data = np.random.rand(n_dim, n_points)

    # Testovací pole = 10**( np.random.normal(N, loc = -10, scale=3))
    QA_data = 10**(np.random.normal(loc=-10, scale=3, size=(n_points, n_samples_A)))

    # případně zvětšit scale pro větší rozdíl průměrů polí
    QB_data = 10**(np.random.normal(loc=-10, scale=5, size=(n_points, n_samples_B)))

    # Definujte vstupní formát náhodných polí (knihovna xarray). 
    # Var: X coords:[i_dim, i_point] 
    # Var: QA coords:[i_point, i_sample]
    # Var: QB coords:[i_point, i_sample]
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

    q_a_vals = ds["QA"].values
    q_b_vals = ds["QB"].values
    x_vals = ds["X"].sel(i_dim="x").values
    y_vals = ds["X"].sel(i_dim="y").values

    return q_a_vals, q_b_vals, vizualizace, x_vals, y_vals


@app.cell(hide_code=True)
def _(mo):
    # První okno: histogramy pro A a B
    mo.md("""
    ### Histogramy pro A a B
    """)
    return


@app.cell(hide_code=True)
def _(q_a_vals, q_b_vals, vizualizace):
    fig_1 = vizualizace.plot_histograms(q_a_vals, q_b_vals)
    fig_1
    return


@app.cell(hide_code=True)
def _(mo):
    # Výběr průměrování
    # .. zkusit vyrobit interaktivní přepínání velikosti průměrovacích buňek
    # … přepínání typu průměru
    mean_dropdown = mo.ui.dropdown(options=["Aritmetický", "Geometrický", "Harmonický"], value="Aritmetický", label="Výběr průměrování")
    grid_slider = mo.ui.slider(start=2, stop=30, step=2, value=10, label="Rozlišení")

    ui_panel = mo.md(f"""
    ### Výběr průměrování a rozlišení
    {mean_dropdown}

    {grid_slider}
    """)
    return grid_slider, mean_dropdown, ui_panel


@app.cell(hide_code=True)
def _(ui_panel):
    ui_panel
    return


@app.cell(hide_code=True)
def _(mo):
    # Druhé okno: průměr vlevo pro A, průměr vpravo pro B, obojí s vybíracími prvky + zobrazit logaritmus z dat.
    mo.md("""
    ### Průměr vlevo pro A, průměr vpravo pro B, obojí s vybíracími prvky + zobrazit logaritmus z dat.
    """)
    return


@app.cell(hide_code=True)
def _(
    grid_slider,
    mean_dropdown,
    q_a_vals,
    q_b_vals,
    vizualizace,
    x_vals,
    y_vals,
):
    fig_2 = vizualizace.plot_window_2_means(x_vals, y_vals, q_a_vals, q_b_vals, mean_dropdown.value, grid_slider.value)
    fig_2
    return


@app.cell(hide_code=True)
def _(mo):
    # Třetí okno: rozptyly pro A a B
    mo.md("""
    ### Rozptyly pro A a B
    """)
    return


@app.cell(hide_code=True)
def _(grid_slider, q_a_vals, q_b_vals, vizualizace, x_vals, y_vals):
    fig_3 = vizualizace.plot_window_3_variances(x_vals, y_vals, q_a_vals, q_b_vals, grid_slider.value)
    fig_3
    return


@app.cell(hide_code=True)
def _(mo):
    # čtvrté okno: T-statistika + p-hodnota pro vybraný typ průměru + rozlišení
    mo.md("""
    ### T-statistika + p-hodnota pro vybraný typ průměru + rozlišení
    """)
    return


@app.cell(hide_code=True)
def _(
    grid_slider,
    mean_dropdown,
    q_a_vals,
    q_b_vals,
    vizualizace,
    x_vals,
    y_vals,
):
    fig_4 = vizualizace.plot_window_4_ttest(x_vals, y_vals, q_a_vals, q_b_vals, mean_dropdown.value, grid_slider.value)
    fig_4
    return


@app.cell(hide_code=True)
def _(mo):
    # Páté okno: počet dat v buňkách + Wasserstein distance
    mo.md("""
    ### Počet dat v buňkách + Wasserstein distance
    """)
    return


@app.cell(hide_code=True)
def _(grid_slider, q_a_vals, q_b_vals, vizualizace, x_vals, y_vals):
    fig_5 = vizualizace.plot_window_5_counts_wasserstein(x_vals, y_vals, q_a_vals, q_b_vals, grid_slider.value)
    fig_5
    return


@app.cell(hide_code=True)
def _(mo):
    # Bod: Vypočtěte různé druhy průměrů (aritmetický, geometrický, harmonický) na podčtvercích (multi-scale) pro různé velikosti oken.
    mo.md("""
    ### Multi-scale analýza (závislost na velikosti oken)
    """)
    return


@app.cell(hide_code=True)
def _(q_a_vals, q_b_vals, vizualizace, x_vals, y_vals):
    fig_multi_maps, fig_multi_line = vizualizace.plot_multiscale_analysis(x_vals, y_vals, q_a_vals, q_b_vals)
    return fig_multi_line, fig_multi_maps


@app.cell(hide_code=True)
def _(fig_multi_maps):
    fig_multi_maps
    return


@app.cell(hide_code=True)
def _(fig_multi_line):
    fig_multi_line
    return


if __name__ == "__main__":
    app.run()

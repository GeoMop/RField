import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d, ttest_ind, wasserstein_distance
from typing import Tuple, Any
import xarray as xr
import pandas as pd
import hvplot.xarray
import hvplot.pandas
import holoviews as hv

# Zapnutí interaktivního režimu pro hvplot grafy (lupa, posun, uložení)
hv.extension('bokeh')

import warnings
warnings.filterwarnings('ignore')

# Refaktorovat kód -> výpočty a složitější grafy do samostatného modulu/ modulů.

class MeanCalculator:
    # Modularize - create a class representing a "mean"
    
    @staticmethod
    def arithmetic(q: np.ndarray) -> float:
        # aritmetický průměr : A_1(q) = sum_i (q_i) / N
        return float(np.mean(q))

    @staticmethod
    def geometric(q: np.ndarray) -> float:
        # geometrický průměr : A_log(q) = exp( (sum_i log(q_i)) / N )  = (prod_i q_i)**(1/N)
        # Omezení hodnot zdola (1e-10), aby se předešlo chybě logaritmu z nuly
        safe_q = np.clip(q, 1e-10, None)
        return float(np.exp(np.mean(np.log(safe_q))))

    @staticmethod
    def harmonic(q: np.ndarray) -> float:
        # Harmonický průměr: A_inv(q) = (( sum_i q_i**(-1)) /N)**(-1)
        # Omezení hodnot zdola, aby se předešlo dělení nulou
        safe_q = np.clip(q, 1e-10, None)
        return float(1.0 / np.mean(1.0 / safe_q))


def create_binned_grid(x: np.ndarray, y: np.ndarray, values: np.ndarray, num_bins: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    x - field A, has shape (N,N)
    Přidat k hlavním funkcím doc string s popisem vstupních parametrů.
    """
    bins = (num_bins, num_bins)
    
    # Seskupení náhodných bodů do 2D mřížky (binning) s výpočtem průměru
    binned = binned_statistic_2d(x, y, values, statistic='mean', bins=bins)
    
    # Compute mean of whole array and use it as default value.
    stat = binned.statistic
    default_val = np.nanmean(values)
    
    # Vyplnění prázdných čtverců (kde nebyly žádné body a vzniklo NaN) celkovým průměrem
    stat = np.where(np.isnan(stat), default_val, stat)
    
    return stat, binned.x_edge, binned.y_edge

def bin_single_field(X_coords: Any, data_var: Any, num_bins: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    x - field A, has shape (N,N)
    """
    x = X_coords.sel(i_dim="x").values
    y = X_coords.sel(i_dim="y").values
    grid, x_edges, y_edges = create_binned_grid(x, y, data_var.values, num_bins)
    return grid, x_edges, y_edges

def bin_all_samples(X_coords: Any, data_var: Any, num_bins: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    x - field A, has shape (N,N)
    """
    x = X_coords.sel(i_dim="x").values
    y = X_coords.sel(i_dim="y").values
    n_samples = data_var.shape[1]
    
    # Vytvoření 3D pole pro uložení binningu všech vzorků (samples)
    grid_3d = np.zeros((num_bins, num_bins, n_samples))
    for i in range(n_samples):
        grid_3d[:, :, i], x_edges, y_edges = create_binned_grid(x, y, data_var.values[:, i], num_bins)
        
    return grid_3d, x_edges, y_edges

def plot_means_two_columns(x_edges: np.ndarray, y_edges: np.ndarray, grid_mean_A: np.ndarray, grid_mean_B: np.ndarray):
    # Dva sloupce, vlevo field A, vpravo field B
    # Pro grafy průměru a rozptylu použít imgshow nebo podobnou funkci
    da_A = xr.DataArray(grid_mean_A.T, coords=[('y', y_edges[:-1]), ('x', x_edges[:-1])], name='Průměr_A')
    da_B = xr.DataArray(grid_mean_B.T, coords=[('y', y_edges[:-1]), ('x', x_edges[:-1])], name='Průměr_B')
    
    plot1 = da_A.hvplot.image(x='x', y='y', cmap='viridis', title="Průměr pole A", width=400, height=350)
    plot2 = da_B.hvplot.image(x='x', y='y', cmap='viridis', title="Průměr pole B", width=400, height=350)
    
    return plot1 + plot2

def plot_variances_two_columns(x_edges: np.ndarray, y_edges: np.ndarray, grid_var_A: np.ndarray, grid_var_B: np.ndarray):
    # mapa průměru a rozptylu pole (řezu)
    da_A = xr.DataArray(grid_var_A.T, coords=[('y', y_edges[:-1]), ('x', x_edges[:-1])], name='Rozptyl_A')
    da_B = xr.DataArray(grid_var_B.T, coords=[('y', y_edges[:-1]), ('x', x_edges[:-1])], name='Rozptyl_B')
    
    plot1 = da_A.hvplot.image(x='x', y='y', cmap='magma', title="Rozptyl pole A", width=400, height=350)
    plot2 = da_B.hvplot.image(x='x', y='y', cmap='magma', title="Rozptyl pole B", width=400, height=350)
    
    return plot1 + plot2

def plot_histograms(data_A: np.ndarray, data_B: np.ndarray):
    # histogram
    df_A = pd.DataFrame({'QA': data_A.flatten()})
    df_B = pd.DataFrame({'QB': data_B.flatten()})
    
    plot1 = df_A.hvplot.hist('QA', bins=30, color='skyblue', title="Histogram pole A", width=400, height=300)
    plot2 = df_B.hvplot.hist('QB', bins=30, color='salmon', title="Histogram pole B", width=400, height=300)
    
    return plot1 + plot2

def plot_point_counts(x: np.ndarray, y: np.ndarray, num_bins: int = 20):
    # Přidat zobrazení počtu vstupních bodů v masce.
    bins = (num_bins, num_bins)
    
    # Výpočet počtu bodů spadajících do každé buňky (statistic='count')
    binned = binned_statistic_2d(x, y, None, statistic='count', bins=bins)
    
    da_c = xr.DataArray(binned.statistic.T, coords=[('y', binned.y_edge[:-1]), ('x', binned.x_edge[:-1])], name='Počet_bodů')
    plot1 = da_c.hvplot.image(x='x', y='y', cmap='plasma', title="Počet vstupních bodů v masce", width=500, height=400)
    return plot1

def perform_ttest(grid_A: np.ndarray, grid_B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Buňka 3 : statistický test (Two-sample t-tests , ) pro odpovídající dvojice pixelů průměru pro více polí ve skupině A a skupině B, může být různý počet. Viz. scipy.stats.testt_ind; a.shape = (nx, ny, 10) b.shape = (nx,ny, 20), axis=-1, equal_var=False, alternative='two-sided'
    # ? test_ind s ignorováním NaN nebo pro maskovaná pole. np.masked_array
    
    # nan_policy='omit' umožňuje ignorovat hodnoty NaN při výpočtu statistiky
    t_stat, p_value = ttest_ind(grid_A, grid_B, axis=-1, equal_var=False, alternative='two-sided', nan_policy='omit')
    return t_stat, p_value

def plot_ttest_results(x_edges: np.ndarray, y_edges: np.ndarray, t_stat: np.ndarray, p_value: np.ndarray):
    # Grafy pro T (attribute statistics) a p-value.
    da_t = xr.DataArray(t_stat.T, coords=[('y', y_edges[:-1]), ('x', x_edges[:-1])], name='T_statistika')
    da_p = xr.DataArray(p_value.T, coords=[('y', y_edges[:-1]), ('x', x_edges[:-1])], name='p_hodnota')
    
    plot1 = da_t.hvplot.image(x='x', y='y', cmap='coolwarm', title="T-statistika", width=400, height=350)
    
    # Range pro p-value (0, 1). Nastavení limitů barev přes clim
    plot2 = da_p.hvplot.image(x='x', y='y', cmap='RdYlGn', title="p-hodnota", width=400, height=350, clim=(0, 1))
    
    return plot1 + plot2

def plot_interactive_mean(x: np.ndarray, y: np.ndarray, q_values: np.ndarray, mean_type: str, grid_size: int = 10):
    # … přepínání typu průměru
    # Compute mean of whole array and use it as default value.
    if mean_type == "Aritmetický": default_val = MeanCalculator.arithmetic(q_values)
    elif mean_type == "Geometrický": default_val = MeanCalculator.geometric(q_values)
    else: default_val = MeanCalculator.harmonic(q_values)

    A_mean = np.full((grid_size, grid_size), default_val)
    dx = 1.0 / grid_size
    dy = 1.0 / grid_size

    for i in range(grid_size):
        for j in range(grid_size):
            # Booleovská maska pro výběr bodů, které spadají přesně do aktuálního čtverce (okna)
            mask = (x >= i*dx) & (x < (i+1)*dx) & (y >= j*dy) & (y < (j+1)*dy)
            q_in_window = q_values[mask]

            # Výpočet vybraného typu průměru pouze pokud jsou ve čtverci body
            if len(q_in_window) > 0:
                if mean_type == "Aritmetický":
                    A_mean[j, i] = MeanCalculator.arithmetic(q_in_window)
                elif mean_type == "Geometrický":
                    A_mean[j, i] = MeanCalculator.geometric(q_in_window)
                elif mean_type == "Harmonický":
                    A_mean[j, i] = MeanCalculator.harmonic(q_in_window)

    da = xr.DataArray(A_mean, coords=[('y', np.arange(grid_size)), ('x', np.arange(grid_size))], name=mean_type)
    return da.hvplot.image(x='x', y='y', cmap='viridis', title=f"Typ průměru: {mean_type}", width=500, height=400)

def plot_wasserstein_distance(x: np.ndarray, y: np.ndarray, q_a_vals: np.ndarray, q_b_vals: np.ndarray, grid_size: int = 10):
    # 1D samples distance (even for different sizes): scipy.stats.wasserstein_distance; use p=1, where fast sweep algorithm is possible. Numerical comparison of tho set of samples within a single cell.
    # Compute mean of whole array and use it as default value.
    
    # Výpočet Wassersteinovy vzdálenosti (Earth Mover's Distance)
    default_val = wasserstein_distance(q_a_vals.flatten(), q_b_vals.flatten())
    w_map = np.full((grid_size, grid_size), default_val)
    dx = 1.0 / grid_size
    dy = 1.0 / grid_size

    for i in range(grid_size):
        for j in range(grid_size):
            mask = (x >= i*dx) & (x < (i+1)*dx) & (y >= j*dy) & (y < (j+1)*dy)

            if np.sum(mask) > 0:
                qa_w = q_a_vals[mask, :].flatten()
                qb_w = q_b_vals[mask, :].flatten()
                w_map[j, i] = wasserstein_distance(qa_w, qb_w)

    da = xr.DataArray(w_map, coords=[('y', np.arange(grid_size)), ('x', np.arange(grid_size))], name='Wasserstein')
    return da.hvplot.image(x='x', y='y', cmap='plasma', title="Wasserstein distance", width=500, height=400)

def plot_geom_and_log(x: np.ndarray, y: np.ndarray, q_values: np.ndarray, grid_size: int = 10):
    # Porovnání průměrů -> geometrický průměr + logaritmus průměru do grafu
    # Compute mean of whole array and use it as default value.
    default_geom = MeanCalculator.geometric(q_values)
    default_log = np.log(MeanCalculator.arithmetic(q_values) + 1e-10)
    
    A_geom = np.full((grid_size, grid_size), default_geom)
    A_log = np.full((grid_size, grid_size), default_log)

    dx = 1.0 / grid_size
    dy = 1.0 / grid_size

    for i in range(grid_size):
        for j in range(grid_size):
            mask = (x >= i*dx) & (x < (i+1)*dx) & (y >= j*dy) & (y < (j+1)*dy)
            q_in_window = q_values[mask]

            if len(q_in_window) > 0:
                A_geom[j, i] = MeanCalculator.geometric(q_in_window)
                A_log[j, i] = np.log(MeanCalculator.arithmetic(q_in_window) + 1e-10)

    da_geom = xr.DataArray(A_geom, coords=[('y', np.arange(grid_size)), ('x', np.arange(grid_size))], name='Geometricky')
    da_log = xr.DataArray(A_log, coords=[('y', np.arange(grid_size)), ('x', np.arange(grid_size))], name='Logaritmus')
    
    plot1 = da_geom.hvplot.image(x='x', y='y', cmap='viridis', title="Geometrický průměr", width=400, height=350)
    plot2 = da_log.hvplot.image(x='x', y='y', cmap='viridis', title="Logaritmus průměru", width=400, height=350)
    
    return plot1 + plot2

def multiscale_analysis(x: np.ndarray, y: np.ndarray, q_a_vals: np.ndarray, q_b_vals: np.ndarray) -> Tuple[plt.Figure, plt.Figure]:
    grid_sizes = [2, 4, 8]
    p_values_all = {'arith': [], 'geom': [], 'harm': []}

    fig1, axes1 = plt.subplots(len(grid_sizes), 3, figsize=(15, 5 * len(grid_sizes)))
    
    # Compute mean of whole array and use it as default value.
    default_p_val = 1.0

    for idx, gs in enumerate(grid_sizes):
        dx = 1.0 / gs
        dy = 1.0 / gs
        
        p_map_arith = np.full((gs, gs), default_p_val)
        p_map_geom = np.full((gs, gs), default_p_val)
        p_map_harm = np.full((gs, gs), default_p_val)

        for i in range(gs):
            for j in range(gs):
                mask = (x >= i*dx) & (x < (i+1)*dx) & (y >= j*dy) & (y < (j+1)*dy)
                
                if np.sum(mask) > 0:
                    qa_w = q_a_vals[mask, :]
                    qb_w = q_b_vals[mask, :]
                    
                    qa_arith = np.mean(qa_w, axis=0)
                    qb_arith = np.mean(qb_w, axis=0)
                    
                    qa_w_safe = np.clip(qa_w, 1e-10, None)
                    qb_w_safe = np.clip(qb_w, 1e-10, None)
                    
                    qa_geom = np.exp(np.mean(np.log(qa_w_safe), axis=0))
                    qb_geom = np.exp(np.mean(np.log(qb_w_safe), axis=0))
                    
                    qa_harm = 1.0 / np.mean(1.0 / qa_w_safe, axis=0)
                    qb_harm = 1.0 / np.mean(1.0 / qb_w_safe, axis=0)
                    
                    _, p_a = ttest_ind(qa_arith, qb_arith, equal_var=False)
                    _, p_g = ttest_ind(qa_geom, qb_geom, equal_var=False)
                    _, p_h = ttest_ind(qa_harm, qb_harm, equal_var=False)
                    
                    p_map_arith[j, i] = p_a
                    p_map_geom[j, i] = p_g
                    p_map_harm[j, i] = p_h
                    
        p_values_all['arith'].append(np.nanmean(p_map_arith))
        p_values_all['geom'].append(np.nanmean(p_map_geom))
        p_values_all['harm'].append(np.nanmean(p_map_harm))
        
        # Vykreslení multi-scale map pro každou velikost mřížky
        im_a = axes1[idx, 0].imshow(p_map_arith, origin='lower', extent=[0, 1, 0, 1], cmap='RdYlGn', vmin=0, vmax=1)
        axes1[idx, 0].set_ylabel(f"Mřížka {gs}x{gs}")
        if idx == 0: axes1[idx, 0].set_title("Aritmetický")
        plt.colorbar(im_a, ax=axes1[idx, 0])
        
        im_g = axes1[idx, 1].imshow(p_map_geom, origin='lower', extent=[0, 1, 0, 1], cmap='RdYlGn', vmin=0, vmax=1)
        if idx == 0: axes1[idx, 1].set_title("Geometrický")
        plt.colorbar(im_g, ax=axes1[idx, 1])
        
        im_h = axes1[idx, 2].imshow(p_map_harm, origin='lower', extent=[0, 1, 0, 1], cmap='RdYlGn', vmin=0, vmax=1)
        if idx == 0: axes1[idx, 2].set_title("Harmonický")
        plt.colorbar(im_h, ax=axes1[idx, 2])

    plt.tight_layout()
    
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(grid_sizes, p_values_all['arith'], marker='o', label='Aritmetický')
    ax2.plot(grid_sizes, p_values_all['geom'], marker='s', label='Geometrický')
    ax2.plot(grid_sizes, p_values_all['harm'], marker='^', label='Harmonický')
    ax2.set_xticks(grid_sizes)
    plt.legend()
    
    return fig1, fig2
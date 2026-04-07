import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, wasserstein_distance
import xarray as xr
import pandas as pd
import hvplot.pandas
import hvplot.xarray
import holoviews as hv

from vypocty import MeanCalculator, get_mean_func

hv.extension('bokeh')

import warnings
warnings.filterwarnings('ignore')

def plot_histograms(q_a: np.ndarray, q_b: np.ndarray):
    """
    Co to je: Vykreslení histogramů (První okno).
    Typ výstupu: HoloViews/hvplot objekt.
    S čím pracuje: Zplošťuje 2D pole hodnot na 1D a vytváří pandas DataFrame.
    Kde bere data: Z argumentů 'q_a' a 'q_b', které sem pošle hlavní soubor 'vyvod_marimo.py'.
    """
    df_A = pd.DataFrame({'QA': q_a.flatten()})
    df_B = pd.DataFrame({'QB': q_b.flatten()})
    
    p1 = df_A.hvplot.hist('QA', bins=30, color='skyblue', title="Histogram A", width=400, height=300)
    p2 = df_B.hvplot.hist('QB', bins=30, color='salmon', title="Histogram B", width=400, height=300)
    return p1 + p2

def plot_window_2_means(x: np.ndarray, y: np.ndarray, qa: np.ndarray, qb: np.ndarray, mean_type: str, grid_size: int):
    """
    Co to je: Výpočet a vykreslení 2D map průměrů a logaritmů (Druhé okno).
    Typ výstupu: HoloViews/hvplot layout (4 grafy vedle sebe).
    S čím pracuje: Filtruje prostorová data (x, y) do mřížky podle 'grid_size' a aplikuje 'mean_type'.
    Kde bere data: Koordináty a hodnoty přicházejí z 'vyvod_marimo.py' přes parametry funkce.
    """
    mean_func = get_mean_func(mean_type)
    A_mean, B_mean = np.full((grid_size, grid_size), np.nan), np.full((grid_size, grid_size), np.nan)
    A_log, B_log = np.full((grid_size, grid_size), np.nan), np.full((grid_size, grid_size), np.nan)
    
    dx, dy = 1.0 / grid_size, 1.0 / grid_size
    
    for i in range(grid_size):
        for j in range(grid_size):
            mask = (x >= i*dx) & (x < (i+1)*dx) & (y >= j*dy) & (y < (j+1)*dy)
            qa_w, qb_w = qa[mask], qb[mask]
            
            if len(qa_w) > 0:
                A_mean[j, i] = mean_func(qa_w)
                B_mean[j, i] = mean_func(qb_w)
                A_log[j, i] = np.log10(np.clip(MeanCalculator.arithmetic(qa_w), 1e-10, None))
                B_log[j, i] = np.log10(np.clip(MeanCalculator.arithmetic(qb_w), 1e-10, None))

    A_mean = np.where(np.isnan(A_mean), np.nanmean(A_mean), A_mean)
    B_mean = np.where(np.isnan(B_mean), np.nanmean(B_mean), B_mean)
    A_log = np.where(np.isnan(A_log), np.nanmean(A_log), A_log)
    B_log = np.where(np.isnan(B_log), np.nanmean(B_log), B_log)

    coords = [('y', np.arange(grid_size)), ('x', np.arange(grid_size))]
    da_A = xr.DataArray(A_mean, coords=coords, name='Průměr_A')
    da_B = xr.DataArray(B_mean, coords=coords, name='Průměr_B')
    da_A_log = xr.DataArray(A_log, coords=coords, name='Log_A')
    da_B_log = xr.DataArray(B_log, coords=coords, name='Log_B')
    
    p1 = da_A.hvplot.image(x='x', y='y', cmap='viridis', title=f"A: {mean_type}", width=350, height=300)
    p2 = da_B.hvplot.image(x='x', y='y', cmap='viridis', title=f"B: {mean_type}", width=350, height=300)
    p3 = da_A_log.hvplot.image(x='x', y='y', cmap='plasma', title="Logaritmus (A)", width=350, height=300)
    p4 = da_B_log.hvplot.image(x='x', y='y', cmap='plasma', title="Logaritmus (B)", width=350, height=300)
    
    return (p1 + p2 + p3 + p4).cols(2)

def plot_window_3_variances(x: np.ndarray, y: np.ndarray, qa: np.ndarray, qb: np.ndarray, grid_size: int):
    """
    Co to je: Výpočet a vykreslení rozptylu (Třetí okno).
    Typ výstupu: HoloViews/hvplot layout.
    S čím pracuje: Rozděluje data do mřížky a počítá numpy.var().
    Kde bere data: Hodnoty a pozice (x, y, qa, qb) dodává hlavní soubor.
    """
    A_var, B_var = np.full((grid_size, grid_size), np.nan), np.full((grid_size, grid_size), np.nan)
    dx, dy = 1.0 / grid_size, 1.0 / grid_size
    
    for i in range(grid_size):
        for j in range(grid_size):
            mask = (x >= i*dx) & (x < (i+1)*dx) & (y >= j*dy) & (y < (j+1)*dy)
            qa_w, qb_w = qa[mask], qb[mask]
            
            if len(qa_w) > 0:
                A_var[j, i] = np.var(qa_w)
                B_var[j, i] = np.var(qb_w)

    A_var = np.where(np.isnan(A_var), np.nanmean(A_var), A_var)
    B_var = np.where(np.isnan(B_var), np.nanmean(B_var), B_var)

    coords = [('y', np.arange(grid_size)), ('x', np.arange(grid_size))]
    da_A = xr.DataArray(A_var, coords=coords, name='Rozptyl_A')
    da_B = xr.DataArray(B_var, coords=coords, name='Rozptyl_B')
    
    p1 = da_A.hvplot.image(x='x', y='y', cmap='magma', title="Rozptyl A", width=400, height=300)
    p2 = da_B.hvplot.image(x='x', y='y', cmap='magma', title="Rozptyl B", width=400, height=300)
    return p1 + p2

def plot_window_4_ttest(x: np.ndarray, y: np.ndarray, qa: np.ndarray, qb: np.ndarray, mean_type: str, grid_size: int):
    """
    Co to je: Výpočet T-statistiky a P-hodnoty (Čtvrté okno).
    Typ výstupu: HoloViews/hvplot layout s omezením barvy clim=(0, 1).
    S čím pracuje: Porovnává distribuce vzorků pomocí scipy.stats.ttest_ind.
    Kde bere data: Základní data přijdou z parametrů, statistiku si funkce počítá sama nad těmito daty.
    """
    mean_func = get_mean_func(mean_type)
    t_map, p_map = np.full((grid_size, grid_size), np.nan), np.full((grid_size, grid_size), np.nan)
    dx, dy = 1.0 / grid_size, 1.0 / grid_size
    
    for i in range(grid_size):
        for j in range(grid_size):
            mask = (x >= i*dx) & (x < (i+1)*dx) & (y >= j*dy) & (y < (j+1)*dy)
            if np.sum(mask) > 0:
                qa_w, qb_w = qa[mask, :], qb[mask, :]
                qa_m, qb_m = mean_func(qa_w, axis=0), mean_func(qb_w, axis=0)
                
                t_s, p_v = ttest_ind(qa_m, qb_m, equal_var=False, nan_policy='omit')
                t_map[j, i] = t_s
                p_map[j, i] = p_v

    t_map = np.where(np.isnan(t_map), np.nanmean(t_map), t_map)
    p_map = np.where(np.isnan(p_map), 1.0, p_map)

    coords = [('y', np.arange(grid_size)), ('x', np.arange(grid_size))]
    da_t = xr.DataArray(t_map, coords=coords, name='T-statistika')
    da_p = xr.DataArray(p_map, coords=coords, name='p-hodnota')
    
    p1 = da_t.hvplot.image(x='x', y='y', cmap='coolwarm', title=f"T-statistika ({mean_type})", width=400, height=300)
    p2 = da_p.hvplot.image(x='x', y='y', cmap='RdYlGn', title=f"p-hodnota ({mean_type})", clim=(0, 1), width=400, height=300)
    return p1 + p2

def plot_window_5_counts_wasserstein(x: np.ndarray, y: np.ndarray, qa: np.ndarray, qb: np.ndarray, grid_size: int):
    """
    Co to je: Výpočet hustoty bodů a Wassersteinovy vzdálenosti (Páté okno).
    Typ výstupu: HoloViews/hvplot layout.
    S čím pracuje: scipy.stats.wasserstein_distance pro 1D vzorky uvnitř buňky.
    Kde bere data: Parametry obsahují pole z hlavního generátoru dat.
    """
    counts, w_map = np.full((grid_size, grid_size), 0.0), np.full((grid_size, grid_size), np.nan)
    dx, dy = 1.0 / grid_size, 1.0 / grid_size
    
    for i in range(grid_size):
        for j in range(grid_size):
            mask = (x >= i*dx) & (x < (i+1)*dx) & (y >= j*dy) & (y < (j+1)*dy)
            pts = np.sum(mask)
            counts[j, i] = pts
            
            if pts > 0:
                qa_w, qb_w = qa[mask, :].flatten(), qb[mask, :].flatten()
                w_map[j, i] = wasserstein_distance(qa_w, qb_w)

    w_map = np.where(np.isnan(w_map), np.nanmean(w_map), w_map)

    coords = [('y', np.arange(grid_size)), ('x', np.arange(grid_size))]
    da_c = xr.DataArray(counts, coords=coords, name='Počet_bodů')
    da_w = xr.DataArray(w_map, coords=coords, name='Wasserstein')
    
    p1 = da_c.hvplot.image(x='x', y='y', cmap='plasma', title="Počet dat v buňkách", width=400, height=300)
    p2 = da_w.hvplot.image(x='x', y='y', cmap='plasma', title="Wasserstein distance", width=400, height=300)
    return p1 + p2

def plot_multiscale_analysis(x: np.ndarray, y: np.ndarray, qa: np.ndarray, qb: np.ndarray):
    """
    Co to je: Vykreslí multi-scale závislosti pro pevné velikosti oken (Šesté okno).
    Typ výstupu: Matplotlib Figure (dva objekty: mapy a čárový graf).
    S čím pracuje: Testuje fixní velikosti mřížky [2, 4, 8].
    Kde bere data: Ze zadaných parametrů funkce (QA a QB data).
    """
    grid_sizes = [2, 4, 8]
    p_values_all = {'arith': [], 'geom': [], 'harm': []}

    fig1, axes1 = plt.subplots(len(grid_sizes), 3, figsize=(15, 5 * len(grid_sizes)))
    default_p_val = 1.0

    for idx, gs in enumerate(grid_sizes):
        dx, dy = 1.0 / gs, 1.0 / gs
        p_map_arith, p_map_geom, p_map_harm = np.full((gs, gs), default_p_val), np.full((gs, gs), default_p_val), np.full((gs, gs), default_p_val)

        for i in range(gs):
            for j in range(gs):
                mask = (x >= i*dx) & (x < (i+1)*dx) & (y >= j*dy) & (y < (j+1)*dy)
                
                if np.sum(mask) > 0:
                    qa_w, qb_w = qa[mask, :], qb[mask, :]
                    qa_arith, qb_arith = np.mean(qa_w, axis=0), np.mean(qb_w, axis=0)
                    
                    qa_w_safe, qb_w_safe = np.clip(qa_w, 1e-10, None), np.clip(qb_w, 1e-10, None)
                    qa_geom, qb_geom = np.exp(np.mean(np.log(qa_w_safe), axis=0)), np.exp(np.mean(np.log(qb_w_safe), axis=0))
                    qa_harm, qb_harm = 1.0 / np.mean(1.0 / qa_w_safe, axis=0), 1.0 / np.mean(1.0 / qb_w_safe, axis=0)
                    
                    _, p_a = ttest_ind(qa_arith, qb_arith, equal_var=False, nan_policy='omit')
                    _, p_g = ttest_ind(qa_geom, qb_geom, equal_var=False, nan_policy='omit')
                    _, p_h = ttest_ind(qa_harm, qb_harm, equal_var=False, nan_policy='omit')
                    
                    p_map_arith[j, i] = p_a
                    p_map_geom[j, i] = p_g
                    p_map_harm[j, i] = p_h
                    
        p_values_all['arith'].append(np.nanmean(p_map_arith))
        p_values_all['geom'].append(np.nanmean(p_map_geom))
        p_values_all['harm'].append(np.nanmean(p_map_harm))
        
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
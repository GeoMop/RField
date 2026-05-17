import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d, ttest_ind
from scipy.interpolate import griddata
import pywt
import warnings
warnings.filterwarnings('ignore')

def create_binned_grid(x, y, values, num_bins=20):
    bins = (num_bins, num_bins)
    binned = binned_statistic_2d(x, y, values, statistic='mean', bins=bins)
    return binned.statistic, binned.x_edge, binned.y_edge

def bin_single_field(X_coords, data_var, num_bins=20):
    x = X_coords.sel(i_dim="x").values
    y = X_coords.sel(i_dim="y").values
    grid, x_edges, y_edges = create_binned_grid(x, y, data_var.values, num_bins)
    return grid, x_edges, y_edges

def bin_all_samples(X_coords, data_var, num_bins=20):
    x = X_coords.sel(i_dim="x").values
    y = X_coords.sel(i_dim="y").values
    n_samples = data_var.shape[1]
    
    grid_3d = np.zeros((num_bins, num_bins, n_samples))
    for i in range(n_samples):
        grid_3d[:, :, i], x_edges, y_edges = create_binned_grid(x, y, data_var.values[:, i], num_bins)
        
    return grid_3d, x_edges, y_edges

def plot_means_two_columns(x_edges, y_edges, grid_mean_A, grid_mean_B):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    im1 = axes[0].pcolormesh(x_edges, y_edges, grid_mean_A.T, cmap='viridis', shading='flat')
    axes[0].set_title("Průměr pole A")
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].pcolormesh(x_edges, y_edges, grid_mean_B.T, cmap='viridis', shading='flat')
    axes[1].set_title("Průměr pole B")
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    return fig

def perform_ttest(grid_A, grid_B):
    t_stat, p_value = ttest_ind(grid_A, grid_B, axis=-1, equal_var=False, alternative='two-sided')
    return t_stat, p_value

def plot_ttest_results(x_edges, y_edges, t_stat, p_value):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    im1 = axes[0].pcolormesh(x_edges, y_edges, t_stat.T, cmap='coolwarm', shading='flat')
    axes[0].set_title("T-statistika")
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].pcolormesh(x_edges, y_edges, p_value.T, cmap='RdYlGn', shading='flat', vmin=0, vmax=0.1)
    axes[1].set_title("p-hodnota")
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    return fig

def multiscale_analysis(x, y, q_a_vals, q_b_vals):
    grid_sizes = [2, 4, 8]
    p_values_all = {'arith': [], 'geom': [], 'harm': []}

    fig1, axes1 = plt.subplots(len(grid_sizes), 3, figsize=(15, 5 * len(grid_sizes)))

    for idx, gs in enumerate(grid_sizes):
        dx = 1.0 / gs
        dy = 1.0 / gs
        
        p_map_arith = np.full((gs, gs), np.nan)
        p_map_geom = np.full((gs, gs), np.nan)
        p_map_harm = np.full((gs, gs), np.nan)

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
    ax2.set_xlabel("Velikost okna (počet dělení)")
    ax2.set_ylabel("Průměrná p-hodnota")
    ax2.set_title("Závislost p-hodnoty na velikosti okna")
    plt.legend()
    
    return fig1, fig2


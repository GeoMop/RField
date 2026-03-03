import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# --- Konfigurační parametry ---
n_points = 300   # Počet bodů v prostoru
n_samples = 100  # Počet realizací (vzorků)
n_dim = 2        # Rozměrnost prostoru (x, y)

# --- Generování syntetických dat ---
# Náhodné souřadnice bodů v jednotkovém čtverci [0, 1] x [0, 1]
X_data = np.random.rand(n_dim, n_points)

# Pole QA: generováno z rovnoměrného rozdělení [0, 1]
QA_data = np.random.rand(n_points, n_samples)

# Pole QB: generováno z normálního rozdělení (μ=0.5, σ=0.15)
QB_data = np.random.normal(loc=0.5, scale=0.15, size=(n_points, n_samples))

# --- Vytvoření struktury xarray Dataset ---
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

# --- Statistická analýza ---
# Výpočet průměru a rozptylu podél dimenze realizací (i_sample)
stats = xr.Dataset({
    "mean_QA": ds["QA"].mean(dim="i_sample"),
    "var_QA":  ds["QA"].var(dim="i_sample"),
    "mean_QB": ds["QB"].mean(dim="i_sample"),
    "var_QB":  ds["QB"].var(dim="i_sample")
})

# Extraxe souřadnic pro vizualizaci
x_coords = ds["X"].sel(i_dim="x")
y_coords = ds["X"].sel(i_dim="y")

# --- Vizualizace ---
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Porovnání polí QA a QB na neregulární síti", fontsize=16)

# Horní řádek: Analýza pole QA
sc1 = axes[0, 0].scatter(x_coords, y_coords, c=stats["mean_QA"], cmap='viridis', s=20)
axes[0, 0].set_title("QA: Mapa průměru")
plt.colorbar(sc1, ax=axes[0, 0])

sc2 = axes[0, 1].scatter(x_coords, y_coords, c=stats["var_QA"], cmap='magma', s=20)
axes[0, 1].set_title("QA: Mapa rozptylu")
plt.colorbar(sc2, ax=axes[0, 1])

axes[0, 2].hist(ds["QA"].values.flatten(), bins=30, color='skyblue', edgecolor='black')
axes[0, 2].set_title("QA: Histogram všech hodnot")

# Spodní řádek: Analýza pole QB
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
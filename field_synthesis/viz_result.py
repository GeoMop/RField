import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import os

# --- Налаштування для коректного відображення вікна на Windows ---
import matplotlib
matplotlib.use('TkAgg') # Примусово використовуємо Tkinter інтерфейс
# ---------------------------------------------------------------

def visualize_full_3d_synthesis(zarr_path="synthesis_results.zarr"):
    """
    Načte vygenerovaná data ze Zarr a zobrazí kompletní mřížku 3x3 2D řezů (XY)
    podél osy Z, čímž vizualizuje celou 3D strukturu masivu.
    """
    print(f"Otevírám Zarr dataset pro kompletní 3D vizualizaci: {zarr_path}")
    
    if not os.path.exists(zarr_path):
        print(f"Chyba: Soubor '{zarr_path}' nebyl nalezen. Nejdříve spusťte main.py.")
        return

    # 1. Načtení dat pomocí xarray
    try:
        ds = xr.open_zarr(zarr_path)
    except Exception as e:
        print(f"Chyba při otevírání Zarr: {e}")
        return

    print("Data úspěšně načtena.")
    
    # 2. Výběr první realizace (field_idx=0)
    # mixed_fields má tvar [počet_polí, počet_bodů] -> [2000, 729]
    field_flat = ds['mixed_fields'].isel(field_idx=0).values

    # 3. Definice rozměrů mřížky (9x9x9 = 729)
    nx, ny, nz = 9, 9, 9

    if field_flat.shape[0] != (nx * ny * nz):
        print(f"Chyba: Tvar dat ({field_flat.shape[0]}) neodpovídá mřížce 9x9x9.")
        return

    # 4. Reshape (přeměna) 'placatého' pole zpět do 3D objemu [Z, Y, X]
    volume = field_flat.reshape((nz, ny, nx))

    print(f"Vytvářím mřížku grafů 3x3 pro zobrazení všech {nz} řezů подél osy Z.")

    # 5. Příprava plochy pro vykreslení (Grid 3x3 subplotů)
    # sharex, sharey zajistí stejná měřítka na osách všech grafů
    fig, axes = plt.subplots(3, 3, figsize=(12, 12), sharex=True, sharey=True)
    fig.suptitle('Syntetizovaný 3D masiv: XY řezy podél výšky Z (0 až 8)', fontsize=16, fontweight='bold')

    # Налаштуємо єдину барвну шкалу для всіх графіків для коректного порівняння
    # Використовуємо мінімум і максимум всього об'єму
    vmin, vmax = np.nanmin(volume), np.nanmax(volume)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = 'jet' # 'jet' добре показує контраст між зонами порід

    # Створимо індекси для сітки (для pcolormesh, щоб було як на вашому скріншоті)
    x_idx = np.arange(nx)
    y_idx = np.arange(ny)
    xx, yy = np.meshgrid(x_idx, y_idx)

    # 6. Smyčka přes všechny vrstvy (slices) osy Z
    for z in range(nz):
        # Výpočet pozice в gridu 3x3
        row = z // 3
        col = z % 3
        ax = axes[row, col]

        # Výběr XY řezu pro aktuální Z
        slice_data = volume[z, :, :]

        # Vykreslení за допомогою pcolormesh (як на image_1.png)
        # edgecolors і linewidth додають тонку чорну сітку навколо 'пікселів'
        im = ax.pcolormesh(xx, yy, slice_data, cmap=cmap, norm=norm, shading='auto', edgecolors='black', linewidth=0.2)

        ax.set_title(f'Vrstva Z = {z}', fontsize=12)

        # Налаштування підписів осей тільки на краях для чистоти картинки
        if col == 0:
            ax.set_ylabel('Y index')
        if row == 2:
            ax.set_xlabel('X index')

        # Налаштування tickів на осях 0, 1, ..., 8
        ax.set_xticks(np.arange(nx))
        ax.set_yticks(np.arange(ny))

    # 7. Přidání jedné společné barevné lišty (Colorbar)
    # rect коригує площу subplotів, щоб там lišta вмістилася
    fig.tight_layout(rect=[0, 0.03, 0.9, 0.96])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax, label='Tenzorová složka 0 (Smíchané hodnoty)')

    # Збереження результату
    save_name = 'synthesized_field_all_slices.png'
    plt.savefig(save_name, dpi=300)
    
    print(f"Hotovo! Комплексна візуалізація 3D об'єму збережена як '{save_name}'.")
    plt.show()

if __name__ == "__main__":
    visualize_full_3d_synthesis()
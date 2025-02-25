import os
import numpy as np
import matplotlib.pyplot as plt


fig, axs = plt.subplots(2, 3)
fig.delaxes(axs[1, 2])

for protein_idx, protein_name in enumerate(["c-met", "brd4", "esr1", "acaa1", "tub7"]):
    protein = False
    for file in os.listdir('logs'):
        affins = np.zeros((10, 50)) * np.nan
        hbonds = np.zeros((10, 50)) * np.nan
        current_trial = 0
        i = 0
        for line in open(f'logs/{file}', 'r'):
            if 'Starting protein' in line:
                if protein_name in line:
                    protein = True
                else:
                    protein = False
            if protein:
                if 'Starting trial' in line:
                    current_trial += 1
                    i = 0
                if 'Docking result' in line:
                    if not 'failed' in line:
                        affin, hbond = line.split()[-3:-1]
                        affins[current_trial - 1][i] = float(affin)
                        hbonds[current_trial - 1][i] = float(hbond)
                    i += 1
        axs[protein_idx // 3][protein_idx % 3].plot(np.nanmean(affins, axis=0), label=file.replace('.txt', ''))
        interval = 1.96 * np.nanstd(affins, axis=0) / np.sqrt(affins.shape[0])
        axs[protein_idx // 3][protein_idx % 3].fill_between(np.arange(affins.shape[1]), np.nanmean(affins, axis=0) - interval, np.nanmean(affins, axis=0) + interval, alpha=0.2)
        axs[protein_idx // 3][protein_idx % 3].set_title(protein_name)
plt.legend()
plt.tight_layout()
plt.savefig('results.png')
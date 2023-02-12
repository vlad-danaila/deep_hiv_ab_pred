from deep_hiv_ab_pred.preprocessing.aminoacids import one_hot
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

data = one_hot()

matplotlib.rc('font', size=5)

fig, ax = plt.subplots()
im = ax.imshow(data)

ax.set_xticks(np.arange(data.values.shape[1]))
ax.set_yticks(np.arange(len(list(data.index))))

ax.set_xticklabels(['Steric parameter', 'Polarizability', 'Volume', 'Hydrophobicity',
                    'Isoelectric point', 'Helix probability', 'Sheet probability'])
ax.set_yticklabels(list(data.index))

ax.tick_params(top=True, bottom=False,
               labeltop=True, labelbottom=False)

plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
         rotation_mode="anchor")

for edge, spine in ax.spines.items():
    spine.set_visible(False)

kw = dict(horizontalalignment="center", verticalalignment="center", fontsize = 5)
valfmt = matplotlib.ticker.StrMethodFormatter("{x:.2f}")
threshold = 0
textcolors=["white", "black"]

texts = []
for i in range(data.values.shape[0]):
    for j in range(data.values.shape[1]):
        kw.update(color=textcolors[data.values[i, j] > threshold])
        text = ax.text(j, i, valfmt(data.values[i, j], None), **kw)
        texts.append(text)

from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)

fig.set_size_inches(3, 6)
fig.tight_layout()
fig.savefig('amino_one_hot.pdf')
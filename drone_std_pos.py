from matplotlib import pyplot as pl
import numpy as np

fig = pl.figure()
ax = fig.add_subplot(projection="polar")

for i in np.arange(0, 10):
    ax.scatter([i * 40 * np.pi / 180], [1], c="b")

    ax.plot([0, i * 40 * np.pi / 180], [0, 1], c="g")

    print(f"A_{i + 1}=({np.cos(i * 40 * np.pi / 180)},{np.sin(i * 40 * np.pi / 180)})")

pl.show()

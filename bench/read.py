#!/usr/bin/env python3
import pandas as pd
import math
import matplotlib.pyplot as plt
from pathlib import Path

this = Path(__file__)
this.resolve()
root = this.parent

data = pd.read_json(root/"results.json", orient = "records")
timeout = data["param_timeout"].values[0]

ok_popcon = data.loc[lambda line: (line["invalid"] == False)][lambda line: line["timeout"] == False][lambda line: line["param_goal"] == "popcon"]

cmap = plt.get_cmap("gist_ncar")

def relax(ax1, bfs: bool):
    ax1.set_xlabel('relax ' + ("bfs" if bfs else "dfs"))
    ax1.set_ylabel('time (s)')
    ax1.set_yscale('log')
    ax1.axhline(y=timeout, color="black", dashes = [10, 20])
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('up/low (dashed)')  # we already handled the x-label with ax1
    ax2.set_yscale('log')
    ax2.axline((0, 1), (1, 2), color="grey", dashes = [1, 5, 10, 5])
    files = sorted(list(set(ok_popcon["param_file"])))
    for i, file in enumerate(files):
        this = ok_popcon.loc[lambda line: line["param_file"] == file][lambda line: (line["param_relax_bfs"]==bfs) | (line["param_relax"] == 0)][lambda line: (line["param_relax"] == 0)|(line["param_exact"]==False)]
        mc = data.loc[lambda line: line["param_file"] == file][lambda line: line["param_goal"] == "modelcount"]["time"].values[0]
        color = cmap(i/len(files))
        ax1.plot([0], [mc], "o", alpha=0.5, color=color)
        ax1.plot(this["param_relax"], this["time"], color=color, label=file)
        ax2.plot(this["param_relax"], this["up_to_low_ratio_percent_last"]/100, dashes=[1], color=color)

    (min, max) = ax2.get_ylim()
    ax2.set_yticks([10**i for i in range(math.floor(math.log10(min)), math.ceil(math.log10(max))+1)])
    ax1.legend()

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
relax(ax1, False)
relax(ax2, True)
plt.show()

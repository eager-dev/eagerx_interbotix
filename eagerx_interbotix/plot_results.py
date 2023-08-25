# Common imports
import os
import yaml
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# Import seaborn
# Apply the default theme
import seaborn as sns

sns.set(style="whitegrid", font_scale=1.5)
cmap = plt.rcParams["axes.prop_cycle"].by_key()["color"]
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 9
plt.rcParams["font.size"] = 12
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8
plt.rcParams["xtick.major.pad"] = -2.0
plt.rcParams["ytick.major.pad"] = -2.0
plt.rcParams["lines.linewidth"] = 1.3
plt.rcParams["axes.xmargin"] = 0.0
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


if __name__ == "__main__":
    # d = {}

    # Get root path
    root = Path(__file__).parent.parent

    # Load config
    cfg_path = root / "cfg" / "eval.yaml"
    with open(str(cfg_path), "r") as f:
        cfg = yaml.safe_load(f)
    train_cfg_path = root / "cfg" / "train.yaml"
    with open(str(train_cfg_path), "r") as f:
        train_cfg = yaml.safe_load(f)

    # Get parameters
    repetitions = 3
    cfg["settings"] = train_cfg["settings"]

    fig, ax = plt.subplots()
    ax.figure.set_size_inches(7.00137, 7.00137 / 2.5)
    ax.set_ylabel("Mean Distance to Goal (m)")
    width = 0.25
    xtickslabels = []

    for idx, setting in enumerate(cfg["settings"].keys()):
        name = cfg["settings"][setting]["name"].replace("+", "\n")
        xtickslabels.append(name)
        for mode in ["sim", "real"]:
            results = []
            for repetition in range(repetitions):
                eval_log_dir = root / "exps" / "eval" / "runs" / f"{setting}_{repetition}"
                eval_file = eval_log_dir / "eval.yaml"
                obs_file = eval_log_dir / f"{mode}_obs.pkl"

                # Check if evaluation already done
                if os.path.exists(eval_file):
                    eval_results = yaml.safe_load(open(str(eval_file), "r"))
                else:
                    eval_results = {}
                if eval_results is not None and mode in eval_results.keys() and "results" in eval_results[mode].keys():
                    mean_result = eval_results[mode]["mean"]
                    results.append(mean_result)

            if len(results) > 0:
                results = np.array(results)
                mean = np.mean(results, axis=0)
                std = np.std(results, axis=0)
                x = idx - width / 2 if mode == "sim" else idx + width / 2
                color = "c" if mode == "sim" else "m"
                ax.bar(x, mean, label=f"{setting}_{mode}", yerr=std, width=width, color=color)
    ax.set(xticks=np.arange(len(cfg["settings"].keys())), xticklabels=xtickslabels)
    ax.grid(axis="x")
    sim_patch = mpatches.Patch(color="c", label="sim")
    real_patch = mpatches.Patch(color="m", label="real")
    std_patch = mlines.Line2D([], [], color="k", marker="_", linestyle="None", markersize=10, label="$\sigma$")
    ax.legend(handles=[sim_patch, real_patch, std_patch])
    [label.set_fontweight("bold") for label in ax.get_xticklabels()]
    plt.savefig("results.pdf", bbox_inches="tight")
    plt.show()

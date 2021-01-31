import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class PlotManager:

    def __init__(self):
        pass

    results_folder = os.path.join(os.path.dirname(__file__), "plot")

    markers = ["o", "x", "^", "P", "v", "<", ">"]
    colors = ["g", "b", "r", "c", "m", "y", "k"]

    @staticmethod
    def plot_compare(x, y_list, models_names):
        min_min = 1
        max_max = 0
        for i, y in enumerate(y_list):
            plt.plot(x,
                     y,
                     marker=PlotManager.markers[i % len(PlotManager.markers)],
                     c=PlotManager.colors[i % len(PlotManager.colors)],
                     alpha=0.5,
                     label="Historical values" if i == 0 else "Model's prediction ({})".format(models_names[i].split("_")[1]))
            if min(y) < min_min:
                min_min = min(y)
            if max(y) > max_max:
                max_max = max(y)
        plt.xlabel("Year (t)")
        plt.ylabel("RCW value")
        plt.ylim(min_min*0.95, max_max*1.05)
        plt.legend()
        plt.savefig(os.path.join(PlotManager.results_folder,
                                 "models_graph_compare.png"))
        plt.close()

    @staticmethod
    def feature_importance(feature_name_val: dict, model_name: str):
        labels = [key for key, val in feature_name_val.items()]
        vals = [val for key, val in feature_name_val.items()]

        labels, vals = (list(t) for t in zip(*sorted(zip(labels, vals), key=lambda x: x[1])))

        x = np.arange(len(labels))  # the label locations

        fig, ax = plt.subplots(figsize=(16, 9))
        ax.barh(x, vals, 0.8, color="blue")

        # Remove axes splines
        for s in ['top', 'bottom', 'left', 'right']:
            ax.spines[s].set_visible(False)

        # Remove x, y Ticks
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')

        # Add padding between axes and labels
        ax.xaxis.set_tick_params(pad=5)
        ax.yaxis.set_tick_params(pad=10)

        # Add x, y gridlines
        ax.grid(b=True,
                axis="x",
                color='black',
                linestyle='-',
                linewidth=1,
                alpha=0.75)

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_xlabel('Importance')
        ax.set_yticks(x)
        ax.set_yticklabels(labels)
        ax.set_xticks([i / 20 for i in range(21)])
        ax.legend()
        fig.tight_layout()
        plt.savefig(os.path.join(PlotManager.results_folder, "feature_importance_{}.png".format(model_name)))
        plt.close()

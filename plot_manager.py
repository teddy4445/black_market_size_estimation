import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class PlotManager:

    def __init__(self):
        pass

    results_folder = os.path.join(os.path.dirname(__file__), "plot")

    markers = ["o", "x", "^", "P", "v", "<", ">"]
    colors = ["k", "b", "r", "g", "c", "m", "y", ]

    @staticmethod
    def analyze_data(df, model_type):

        g = sns.pairplot(df, corner=True, hue="pred")
        g.map_lower(sns.kdeplot, levels=1, color=".2")
        plt.savefig(os.path.join(PlotManager.results_folder,
                                 "data_pair_lot_{}.png".format(model_type)))
        plt.close()

        corr = df.corr()
        for name in list(corr):
            df[name] = df[name].abs()
        sns.heatmap(corr,
                    cmap='coolwarm',
                    vmin=0,
                    vmax=1)
        plt.savefig(os.path.join(PlotManager.results_folder,
                                 "data_features_heat_map_{}.png".format(model_type)))
        plt.close()

    @staticmethod
    def plot_compare(x, y_list, models_names, train_size):
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
        plt.plot([(x[train_size] + x[train_size+1])/2, (x[train_size] + x[train_size+1])/2],
                 [min_min * 0.95, max_max * 1.05],
                 c="gray",
                 markersize=8,
                 label="Forecasting Start")
        plt.xlabel("Year (t)")
        plt.ylabel("RCW value")
        plt.ylim(min_min*0.95, max_max*1.05)
        plt.legend()
        plt.savefig(os.path.join(PlotManager.results_folder,
                                 "models_graph_compare.png"))
        plt.close()

        min_min = 1
        max_max = 0
        for i, y in enumerate(y_list):
            plt.plot(x[train_size+1:],
                     y[train_size+1:],
                     marker=PlotManager.markers[i % len(PlotManager.markers)],
                     c=PlotManager.colors[i % len(PlotManager.colors)],
                     alpha=0.5,
                     label="Historical values" if i == 0 else "Model's prediction ({})".format(models_names[i].split("_")[1]))
            if min(y[train_size+1:]) < min_min:
                min_min = min(y[train_size+1:])
            if max(y[train_size+1:]) > max_max:
                max_max = max(y[train_size+1:])
        plt.xlabel("Year (t)")
        plt.ylabel("RCW value")
        plt.ylim(min_min*0.95, max_max*1.05)
        plt.legend()
        plt.savefig(os.path.join(PlotManager.results_folder,
                                 "models_graph_compare_only_prediction.png"))
        plt.close()

    @staticmethod
    def black_market_size(models: dict,
                          vop: list,
                          gdp: list,
                          x: dict,
                          model_name: str,
                          start_year: int):
        all_x = list(models["all"].predict(x=x["all"]))
        no_tax_x = list(models["no_tax"].predict(x=x["no_tax"]))
        no_crime_x = list(models["no_crime"].predict(x=x["no_crime"]))
        no_self_employ_x = list(models["no_self_employ"].predict(x=x["no_self_employ"]))

        y = [PlotManager.get_black_market_size(rcw_full_model=all_x[i],
                                               rcw_without_tax_model=no_tax_x[i],
                                               rcw_without_crime_model=no_crime_x[i],
                                               rcw_without_self_employ_model=no_self_employ_x[i],
                                               vop=vop[i],
                                               gdp=gdp[i])
             for i in range(len(all_x))]

        plt.plot([start_year + i for i in range(len(y))],
                 y,
                 marker=PlotManager.markers[0],
                 c=PlotManager.colors[0],
                 alpha=0.5,
                 label="Black Market Size ({})".format(model_name))
        plt.xlabel("Year (t)")
        plt.ylabel("Black Market Size (NIS)")
        plt.ylim(0, 100)
        plt.legend()
        plt.savefig(os.path.join(PlotManager.results_folder,
                                 "black_market_size_model_{}.png".format(model_name)))
        plt.close()

        with open(os.path.join(os.path.dirname(__file__), "data", "black_market_size_model_{}.csv".format(model_name)), "w") as data_file:
            data_file.write("year,percent\n")
            for i in range(len(y)):
                data_file.write("{},{:.2f}\n".format(start_year+i, y[i]))

    @staticmethod
    def black_market_size_from_file(file_path: str,
                                    model_name: int):

        x = []
        y = []
        with open(file_path, "r") as data_file:
            is_first = True
            for line in data_file.readlines():
                if is_first:
                    is_first = False
                    continue
                items = line.split(",")
                x.append(int(items[0]))
                y.append(float(items[1]))

        fig, ax = plt.subplots()
        plt.bar(x,
                 y,
                width=0.7,
                color="black")
        plt.xlabel("Year (t)")
        plt.ylabel("The share of the non-observed economy out of GDP")
        plt.ylim(10, max(y) * 1.05)
        plt.legend()
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        plt.savefig(os.path.join(PlotManager.results_folder,
                                 "black_market_size_model_from_file_{}.png".format(model_name)))
        plt.close()

    @staticmethod
    def black_market_size_from_file(file_path: str,
                                    model_name: int):

        x = []
        y1 = []
        y2 = []
        y3 = []
        y4 = []
        with open(file_path, "r") as data_file:
            is_first = True
            for line in data_file.readlines():
                if is_first:
                    is_first = False
                    continue
                items = line.split(",")
                x.append(int(items[0]))
                y1.append(float(items[1]))
                y2.append(float(items[2]))
                y3.append(float(items[3]))
                y4.append(float(items[1]) - float(items[2]) - float(items[3]))
        y = [y1, y2, y3, y4]
        y2_bottom = [y2[i] + y3[i] for i in range(len(y2))]

        fig, ax = plt.subplots()
        labels = ["Taxes", "Crime", "Self Employment"]
        width = 0.75
        for i in range(1, 4):
            plt.bar(x,
                    y[i],
                    bottom=None if i == 1 else y2 if i == 2 else y2_bottom,
                    width=width,
                    color=PlotManager.colors[i],
                    label=labels[i-1])
        plt.xlabel("Year (t)")
        plt.ylabel("The share of the non-observed economy out of GDP")
        plt.ylim(0, 25)
        plt.legend()
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        plt.savefig(os.path.join(PlotManager.results_folder,
                                 "black_market_size_model_from_file_break_down{}.png".format(model_name)))
        plt.close()

    @staticmethod
    def get_black_market_size(rcw_full_model: float,
                              rcw_without_tax_model: float,
                              rcw_without_crime_model: float,
                              rcw_without_self_employ_model: float,
                              vop: float,
                              gdp: float):
        gdp = float(gdp.replace(",", ""))
        delta1 = (rcw_full_model - rcw_without_tax_model) * 1000 / vop * gdp
        delta2 = (rcw_full_model - rcw_without_crime_model) * 1000 / vop * gdp
        delta3 = (rcw_full_model - rcw_without_self_employ_model) * 1000 / vop* gdp
        return abs(delta1 + delta2 + delta3) % 100

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


if __name__ == '__main__':
    PlotManager.black_market_size_from_file(file_path=os.path.join(os.path.dirname(__file__), "data", "black_market_size_model_rfr.csv"),
                                            model_name="rfr")

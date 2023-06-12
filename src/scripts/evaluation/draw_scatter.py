from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils.confidence_intervals_plotting import plot_confidences_intervals
from src.utils.distribution_plotting import DistributionPlotTypology, draw_distribution_plot

sns.set_theme(style="darkgrid")


def draw_scatter_plot(data_df: pd.DataFrame,
                      title: str,
                      x_col: str,
                      y_col: str,
                      hue_col: Optional[str] = None):
    p = sns.scatterplot(data=data_df,
                        x=x_col,
                        y=y_col,
                        hue=hue_col,
                        s=150)
    p.set_title(title, weight='bold').set_fontsize('18')
    _, xlabels = plt.xticks()
    ylabels = [round(y, 1) for y in p.get_yticks().tolist()]
    p.set_xlabel(x_col, weight="bold")
    p.set_xticklabels(xlabels, size=12)
    p.set_ylabel(y_col, weight="bold")
    p.set_yticklabels(ylabels, size=12)
    plt.legend()
    plt.show()
    plt.close()


if __name__ == '__main__':

    link_pred = [
        0.420, 0.391, 0.433, 0.387, 0.435, 0.430, 0.561, 0.506, 0.416,  # codex small
        0.403, 0.504, 0.369, 0.270, 0.428, 0.434, 0.547, 0.477, 0.368,  # wn18rr
        0.412, 0.444, 0.300, 0.321, 0.322, 0.429, 0.487, 0.344, 0.450,  # fb15k-237
    ]

    link_del = [
        0.728, 0.295, 0.536, 0.522, 0.350, 0.479, 0.550, 0.650, 0.754,  # codex small
        0.009, 0.015, 0.006, 0.014, 0.009, 0.021, 0.016, 0.062, 0.043,  # wn18rr
        0.302, 0.222, 0.112, 0.446, 0.095, 0.269, 0.160, 0.270, 0.413,  # fb15k-237
    ]

    datasets = ["CODEX SMALL"] * 9 + ["WN18RR"] * 9 + ["FB15K-237"] * 9

    df = pd.DataFrame({
        "Link Prediction": link_pred,
        "Link Deletion": link_del,
        "Datasets": datasets,
    })

    draw_scatter_plot(
        data_df=df,
        title="Correlation between the LP and LD tasks (Hits@10)",
        x_col="Link Prediction",
        y_col="Link Deletion",
        hue_col="Datasets",
    )
    exit(0)

    hits_10_values = {
        "AutoSF": [0.420, 0.403, 0.412, 0.728, 0.009, 0.302, 0.928, 0.716, 0.972],
        "BoxE": [0.391, 0.504, 0.444, 0.295, 0.015, 0.222, 0.928, 0.795, 0.974],
        "ComplEx": [0.433, 0.369, 0.300, 0.536, 0.006, 0.112, 0.890, 0.689, 0.932],
        # "ConvE": [0.286, 0.439, 0, 0.648, 0.018, 0, 0.695, 0.801, 0],
        "DistMult": [0.387, 0.270, 0.321, 0.522, 0.014, 0.446, 0.904, 0.663, 0.975],
        "HolE": [0.435, 0.428, 0.322, 0.350, 0.009, 0.095, 0.879, 0.786, 0.943],
        "PairRE": [0.430, 0.434, 0.429, 0.479, 0.021, 0.269, 0.906, 0.780, 0.965],
        "RotatE": [0.561, 0.547, 0.487, 0.550, 0.016, 0.160, 0.908, 0.801, 0.943],
        "TransE": [0.506, 0.477, 0.344, 0.650, 0.062, 0.270, 0.937, 0.889, 0.976],
        "TransH": [0.416, 0.368, 0.450, 0.754, 0.043, 0.413, 0.932, 0.754, 0.981],
    }

    for k in [
        DistributionPlotTypology.VIOLIN_PLOT,
        DistributionPlotTypology.BOX_PLOT,
        DistributionPlotTypology.SCATTER_PLOT
    ]:
        draw_distribution_plot(
            label_values_map=hits_10_values,
            title="Performance Analysis of KGE Models across Datasets and Tasks (Hits@10)",
            plot_type=k,
            orient="v",
            show_flag=True,
            out_path=None,
        )

    plot_confidences_intervals(
        label_values_map=hits_10_values,
        title="Performance Analysis of KGE Models across Datasets and Tasks (Hits@10)",
        use_median=True,
        use_mean=False,
        percentile_min=25,
        percentile_max=75,
        line_color="#2187bb",
        point_color="#f44336",
        horizontal_line_width=0.25,
        round_digits=4
    )

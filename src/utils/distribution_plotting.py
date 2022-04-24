from enum import Enum
from typing import Dict, Sequence, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="darkgrid")  # white, dark, whitegrid, darkgrid, ticks


class DistributionPlotTypology(Enum):
    VIOLIN_PLOT: int = 1
    BOX_PLOT: int = 2
    SCATTER_PLOT: int = 3


def draw_distribution_plot(
        label_values_map: Dict[str, Sequence[float]],
        title: str,
        plot_type: DistributionPlotTypology = DistributionPlotTypology.BOX_PLOT,
        orient: str = "v",
        palette: Optional[str] = None,
        show_flag: bool = False,
        out_path: Optional[str] = None
):
    a4_dims = (11.7, 8.27)
    fig, ax = plt.subplots(figsize=a4_dims)
    df_data = pd.DataFrame(data=label_values_map).reset_index(drop=True)
    if plot_type.value == 1:
        p = sns.violinplot(x=None,
                           y=None,
                           hue=None,
                           data=df_data,
                           order=None,
                           hue_order=None,
                           bw='scott',  # scott, silverman
                           cut=2,
                           scale='area',  # area, count, width
                           scale_hue=True,
                           gridsize=1000,
                           width=0.8,
                           inner='box',  # box, quartile, point, stick, None
                           split=False,
                           dodge=True,
                           orient=orient,  # "v" | "h"
                           linewidth=None,
                           color=None,
                           palette=palette,
                           saturation=0.6,
                           ax=ax)
    elif plot_type.value == 2:
        p = sns.boxplot(x=None,
                        y=None,
                        hue=None,
                        data=df_data,
                        order=None,
                        hue_order=None,
                        orient=orient,
                        color=None,
                        palette=palette,
                        saturation=0.75,
                        width=0.8,
                        dodge=True,
                        fliersize=5,
                        linewidth=None,
                        whis=1.5,
                        ax=ax)
    elif plot_type.value == 3:
        p = sns.stripplot(x=None,
                          y=None,
                          hue=None,
                          data=df_data,
                          order=None,
                          hue_order=None,
                          jitter=True,
                          dodge=False,
                          orient=orient,
                          color=None,
                          palette=palette,
                          size=5,
                          edgecolor='gray',
                          linewidth=0,
                          ax=ax)
    else:
        raise ValueError("Invalid plot_type!")
    p.set_title(title, weight='bold').set_fontsize('18')
    _, xlabels = plt.xticks()
    ylabels = [round(y, 1) for y in p.get_yticks().tolist()]
    p.set_xticklabels(xlabels, size=16, weight='bold')
    p.set_yticklabels(ylabels, size=14)
    if show_flag:
        plt.show()
        plt.close()
    if out_path:
        plt.savefig(out_path)


if __name__ == '__main__':
    records = {
        "a": [1, 2, 5, 8, 10, 7, 7, 8],
        "b": [5, 5, 5, 5, 5, 5, 5, 5],
        "c": [10, 3, 4, 5, 9, 8, 9, 1],
        "d": [0, 4, 3, 9, 9, 6, 5, 4],
        "e": [0, 0, 0, 0, 1, 1, 1, 1]
    }

    for k in [
        DistributionPlotTypology.VIOLIN_PLOT,
        DistributionPlotTypology.BOX_PLOT,
        DistributionPlotTypology.SCATTER_PLOT
    ]:
        draw_distribution_plot(
            label_values_map=records,
            title="AAA",
            plot_type=k,
            orient="v",
            show_flag=True,
            out_path=None,
        )

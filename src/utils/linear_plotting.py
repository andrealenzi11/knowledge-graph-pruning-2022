from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import seaborn as sns


def plot_linear_chart(name_values_map: Dict[str, List[float]]):
    plt.style.use('ggplot')
    colors_list = sns.color_palette(palette="deep", n_colors=(len(name_values_map)))
    print(f"colors_list size: {len(colors_list)}")
    # input validation
    values_sizes = []
    for k, v in name_values_map.items():
        assert isinstance(k, str)
        assert isinstance(v, list)
        values_sizes.append(len(v))
    if len(set(values_sizes)) != 1:
        raise ValueError(f"the input values have a different number of elements! \n"
                         f"sizes = {values_sizes}")
    # interpolation and plotting
    i = 0
    for k, v in name_values_map.items():
        col_index = i % len(colors_list)
        color = colors_list[col_index]
        print(k, "|", col_index, "|", color, "|", v)
        x_new = np.arange(0, values_sizes[0], 1)
        f1 = interpolate.interp1d(x_new, v)
        plt.plot(x_new, v, 'o', color=color)
        plt.plot(x_new, f1(x_new), '--', linewidth=1.75, color=color, label=k)
        i += 1
    # chart elements definition
    plt.grid(True)
    plt.legend(loc='best')
    plt.title("A title", fontsize=18, fontweight='heavy', color='darkred')
    plt.xlabel("name of the X axis", fontsize=14, fontweight='bold')
    plt.ylabel('name of the Y axis', fontsize=14, fontweight='bold')
    # plt.axis([-0.2, 5.2, -0.2, 120])
    plt.tight_layout()
    plt.subplots_adjust(left=0.060, bottom=0.080, right=0.9890, top=0.940)
    plt.show()
    plt.close()


if __name__ == '__main__':
    print(len(sns.color_palette()))
    example_diz = {
        "a": [0.3, 0.8, 0.5],
        "b": [1.3, 1.8, 1.5],
        "c": [2.3, 2.8, 2.5],
        "d": [3.3, 3.8, 3.5],
        "e": [4.3, 4.8, 4.5],
    }
    plot_linear_chart(name_values_map=example_diz)

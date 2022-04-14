# -*- coding: utf-8 -*-
from collections import Sequence
from typing import Optional

import matplotlib.pyplot as plt


def autolabel(rects):
    """ Attach a text label above each bar displaying its height """
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2., 1.0 * height,
                 '  %s' % str(round(height, 4)), ha='center', va='bottom', rotation=90)


def plot_vertical_bar_chart(keys: Sequence,
                            values: Sequence,
                            title: str,
                            x_label: str = "X label",
                            y_label: str = "Y label",
                            legend_label: Optional[str] = None,
                            xticks_rotation: int = 60,
                            color: str = "darkcyan",
                            left: Optional[float] = None,
                            bottom: Optional[float] = None,
                            right: Optional[float] = None,
                            top: Optional[float] = None,
                            wspace: Optional[float] = None,
                            hspace: Optional[float] = None,
                            xlim_lo: Optional[float] = None,
                            xlim_hi: Optional[float] = None,
                            ylim_lo: Optional[float] = None,
                            ylim_hi: Optional[float] = None):
    """ Plot vertical bar chart (histogram) """
    plt.style.use(["ggplot"])
    rect1 = plt.bar(keys, values, color=color, align='center', label=legend_label)
    plt.title(label=title, fontweight="bold", size=16)
    plt.xlabel(xlabel=x_label, fontweight="bold", size=13)
    plt.ylabel(ylabel=y_label, fontweight="bold", size=13)
    plt.xticks(keys, rotation=xticks_rotation, rotation_mode="anchor", ha="right", fontweight="bold")
    plt.xlim(xlim_lo, xlim_hi)
    plt.ylim(ylim_lo, ylim_hi)
    plt.legend(loc="best")
    autolabel(rect1)
    plt.tight_layout()
    plt.subplots_adjust(left=left,
                        bottom=bottom,
                        right=right,
                        top=top,
                        wspace=wspace,
                        hspace=hspace)
    plt.show()
    plt.close()


def plot_horizontal_bar_chart(keys: Sequence,
                              values: Sequence,
                              title: str,
                              x_label: str = "X label",
                              y_label: str = "Y label",
                              color: str = "slategray",
                              left: Optional[float] = None,
                              bottom: Optional[float] = None,
                              right: Optional[float] = None,
                              top: Optional[float] = None,
                              wspace: Optional[float] = None,
                              hspace: Optional[float] = None,
                              xlim_lo: Optional[float] = None,
                              xlim_hi: Optional[float] = None,
                              ylim_lo: Optional[float] = None,
                              ylim_hi: Optional[float] = None):
    """ Plot horizontal bar chart """
    plt.style.use(['ggplot'])
    plt.title(label=title, fontweight="bold", size=16)
    plt.xlabel(xlabel=x_label, fontweight="bold")
    plt.ylabel(ylabel=y_label, fontweight="bold")
    plt.barh(keys, values, color=color)
    for iX, vX in enumerate(values):
        plt.text(vX, iX, " " + str(round(vX, 4)), color='blue', va='center', fontweight='bold')
    plt.xlim(xlim_lo, xlim_hi)
    plt.ylim(ylim_lo, ylim_hi)
    plt.locator_params(axis='x', nbins=5)
    plt.tight_layout()
    plt.subplots_adjust(left=left,
                        bottom=bottom,
                        right=right,
                        top=top,
                        wspace=wspace,
                        hspace=hspace)
    plt.show()
    plt.close()

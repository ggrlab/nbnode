import os
from typing import List

import matplotlib
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
from dtreeviz.trees import DTreeVizAPI
from matplotlib.colors import Colormap, LinearSegmentedColormap
from pydotplus.graphviz import Dot

from nbnode_pyscaffold.plot.shifted_colormap import shifted_colormap

matplotlib.use("AGG")


# from https://stackoverflow.com/questions/12472338/flattening-a-list-recursively
def flatten(S: List):
    if S == []:
        return S
    if isinstance(S[0], list):
        return flatten(S[0]) + flatten(S[1:])
    return S[:1] + flatten(S[1:])


class LinearShiftedColormap:
    def __init__(
        self,
        range_min: float = -1,
        range_max: float = 1,
        base_cmap=matplotlib.cm.RdBu_r,
        name: str = "shifted",
    ):
        self.cmap = shifted_colormap(
            base_cmap, min_val=range_min, max_val=range_max, name=name
        )
        self.range_min = range_min
        self.range_max = range_max


def plot_save_unified(any_plot, file: str, **kwargs):
    if isinstance(any_plot, Dot):
        if not file.endswith(".pdf"):
            file = file.rsplit(".", maxsplit=1)[0] + ".pdf"
        any_plot.write_pdf(file, **kwargs)
    elif isinstance(any_plot, DTreeVizAPI):
        if os.name == "nt" and not file.endswith(".svg"):
            file = file.rsplit(".", maxsplit=1)[0] + ".svg"
        any_plot.save(file)
        os.remove(
            file.rsplit(".", maxsplit=1)[0]
        )  # I do not know why there is an additional file created...
    else:
        any_plot.savefig(file, **kwargs)

import os
from typing import List

import matplotlib
import matplotlib.cm
from dtreeviz.trees import DTreeVizAPI
from pydotplus.graphviz import Dot

from nbnode.plot.shifted_colormap import shifted_colormap

matplotlib.use("AGG")


# from https://stackoverflow.com/questions/12472338/flattening-a-list-recursively
def flatten(S: List):
    """Flatten any list of (nested) lists.

    Flattens further until the element to flatten is not a list.

    Args:
        S (List):
            A list containing elements, lists and/or nested lists.

    Returns:
        _type_:
            A flattened list.
    """
    if S == []:
        return S
    if isinstance(S[0], list):
        return flatten(S[0]) + flatten(S[1:])
    return S[:1] + flatten(S[1:])


class LinearShiftedColormap:
    """
    A linear shifted colormap.

    Takes a base colormap and scales/stretches it between min_val and max_val.
    """

    def __init__(
        self,
        range_min: float = -1,
        range_max: float = 1,
        base_cmap=matplotlib.cm.RdBu_r,
        name: str = "shifted",
    ):
        """Initialize a LinearShiftedColormap.

        Args:
            range_min (float, optional):
                The minimum value of the colormap. Defaults to -1.
            range_max (float, optional):
                The maximum value of the colormap. Defaults to 1.
            base_cmap (_type_, optional):
                The base colormap to use. You can use any colormap from matplotlib::

                    import matplotlib as mpl

                    new_base_cmap = mpl.colormaps.get_cmap("viridis")
                    shifted_cmap = LinearShiftedColormap(base_cmap=new_base_cmap)

                Defaults to matplotlib.cm.RdBu_r.
            name (str, optional):
                The name of the colormap. Defaults to "shifted".
        """
        self.cmap = shifted_colormap(
            base_cmap, min_val=range_min, max_val=range_max, name=name
        )
        self.range_min = range_min
        self.range_max = range_max


def plot_save_unified(any_plot, file: str, **kwargs):
    """
    Save a plot in a unified way.

    Args:
        any_plot (_type_):
            A plot object. E.g. a matplotlib plot/figure.
        file (str):
            The file to save the plot to.
    """
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

import numpy as np
from matplotlib.colors import Colormap, LinearSegmentedColormap


def shifted_colormap(cmap, min_val, max_val, name) -> Colormap:
    """Function to offset the "center" of a colormap.

    Useful for data with a negative min and positive max and you want the middle of the
    colormap's dynamic range to be at zero.

    Adapted from
    https://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib
    Post of DaveTheScientist


    Args:
      cmap :
        The matplotlib colormap to be altered.
      start :
        Offset from lowest point in the colormap's range.
        Defaults to 0.0 (no lower ofset). Should be between
        0.0 and `midpoint`.
      midpoint :
        The new center of the colormap. Defaults to
        0.5 (no shift). Should be between 0.0 and 1.0. In
        general, this should be  1 - vmax/(vmax + abs(vmin))
        For example if your data range from -15.0 to +5.0 and
        you want the center of the colormap at 0.0, `midpoint`
        should be set to  1 - 5/(5 + 15)) or 0.75
      stop :
        Offset from highets point in the colormap's range.
        Defaults to 1.0 (no upper ofset). Should be between
        `midpoint` and 1.0.

    Returns:
        Colormap: _description_
    """
    epsilon = 0.001
    start, stop = 0.0, 1.0
    min_val, max_val = min(0.0, min_val), max(0.0, max_val)  # Edit #2
    # sometimes, the values are all exactly identical (or 0)
    # Therefore, add a small value that we never have a division through zero.
    midpoint = 1.0 - max_val / (max_val + abs(min_val) + 0.0001)

    cdict = {"red": [], "green": [], "blue": [], "alpha": []}

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack(
        [
            np.linspace(0.0, midpoint, 128, endpoint=False),
            np.linspace(midpoint, 1.0, 129, endpoint=True),
        ]
    )

    for ri, si in zip(reg_index, shift_index):
        if abs(si - midpoint) < epsilon:
            r, g, b, a = cmap(0.5)  # 0.5 = original midpoint.
        else:
            r, g, b, a = cmap(ri)
        cdict["red"].append((si, r, r))
        cdict["green"].append((si, g, g))
        cdict["blue"].append((si, b, b))
        cdict["alpha"].append((si, a, a))

    newcmap = LinearSegmentedColormap(name, cdict)
    # plt.register_cmap(cmap=newcmap)   # I think I do not need the registering after
    # I overgive the colormaps anyways
    # and do NOT refer to them by their name.

    return newcmap

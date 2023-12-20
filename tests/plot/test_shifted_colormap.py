def test_shifted_colormap():
    import matplotlib

    from nbnode.plot.shifted_colormap import shifted_colormap

    new_cmap = shifted_colormap(
        cmap=matplotlib.cm.RdBu_r,
        min_val=-5,
        max_val=10,
        name="NoName",
    )
    print(new_cmap)


def test_linear_shifted_colormap():
    import matplotlib

    from nbnode.plot.utils import LinearShiftedColormap

    LinearShiftedColormap()
    LinearShiftedColormap(range_min=-1, range_max=1)
    LinearShiftedColormap(range_min=-1, range_max=1, base_cmap=matplotlib.cm.RdBu_r)
    LinearShiftedColormap(
        range_min=-1, range_max=1, base_cmap=matplotlib.cm.RdBu_r, name="shifted"
    )
    LinearShiftedColormap(
        range_min=-10, range_max=10, base_cmap=matplotlib.cm.RdBu_r, name="shifted"
    )

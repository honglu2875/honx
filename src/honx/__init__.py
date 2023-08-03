from ._min_dist import dijkstra_stl


def dijkstra(*args, **kwargs):
    return dijkstra_stl(*args, **kwargs)


__version__ = "0.0.1"
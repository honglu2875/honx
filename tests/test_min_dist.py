from honx import dijkstra


def test_dijkstra():
    # dijkstra: args: (graph: list[list[tuple[int, int]]], src: int) -> list[int]
    graph = [
        [(1, 1), (2, 4)],
        [(0, 1), (2, 4)],
        [(0, 2), (1, 3)],
    ]
    src = 0
    dist = dijkstra(graph, src)
    print(dist)
    raise NotImplementedError
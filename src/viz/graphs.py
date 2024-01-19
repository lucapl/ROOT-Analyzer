import matplotlib.pyplot as plt
import networkx as nx


def i_to_x_y(i):
    return i % 4, 2 - (i // 4)


def draw_map(board_info):
    g = nx.grid_2d_graph(4, 3)
    g.remove_edges_from(list(g.edges()))

    labels = {i_to_x_y(i): i for i in range(12)}

    for i, connections in enumerate(board_info["connections"]):
        x, y = i_to_x_y(i)
        for j in connections:
            z, w = i_to_x_y(j)
            g.add_edge((x, y), (z, w))

    pos = dict((n, n) for n in g.nodes())
    nx.draw_networkx(g, pos=pos, labels=labels)
    plt.axis('off')
    plt.show()

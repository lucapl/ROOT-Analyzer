import matplotlib.pyplot as plt
import networkx as nx


def i_to_x_y(i):
    return i % 4, 2 - (i // 4)


def draw_map(board_info):
    G_map = nx.grid_2d_graph(4, 3)
    G_map.remove_edges_from(list(G_map.edges()))

    labels = {i_to_x_y(i): i for i in range(12)}

    for i, connections in enumerate(board_info["connections"]):
        x, y = i_to_x_y(i)
        for j in connections:
            z, w = i_to_x_y(j)
            G_map.add_edge((x, y), (z, w))

    pos = dict((n, n) for n in G_map.nodes())
    nx.draw_networkx(G_map, pos=pos, labels=labels)
    plt.axis('off')
    plt.show()

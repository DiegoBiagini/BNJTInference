import sys

import matplotlib.pyplot as plt
import networkx as nx

import util

"""
Command line utility to visualize a bayesian net and a junction tree, previously created and saved in a file

Use:
visualize.py model.dat
"""

def main():
    # Check arguments
    if len(sys.argv) != 2:
        raise ValueError("Not enough arguments")

    model_path = sys.argv[1]

    bnet , jtree = util.load_model(model_path)
    # Bayes net drawing
    net = nx.DiGraph()

    for var in bnet.get_variables():
        net.add_node(var.name)

    graph = bnet.get_graph()
    for var in graph.keys():
        for el in graph[var]:
            net.add_edge(var.name, el.name)

    nx.draw(net , node_size=2000, node_color="white", edge_color="black",  cmap = plt.get_cmap('jet')  ,pos=nx.planar_layout(net),with_labels=True)
    ax = plt.gca()  # to get the current axis
    ax.collections[0].set_edgecolor("#000000")

    manager = plt.get_current_fig_manager()
    manager.window.state('zoomed')
    plt.show()

    # Junction tree drawing

    tree = nx.Graph()
    cliques, separators = jtree.get_cliques_and_seps()

    clique_names = [c.node_vars_to_string() for c in cliques]
    sep_names = [sep.node_vars_to_string() for sep in separators]

    for clique in cliques:
        tree.add_node(clique.node_vars_to_string())
        for sep in clique.get_neighbours():
            tree.add_node(sep.node_vars_to_string())
            tree.add_edge(clique.node_vars_to_string(), sep.node_vars_to_string())

    pos = nx.planar_layout(tree)
    nx.draw_networkx_nodes(tree, pos, nodelist=clique_names, node_shape='o', node_size=2000, node_color="white", edge_color="black",  cmap = plt.get_cmap('jet'),with_labels=True)
    nx.draw_networkx_nodes(tree, pos, nodelist=sep_names, node_shape='^', node_size=2000, node_color="white", edge_color="black",  cmap = plt.get_cmap('jet'),with_labels=True)
    nx.draw_networkx_edges(tree, pos)
    nx.draw_networkx_labels(tree, pos)

    ax = plt.gca()  # to get the current axis
    ax.collections[0].set_edgecolor("#000000")
    ax.collections[1].set_edgecolor("#000000")

    manager = plt.get_current_fig_manager()
    manager.window.state('zoomed')
    plt.show()

if __name__ == '__main__':
    main()
from __future__ import annotations

import numpy as np
import networkx as nx

from typing import List
from networkx import Graph, DiGraph

def displayCircular(G):
    return nx.draw_circular(
        G,
        with_labels=True,
        node_size=1000,
        node_color="c",
        width=0.8,
        font_size=14,
    )


def displayGraph(G, kij):
    import matplotlib.pyplot as plt
    labels = dict(zip(G.nodes, G.nodes))
    pos = nx.circular_layout(G)

    cmap = plt.cm.Blues
    edge_colors = [np.power(kij[x[0], x[1]], 0.25) for x in list(G.edges)]
    edges = nx.draw_networkx_edges(
        G,
        pos,
        edge_color=edge_colors,
        edge_cmap=cmap,
        width=1,
    )

    nx.draw_networkx_nodes(G, pos, node_size=500, node_color="black")
    labels = dict(zip(G.nodes, G.nodes))
    nx.draw_networkx_labels(G, pos, labels, font_size=15, font_color="whitesmoke")
    # nx.draw_networkx_edges(
    #     fgraph,
    #     pos,
    #     edgelist=list(fgraph.edges),
    #     edge_color="tab:red",
    #     width=3,
    #     alpha=0.5,
    # )
    plt.show()
    return 0


def displayGraphHighlight(G, kij, fgraph):
    import matplotlib.pyplot as plt
    labels = dict(zip(G.nodes, G.nodes))
    pos = nx.circular_layout(G)

    cmap = plt.cm.Blues
    edge_colors = [np.power(kij[x[0], x[1]], 0.25) for x in list(G.edges)]
    edges = nx.draw_networkx_edges(
        G,
        pos,
        edge_color=edge_colors,
        edge_cmap=cmap,
        width=1,
    )

    nx.draw_networkx_nodes(G, pos, node_size=500, node_color="black")
    labels = dict(zip(G.nodes, G.nodes))
    nx.draw_networkx_labels(G, pos, labels, font_size=15, font_color="whitesmoke")
    nx.draw_networkx_edges(
        fgraph,
        pos,
        edgelist=list(fgraph.edges),
        edge_color="tab:red",
        width=3,
        alpha=0.5,
    )
    plt.show()
    return 0


def fromOrderToDiGraph(order):
    nodes = len(order)
    edges = []
    for i in range(nodes - 1):
        edges.append((order[i], order[i + 1]))
    # print(edges)
    return nx.DiGraph(edges)


def addEdgesByGreedySearch(graph, kij, maxdes=1):
    debug = False
    graph2 = graph.copy()
    order = list(graph2.nodes)
    nodes = len(order)
    for node in order:
        ancestors = list(nx.ancestors(graph2, node))
        adj = list(graph2.adj[node])  # childen
        weights = kij[node, :]
        wvec = []
        ivec = []
        for i in range(nodes):
            if i == node:
                continue  # no self edge
            if i in adj:
                continue  # new children cannot be in adj
            if i in ancestors:
                continue  # edge go back is not allowed
            ivec.append(i)
            wvec.append(weights[i])
        ivec = np.array(ivec)
        wvec = np.array(wvec)
        ord = np.argsort(wvec)[-1::-1]
        wvec = wvec[ord]
        ivec = ivec[ord]
        numnew = min(len(ord), maxdes)
        childnew = ivec[:numnew]
        graph2.add_edges_from([(node, x) for x in childnew])
        if debug:
            print("node=", node)
            print("ivec=", ivec)
            print("wvec=", wvec)
            print("ord=", ord)
            print("numnew=", numnew)
            print("childnew=", childnew)
            print()
    return graph2


def fromKijToGraph(kij):
    nodes = kij.shape[0]
    edges = []
    for i in range(nodes):
        for j in range(i):
            edges.append((j, i))
    graph = nx.Graph()
    graph.add_edges_from(edges)
    return graph


def num_count(graph:DiGraph) -> List[int]:
    """
    to calculate the pos. of site i in param M
    """
    num = [0] * len(list(graph.nodes))
    all_in_num = 0
    for i in list(graph.nodes):
        all_in = list(graph.predecessors(str(i)))
        all_in_num += len(all_in)
        num[int(i)] = all_in_num
    return num

def checkgraph(graph1: DiGraph, graph2: DiGraph) -> List[List[int]]:
    """
    check graph1 ⊂ graph2
    RETURN(List[List[Int]]):
    the index of graph1's edges in graph2's edges
    """
    graph1_node = list(graph1.nodes)
    graph2_node = list(graph2.nodes)
    assert graph1_node == graph2_node
    add_edge = []
    for site in graph1_node:
        graph1_pre_site = list(graph1.predecessors(site))
        graph2_pre_site = list(graph2.predecessors(site))
        assert set(graph1_pre_site).issubset(set(graph2_pre_site))
        # print("graph1 is subset of graph2?", set(graph1_pre_site).issubset(set(graph2_pre_site)))
        # print(site, "=big=>", graph2_pre_site)
        # print(site, "=big=>", graph2_pre_site[:len(graph1_pre_site)])
        # print(site, "=small=>", graph1_pre_site)
        edge_index = [graph2_pre_site.index(element) for element in graph1_pre_site]
        # print(edge_index)
        add_edge.append(edge_index)  # 这个顺序是按照graph对应一维顺序排列的
    return add_edge

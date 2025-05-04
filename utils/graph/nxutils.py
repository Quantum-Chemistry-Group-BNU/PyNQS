from __future__ import annotations

import numpy as np
import networkx as nx

from heapq import heapify, heappush, heappop
from typing import List, Tuple, Dict
from networkx import Graph, DiGraph

def displayCircular(G):
    """
    draw picture looked like regular polygon by graph G.
    """
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
    """
    func "displayGraphHighlight" without egdes between two vertexes.
    """
    labels = dict(zip(G.nodes, G.nodes))
    pos = nx.circular_layout(G)

    cmap = plt.cm.Blues
    edge_colors = [np.power(kij[x[0], x[1]], 0.25) for x in list(G.edges)]
    nx.draw_networkx_edges(
        G=G,
        pos=pos,
        edge_color=edge_colors,
        edge_cmap=cmap,
        width=1,
    )

    nx.draw_networkx_nodes(G, pos, node_size=500, node_color="black")
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

    # drwa other edges
    cmap = plt.cm.Blues
    edge_colors = [np.power(kij[x[0], x[1]], 0.25) for x in list(G.edges)]
    nx.draw_networkx_edges(
        G,
        pos,
        edge_color=edge_colors,
        edge_cmap=cmap,
        width=1,
    )

    # draw the order along vertexes
    # draw vertexes & their labels
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color="black")
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


def fromOrderToDiGraph(order):
    # to generate a digraph along the ver
    nodes = len(order)
    edges = []
    for i in range(nodes - 1):
        edges.append((order[i], order[i + 1]))
    # print(edges)
    return nx.DiGraph(edges)


def addEdgesByGreedySearch(graph, kij, maxdes=1, tensor=False, debug=False):
    graph2 = graph.copy()
    order = list(graph2.nodes)
    nodes = len(order)
    for node in order:
        ancestors = list(nx.ancestors(graph2, node))  # list of vertexed before <node>
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
            if tensor:
                if len(list(graph2.predecessors(i))) >= 2:
                    continue  # the predecessor of i should less than 2
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
    """
    generate all the edges from kij matrix
    """
    nodes = kij.shape[0]
    edges = []
    # generate connections from kij matrix
    for i in range(nodes):
        for j in range(i):
            edges.append((j, i))
    # obtain digraph
    graph = nx.DiGraph()
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
    add_edge: dict[list[int]] = {}
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
        add_edge[site] = edge_index  # order along 1d order(nodes' order)
    return add_edge


def scan_matrix(graph: DiGraph, different_h = False):
    """
    return (list[predecessors, successors], dict{dict}) of one given site
    """
    graph_node = list(graph.nodes)
    martix_index = {}
    if different_h:
        start = len(list(graph.successors(graph_node[0])))
    else:
        start = 1
    num_site = -1 * start
    for site in graph_node:
        index:dict = {}
        if different_h:
            for o in list(graph.successors(site)):
                index_in = {}
                for i in list(graph.predecessors(site)):
                    index_in[i] = num_site
                    num_site += 1
                index[o] = index_in
        else:
            index_in = {}
            for i in list(graph.predecessors(site)):
                index_in[i] = num_site
                num_site += 1
            for o in list(graph.successors(site)):
                index[o] = index_in
        martix_index[site] = ([list(graph.predecessors(site)), list(graph.successors(site))], index)
    return (martix_index, num_site+start)


def scan_eta(graph: DiGraph, different_h = False):
    '''
    return (eta_index[site][out], end)
    '''
    graph_node = list(graph.nodes)
    eta_index = {}
    num_site= len(graph_node)
    for index_site, site in enumerate(graph_node):
        index = {}
        if site == graph_node[-1]:
            index[graph_node[0]] = int(graph_node[-1])
        else:
            for o in list(graph.successors(site)):
                if o == graph_node[index_site+1]:
                    index[o] = int(site)
                else:
                    index[o] = num_site
                    if different_h:
                        num_site += 1
                    else:
                        index[o] = index[graph_node[index_site+1]]
        eta_index[site] = ([list(graph.predecessors(site)), list(graph.successors(site))], index)
    return (eta_index, num_site)

def scan_tensor(graph: DiGraph, max_degree: int = 2):
    """
    return the the node of graph which can have tensor term
    """
    graph_node = list(graph.nodes)
    tensor_index = []
    for site in graph_node:
        if len(list(graph.predecessors(site))) > 1 and len(list(graph.predecessors(site))) <= max_degree:
            tensor_index.append(site)
    return tensor_index


def check_tesnor(graph: DiGraph):
    """
    check this graph can use non-cmpr-tensor or not
    """
    graph_node = list(graph.nodes)
    result = True
    for site in graph_node:
        if len(list(graph.predecessors(site))) >= 2:
            result = False
    return result

def calculate_min_hidden_states(
    graph: nx.DiGraph,
) -> Tuple[int, Dict[int,int], Dict[int,int]]:

    topo = list(map(int, graph.adj))
    # timestamp
    start = {u: i for i, u in enumerate(topo)}
    mapping = {node: int(node) for node in graph.nodes}
    graph = nx.relabel_nodes(graph, mapping)
    end = {
        u: max((start[v] for v in graph.successors(u)), default=start[u])
        for u in graph.nodes()
    }

    # scanning line
    events: List[Tuple[int,int]] = []
    for u in topo:
        l, r = start[u], end[u]
        events.append((l, +1))
        events.append((r, -1))  # +ε release
    events.sort(key=lambda x: (x[0], x[1]))

    active = max_active = 0
    for _, delta in events:
        active += delta
        if active > max_active:
            max_active = active

    return max_active, start, end

def allocate_registers(
    graph: nx.DiGraph,
) -> Tuple[Dict[int,int], Dict[int,int], int, Dict[int,int]]:
    # interval graph coloring / register allocation
    max_regs, start, end = calculate_min_hidden_states(graph)

    print(f"start {start}")
    print(f"end {end}")
    print(f"max-regs: {max_regs}")

    topo = list(map(int, graph.adj))
    free_regs = list(range(max_regs))
    heapify(free_regs)
    heap_active: List[Tuple[int,int,int]] = []  # (release_time, node, reg_id)
    regs_map: Dict[int,int] = {}

    for u in topo:
        now = start[u]
        while heap_active and heap_active[0][0] <= now:
            _, expired_u, reg = heappop(heap_active)
            free_regs.append(reg)

        # print(free_regs)
        reg_u = heappop(free_regs)
        regs_map[u] = reg_u
        heappush(heap_active, (end[u], u, reg_u))

    return start, end, max_regs, regs_map
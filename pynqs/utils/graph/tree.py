import networkx as nx
import matplotlib.pyplot as plt


def _format_spin_label(value):
    if value % 2 == 0:
        return str(value // 2)
    return f"{value}/2"


def draw_tree(
    tree,
    figsize=None,
    node_size=3200,
    font_size=22,
    width=None,
    level_gap=None,
    linewidths=2.5,
    edge_width=4.0,
):
    """
    tree: list of [left, right, value]
    e.g.
        tree = [
            [1, 2, 1],
            [None, None, 2],
            [3, 4, 3],
            [None, None, 1],
            [None, None, 2],
        ]
    """

    children = set()
    for left, right, _ in tree:
        if left is not None:
            children.add(left)
        if right is not None:
            children.add(right)

    all_nodes = set(range(len(tree)))
    roots = list(all_nodes - children)
    root = roots[0]

    G = nx.DiGraph()
    labels = {}

    for i, (left, right, value) in enumerate(tree):
        G.add_node(i)
        labels[i] = _format_spin_label(value)

        if left is not None:
            G.add_edge(i, left)
        if right is not None:
            G.add_edge(i, right)

    pos = {}
    leaf_order = []
    depth_counts = {}

    def collect_tree_stats(node, depth=0):
        depth_counts[depth] = depth_counts.get(depth, 0) + 1
        left, right, _ = tree[node]
        if left is None and right is None:
            leaf_order.append(node)
            return
        if left is not None:
            collect_tree_stats(left, depth + 1)
        if right is not None:
            collect_tree_stats(right, depth + 1)

    collect_tree_stats(root)
    n_levels = max(depth_counts) + 1
    n_leaf = len(leaf_order)

    if width is None:
        width = max(1.9, 1.15 * max(1, n_leaf - 1) + 0.45)
    if level_gap is None:
        max_depth = max(1, n_levels - 1)
        level_gap = min(1.35, max(0.8, 2.5 / max_depth))
    if figsize is None:
        figsize = (
            max(2.8, 0.95 * width + 0.7),
            max(2.8, 1.2 * level_gap * max(1, n_levels - 1) + 1.0),
        )

    if n_leaf == 1:
        leaf_x = {leaf_order[0]: 0.0}
    else:
        xs = [-width / 2 + width * i / (n_leaf - 1) for i in range(n_leaf)]
        leaf_x = {node: x for node, x in zip(leaf_order, xs)}

    def dfs(node, depth):
        left, right, _ = tree[node]
        if left is None and right is None:
            x_center = leaf_x[node]
        else:
            child_x = []
            if left is not None:
                child_x.append(dfs(left, depth + 1))
            if right is not None:
                child_x.append(dfs(right, depth + 1))
            x_center = sum(child_x) / len(child_x)
        pos[node] = (x_center, -depth * level_gap)
        return x_center

    dfs(root, 0)

    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx(
        G,
        pos=pos,
        labels=labels,
        node_size=node_size,
        node_color="white",
        edgecolors="black",
        width=edge_width,
        linewidths=linewidths,
        arrows=False,
        font_size=font_size,
        font_weight="bold",
        ax=ax,
    )

    xs = [xy[0] for xy in pos.values()]
    ys = [xy[1] for xy in pos.values()]
    x_spacing = width / max(1, n_leaf - 1)
    axis_width_in = max(figsize[0] * 0.9, 1e-6)
    node_radius_pt = (node_size / 3.141592653589793) ** 0.5
    node_radius_in = node_radius_pt / 72.0
    data_per_in = max(width, 1e-6) / axis_width_in
    node_radius_x = node_radius_in * data_per_in
    x_pad = max(0.45, 0.6 * x_spacing, 1.35 * node_radius_x)
    y_pad = max(0.35, level_gap * 0.55)
    ax.set_xlim(min(xs) - x_pad, max(xs) + x_pad)
    ax.set_ylim(min(ys) - y_pad, max(ys) + y_pad)
    ax.set_axis_off()
    fig.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.03)
    plt.show()

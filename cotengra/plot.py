import math
import functools
import importlib
import itertools
import collections

import numpy as np


def show_and_close(fn):
    @functools.wraps(fn)
    def wrapped(*args, show_and_close=True, **kwargs):
        import matplotlib.pyplot as plt

        fig, ax = fn(*args, **kwargs)

        if fig is not None:
            if show_and_close:
                plt.show()
                plt.close(fig)

        return fig, ax

    return wrapped


NEUTRAL_STYLE = {
    "axes.edgecolor": (0.5, 0.5, 0.5),
    "axes.facecolor": (0, 0, 0, 0),
    "axes.grid": True,
    "axes.labelcolor": (0.5, 0.5, 0.5),
    "axes.spines.right": False,
    "axes.spines.top": False,
    "figure.facecolor": (0, 0, 0, 0),
    "grid.alpha": 0.1,
    "grid.color": (0.5, 0.5, 0.5),
    "legend.frameon": False,
    "text.color": (0.5, 0.5, 0.5),
    "xtick.color": (0.5, 0.5, 0.5),
    "xtick.minor.visible": True,
    "ytick.color": (0.5, 0.5, 0.5),
    "ytick.minor.visible": True,
}


def use_neutral_style(fn):

    @functools.wraps(fn)
    def new_fn(*args, use_neutral_style=True, **kwargs):
        import matplotlib as mpl

        if not use_neutral_style:
            return fn(*args, **kwargs)

        with mpl.rc_context(NEUTRAL_STYLE):
            return fn(*args, **kwargs)

    return new_fn


def plot_trials_alt(self, y=None, width=800, height=300):
    """Plot the trials interactively using altair."""
    import altair as alt
    import pandas as pd

    df = self.to_df()

    if y is None:
        y = self.minimize

    if y == "size":
        best_y = math.log2(self.best[y])
        ylabel = "log2[SIZE]"
    if y == "flops":
        best_y = math.log10(self.best[y])
        ylabel = "log10[FLOPS]"

    hline = (
        alt.Chart(pd.DataFrame({"best": [best_y]}))
        .mark_rule(strokeDash=[2, 2], color="grey")
        .encode(y="best:Q")
    )

    scatter = (
        alt.Chart(df)
        .mark_point()
        .encode(
            x="run:Q",
            y=alt.Y("{}:Q".format(y), title=ylabel),
            size=alt.Size(
                "random_strength:Q",
                scale=alt.Scale(range=[50, 150], type="log"),
                legend=None,
            ),
            color="method:N",
            tooltip=list(df.columns),
        )
    )

    return (
        (hline + scatter)
        .properties(
            width=width,
            height=height,
        )
        .configure_axis(gridColor="rgb(248, 248, 248)")
    ).interactive()


_scatter_labels = {
    "size": "log2[SIZE]",
    "flops": "log10[FLOPS]",
    "write": "log10[WRITE]",
}


@show_and_close
@use_neutral_style
def plot_scatter(
    self,
    x="size",
    y="flops",
    figsize=(5, 5),
):
    import matplotlib.pyplot as plt
    import cotengra as ctg

    factor = None
    if x not in ("trial", "score"):
        xminimize = ctg.scoring.get_score_fn(x)
        x = getattr(xminimize, 'name', x)
        factor = getattr(xminimize, 'factor', 64)
    if y not in ("trial", "score"):
        yminimize = ctg.scoring.get_score_fn(y)
        y = getattr(yminimize, 'name', y)
        factor = getattr(yminimize, 'factor', 64)

    if factor is None:
        factor = 64

    N = len(self.scores)
    data = collections.defaultdict(lambda: collections.defaultdict(list))

    for i in range(N):
        method = self.method_choices[i]
        data[method]["trial"].append(i)
        data[method]["score"].append(self.scores[i])
        data[method]["size"].append(math.log2(self.costs_size[i]))
        f = self.costs_flops[i]
        w = self.costs_write[i]
        data[method]["flops"].append(math.log10(f))
        data[method]["write"].append(math.log10(w))
        data[method]["combo"].append(math.log10(f + factor * w))

    def parse_label(z):
        if z == "size":
            return "log2[SIZE]"
        if z in ("flops", "write"):
            return f"log10[{z.upper()}]"
        if z == "combo":
            return f"log10[FLOPS + {factor} * WRITE]"
        return z.upper()

    xlabel = parse_label(x)
    ylabel = parse_label(y)

    markers = itertools.cycle("oXsvPD^h*p<d8>H")

    fig, ax = plt.subplots(figsize=figsize)

    for method, datum in sorted(data.items()):
        ax.scatter(
            datum[x],
            datum[y],
            marker=next(markers),
            label=method,
            edgecolor="white",
        )

    ax.grid(True, color=(0.5, 0.5, 0.5), which="major", alpha=0.1)
    ax.set_axisbelow(True)

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=(len(data) * 6) // len(data),
        framealpha=0,
        handlelength=0,
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return fig, ax


def plot_trials(
    self,
    y="score",
    figsize=(8, 3),
    **kwargs,
):
    return plot_scatter(
        self,
        x="trial",
        y=y,
        figsize=figsize,
        **kwargs,
    )


def plot_scatter_alt(
    self,
    x="size",
    y="flops",
    color="run:Q",
    color_scheme="purplebluegreen",
    shape="method:N",
    width=400,
    height=400,
):
    """Plot the trials total flops vs max size."""
    import altair as alt

    df = self.to_df()
    scatter = (
        alt.Chart(df)
        .mark_point()
        .encode(
            x=alt.X(x, title=_scatter_labels[x], scale=alt.Scale(zero=False)),
            y=alt.Y(y, title=_scatter_labels[y], scale=alt.Scale(zero=False)),
            size=alt.Size(
                "random_strength:Q",
                scale=alt.Scale(range=[50, 150], type="log"),
                legend=None,
            ),
            shape=alt.Shape(shape),
            color=alt.Color(color, scale=alt.Scale(scheme=color_scheme)),
            tooltip=list(df.columns),
        )
    )
    return (
        scatter.properties(
            width=width,
            height=height,
        ).configure_axis(gridColor="rgb(248, 248, 248)")
    ).interactive()


def tree_to_networkx(tree):
    import networkx as nx

    G = nx.Graph()

    for c, (p, l, r) in enumerate(tree.traverse()):
        G.add_edge(
            p, l,
            size=math.log10(tree.get_size(l) + 1) + 1,
            weight=len(l),
            contraction=c,
        )
        G.add_edge(
            p, r,
            size=math.log10(tree.get_size(r) + 1) + 1,
            weight=len(r),
            contraction=c,
        )
        G.nodes[p]["contraction"] = min(c, G.nodes[p].get("contraction", c))
        G.nodes[l]["contraction"] = min(c, G.nodes[l].get("contraction", c))
        G.nodes[r]["contraction"] = min(c, G.nodes[r].get("contraction", c))

    for node in tree.info:
        G.nodes[node]["flops"] = math.log10(tree.get_flops(node) + 1) + 1

    return G


def hypergraph_compute_plot_info_G(
    H,
    G,
    highlight=(),
    node_color=(0.5, 0.5, 0.5, 1.0),
    edge_color=(0.0, 0.0, 0.0),
    edge_alpha=1 / 3,
    colormap="Spectral_r",
    centrality=False,
    sliced_inds=(),
):
    """Imbue the networkx representation, ``G``, of hypergraph, ``H`` with
    relevant plot information as node and edge attributes.
    """
    import matplotlib as mpl
    import matplotlib.cm

    for e in G.edges:
        ix = G.edges[e]["ind"]
        width = math.log2(H.size_dict.get(ix, 2))
        color = (
            (*edge_color[:3], edge_alpha)
            if ix not in highlight
            else (1.0, 0.0, 1.0, edge_alpha**0.5)
        )
        label = ix if not G.edges[e]["hyperedge"] else ""

        G.edges[e]["color"] = color
        G.edges[e]["width"] = width
        G.edges[e]["label"] = label
        G.edges[e]["style"] = "dotted" if ix in sliced_inds else "solid"

    if centrality:
        if centrality == "resistance":
            Cs = H.resistance_centrality()
        elif centrality == "simple":
            Cs = H.simple_centrality()
        else:
            Cs = centrality

        if isinstance(colormap, mpl.colors.Colormap):
            cmap = colormap
        else:
            cmap = getattr(matplotlib.cm, colormap)

    for nd in G.nodes:
        if G.nodes[nd]["hyperedge"]:
            color = (0.0, 0.0, 0.0, 0.0)
            label = str(nd)
        else:
            if centrality:
                c = Cs[nd]
                G.nodes[nd]["centrality"] = c
                color = cmap(c)
            else:
                color = node_color
            label = f"{nd}"  # H.inputs[nd]

        G.nodes[nd]["color"] = color
        G.nodes[nd]["label"] = label


def rotate(xy, theta):
    """Return a rotated set of points."""
    s = np.sin(theta)
    c = np.cos(theta)

    xyr = np.empty_like(xy)
    xyr[:, 0] = c * xy[:, 0] - s * xy[:, 1]
    xyr[:, 1] = s * xy[:, 0] + c * xy[:, 1]

    return xyr


def span(xy):
    """Return the vertical span of the points."""
    return xy[:, 1].max() - xy[:, 1].min()


def massage_pos(pos, nangles=12, flatten=False):
    """Rotate a position dict's points to cover a small vertical span"""
    xy = np.empty((len(pos), 2))
    for i, (x, y) in enumerate(pos.values()):
        xy[i, 0] = x
        xy[i, 1] = y

    thetas = np.linspace(0, 2 * np.pi, nangles, endpoint=False)
    rxys = (rotate(xy, theta) for theta in thetas)
    rxy0 = min(rxys, key=lambda rxy: span(rxy))

    if flatten:
        rxy0[:, 1] /= 2

    return dict(zip(pos, rxy0))


def layout_pygraphviz(
    G,
    prog="neato",
    dim=2,
    **kwargs,
):
    # TODO: fix nodes with pin attribute
    # TODO: initial positions
    # TODO: max iters
    # TODO: spring parameter
    import pygraphviz as pgv

    aG = pgv.AGraph()
    mapping = {}
    for nodea, nodeb in G.edges():
        s_nodea = str(nodea)
        s_nodeb = str(nodeb)
        mapping[s_nodea] = nodea
        mapping[s_nodeb] = nodeb
        aG.add_edge(s_nodea, s_nodeb)

    kwargs = {}

    if dim == 2.5:
        kwargs["dim"] = 3
        kwargs["dimen"] = 2
    else:
        kwargs["dim"] = kwargs["dimen"] = dim
    args = " ".join(f"-G{k}={v}" for k, v in kwargs.items())

    # run layout algorithm
    aG.layout(prog=prog, args=args)

    # extract layout
    pos = {}
    for snode, node in mapping.items():
        spos = aG.get_node(snode).attr["pos"]
        pos[node] = tuple(map(float, spos.split(",")))

    # normalize to unit square
    xmin = ymin = zmin = float("inf")
    xmax = ymax = zmaz = float("-inf")
    for x, y, *maybe_z in pos.values():
        xmin = min(xmin, x)
        xmax = max(xmax, x)
        ymin = min(ymin, y)
        ymax = max(ymax, y)
        for z in maybe_z:
            zmin = min(zmin, z)
            zmaz = max(zmaz, z)

    for node, (x, y, *maybe_z) in pos.items():
        pos[node] = (
            2 * (x - xmin) / (xmax - xmin) - 1,
            2 * (y - ymin) / (ymax - ymin) - 1,
            *(2 * (z - zmin) / (zmaz - zmin) - 1 for z in maybe_z),
        )

    return pos


HAS_FA2 = importlib.util.find_spec("fa2") is not None
HAS_PYGRAPHVIZ = importlib.util.find_spec("pygraphviz") is not None


def get_nice_pos(
    G,
    *,
    dim=2,
    layout="auto",
    initial_layout="auto",
    iterations="auto",
    k=None,
    use_forceatlas2=False,
    flatten=False,
):
    if (layout == "auto") and HAS_PYGRAPHVIZ:
        layout = "neato"

    if layout in ("dot", "neato", "fdp", "sfdp"):
        pos = layout_pygraphviz(G, prog=layout, dim=dim)

        if dim == 2:
            pos = massage_pos(pos, flatten=flatten)

        return pos

    import networkx as nx

    if layout != "auto":
        initial_layout = layout
        iterations = 0

    if initial_layout == "auto":
        # automatically select
        if len(G) <= 100:
            # usually nicest
            initial_layout = "kamada_kawai"
        else:
            # faster, but not as nice
            initial_layout = "spectral"

    if iterations == "auto":
        # the smaller the graph, the more iterations we can afford
        iterations = max(200, 1000 - len(G))

    if dim == 2.5:
        dim = 3
        project_back_to_2d = True
    else:
        project_back_to_2d = False

    # use spectral or other layout as starting point
    ly_opts = {"dim": dim} if dim != 2 else {}
    pos0 = getattr(nx, initial_layout + "_layout")(G, **ly_opts)

    # and then relax remaining using spring layout
    if iterations:
        if use_forceatlas2 is True:
            # turn on for more than 1 node
            use_forceatlas2 = 1
        elif use_forceatlas2 in (0, False):
            # never turn on
            use_forceatlas2 = float("inf")

        should_use_fa2 = HAS_FA2 and (len(G) > use_forceatlas2) and (dim == 2)

        if should_use_fa2:
            from fa2 import ForceAtlas2

            # NB: some versions of fa2 don't support the `weight_attr` option
            pos = ForceAtlas2(verbose=False).forceatlas2_networkx_layout(
                G, pos=pos0, iterations=iterations
            )
        else:
            pos = nx.spring_layout(
                G,
                pos=pos0,
                k=k,
                dim=dim,
                iterations=iterations,
            )
    else:
        pos = pos0

    if project_back_to_2d:
        # project back to 2d
        pos = {k: v[:2] for k, v in pos.items()}
        dim = 2

    if dim == 2:
        # finally rotate them to cover a small vertical span
        pos = massage_pos(pos)

    return pos


@show_and_close
@use_neutral_style
def plot_tree(
    tree,
    layout="ring",
    hypergraph_layout="auto",
    hypergraph_layout_opts=None,
    k=0.01,
    iterations=500,
    span=None,
    order=None,
    order_y_pow=1.0,
    edge_scale=1.0,
    node_scale=1.0,
    highlight=(),
    edge_color="size",
    edge_colormap="GnBu",
    edge_max_width=None,
    node_colormap="YlOrRd",
    node_color="flops",
    node_max_size=None,
    figsize=(5, 5),
    raw_edge_color=None,
    raw_edge_alpha=None,
    tree_root_height=True,
    tree_alpha=0.8,
    colorbars=True,
    plot_raw_graph=True,
    plot_leaf_labels=False,
    ax=None,
):
    """Plot a contraction tree using matplotlib."""
    import networkx as nx
    import matplotlib as mpl
    from matplotlib import pyplot as plt

    hypergraph_layout_opts = (
        {} if hypergraph_layout_opts is None else dict(hypergraph_layout_opts)
    )
    hypergraph_layout_opts.setdefault("layout", hypergraph_layout)

    if raw_edge_color is None:
        raw_edge_color = (0.5, 0.5, 0.5)
    if raw_edge_alpha is None:
        raw_edge_alpha = 0.5

    # draw the contraction tree
    G_tree = tree_to_networkx(tree)
    leaves = tree.get_leaves_ordered()

    # set the tree edge and node sizes
    edge_weights = [
        edge_scale * 0.8 * G_tree.edges[e]["size"] for e in G_tree.edges
    ]
    node_weights = [
        node_scale * 8.0 * G_tree.nodes[n]["flops"] for n in G_tree.nodes
    ]

    # tree edge colors
    if edge_color == "order":
        ew_range = 0, tree.N - 1
        enorm = mpl.colors.Normalize(*ew_range, clip=True)
        if not isinstance(edge_colormap, mpl.colors.Colormap):
            edge_colormap = getattr(mpl.cm, edge_colormap)
        emapper = mpl.cm.ScalarMappable(norm=enorm, cmap=edge_colormap)
        edge_colors = [
            emapper.to_rgba(d['contraction'])
            for _, _, d in G_tree.edges(data=True)
        ]
    else:
        ew_range = min(edge_weights), max(edge_weights)
        enorm = mpl.colors.Normalize(*ew_range, clip=True)
        if not isinstance(edge_colormap, mpl.colors.Colormap):
            edge_colormap = getattr(mpl.cm, edge_colormap)
        emapper = mpl.cm.ScalarMappable(norm=enorm, cmap=edge_colormap)
        edge_colors = [emapper.to_rgba(x) for x in edge_weights]

    # tree node colors
    if node_color == "order":
        nw_range = 0, tree.N - 1
        nnorm = mpl.colors.Normalize(*nw_range, clip=True)
        if not isinstance(node_colormap, mpl.colors.Colormap):
            node_colormap = getattr(mpl.cm, node_colormap)
        nmapper = mpl.cm.ScalarMappable(norm=enorm, cmap=node_colormap)
        node_colors = [
            nmapper.to_rgba(d['contraction'])
            for _, d in G_tree.nodes(data=True)
        ]
    else:
        nw_range = min(node_weights), max(node_weights)
        nnorm = mpl.colors.Normalize(*nw_range, clip=True)
        if not isinstance(node_colormap, mpl.colors.Colormap):
            node_colormap = getattr(mpl.cm, node_colormap)
        nmapper = mpl.cm.ScalarMappable(norm=nnorm, cmap=node_colormap)
        node_colors = [nmapper.to_rgba(x) for x in node_weights]

    # plot the raw connectivity of the underlying graph
    if plot_raw_graph:
        H_tn = tree.get_hypergraph()
        G_tn = H_tn.to_networkx(as_tree_leaves=True)
        hypergraph_compute_plot_info_G(
            H_tn,
            G_tn,
            highlight=highlight,
            edge_color=raw_edge_color,
            edge_alpha=raw_edge_alpha,
            centrality=False,
            sliced_inds=tree.sliced_inds,
        )
        any_hyper = G_tn.graph["any_hyper"]

    if layout == "tent":
        # place raw graph first
        hypergraph_layout_opts.setdefault("flatten", True)
        pos = get_nice_pos(G_tn, **hypergraph_layout_opts)

        xmin = min(v[0] for v in pos.values())
        xmax = max(v[0] for v in pos.values())

        if span is not None:
            # place the intermediates vertically above the leaf nodes that they
            # are mapped to in the span
            ymax = max(v[1] for v in pos.values())
            if span is True:
                span = tree.get_spans()[0]
            for node in G_tree.nodes:
                if len(node) == 1:
                    continue
                raw_pos = pos[span[node]]
                pos[node] = (
                    raw_pos[0],
                    ymax + len(node) * (xmax - xmin) / tree.N,
                )

        elif order is not None:
            if order is True:
                order = None
            ymax = max(v[1] for v in pos.values())
            for i, (p, _, _) in enumerate(tree.traverse(order)):
                x_av, y_av = 0.0, 0.0
                for ti in p:
                    coo_i = pos[frozenset([ti])]
                    x_av += coo_i[0] / len(p)
                    y_av += coo_i[1] / len(p)
                y_av = (
                    ymax
                    + float(tree_root_height)
                    * (xmax - xmin)
                    * ((i + 1) / tree.N) ** order_y_pow
                )
                pos[p] = (x_av, y_av)

        else:
            # place the top of the tree
            pos[tree.root] = (0, float(tree_root_height) * (xmax - xmin))
            # layout the tree nodes between bottom and top
            # first need to filter out TN nodes not appearing in tree
            tree_pos = {k: v for k, v in pos.items() if k in G_tree.nodes}
            pos.update(
                nx.spring_layout(
                    G_tree,
                    fixed=tree_pos,
                    pos=tree_pos,
                    k=k,
                    iterations=iterations,
                )
            )

    elif layout == "ring":
        # work out a layout based on leaves in circle
        pos = {tree.root: (0, 0)}
        for i, x in enumerate(leaves):
            pos[x] = (
                math.sin(2 * math.pi * i / tree.N),
                math.cos(2 * math.pi * i / tree.N),
            )
        # layout the remaining tree nodes
        pos = nx.spring_layout(
            G_tree, fixed=pos, pos=pos, k=k, iterations=iterations
        )
        # if there are hyperedges layout the faux-nodes
        if plot_raw_graph and any_hyper:
            fixed_raw = {k: v for k, v in pos.items() if k in leaves}
            pos.update(
                nx.spring_layout(
                    G_tn,
                    fixed=fixed_raw,
                    pos=fixed_raw,
                    k=k,
                    iterations=iterations,
                )
            )

    elif layout == "span":
        # place raw graph first
        hypergraph_layout_opts.setdefault("flatten", False)
        pos = get_nice_pos(G_tn, **hypergraph_layout_opts)
        if span is None:
            span = tree.get_spans()[0]
        pos.update({node: pos[span[node]] for node in G_tree.nodes})

    created_ax = ax is None
    if created_ax:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_aspect("equal")
        ax.axis("off")
    else:
        fig = None

    if plot_raw_graph:
        nx.draw_networkx_edges(
            G_tn,
            pos=pos,
            ax=ax,
            width=[G_tn.edges[e]["width"] for e in G_tn.edges],
            style=[G_tn.edges[e]["style"] for e in G_tn.edges],
            edge_color=[G_tn.edges[e]["color"] for e in G_tn.edges],
        )

    if plot_leaf_labels:
        nx.draw_networkx_labels(
            G_tn,
            pos=pos,
            ax=ax,
            labels={k: int(next(iter(k))) for k in G_tn.nodes},
        )

    if edge_max_width is not None:
        edge_weights = [min(e, edge_max_width) for e in edge_weights]
    if node_max_size is not None:
        node_weights = [min(n, node_max_size) for n in node_weights]

    nx.draw_networkx_edges(
        G_tree,
        pos=pos,
        ax=ax,
        width=edge_weights,
        edge_color=edge_colors,
        alpha=tree_alpha,
    )
    nx.draw_networkx_nodes(
        G_tree,
        pos=pos,
        ax=ax,
        node_size=node_weights,
        node_color=node_colors,
        alpha=tree_alpha,
    )

    if colorbars and created_ax:
        min_size = math.log2(min(tree.get_size(x) for x in tree.info))
        max_size = math.log2(max(tree.get_size(x) for x in tree.info))
        edge_norm = mpl.colors.Normalize(vmin=min_size, vmax=max_size)
        ax_l = fig.add_axes([-0.02, 0.25, 0.02, 0.5])
        cb_l = mpl.colorbar.ColorbarBase(
            ax_l, cmap=edge_colormap, norm=edge_norm
        )
        cb_l.outline.set_visible(False)
        ax_l.yaxis.tick_left()
        ax_l.set(title="log2[SIZE]")

        min_flops = math.log10(
            min(max(tree.get_flops(x), 1) for x in tree.info)
        )
        max_flops = math.log10(
            max(max(tree.get_flops(x), 1) for x in tree.info)
        )
        node_norm = mpl.colors.Normalize(vmin=min_flops, vmax=max_flops)
        ax_r = fig.add_axes([1.0, 0.25, 0.02, 0.5])
        cb_r = mpl.colorbar.ColorbarBase(
            ax_r, cmap=node_colormap, norm=node_norm
        )
        cb_r.outline.set_visible(False)
        ax_r.yaxis.tick_right()
        ax_r.set(title="log10[FLOPS]")

    # with warnings.catch_warnings():
    #     warnings.simplefilter('ignore', UserWarning)
    #     plt.tight_layout()

    return fig, ax


@functools.wraps(plot_tree)
def plot_tree_ring(tree, **kwargs):
    kwargs.setdefault("tree_alpha", 0.97)
    kwargs.setdefault("raw_edge_alpha", 1 / 10)
    return plot_tree(tree, "ring", **kwargs)


@functools.wraps(plot_tree)
def plot_tree_tent(tree, **kwargs):
    kwargs.setdefault("tree_alpha", 0.7)
    kwargs.setdefault("raw_edge_alpha", 1 / 5)
    kwargs.setdefault("edge_scale", 1 / 2)
    kwargs.setdefault("node_scale", 1 / 3)
    return plot_tree(tree, "tent", **kwargs)


@functools.wraps(plot_tree)
def plot_tree_span(tree, **kwargs):
    kwargs.setdefault("edge_colormap", "viridis")
    kwargs.setdefault("edge_scale", 2)
    kwargs.setdefault("edge_max_width", 3)
    kwargs.setdefault("node_colormap", "plasma")
    kwargs.setdefault("node_scale", 2)
    kwargs.setdefault("node_max_size", 30)
    return plot_tree(tree, "span", **kwargs)


def tree_to_df(tree):
    import pandas as pd

    sizes = []
    isizes = []
    psizes = []
    flops = []
    stages = []
    scalings = []
    ranks = []

    for k in tree.info:
        if tree.get_flops(k) == 0:
            continue
        sizes.append(max(1, tree.get_size(k)))
        isizes.append(sum(map(tree.get_size, tree.children[k])))
        psizes.append(sizes[-1] + isizes[-1])
        flops.append(max(1, tree.get_flops(k)))
        stages.append(len(k))
        ranks.append(len(tree.get_legs(k)))
        scalings.append(len(tree.get_involved(k)))

    return pd.DataFrame(
        {
            "out-size": sizes,
            "input-size": isizes,
            "peak-size": psizes,
            "flops": flops,
            "stage": stages,
            "rank": ranks,
            "scaling": scalings,
        }
    )


@show_and_close
def plot_contractions(
    tree,
    x="peak-size",
    y="flops",
    color="stage",
    size="scaling",
    point_opacity=0.8,
    color_scheme="viridis_r",
    x_scale="log",
    y_scale="log",
    figsize=(6, 4),
):
    from matplotlib import pyplot as plt
    import matplotlib as mpl
    import seaborn as sns

    df = tree_to_df(tree)
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    sns.scatterplot(
        x=x,
        y=y,
        hue=color,
        size=size,
        palette=color_scheme,
        data=df,
        alpha=point_opacity,
    )

    ax.set(yscale=y_scale, xscale=x_scale)
    ax.grid(True, c=(0.98, 0.98, 0.98))
    ax.set_axisbelow(True)
    ax.get_legend().remove()

    ax_cb = fig.add_axes([1.02, 0.25, 0.02, 0.55])
    ax_cb.set(title="Stage")
    if not isinstance(color_scheme, mpl.colors.Colormap):
        color_scheme = getattr(mpl.cm, color_scheme)
    cb = mpl.colorbar.ColorbarBase(ax_cb, cmap=color_scheme)
    cb.outline.set_visible(False)

    # with warnings.catch_warnings():
    #     warnings.simplefilter('ignore', UserWarning)
    #     plt.tight_layout()

    return fig, ax


def plot_contractions_alt(
    tree,
    x="peak-size",
    y="flops",
    color="stage",
    size="scaling",
    width=400,
    height=400,
    point_opacity=0.8,
    color_scheme="lightmulti",
    x_scale="log",
    y_scale="log",
    color_scale="log",
    size_scale="linear",
):
    import altair as alt

    df = tree_to_df(tree)
    chart = alt.Chart(df)

    encoding = chart.encode(
        x=alt.X(x, scale=alt.Scale(type=x_scale, padding=10)),
        y=alt.Y(y, scale=alt.Scale(type=y_scale, padding=10)),
        color=alt.Color(
            color, scale=alt.Scale(scheme=color_scheme, type=color_scale)
        ),
        size=alt.Size(size, scale=alt.Scale(type=size_scale)),
        tooltip=list(df.columns),
    )
    plot = encoding.mark_point(opacity=point_opacity)

    return (
        plot.configure_axis(gridColor="rgb(248,248,248)")
        .properties(width=width, height=height)
        .interactive()
    )


def slicefinder_to_df(slice_finder, relative_flops=False):
    import pandas as pd

    all_ccs = sorted(slice_finder.costs.values(), key=lambda x: -x.size)
    maxsizes = [math.log2(x.size) for x in all_ccs]
    if relative_flops:
        newflops = [
            float(x.total_flops) / float(slice_finder.cost0.total_flops)
            for x in all_ccs
        ]
    else:
        newflops = [math.log10(x.total_flops) for x in all_ccs]

    numslices = [math.log2(x.nslices) for x in all_ccs]

    return pd.DataFrame(
        {
            "log2[SIZE]": maxsizes,
            "log10[FLOPS]": newflops,
            "log2[NSLICES]": numslices,
        }
    )


@show_and_close
def plot_slicings(
    slice_finder,
    color_scheme="RdYlBu_r",
    relative_flops=False,
    figsize=(6, 3),
    point_opacity=0.8,
):
    import matplotlib as mpl
    from matplotlib import pyplot as plt
    import seaborn as sns

    df = slicefinder_to_df(slice_finder, relative_flops=relative_flops)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    sns.scatterplot(
        x="log2[SIZE]",
        y="log10[FLOPS]",
        hue="log2[NSLICES]",
        palette=color_scheme,
        data=df,
        alpha=point_opacity,
    )

    ax.set(xlim=(max(df["log2[SIZE]"] + 1), min(df["log2[SIZE]"] - 1)))
    ax.grid(True, c=(0.98, 0.98, 0.98))
    ax.set_axisbelow(True)
    ax.get_legend().remove()

    ax_cb = fig.add_axes([1.02, 0.25, 0.02, 0.55])
    nm = mpl.colors.Normalize(
        vmin=df["log2[NSLICES]"].min(), vmax=df["log2[NSLICES]"].max()
    )
    if not isinstance(color_scheme, mpl.colors.Colormap):
        color_scheme = getattr(mpl.cm, color_scheme)
    cb = mpl.colorbar.ColorbarBase(ax_cb, cmap=color_scheme, norm=nm)
    cb.outline.set_visible(False)

    # with warnings.catch_warnings():
    #     warnings.simplefilter('ignore', UserWarning)
    #     plt.tight_layout()

    return fig, ax


def plot_slicings_alt(
    slice_finder, color_scheme="redyellowblue", relative_flops=False
):
    import altair as alt

    df = slicefinder_to_df(slice_finder, relative_flops=relative_flops)

    df["size"] = 2 ** df["log2[SIZE]"]
    df["flops"] = 10 ** df["log10[FLOPS]"]
    df["numslices"] = 2 ** df["log2[NSLICES]"]

    return (
        alt.Chart(df)
        .mark_point()
        .encode(
            x=alt.X(
                "size:Q",
                sort="descending",
                scale=alt.Scale(type="log", zero=False),
            ),
            y=alt.Y("flops:Q", scale=alt.Scale(type="log")),
            color=alt.Color(
                "numslices:Q",
                scale=alt.Scale(type="log", scheme=color_scheme),
                sort="descending",
            ),
            tooltip=list(df.columns),
        )
        .configure_axis(gridColor="rgb(248,248,248)")
        .interactive()
    )


@show_and_close
@use_neutral_style
def plot_hypergraph(
    H,
    *,
    highlight=(),
    centrality="simple",
    colormap="plasma",
    pos=None,
    dim=2,
    layout="auto",
    initial_layout="auto",
    iterations="auto",
    k=None,
    use_forceatlas2=False,
    flatten=False,
    node_size=None,
    node_color=(0.5, 0.5, 0.5, 1.0),
    edge_alpha=1 / 3,
    edge_style="solid",
    hyperedge_style="dashed",
    draw_edge_labels=None,
    edge_labels_font_size=8,
    edge_labels_font_family="monospace",
    info=None,
    ax=None,
    figsize=(5, 5),
):
    import networkx as nx
    import matplotlib as mpl
    from matplotlib import pyplot as plt
    from matplotlib.colors import to_rgb

    font_color = edge_color = (0.5, 0.5, 0.5, 0.9)

    # set the size of the nodes
    if node_size is None:
        node_size = 1000 / len(H) ** 0.7
    node_outline_size = min(3, node_size**0.5 / 5)

    G = H.to_networkx()
    hypergraph_compute_plot_info_G(
        H=H,
        G=G,
        highlight=highlight,
        node_color=node_color,
        edge_color=edge_color,
        edge_alpha=edge_alpha,
        colormap=colormap,
        centrality=centrality,
    )

    if pos is None:
        pos = get_nice_pos(
            G,
            dim=dim,
            layout=layout,
            initial_layout=initial_layout,
            iterations=iterations,
            k=k,
            use_forceatlas2=use_forceatlas2,
            flatten=flatten,
        )

    created_ax = ax is None
    if created_ax:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_aspect("equal")
        ax.axis("off")
    else:
        fig = None

    nx.draw_networkx_nodes(
        G,
        pos=pos,
        ax=ax,
        node_size=node_size,
        node_color=[G.nodes[nd]["color"] for nd in G.nodes],
        edgecolors=[
            tuple(
                (1.0 if i == 3 else 0.8) * c
                for i, c in enumerate(G.nodes[nd]["color"])
            )
            for nd in G.nodes
        ],
        linewidths=node_outline_size,
    )
    nx.draw_networkx_edges(
        G,
        pos=pos,
        ax=ax,
        width=[G.edges[e]["width"] for e in G.edges],
        edge_color=[G.edges[e]["color"] for e in G.edges],
        style=[
            hyperedge_style if G.edges[e]["hyperedge"] else edge_style
            for e in G.edges
        ],
    )

    if draw_edge_labels is None:
        draw_edge_labels = len(H) <= 20

    if draw_edge_labels:
        nx.draw_networkx_labels(
            G,
            pos=pos,
            ax=ax,
            labels=dict(
                zip(G.nodes, (G.nodes[nd]["label"] for nd in G.nodes))
            ),
            font_size=edge_labels_font_size,
            font_color=font_color,
            font_family=edge_labels_font_family,
            bbox={
                "color": to_rgb(mpl.rcParams["figure.facecolor"]),
                "alpha": 0.0,
            },
        )
        nx.draw_networkx_edge_labels(
            G,
            pos=pos,
            ax=ax,
            edge_labels=dict(
                zip(G.edges, (G.edges[e]["label"] for e in G.edges))
            ),
            font_size=edge_labels_font_size,
            font_color=font_color,
            font_family=edge_labels_font_family,
            bbox={
                "color": to_rgb(mpl.rcParams["figure.facecolor"]),
                "alpha": 0.0,
            },
        )

    if info is not None and "pos" in info:
        info["pos"] = pos

    return fig, ax


@show_and_close
def plot_tree_rubberband(
    tree,
    order=None,
    colormap="Spectral",
    with_edge_labels=None,
    with_node_labels=None,
    highlight=(),
    centrality=False,
    layout="auto",
    node_size=None,
    node_color=(0.5, 0.5, 0.5, 1.0),
    edge_alpha=1 / 3,
    edge_style="solid",
    hyperedge_style="dashed",
    draw_edge_labels=None,
    edge_labels_font_size=8,
    edge_labels_font_family="monospace",
    iterations=500,
    ax=None,
    figsize=(5, 5),
):
    """Plot a ``ContractionTree`` using 'rubberbands' to represent intermediate
    contractions / subgraphs - requires ``quimb``. This can be intuitive
    for small and planar contractions.
    """
    from quimb.experimental.schematic import Drawing
    import matplotlib as mpl

    H = tree.get_hypergraph()
    info = {"pos": None}
    fig, ax = plot_hypergraph(
        H,
        highlight=highlight,
        centrality=centrality,
        layout=layout,
        node_size=node_size,
        node_color=node_color,
        edge_alpha=edge_alpha,
        edge_style=edge_style,
        hyperedge_style=hyperedge_style,
        draw_edge_labels=draw_edge_labels,
        edge_labels_font_size=edge_labels_font_size,
        edge_labels_font_family=edge_labels_font_family,
        iterations=iterations,
        figsize=figsize,
        info=info,
        show_and_close=False,
    )
    pos = info["pos"]

    if isinstance(colormap, str):
        cmap = mpl.cm.get_cmap(colormap)
    else:
        cmap = colormap

    d = Drawing(ax=ax)

    counts = collections.defaultdict(int)
    for i, (p, _, _) in enumerate(tree.traverse()):
        pts = [pos[node] for node in p]
        for node in p:
            counts[node] += 1
        radius = [0.05 + 0.01 * counts[node] for node in p]
        prog = i / (tree.N - 2)
        color = cmap(prog)
        d.patch_around(
            pts,
            resolution=20,
            radius=radius,
            edgecolor=color,
            facecolor='none',
            linestyle='-',
            zorder=-prog,
        )

    return fig, ax

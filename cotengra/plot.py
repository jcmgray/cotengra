"""Hypergraph, optimizer, and contraction tree visualization tools."""

import collections
import functools
import importlib
import itertools
import math


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
    cumulative_time=False,
    plot_best=False,
    figsize=(5, 5),
):
    import matplotlib.pyplot as plt

    import cotengra as ctg
    from cotengra.schematic import hash_to_color

    factor = None
    if x not in ("trial", "score", "time"):
        xminimize = ctg.scoring.get_score_fn(x)
        x = getattr(xminimize, "name", x)
        factor = getattr(xminimize, "factor", 64)
    if y not in ("trial", "score"):
        yminimize = ctg.scoring.get_score_fn(y)
        y = getattr(yminimize, "name", y)
        factor = getattr(yminimize, "factor", 64)

    if factor is None:
        factor = 64

    N = len(self.scores)
    data = collections.defaultdict(lambda: collections.defaultdict(list))

    ttotal = 0
    best = float("inf")
    bestx = []
    besty = []

    for i in range(N):
        method = self.method_choices[i]
        data_method = data[method]

        data_method["trial"].append(i)
        if cumulative_time:
            ttotal += self.times[i]
            data_method["time"].append(ttotal)
        else:
            data_method["time"].append(self.times[i])

        scorei = self.scores[i]
        sizei = math.log2(self.costs_size[i])
        f = self.costs_flops[i]
        w = self.costs_write[i]
        flopsi = math.log10(f)
        writei = math.log10(w)
        comboi = math.log10(f + factor * w)

        data_method["score"].append(scorei)
        data_method["size"].append(sizei)
        data_method["flops"].append(flopsi)
        data_method["write"].append(writei)
        data_method["combo"].append(comboi)

        if data_method[y][-1] < best:
            bestx.append(data_method[x][-1])
            besty.append(best)
            best = data_method[y][-1]
            bestx.append(data_method[x][-1])
            besty.append(best)

    bestx.append(data_method[x][-1])
    besty.append(best)

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
            color=hash_to_color(method),
            marker=next(markers),
            label=method,
            edgecolor="white",
        )

    if plot_best:
        ax.plot(
            bestx,
            besty,
            color=(0, 0.7, 0.3, 0.5),
            zorder=10,
        )
        ax.text(
            bestx[-1],
            besty[-1],
            f"{besty[-1]:.2f}",
            ha="left",
            va="center",
            color=(0, 0.7, 0.3),
            fontsize=8,
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
    *,
    x="trial",
    y="score",
    figsize=(8, 3),
    cumulative_time=True,
    plot_best=True,
    **kwargs,
):
    return plot_scatter(
        self,
        x=x,
        y=y,
        figsize=figsize,
        cumulative_time=cumulative_time,
        plot_best=plot_best,
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


def logxextrapolate(p1, p2, x):
    x1, y1 = p1
    x2, y2 = p2
    # Calculate the slope (m) and intercept (b) in log-x space
    m = (y2 - y1) / (math.log(x2) - math.log(x1))
    b = y1 - m * math.log(x1)
    return m * math.log(x) + b


def mapper(y, x, mn, mx):
    return (x, (y - mn) / (mx - mn))


def mapper_cat(c, x, lookup):
    return (x, lookup[c])


@show_and_close
def plot_parameters_parallel(
    self,
    method,
    colormap="Spectral",
    smoothing=0.5,
    rasterized=None,
    **drawing_opts,
):
    """Plot the parameter choices for a given method in parallel coordinates.

    Parameters
    ----------
    method : str
        The method to plot the parameter choices for.
    colormap : str, optional
        The colormap to use for coloring the curves.
    smoothing : float, optional
        The amount of smoothing to apply to the curves.
    rasterized : bool, optional
        Whether to rasterize the plot.
    drawing_opts : dict, optional
        Additional options to pass to the Drawing constructor.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plot.
    ax : matplotlib.axes.Axes
        The axes containing the plot.
    """
    import matplotlib as mpl
    from cotengra.schematic import Drawing

    # collect only rows with the same method
    columns = {"score": []}
    for m, p, s in zip(self.method_choices, self.param_choices, self.scores):
        if m != method:
            continue
        for k, v in p.items():
            try:
                columns[k].append(v)
            except KeyError:
                columns[k] = [v]
        columns["score"].append(s)

    ncol = len(columns)
    nrow = len(columns["score"])

    # determine ranges for mapping parameters into [0, 1]
    ranges = {}
    mappers = {}
    order = sorted(columns, key=lambda k: (k == "score", k))
    for x, k in enumerate(order):
        vs = columns[k]
        if isinstance(vs[0], str):
            # categorical
            values = sorted(set(vs))
            ranges[k] = values
            lookup = {
                c: (0.5 if len(values) == 1 else x / (len(values) - 1))
                for x, c in enumerate(values)
            }
            mappers[k] = functools.partial(mapper_cat, x=x, lookup=lookup)
        else:
            # numerical
            mn = min(vs)
            mx = max(vs)
            ranges[k] = (mn, mx)
            mappers[k] = functools.partial(mapper, x=x, mn=mn, mx=mx)

    drawing_opts.setdefault("figsize", (2 * ncol, 2 * ncol))
    d = Drawing(**drawing_opts)

    # draw the curves through the mapped points
    # XXX: use np.nanquantile to map score colors, ignoring outliers
    cmap = mpl.colormaps.get_cmap(colormap)
    linewidth = logxextrapolate((1, 2), (1000, 0.5), nrow)
    alpha = logxextrapolate((1, 1.0), (1000, 0.25), nrow)
    if rasterized is None:
        rasterized = nrow > 64

    for i in range(nrow):
        coos = [mappers[k](columns[k][i]) for k in order]
        # last column is the score
        mapped_score = coos[-1][1]
        c = cmap(1 - mapped_score)
        d.curve(
            coos,
            color=c,
            alpha=alpha,
            linewidth=linewidth,
            smoothing=smoothing,
            # make best and worst appear on top, slightly prefer best
            zorder=1 + abs(0.51 - mapped_score),
            rasterized=rasterized,
        )

    # label the axes and data ranges
    for j, k in enumerate(ranges):
        d.line((j, 0), (j, 1), color=(0.5, 0.5, 0.5), linewidth=0.5)
        d.text((j, 1.2), k, color=(0.5, 0.5, 0.5))
        rk = ranges[k]
        if isinstance(rk, list):
            # categorical
            for o in rk:
                d.text(
                    mappers[k](o),
                    o,
                    color=(0.5, 0.5, 0.5),
                    va="bottom",
                    zorder=10,
                )
        else:
            # numerical
            d.text((j, 1.1), f"{ranges[k][1]:.4g}", color=(0.5, 0.5, 0.5))
            d.text((j, -0.1), f"{ranges[k][0]:.4g}", color=(0.5, 0.5, 0.5))

    return d.fig, d.ax


def tree_to_networkx(tree):
    import networkx as nx

    G = nx.Graph()

    for c, (p, l, r) in enumerate(tree.traverse()):
        G.add_edge(
            p,
            l,
            size=math.log10(tree.get_size(l) + 1) + 1,
            weight=len(l),
            contraction=c,
        )
        G.add_edge(
            p,
            r,
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
    edge_style="-",
    hyperedge_style="--",
    sliced_style=":",
):
    """Imbue the networkx representation, ``G``, of hypergraph, ``H`` with
    relevant plot information as node and edge attributes.
    """
    import matplotlib as mpl
    import matplotlib.cm

    if edge_color is False:
        edge_color = (0.5, 0.5, 0.5)

    if edge_color is True:
        from .schematic import hash_to_color

        def _edge_colorer(ix):
            if ix in highlight:
                return (1.0, 0.0, 1.0, edge_alpha**0.5)
            return hash_to_color(ix)

    else:
        rgb = (*mpl.colors.to_rgb(edge_color), edge_alpha)

        def _edge_colorer(ix):
            if ix in highlight:
                return (1.0, 0.0, 1.0, edge_alpha**0.5)
            return rgb

    for *_, edata in G.edges(data=True):
        ix = edata["ind"]
        width = math.log2(H.size_dict.get(ix, 2))
        color = _edge_colorer(ix)

        if edata["hyperedge"] or edata["output"]:
            label = ""
        else:
            label = f"{ix}"

        if ix in sliced_inds:
            style = sliced_style
        elif edata["hyperedge"]:
            style = hyperedge_style
        else:
            style = edge_style

        edata["color"] = color
        edata["width"] = width
        edata["label"] = label
        edata["style"] = style

        if "multi" in edata:
            multidata = edata["multi"]
            for ix in multidata["inds"]:
                multidata.setdefault("widths", []).append(
                    math.log2(H.size_dict.get(ix, 2))
                )
                multidata.setdefault("colors", []).append(_edge_colorer(ix))
                # multiedges can only be inner non-hyper indices
                multidata.setdefault("labels", []).append(f"{ix}")
                multidata.setdefault("styles", []).append(
                    sliced_style if ix in sliced_inds else edge_style
                )

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

    if node_color is False:
        node_color = (0.5, 0.5, 0.5)

    if node_color is True:
        from .schematic import auto_colors

        node_colors = dict(
            zip(sorted(H.nodes), auto_colors(H.get_num_nodes()))
        )

        def _node_colorer(nd):
            return node_colors[nd]

    elif node_color == "centrality":

        def _node_colorer(nd):
            return cmap(Cs[nd])

    else:

        def _node_colorer(nd):
            return node_color

    for nd, ndata in G.nodes(data=True):
        hyperedge = ndata["hyperedge"]
        output = ndata.get("output", False)

        if hyperedge or output:
            color = (0.0, 0.0, 0.0, 0.0)
            if hyperedge and output:
                label = ""
            else:
                label = f"{ndata['ind']}"
        else:
            color = _node_colorer(nd)
            label = f"{nd}"  # H.inputs[nd]

        ndata["color"] = color
        ndata["label"] = label


def rotate(xy, theta):
    """Return a rotated set of points."""
    import numpy as np

    s = np.sin(theta)
    c = np.cos(theta)

    xyr = np.empty_like(xy)
    xyr[:, 0] = c * xy[:, 0] - s * xy[:, 1]
    xyr[:, 1] = s * xy[:, 0] + c * xy[:, 1]

    return xyr


def span(xy):
    """Return the vertical span of the points."""
    return xy[:, 1].max() - xy[:, 1].min()


def massage_pos(pos, nangles=100, flatten=False):
    """Rotate a position dict's points to cover a small vertical span"""
    import numpy as np

    xy = np.empty((len(pos), 2))
    for i, (x, y) in enumerate(pos.values()):
        xy[i, 0] = x
        xy[i, 1] = y

    thetas = np.linspace(0, 2 * np.pi, nangles, endpoint=False)
    rxys = (rotate(xy, theta) for theta in thetas)
    rxy0 = min(rxys, key=span)

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
    import matplotlib as mpl
    import networkx as nx
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
            emapper.to_rgba(d["contraction"])
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
            nmapper.to_rgba(d["contraction"])
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
@use_neutral_style
def plot_contractions(
    tree,
    order=None,
    color_size=(0.6, 0.4, 0.7),
    color_cost=(0.3, 0.7, 0.5),
    figsize=(8, 3),
):
    import matplotlib.pyplot as plt

    sz = sum(tree.get_size(node) for node in tree.gen_leaves())

    sizes = []
    peaks = []
    costs = []
    for p, l, r in tree.traverse(order):
        sz += tree.get_size(p)
        peaks.append(math.log2(sz))
        sz -= tree.get_size(l)
        sz -= tree.get_size(r)
        sizes.append(math.log2(tree.get_size(p)))
        costs.append(math.log10(tree.get_flops(p)))

    cons = list(range(len(peaks)))

    fig, ax = plt.subplots(figsize=figsize)

    ax.set_xlabel("contraction")

    ax.plot(
        cons,
        peaks,
        color=color_size,
        marker="x",
        markersize=3,
        alpha=0.5,
        label="peak",
    )
    ax.plot(
        cons,
        sizes,
        color=color_size,
        marker="o",
        markersize=3,
        alpha=0.5,
        linestyle=(1, (1, 1)),
        label="write",
    )
    M = math.log2(tree.total_write())
    ax.axhline(
        M,
        color=color_size,
        xmax=0.04,
        linewidth=0.8,
    )
    ax.text(
        0,
        M,
        "total write",
        ha="left",
        va="center",
        color=color_size,
    )
    ax.spines["right"].set_color(color_size)
    ax.tick_params(axis="y", colors=color_size)
    ax.set_ylim(0, 1.06 * M)
    ax.set_ylabel("$\\log_2[SIZE]$", color=color_size)

    rax = ax.twinx()
    rax.spines["right"].set_visible(True)
    rax.spines["right"].set_color(color_cost)
    rax.tick_params(axis="y", colors=color_cost)
    rax.plot(
        cons,
        costs,
        color=color_cost,
        marker="s",
        markersize=3,
        alpha=0.5,
        linestyle=(0, (1, 1)),
        label="cost",
    )

    C = tree.contraction_cost(log=10)

    rax.axhline(
        C,
        color=color_cost,
        xmin=0.96,
        linewidth=0.8,
    )
    rax.text(
        cons[-1],
        C,
        "total cost",
        ha="right",
        va="center",
        color=color_cost,
    )
    rax.set_ylim(0, 1.03 * C)
    rax.set_ylabel("$\\log_{10}[COST]$", color=color_cost)

    ax.legend(ncol=2, bbox_to_anchor=(0.4, 1.00), loc="center")
    rax.legend(bbox_to_anchor=(0.7, 1.00), loc="center")

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
    import seaborn as sns
    from matplotlib import pyplot as plt

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
    edge_color=True,
    node_color=True,
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
    node_scale=1.0,
    edge_alpha=1 / 3,
    edge_style="solid",
    hyperedge_style="dashed",
    draw_edge_labels=None,
    fontcolor=(0.5, 0.5, 0.5),
    edge_labels_font_size=8,
    edge_labels_font_family="monospace",
    node_labels_font_size=10,
    node_labels_font_family="monospace",
    info=None,
    ax=None,
    figsize=(5, 5),
):
    from .schematic import Drawing

    if draw_edge_labels is None:
        draw_edge_labels = len(H) <= 20

    G = H.to_networkx()
    hypergraph_compute_plot_info_G(
        H=H,
        G=G,
        highlight=highlight,
        node_color=node_color,
        edge_color=edge_color,
        edge_alpha=edge_alpha,
        edge_style=edge_style,
        colormap=colormap,
        centrality=centrality,
        hyperedge_style=hyperedge_style,
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

    if node_size is None:
        # compute a base size using the position and number of tensors
        # first get plot volume (taken from quimb.tensor.drawing.py):
        node_packing_factor = H.num_nodes**-0.45
        xs, ys, *zs = zip(*pos.values())
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        # if there only a few tensors we don't want to limit the node size
        # because of flatness, also don't allow the plot volume to go to zero
        xrange = max(((xmax - xmin) / 2, node_packing_factor, 0.1))
        yrange = max(((ymax - ymin) / 2, node_packing_factor, 0.1))
        plot_volume = xrange * yrange
        if zs:
            zmin, zmax = min(zs[0]), max(zs[0])
            zrange = max(((zmax - zmin) / 2, node_packing_factor, 0.1))
            plot_volume *= zrange
        # in total we account for:
        #     - user specified scaling
        #     - number of tensors
        #     - how flat the plot area is (flatter requires smaller nodes)
        node_size = 0.2 * node_scale * node_packing_factor * plot_volume**0.5

    if info is not None:
        info["node_size"] = node_size

    d = Drawing(ax=ax, figsize=figsize)

    edge_label_opts = dict(
        color=fontcolor,
        fontsize=edge_labels_font_size,
        family=edge_labels_font_family,
    )
    node_label_opts = dict(
        color=fontcolor,
        fontsize=node_labels_font_size,
        family=node_labels_font_family,
    )

    for n, ndata in G.nodes(data=True):
        d.circle(
            pos[n],
            radius=node_size,
            color=ndata["color"],
        )
        if draw_edge_labels:
            d.text(pos[n], text=ndata["label"], **node_label_opts)

    for na, nb, edata in G.edges(data=True):
        if "multi" not in edata:
            if draw_edge_labels and edata["label"]:
                text = dict(text=edata["label"], **edge_label_opts)
            else:
                text = None

            d.line(
                pos[na],
                pos[nb],
                color=edata["color"],
                linewidth=edata["width"],
                linestyle=edata["style"],
                text=text,
            )
        else:
            import numpy as np

            multidata = edata["multi"]
            colors = (edata["color"], *multidata["colors"])
            widths = (edata["width"], *multidata["widths"])
            styles = (edata["style"], *multidata["styles"])
            labels = (edata["label"], *multidata["labels"])

            ne = len(colors)
            offsets = np.linspace(-0.05 * ne, 0.05 * ne, ne)

            for i in range(ne):
                if draw_edge_labels and labels[i]:
                    text = dict(text=labels[i], **edge_label_opts)
                else:
                    text = None

                d.line_offset(
                    pos[na],
                    pos[nb],
                    offset=offsets[i],
                    color=colors[i],
                    linewidth=widths[i],
                    linestyle=styles[i],
                    text=text,
                )

    if info is not None and "pos" in info:
        info["pos"] = pos

    return d.fig, d.ax


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
    edge_color=(0.5, 0.5, 0.5),
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
    contractions / subgraphs. This can be intuitive for small and / or  planar
    contractions.
    """
    import matplotlib as mpl

    from .schematic import Drawing

    H = tree.get_hypergraph()
    info = {"pos": None}
    fig, ax = plot_hypergraph(
        H,
        highlight=highlight,
        centrality=centrality,
        layout=layout,
        node_size=node_size,
        node_color=node_color,
        edge_color=edge_color,
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
    r0 = info["node_size"]

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
        radius = [r0 + 0.01 * counts[node] for node in p]
        prog = i / (tree.N - 2)
        color = cmap(prog)
        d.patch_around(
            pts,
            resolution=20,
            radius=radius,
            edgecolor=color,
            facecolor="none",
            linestyle="-",
            zorder=-prog,
        )

    return fig, ax


@show_and_close
def plot_tree_flat(
    tree,
    edge_color=True,
    leaf_color=True,
    node_color=(0.5, 0.5, 0.5, 0.5),
    hyperedge_style="dashed",
    multiedge_spread=0.05,
    multiedge_smoothing=0.5,
    multiedge_midlength=0.5,
    fontcolor=(0.5, 0.5, 0.5),
    edge_labels_font_size=6,
    edge_labels_font_family="monospace",
    node_labels_font_size=8,
    node_labels_font_family="monospace",
    show_sliced=True,
    figsize=None,
):
    """Plot a ``ContractionTree`` as a flat, 2D diagram, including all indices
    at every intermediate contraction. This can be useful for small
    contractions, and does not require any graph layout algorithm.

    Parameters
    ----------
    tree : ContractionTree
        The contraction tree to plot.
    edge_color : bool or color, optional
        Whether to color the edges, or a specific color to use. If ``True``
        (default), each edge will be colored according to a hash of its index.
    leaf_color : bool or color, optional
        Whether to color the input nodes, or a specific color to use. If
        ``True`` (default), each leaf node will be colored with an
        automatically generated sequence according to its linear position in
        the input.
    node_color : bool or color, optional
        Whether to color the intermediate nodes, or a specific color to use. If
        ``True`` (default), each intermediate node will be colored with the
        average color of its children.
    hyperedge_style : str, optional
        The linestyle to use for hyperedges, i.e. indices that don't appeary
        exactly twice on either inputs or the output.
    multiedge_spread : float, optional
        The spread of multi-edges between nodes.
    multiedge_smoothing : float, optional
        The smoothing of multi-edges between nodes.
    multiedge_midlength : float, optional
        The midlength of multi-edges between nodes.
    fontcolor : color, optional
        The color to use for edge and node labels.
    edge_labels_font_size : int, optional
        The font size to use for edge labels.
    edge_labels_font_family : str, optional
        The font family to use for edge labels.
    node_labels_font_size : int, optional
        The font size to use for node labels.
    node_labels_font_family : str, optional
        The font family to use for node labels.
    show_sliced : bool, optional
        Whether to list sliced indices at the top left.
    figsize : tuple, optional
        The size of the figure to create, if not specified will be based on the
        number of nodes in the tree.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plot.
    ax : matplotlib.axes.Axes
        The axes containing the plot.
    """
    import math

    import numpy as np

    from .schematic import Drawing, auto_colors, average_color, hash_to_color

    if figsize is None:
        figsize = (2 * tree.N**0.5, 2 * tree.N**0.5)

    edge_label_opts = dict(
        color=fontcolor,
        fontsize=edge_labels_font_size,
        family=edge_labels_font_family,
    )
    node_label_opts = dict(
        color=fontcolor,
        fontsize=node_labels_font_size,
        family=node_labels_font_family,
    )

    d = Drawing(figsize=figsize)

    # order the leaves are contracted in
    leaf_order = {leaf: i for i, leaf in enumerate(tree.get_leaves_ordered())}

    if edge_color is True:
        edge_colors = {ix: hash_to_color(ix) for ix in tree.size_dict}
    else:
        edge_colors = {ix: edge_color for ix in tree.size_dict}

    if leaf_color is True:
        node_colors = {
            leaf: c for leaf, c in zip(tree.gen_leaves(), auto_colors(tree.N))
        }
    else:
        node_colors = {leaf: leaf_color for leaf in tree.gen_leaves()}

    hyperedges = {ix for ix, cnt in tree.appearances.items() if cnt != 2}

    # position of each node
    pos = {}
    for i, (p, l, r) in enumerate(tree.traverse(), 1):
        if len(l) == 1:
            # left is a leaf
            xyl = pos[l] = (leaf_order[l], i - 1)
            (tid,) = l
            d.circle(xyl, color=node_colors[l])
            d.text(xyl, str(tid), **node_label_opts)
        else:
            xyl = pos[l]

        if len(r) == 1:
            # right is a leaf
            xyr = pos[r] = (leaf_order[r], i - 1)
            (tid,) = r
            d.circle(xyr, color=node_colors[r])
            d.text(xyr, str(tid), **node_label_opts)
        else:
            xyr = pos[r]

        # position of parent is average of children
        xyp = ((xyl[0] + xyr[0]) / 2, i)
        pos[p] = xyp

        if node_color is True:
            # average color of children
            node_colors[p] = average_color((node_colors[l], node_colors[r]))
        else:
            node_colors[p] = node_color

        for xyc, edges in [
            (xyl, sorted(tree.get_legs(l), reverse=True)),
            (xyr, sorted(tree.get_legs(r))),
        ]:
            ne = len(edges)

            if ne == 1:
                offsets = [0.0]
                text_centers = [0.5]
            else:
                offsets = np.linspace(
                    -multiedge_spread * ne, multiedge_spread * ne, ne
                )
                text_centers = np.linspace(3 / 4, 1 / 4, ne)

            for ix, offset, text_center in zip(edges, offsets, text_centers):
                d.line_offset(
                    xyc,
                    xyp,
                    offset,
                    relative=False,
                    color=edge_colors[ix],
                    linewidth=math.log2(tree.size_dict.get(ix, 2)),
                    linestyle=hyperedge_style if ix in hyperedges else "-",
                    smoothing=multiedge_smoothing,
                    midlength=multiedge_midlength,
                    text=dict(text=ix, center=text_center, **edge_label_opts),
                )

        # draw intermediate node
        d.circle(xyp, color=node_colors[p])

    ne = len(tree.get_legs(tree.root))
    if ne:
        xyp = pos[tree.root]
        offsets = np.linspace(
            -multiedge_spread * ne, multiedge_spread * ne, ne
        )
        for ix, offset in zip(tree.get_legs(tree.root), offsets):
            xym = (xyp[0] + offset, tree.N - 0.5)
            xyo = (xyp[0] + offset, tree.N)
            d.curve(
                [xyp, xym, xyo],
                color=edge_colors[ix],
                zorder=0,
                linewidth=math.log2(tree.size_dict.get(ix, 2)),
                linestyle=hyperedge_style if ix in hyperedges else "-",
            )
            d.text(xyo, f"{ix}\n", **edge_label_opts)

    if tree.has_preprocessing():
        from .utils import node_from_single

        for i in tree.preprocessing:
            node = node_from_single(i)
            x, y = pos[node]
            d.circle(
                (x, y - 1),
                color=node_colors[node],
            )
            edges = [
                ix for ix in tree.inputs[i][::-1] if ix not in tree.sliced_inds
            ]
            ne = len(edges)

            if ne == 1:
                offsets = [0.0]
                text_centers = [0.5]
            else:
                offsets = np.linspace(
                    -multiedge_spread * ne, multiedge_spread * ne, ne
                )
                text_centers = np.linspace(3 / 4, 1 / 4, ne)

            for ix, offset, text_center in zip(edges, offsets, text_centers):
                d.line_offset(
                    (x, y - 1),
                    (x, y),
                    offset,
                    relative=False,
                    color=edge_colors[ix],
                    linewidth=math.log2(tree.size_dict.get(ix, 2)),
                    linestyle=hyperedge_style if ix in hyperedges else "-",
                    smoothing=multiedge_smoothing,
                    midlength=multiedge_midlength,
                    text=dict(text=ix, center=text_center, **edge_label_opts),
                )

    if tree.sliced_inds and show_sliced:
        d.label_ax(
            x=0.1,
            y=0.8,
            text=f"$\\sum_{{{','.join(tree.sliced_inds)}}}$",
            color=fontcolor,
        )

    return d.fig, d.ax


@show_and_close
def plot_tree_circuit(
    tree,
    edge_colormap="GnBu",
    edge_max_width=None,
    node_colormap="YlOrRd",
    node_max_size=None,
    figsize=None,
):
    import matplotlib as mpl

    from cotengra.schematic import Drawing

    if figsize is None:
        figsize = (tree.N**0.75, tree.N**0.75)

    d = Drawing(figsize=figsize)

    # edge coloring -> node size
    if edge_max_width is None:
        edge_max_width = math.log2(tree.max_size())

    ew_range = 0, edge_max_width
    enorm = mpl.colors.Normalize(*ew_range, clip=True)
    if not isinstance(edge_colormap, mpl.colors.Colormap):
        edge_colormap = getattr(mpl.cm, edge_colormap)
    emapper = mpl.cm.ScalarMappable(norm=enorm, cmap=edge_colormap)

    # edge coloring -> node flops
    if node_max_size is None:
        node_max_size = math.log2(max(map(tree.get_flops, tree.children)))

    nw_range = 0, node_max_size
    nnorm = mpl.colors.Normalize(*nw_range, clip=True)
    if not isinstance(node_colormap, mpl.colors.Colormap):
        node_colormap = getattr(mpl.cm, node_colormap)
    nmapper = mpl.cm.ScalarMappable(norm=nnorm, cmap=node_colormap)

    pos = {tree.root: (0, 0)}
    queue = [tree.root]
    while queue:
        # process tree in reverse depth first order
        p = queue.pop(0)
        px, py = pos[p]
        l, r = tree.children[p]

        # we do all of right contractions first
        pos[r] = (px - 1, py - 1)
        pos[l] = (px - len(r), py)

        if len(l) > 1:
            queue.append(l)
        else:
            (i,) = l
            d.text(
                pos[l],
                f"{i}",
                color=(0.5, 0.5, 0.5, 0.5),
                fontsize=20 * tree.N**-0.25,
                rotation=-90,
                ha="right",
                va="center",
                family="monospace",
            )
        if len(r) > 1:
            queue.append(r)
        else:
            (i,) = r
            d.text(
                pos[r],
                f"{i}",
                color=(0.5, 0.5, 0.5, 0.5),
                fontsize=20 * tree.N**-0.25,
                rotation=-45,
                ha="right",
                va="top",
                family="monospace",
            )

        lW = math.log2(tree.get_size(l))
        rW = math.log2(tree.get_size(r))
        pC = math.log2(tree.get_flops(p))

        d.line(
            pos[l],
            pos[p],
            color=emapper.to_rgba(lW),
            linewidth=5 * lW / edge_max_width,
        )
        d.line(
            pos[r],
            pos[p],
            color=emapper.to_rgba(rW),
            linewidth=5 * rW / edge_max_width,
        )
        d.circle(
            pos[p],
            color=nmapper.to_rgba(pC),
            radius=0.3 * pC / node_max_size,
            linewidth=0,
        )

    return d.fig, d.ax

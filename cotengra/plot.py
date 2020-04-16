import math
import functools
import collections


def plot_trials(
    self,
    y='score',
    color_scheme='Set2',
    figsize=(8, 3),
    return_fig=False,
):
    from matplotlib import pyplot as plt
    import seaborn as sns

    df = self.to_df()

    if y is None:
        y = self.minimize

    if y == 'score':
        best_y = min(self.scores)
        ylabel = 'Score'
    if y == 'size':
        best_y = math.log2(self.best[y])
        ylabel = 'log2[SIZE]'
    if y == 'flops':
        best_y = math.log10(self.best[y])
        ylabel = 'log10[FLOPS]'
    if y == 'combo':
        best_y = math.log2(self.best['flops']) + math.log2(self.best['size'])
        ylabel = 'log2[FLOPS] + log2[SIZE]'
        df['combo'] = [math.log2(s) + math.log2(f)
                       for s, f in zip(self.sizes, self.costs)]

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.axhline(best_y, color=(0, 0, 0, 0.1), linestyle=':')
    sns.scatterplot(
        y=y,
        x='run',
        data=df,
        ax=ax,
        style='method',
        hue='method',
        palette=color_scheme,
    )

    ax.set(ylabel=ylabel)
    ax.grid(True, c=(0.98, 0.98, 0.98))
    ax.set_axisbelow(True)

    handles, labels = ax.get_legend_handles_labels()
    handles, labels = handles[1:], labels[1:]
    ax.legend(
        handles=handles,
        labels=labels,
        bbox_to_anchor=(0.5, 1.0),
        ncol=(len(labels) * 6) // len(labels),
        loc='lower center',
        columnspacing=1,
        handletextpad=0,
        frameon=False,
    )

    if return_fig:
        return fig


def plot_trials_alt(self, y=None, width=800, height=300):
    """Plot the trials interactively using altair.
    """
    import altair as alt
    import pandas as pd

    df = self.to_df()

    if y is None:
        y = self.minimize

    if y == 'size':
        best_y = math.log2(self.best[y])
        ylabel = 'log2[SIZE]'
    if y == 'flops':
        best_y = math.log10(self.best[y])
        ylabel = 'log10[FLOPS]'

    hline = alt.Chart(
        pd.DataFrame({'best': [best_y]})
    ).mark_rule(strokeDash=[2, 2], color='grey').encode(y='best:Q')

    scatter = (
        alt.Chart(df)
        .mark_point()
        .encode(
            x='run:Q',
            y=alt.Y('{}:Q'.format(y), title=ylabel),
            size=alt.Size(
                'random_strength:Q',
                scale=alt.Scale(range=[50, 150], type='log'),
                legend=None,
            ),
            color='method:N',
            tooltip=list(df.columns)
        )
    )

    return (
        (hline + scatter)
        .properties(
            width=width,
            height=height,
        )
        .configure_axis(
            gridColor='rgb(248, 248, 248)'
        )
    ).interactive()


def plot_scatter(
    self,
    hue='method',
    style='method',
    color_scheme='Set2',
    figsize=(6, 4),
    return_fig=False,
):
    from matplotlib import pyplot as plt
    import seaborn as sns

    df = self.to_df()
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    sns.scatterplot(
        x='size',
        y='flops',
        hue=hue,
        style=style,
        data=df,
        palette=color_scheme,
    )

    ax.set(ylabel='log10[FLOPS]', xlabel='log2[SIZE]',
           xlim=(min(df['size'] - 1), max(df['size'] + 1)))
    ax.grid(True, c=(0.98, 0.98, 0.98))
    ax.set_axisbelow(True)

    handles, labels = ax.get_legend_handles_labels()
    handles, labels = handles[1:], labels[1:]
    ax.legend(
        handles=handles,
        labels=labels,
        bbox_to_anchor=(1.0, 0.5),
        loc='center left',
        columnspacing=1,
        handletextpad=0,
        frameon=False,
    )

    if return_fig:
        return fig


def plot_scatter_alt(
    self,
    color='run:Q',
    color_scheme='purplebluegreen',
    shape='method:N',
    width=400,
    height=400,
):
    """Plot the trials total flops vs max size.
    """
    import altair as alt

    df = self.to_df()
    scatter = (
        alt.Chart(df)
        .mark_point()
        .encode(
            x=alt.X('size:Q', title='log2[SIZE]',
                    scale=alt.Scale(zero=False)),
            y=alt.Y('flops:Q', title='log10[FLOPS]',
                    scale=alt.Scale(zero=False)),
            size=alt.Size(
                'random_strength:Q',
                scale=alt.Scale(range=[50, 150], type='log'),
                legend=None,
            ),
            shape=alt.Shape(shape),
            color=alt.Color(color, scale=alt.Scale(scheme=color_scheme)),
            tooltip=list(df.columns)
        )
    )
    return (
        scatter
        .properties(
            width=width,
            height=height,
        )
        .configure_axis(
            gridColor='rgb(248, 248, 248)'
        )
    ).interactive()


def tree_to_networkx(tree):
    import networkx as nx

    G = nx.Graph()

    for p, l, r in tree.traverse():
        G.add_edge(p, l, size=math.log10(tree.get_size(l) + 1) + 1,
                   weight=len(l))
        G.add_edge(p, r, size=math.log10(tree.get_size(r) + 1) + 1,
                   weight=len(r))

    for node in tree.info:
        G.nodes[node]['flops'] = math.log10(tree.get_flops(node) + 1) + 1

    return G


def plot_tree(
    tree,
    layout='ring',
    k=0.01,
    iterations=500,
    edge_scale=1.0,
    node_scale=1.0,
    highlight=(),
    edge_colormap='GnBu',
    node_colormap='YlOrRd',
    figsize=(5, 5),
    return_fig=False,
    raw_edge_alpha=0.05,
    tree_alpha=0.8,
    colorbars=True,
    plot_raw_graph=True,
):
    """Plot a contraction tree using matplotlib.
    """
    import warnings
    import networkx as nx
    import matplotlib as mpl
    from matplotlib import pyplot as plt

    # draw the contraction tree
    G_tree = tree_to_networkx(tree)
    leaves = tree.get_leaves_ordered()

    # set the tree edge and node sizes
    edge_weights = [edge_scale * 0.8 * G_tree.edges[e]['size'] for e in G_tree.edges]
    node_weights = [node_scale * 8. * G_tree.nodes[n]['flops'] for n in G_tree.nodes]

    # tree edge colors
    ew_range = min(edge_weights), max(edge_weights)
    enorm = mpl.colors.Normalize(*ew_range, clip=True)
    edge_cm = getattr(mpl.cm, edge_colormap)
    emapper = mpl.cm.ScalarMappable(norm=enorm, cmap=edge_cm)
    edge_colors = [emapper.to_rgba(x) for x in edge_weights]

    # tree node colors
    nw_range = min(node_weights), max(node_weights)
    nnorm = mpl.colors.Normalize(*nw_range, clip=True)
    node_cm = getattr(mpl.cm, node_colormap)
    nmapper = mpl.cm.ScalarMappable(norm=nnorm, cmap=node_cm)
    node_colors = [nmapper.to_rgba(x) for x in node_weights]

    # plot the raw connectivity of the underlying graph
    if plot_raw_graph:
        # collect which nodes each edge connects
        ind_map = collections.defaultdict(list)
        for nd in leaves:
            inds, = nd
            for ix in inds:
                ind_map[ix].append(nd)
        # turn into another nx graph
        G_tn = nx.Graph()
        any_hyper = False
        for ix, nodes in ind_map.items():
            width = math.log2(tree.size_dict[ix])
            # regular edge
            if len(nodes) == 2:
                G_tn.add_edge(*nodes, ind=ix, width=width)
            # hyperedge
            else:
                any_hyper = True
                G_tn.add_node(ix)
                for nd in nodes:
                    G_tn.add_edge(ix, nd, ind=ix, width=width)

        edge_widths_raw = [G_tn.edges[e]['width'] for e in G_tn.edges]
        edge_colors_raw = [
            (0., 0., 0., raw_edge_alpha) if G_tn.edges[e]['ind'] not in highlight
            else (1.0, 0.0, 1.0, raw_edge_alpha**0.5)
            for e in G_tn.edges
        ]

    if layout == 'tent':
        # place raw graph first
        pos = nx.kamada_kawai_layout(G_tn)
        pos = nx.spring_layout(G_tn, k=k, iterations=iterations, pos=pos)
        # 'flatten' a bit onto plane
        pos = {k: [v[0], v[1] / 2] for k, v in pos.items()}
        # place the top of the tree
        xmin = min(v[0] for v in pos.values())
        xmax = max(v[0] for v in pos.values())
        pos[tree.root] = (0, 1.0 * (xmax - xmin))
        # layout the tree nodes between bottom and top
        pos.update(nx.spring_layout(
            G_tree, fixed=pos, pos=pos, k=k, iterations=iterations
        ))

    elif layout == 'ring':
        # work out a layout based on leaves in circle
        pos = {tree.root: (0, 0)}
        for i, x in enumerate(leaves):
            pos[x] = (math.sin(2 * math.pi * i / tree.N),
                      math.cos(2 * math.pi * i / tree.N))
        # layout the remaining tree nodes
        pos = nx.spring_layout(G_tree, fixed=pos, pos=pos,
                               k=k, iterations=iterations)
        # if there are hyperedges layout the faux-nodes
        if plot_raw_graph and any_hyper:
            fixed_raw = {k: v for k, v in pos.items() if k in leaves}
            pos.update(nx.spring_layout(G_tn, fixed=fixed_raw, pos=fixed_raw,
                                        k=k, iterations=iterations))

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_aspect('equal')
    ax.axis('off')

    if plot_raw_graph:
        nx.draw_networkx_edges(
            G_tn, pos=pos, ax=ax,
            width=edge_widths_raw,
            edge_color=edge_colors_raw)

    nx.draw_networkx_edges(G_tree, pos=pos, ax=ax, width=edge_weights,
                           edge_color=edge_colors, alpha=tree_alpha)
    nx.draw_networkx_nodes(G_tree, pos=pos, ax=ax, node_size=node_weights,
                           node_color=node_colors, alpha=tree_alpha)

    if colorbars:
        min_size = math.log2(min(tree.get_size(x) for x in tree.info))
        max_size = math.log2(max(tree.get_size(x) for x in tree.info))
        edge_norm = mpl.colors.Normalize(vmin=min_size, vmax=max_size)
        ax_l = fig.add_axes([-0.02, 0.25, 0.02, 0.5])
        cb_l = mpl.colorbar.ColorbarBase(ax_l, cmap=edge_cm, norm=edge_norm)
        cb_l.outline.set_visible(False)
        ax_l.yaxis.tick_left()
        ax_l.set(title='log2[SIZE]')

        min_flops = math.log10(min(max(tree.get_flops(x), 1)
                                   for x in tree.info))
        max_flops = math.log10(max(max(tree.get_flops(x), 1)
                                   for x in tree.info))
        node_norm = mpl.colors.Normalize(vmin=min_flops, vmax=max_flops)
        ax_r = fig.add_axes([1.0, 0.25, 0.02, 0.5])
        cb_r = mpl.colorbar.ColorbarBase(ax_r, cmap=node_cm, norm=node_norm)
        cb_r.outline.set_visible(False)
        ax_r.yaxis.tick_right()
        ax_r.set(title='log10[FLOPS]')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        plt.tight_layout()

    if return_fig:
        return fig


@functools.wraps(plot_tree)
def plot_tree_ring(tree, **kwargs):
    kwargs.setdefault('tree_alpha', 0.97)
    kwargs.setdefault('raw_edge_alpha', 0.03)
    return plot_tree(tree, 'ring', **kwargs)


@functools.wraps(plot_tree)
def plot_tree_tent(tree, **kwargs):
    kwargs.setdefault('tree_alpha', 0.7)
    kwargs.setdefault('raw_edge_alpha', 0.07)
    kwargs.setdefault('edge_scale', 1 / 2)
    kwargs.setdefault('node_scale', 1 / 3)
    return plot_tree(tree, 'tent', **kwargs)


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

    return pd.DataFrame({
        'out-size': sizes,
        'input-size': isizes,
        'peak-size': psizes,
        'flops': flops,
        'stage': stages,
        'rank': ranks,
        'scaling': scalings,
    })


def plot_contractions(
    tree,
    x='peak-size',
    y='flops',
    color='stage',
    size='scaling',
    point_opacity=0.8,
    color_scheme='viridis_r',
    x_scale='log',
    y_scale='log',
    figsize=(6, 4),
    return_fig=False,
):
    from matplotlib import pyplot as plt
    import matplotlib as mpl
    import seaborn as sns
    import warnings

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
    ax_cb.set(title='Stage')
    cm = getattr(mpl.cm, color_scheme)
    cb = mpl.colorbar.ColorbarBase(ax_cb, cmap=cm)
    cb.outline.set_visible(False)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        plt.tight_layout()

    if return_fig:
        return fig


def plot_contractions_alt(
    tree,
    x='peak-size',
    y='flops',
    color='stage',
    size='scaling',
    width=400,
    height=400,
    point_opacity=0.8,
    color_scheme='lightmulti',
    x_scale='log',
    y_scale='log',
    color_scale='log',
    size_scale='linear',
):
    import altair as alt

    df = tree_to_df(tree)
    chart = alt.Chart(df)

    encoding = chart.encode(
        x=alt.X(x, scale=alt.Scale(type=x_scale, padding=10)),
        y=alt.Y(y, scale=alt.Scale(type=y_scale, padding=10)),
        color=alt.Color(color, scale=alt.Scale(scheme=color_scheme,
                                               type=color_scale)),
        size=alt.Size(size, scale=alt.Scale(type=size_scale)),
        tooltip=list(df.columns),
    )
    plot = encoding.mark_point(opacity=point_opacity)

    return (
        plot
        .configure_axis(gridColor='rgb(248,248,248)')
        .properties(width=width, height=height)
        .interactive()
    )


def slicefinder_to_df(slice_finder, relative_flops=False):
    import pandas as pd

    all_ccs = sorted(slice_finder.costs.values(), key=lambda x: -x.size)
    maxsizes = [math.log2(x.size) for x in all_ccs]
    if relative_flops:
        newflops = [float(x.total_flops) /
                    float(slice_finder.cost0.total_flops) for x in all_ccs]
    else:
        newflops = [x.total_flops for x in all_ccs]
    numslices = [x.nslices for x in all_ccs]

    return pd.DataFrame({
        'size': maxsizes,
        'flops': newflops,
        'numslices': numslices,
    })


def plot_slicings(
    slice_finder,
    color_scheme='RdYlBu_r',
    relative_flops=False,
    figsize=(6, 3),
    point_opacity=0.8,
    return_fig=False,
):
    import matplotlib as mpl
    from matplotlib import pyplot as plt
    import seaborn as sns
    import numpy as np
    import warnings

    df = slicefinder_to_df(slice_finder, relative_flops=relative_flops)

    df['log10[FLOPS]'] = np.log10(df['flops'])
    df['log2[NSLICES]'] = np.log2(df['numslices'])

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    sns.scatterplot(
        x='size',
        y='log10[FLOPS]',
        hue='log2[NSLICES]',
        palette=color_scheme,
        data=df,
        alpha=point_opacity,
    )

    ax.set(xlim=(max(df['size'] + 1), min(df['size'] - 1)),
           xlabel='log2[SIZE]')
    ax.grid(True, c=(0.98, 0.98, 0.98))
    ax.set_axisbelow(True)
    ax.get_legend().remove()

    ax_cb = fig.add_axes([1.02, 0.25, 0.02, 0.55])
    ax_cb.set(title='log2[NSLICES]')
    nm = mpl.colors.Normalize(vmin=df['log2[NSLICES]'].min(),
                              vmax=df['log2[NSLICES]'].max())
    cm = getattr(mpl.cm, color_scheme)
    cb = mpl.colorbar.ColorbarBase(ax_cb, cmap=cm, norm=nm)
    cb.outline.set_visible(False)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        plt.tight_layout()

    if return_fig:
        return fig


def plot_slicings_alt(
    slice_finder,
    color_scheme='redyellowblue',
    relative_flops=False
):
    import altair as alt

    df = slicefinder_to_df(slice_finder, relative_flops=relative_flops)

    return alt.Chart(df).mark_point().encode(
        x=alt.X('size:Q', sort='descending',
                scale=alt.Scale(type='linear', zero=False)),
        y=alt.Y('flops:Q', scale=alt.Scale(type='log')),
        color=alt.Color('numslices:Q',
                        scale=alt.Scale(type='log', scheme=color_scheme),
                        sort='descending'),
        tooltip=list(df.columns)
    ).configure_axis(gridColor='rgb(248,248,248)').interactive()

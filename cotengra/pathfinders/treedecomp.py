"""
The following functions are adapted from the repository:

    https://github.com/TheoryInPractice/ConSequences

associated with the paper:

    https://arxiv.org/abs/1807.04599

under the following license:


BSD 3-Clause License

Copyright (c) 2018,  Allison L. Fisher, Timothy D. Goodrich, Blair D. Sullivan,
Andrew L. Wright
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import copy


class TreeDecomposition:
    """
    A named struct for convenience.
    """

    def __init__(self):
        """
        A tree decomposition consisting of a tree of nodes and a lookup table
        mapping decomposition nodes to vertices in the original graph.
        """
        import networkx as nx

        self.tree = nx.Graph()
        self.bags = {}


class EliminationOrdering:
    """
    A named struct for convenience.
    """

    def __init__(self):
        """
        A elimination ordering is an ordered list of vertices.
        """
        self.ordering = []


def _increment_eo(td, eo):
    """
    Given a TreeDecomposition and a (partial) PerfectEliminationOrdering, add
    one more vertex to the eo or recognize that we're
    already done.
    Input:
        td (TreeDecomposition): A tree decomposition.
        eo (PerfectEliminationOrdering): The perfect elimination ordering
        currently being constructed.
    Output:
        The final eo
    """
    while True:
        # Base case: If one node left, add its vertices to the eo
        if td.tree.order() == 1:
            (only_vertex,) = td.tree.nodes()
            eo.ordering.extend(sorted(td.bags[only_vertex]))
            return eo

        # Otherwise we can identify a leaf and its parent
        leaf = next(
            node for node in td.tree.nodes() if td.tree.degree[node] == 1
        )
        parent = min(td.tree.neighbors(leaf))

        # See if there are any vertices in leaf's bag that are not in
        # parent's bag
        vertex_diff = td.bags[leaf] - td.bags[parent]

        # If there's a vertex in the leaf and not in the parent,
        # then remove it from the graph and add it to the eo.
        if vertex_diff:
            next_vertex = min(vertex_diff)
            eo.ordering.append(next_vertex)
            for key in td.bags:
                td.bags[key].discard(next_vertex)

        # Else remove the leaf from the graph
        else:
            td.tree.remove_node(leaf)
            td.bags.pop(leaf)


def td_to_eo(td):
    """
    Generates a perfect elimination ordering from a tree decomposition. The
    algorithm is taken from Markov and Shi Proof of Prop 4.2
    (https://arxiv.org/pdf/quant-ph/0511069.pdf).
    Input:
        td (TreeDecomposition): A tree decomposition for a graph.
    Output:
        eo (PerfectEliminationOrdering): A perfect elimination ordering
        corresponding to the tree decomposition (Note: There may be multiple
        valid eo for a given td).
    """

    # Copy the tree decomposition, my_td will be modified
    my_td = copy.deepcopy(td)

    # Construct the eo
    eo = _increment_eo(my_td, EliminationOrdering())
    return eo


def td_str_to_tree_decomposition(td_str):
    """
    Reads in a .td file contents in PACE format into a TreeDecomposition object
    Input:
        td_filename (str): .td file contents
    Output:
        td (TreeDecomposition): A populated TreeDecomposition object
    """

    td = TreeDecomposition()

    lines = iter(td_str.split("\n"))

    # Ignore comments
    line = next(lines)
    while line[0] == "c":
        line = next(lines)

    # The next line will look like "s td 28 25 95"
    # Currently unused
    # num_nodes, max_bag, num_vertices = map(int, line.split()[2:])

    line = next(lines)
    while line[0] == "b":
        # A bag line will look like:
        # "b 1 1 11 16 41 42 43 44 45"
        node = int(line.split()[1])
        vertices = set(map(int, line.split()[2:]))
        td.bags[node] = vertices
        line = next(lines)

    # Add a node for each bag
    td.tree.add_nodes_from(td.bags)

    # Add the first edge
    td.tree.add_edge(*map(int, line.split()))

    # The remainder of the file is edges
    for line in lines:
        if line:
            td.tree.add_edge(*map(int, line.split()))

    return td

use std::collections::{BTreeSet, BinaryHeap, VecDeque};

use bit_set::BitSet;
use itertools::Itertools;
use rustc_hash::{FxHashMap, FxHashSet};

use pyo3::prelude::*;
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};
use pyo3::wrap_pyfunction;

use u32 as Node;
use FxHashMap as Dict;

// ----------------------------------------------------------------------------

#[pymodule]
#[pyo3(name = "cotengra")]
fn cotengra(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<HyperGraph>()?;
    m.add_function(wrap_pyfunction!(indexes, m)?)?;
    m.add_function(wrap_pyfunction!(nodes_to_centrality, m)?)?;
    m.add_function(wrap_pyfunction!(edges_to_centrality, m)?)?;
    m.add_function(wrap_pyfunction!(optimal_compressed_path, m)?)?;
    Ok(())
}

// ----------------------------------------------------------------------------

#[pyfunction]
fn indexes(s: String) -> PyResult<Vec<usize>> {
    Ok(s.chars()
        .rev()
        .enumerate()
        .filter_map(|(i, b)| if b == '1' { Some(i) } else { None })
        .collect())
}

// ----------------------------------------------------------------------------

#[pyclass]
#[derive(Clone)]
struct HyperGraph {
    #[pyo3(get)]
    nodes: Dict<Node, Vec<String>>,
    #[pyo3(get)]
    edges: Dict<String, Vec<Node>>,
    #[pyo3(get)]
    output: Vec<String>,
    #[pyo3(get)]
    size_dict: Dict<String, u128>,
    node_counter: Node,
    work_edges: Vec<String>,
    work_incidences: Dict<BTreeSet<Node>, Vec<String>>,
}

impl HyperGraph {
    fn from_nodes_edges(
        nodes: Dict<Node, Vec<String>>,
        edges: Dict<String, Vec<Node>>,
        output: Option<Vec<String>>,
        size_dict: Option<Dict<String, u128>>,
    ) -> HyperGraph {
        let output: Vec<String> = match output {
            Some(x) => x,
            None => Vec::new(),
        };
        let size_dict: Dict<String, u128> = match size_dict {
            Some(size_dict) => size_dict,
            None => Dict::default(),
        };
        let node_counter = (nodes.len() - 1) as Node;
        HyperGraph {
            nodes,
            edges,
            output,
            size_dict,
            node_counter,
            work_edges: Vec::new(),
            work_incidences: Dict::default(),
        }
    }

    fn from_nodes(
        nodes: Dict<Node, Vec<String>>,
        output: Option<Vec<String>>,
        size_dict: Option<Dict<String, u128>>,
    ) -> HyperGraph {
        let mut edges: Dict<String, Vec<Node>> = Dict::default();
        for (i, i_edges) in &nodes {
            for e in i_edges {
                edges.entry(e.clone()).or_insert(Vec::new()).push(*i)
            }
        }
        HyperGraph::from_nodes_edges(nodes, edges, output, size_dict)
    }

    fn from_edges(
        edges: Dict<String, Vec<Node>>,
        output: Option<Vec<String>>,
        size_dict: Option<Dict<String, u128>>,
    ) -> HyperGraph {
        let mut nodes: Dict<Node, Vec<String>> = Dict::default();
        for (e, e_nodes) in &edges {
            for &i in e_nodes {
                nodes.entry(i).or_insert(Vec::new()).push(e.clone());
            }
        }
        HyperGraph::from_nodes_edges(nodes, edges, output, size_dict)
    }

    fn edges_size(&self, es: &[String]) -> u128 {
        es.iter()
            .fold(1 as u128, |x, e| x.saturating_mul(self.size_dict[e]))
    }

    fn next_node(&mut self) -> Node {
        self.node_counter += 1;
        while self.nodes.contains_key(&self.node_counter) {
            self.node_counter += 1;
        }
        self.node_counter
    }

    fn add_node(&mut self, inds: Vec<String>, node: Option<Node>) -> Node {
        let i = match node {
            Some(i) => i,
            None => self.next_node(),
        };
        for e in &inds {
            self.edges.entry(e.clone()).or_insert(Vec::new()).push(i);
        }
        self.nodes.insert(i, inds);
        i
    }

    fn remove_node(&mut self, i: Node) -> Vec<String> {
        let inds = self.nodes.remove(&i).unwrap();
        let mut should_delete: bool = false;
        for e in &inds {
            self.edges.entry(e.clone()).and_modify(|v| {
                v.retain(|&j| j != i);
                should_delete = v.len() == 0;
            });
            if should_delete {
                self.edges.remove(e);
            }
        }
        inds
    }

    fn _remove_edge(&mut self, e: &String) {
        for i in &self.edges[e] {
            self.nodes
                .entry(*i)
                .and_modify(|i_edges| i_edges.retain(|d| d != e));
        }
        self.edges.remove(e);
    }

    fn build_neighbor_map(&self) -> Dict<Node, Vec<Node>> {
        self.nodes.keys().map(|&i| (i, self.neighbors(i))).collect()
    }

    fn simple_closeness(
        &self,
        p: f64,
        mu: f64,
        neighbor_map: Option<&Dict<Node, Vec<Node>>>,
    ) -> Dict<Node, f64> {
        let n = self.get_num_edges();

        // temp closure
        let nmap_tmp: Dict<Node, Vec<Node>>;
        let nmap = match neighbor_map {
            Some(x) => x,
            None => {
                nmap_tmp = self.build_neighbor_map();
                &nmap_tmp
            }
        };

        // which nodes have reached which other nodes
        let mut visitors: Dict<Node, BitSet<Node>> = self
            .nodes
            .keys()
            .map(|&x| (x, single_el_bitset(x as usize, n)))
            .collect();
        let mut previous_visitors: Dict<Node, BitSet<Node>>;

        // store the number of unique visitors - the change is this each step
        //    is the number of new shortest paths of length ``d``
        let mut num_visitors: Dict<Node, f64> = self.nodes.keys().map(|&i| (i, 1.0)).collect();

        // the total weighted score - combining num visitors and their distance
        let mut scores: Dict<Node, f64> = self.nodes.keys().map(|&i| (i, 0.0)).collect();

        let sz_stop = (n as f64).powf(p);
        let mut should_stop = false;
        let mut new_nv: f64;

        for d in 0..n {
            // do a parallel update
            previous_visitors = visitors.clone();
            for &i in self.nodes.keys() {
                for &j in nmap[&i].iter() {
                    visitors
                        .entry(i)
                        .and_modify(|v| v.union_with(&previous_visitors[&j]));
                }
                // visitors are worth less the further they've come from
                new_nv = visitors[&i].len() as f64;
                scores.entry(i).and_modify(|v| {
                    *v += (new_nv - num_visitors[&i]) / ((d as f64) + 1.0).powf(mu)
                });
                num_visitors.insert(i, new_nv);

                // once any node has reached a certain number of visitors stop
                should_stop |= new_nv >= sz_stop;
            }
            if should_stop {
                break;
            }
        }
        // finally rescale the values between 0.0 and 1.0
        affine_renorm(&scores)
    }
}

fn single_el_bitset(x: usize, n: usize) -> BitSet<Node> {
    let mut a: BitSet<Node> = BitSet::with_capacity(n);
    a.insert(x);
    a
}

fn affine_renorm(x: &Dict<Node, f64>) -> Dict<Node, f64> {
    let mut mn: f64 = f64::INFINITY;
    let mut mx: f64 = f64::NEG_INFINITY;
    for &v in x.values() {
        mn = f64::min(mn, v);
        mx = f64::max(mx, v);
    }
    if mn == mx {
        mn = 0.0;
        if mx == 0.0 {
            mx = 1.0;
        }
    }
    x.iter().map(|(&k, &v)| (k, (v - mn) / (mx - mn))).collect()
}

#[pymethods]
impl HyperGraph {
    #[new(output = "None", size_dict = "None")]
    fn new(
        nodes: Dict<Node, Vec<String>>,
        output: Option<Vec<String>>,
        size_dict: Option<Dict<String, u128>>,
    ) -> HyperGraph {
        HyperGraph::from_nodes(nodes, output, size_dict)
    }

    fn copy(&self) -> HyperGraph {
        self.clone()
    }

    fn get_num_nodes(&self) -> usize {
        self.nodes.len()
    }

    fn get_num_edges(&self) -> usize {
        self.edges.len()
    }

    fn print_nodes(&self) {
        println!("{:?}", self.nodes);
    }

    fn print_edges(&self) {
        println!("{:?}", self.edges);
    }

    fn node_size(&self, i: Node) -> u128 {
        self.edges_size(&self.nodes[&i])
    }

    fn remove_edge(&mut self, e: String) {
        self._remove_edge(&e);
    }

    fn bond_size(&self, i: Node, j: Node) -> u128 {
        let es: Vec<String> = self.nodes[&i]
            .iter()
            .filter(|&e| self.nodes[&j].contains(e))
            .cloned()
            .collect();
        self.edges_size(&es)
    }

    fn total_node_size(&self) -> u128 {
        self.nodes.values().map(|es| self.edges_size(es)).sum()
    }

    fn neighborhood_size(&self, nodes: Vec<Node>) -> u128 {
        let mut nnodes: Vec<Node> = Vec::new();
        for i in nodes {
            for e in self.get_node(i) {
                for j in self.get_edge(e) {
                    if !nnodes.contains(&j) {
                        nnodes.push(j)
                    }
                }
            }
        }
        nnodes.into_iter().map(|i| self.node_size(i)).sum()
    }

    fn contract_pair_cost(&self, i: Node, j: Node) -> u128 {
        let mut inds = self.get_node(i);
        inds.extend(self.get_node(j));
        inds = inds.into_iter().unique().collect();
        self.edges_size(&inds)
    }

    fn neighborhood_compress_cost(&mut self, chi: u128, nodes: Vec<Node>) -> u128 {
        let mut edges: Vec<String> = Vec::new();
        let mut ignored_nodes: BTreeSet<Node> = BTreeSet::new();
        for i in nodes {
            {
                edges.extend(self.get_node(i));
                ignored_nodes.insert(i);
            }
        }
        edges = edges.into_iter().unique().collect();

        self.work_incidences.clear();
        for e in edges {
            if !self.output.contains(&e) {
                let e_nodes: BTreeSet<Node> = self.edges[&e].iter().cloned().collect();
                self.work_incidences
                    .entry(e_nodes)
                    .or_insert(Vec::new())
                    .push(e);
            }
        }

        // let ignored_nodes: BTreeSet<Node> = nodes.clone().iter().cloned().collect();
        self.work_incidences.remove(&ignored_nodes);

        let mut cost = 0;
        let mut da;
        let mut db;
        let mut outer_edges: Vec<String>;

        for (e_nodes, edges) in self.work_incidences.drain().collect_vec() {
            da = self.edges_size(&edges);
            if da > chi {
                for node in e_nodes {
                    // get edges that
                    outer_edges = self
                        .get_node(node)
                        .into_iter()
                        .filter(|e| !edges.contains(e))
                        .collect();
                    db = self.edges_size(&outer_edges);
                    if da <= db {
                        cost += da.pow(2) * db;
                    } else {
                        cost += da * db.pow(2);
                    }
                }
            }
        }
        cost
    }

    fn get_node(&self, i: Node) -> Vec<String> {
        self.nodes[&i].clone()
    }

    fn has_node(&self, i: Node) -> bool {
        self.nodes.contains_key(&i)
    }

    fn get_edge(&self, e: String) -> Vec<Node> {
        self.edges[&e].clone()
    }

    fn has_edge(&self, e: String) -> bool {
        self.edges.contains_key(&e)
    }

    fn output_nodes(&self) -> Vec<Node> {
        let mut onodes = Vec::new();
        for e in self.output.iter() {
            for i in &self.edges[e] {
                if !onodes.contains(i) {
                    onodes.push(*i);
                }
            }
        }
        onodes
    }

    fn neighbors(&self, i: Node) -> Vec<Node> {
        let mut i_neighbors = Vec::with_capacity(self.nodes[&i].len());
        for e in &self.nodes[&i] {
            for &j in &self.edges[e] {
                if i != j && !i_neighbors.contains(&i) {
                    i_neighbors.push(j);
                }
            }
        }
        i_neighbors
    }

    fn neighbor_edges(&mut self, i: Node) -> Vec<String> {
        self.work_edges.clear();
        for j in self.neighbors(i) {
            for e in &self.nodes[&j] {
                if !self.work_edges.contains(e) {
                    self.work_edges.push(e.clone())
                }
            }
        }
        self.work_edges.drain(..).collect()
    }

    fn contract(&mut self, i: Node, j: Node, node: Option<Node>) -> Node {
        let mut inds_ij = self.remove_node(i);
        inds_ij.extend(self.remove_node(j));
        inds_ij = inds_ij
            .into_iter()
            .unique()
            .filter(|e| self.output.contains(e) || self.edges.contains_key(e))
            .collect();
        self.add_node(inds_ij, node)
    }

    fn compress(&mut self, chi: u128, edges: Option<Vec<String>>) {
        let edges = match edges {
            Some(edges) => edges,
            None => self.edges.keys().cloned().collect(),
        };
        self.work_incidences.clear();
        for e in edges {
            if !self.output.contains(&e) {
                let nodes: BTreeSet<Node> = self.edges[&e].iter().cloned().collect();
                self.work_incidences
                    .entry(nodes)
                    .or_insert(Vec::new())
                    .push(e);
            }
        }
        for (_, es) in self.work_incidences.drain().collect_vec() {
            if es.len() > 1 {
                let new_size = self.edges_size(&es);
                let mut es_it = es.into_iter();
                let e0 = es_it.next().unwrap();
                // let e0 = es.swap_remove(0);
                for e in es_it {
                    self._remove_edge(&e);
                }
                self.size_dict.insert(e0, new_size.min(chi));
            }
        }
    }

    fn compute_contracted_inds(&self, nodes: Vec<Node>) -> Vec<String> {
        let mut inds = Vec::new();
        for i in &nodes {
            for e in &self.nodes[i] {
                if !inds.contains(e)
                    // ind appears on any other node or in output -> keep
                    && (self.edges[e].iter().any(|k| !nodes.contains(k)) || self.output.contains(e))
                {
                    inds.push(e.clone());
                }
            }
        }
        inds
    }

    fn candidate_contraction_size(&mut self, i: Node, j: Node, chi: Option<u128>) -> u128 {
        let mut es = self.compute_contracted_inds(vec![i, j]);
        match chi {
            None => self.edges_size(&es),
            Some(chi) => {
                // if, after contraction, two nodes are incident to the exact same set
                // of nodes, then they are compressable
                self.work_incidences.clear();
                for e in es.drain(..) {
                    let nodes: BTreeSet<Node> = self.edges[&e]
                        .iter()
                        .map(|&k| if k == j { i } else { k })
                        .collect();
                    self.work_incidences
                        .entry(nodes)
                        .or_insert(Vec::new())
                        .push(e);
                }
                self.work_incidences.values().fold(1 as u128, |x, es| {
                    x.saturating_mul(chi.min(self.edges_size(es)))
                })
            }
        }
    }

    fn all_shortest_distances(
        &self,
        nodes: Option<Vec<Node>>,
        neighbor_map: Option<Dict<Node, Vec<Node>>>,
    ) -> Dict<(Node, Node), u32> {
        let mut neighbor_map = match neighbor_map {
            Some(x) => x,
            // None => self.build_neighbor_map(),
            None => Dict::default(),
        };

        let nodes: FxHashSet<Node> = match nodes {
            Some(nodes) => nodes.into_iter().collect(),
            None => self.nodes.keys().cloned().collect(),
        };
        let n = nodes.len();
        let ncomb = (nodes.len().pow(2) - nodes.len()) / 2;

        // which nodes have reached which other nodes
        let mut visitors: Dict<Node, BitSet<Node>> = nodes
            .iter()
            .map(|&x| (x, single_el_bitset(x as usize, n)))
            .collect();

        let mut previous_visitors: Dict<Node, BitSet<Node>>;
        let mut distances = Dict::default();
        let mut any_change;

        for d in 1..self.get_num_nodes() + 1 {
            any_change = false;
            // do a parallel update
            previous_visitors = visitors.clone();
            // only need to iterate over touched region

            for (&i, ivis) in &previous_visitors {
                // send i's visitors to its neighbors
                // for &j in &neighbor_map[&i] {
                for &j in neighbor_map
                    .entry(i)
                    .or_insert_with_key(|&i| self.neighbors(i))
                    .iter()
                {
                    visitors
                        .entry(j)
                        // also populating newly encountered neighbors
                        .or_insert_with(|| {
                            any_change = true;
                            BitSet::with_capacity(n)
                        })
                        .union_with(ivis);
                }
            }
            // all changes in visitors are new distances of d
            for &i in &nodes {
                for j in visitors[&i].difference(&previous_visitors[&i]) {
                    let j = j as Node;
                    if (i < j) & nodes.contains(&j) {
                        distances.insert((i, j), d as u32);
                    }
                    any_change = true;
                }
            }

            if !any_change {
                any_change |= previous_visitors
                    .iter()
                    .map(|(k, v)| v != &visitors[&k])
                    .any(|x| x);
            }

            if (distances.len() == ncomb) | !any_change {
                // we've either calculated all required distances, or there's
                // nothing to be done -> due to disconnected subgraphs
                break;
            }
        }
        distances
    }

    fn all_shortest_distances_condensed(
        &self,
        nodes: Option<Vec<Node>>,
        neighbor_map: Option<Dict<Node, Vec<Node>>>,
    ) -> Vec<u32> {
        let nodes = match nodes {
            Some(nodes) => nodes,
            None => {
                let mut nodes: Vec<Node> = self.nodes.keys().cloned().collect();
                nodes.sort();
                nodes
            }
        };

        let distances = self.all_shortest_distances(Some(nodes.clone()), neighbor_map);
        let mut condensed = Vec::new();

        let n = nodes.len();
        let mut ni: Node;
        let mut nj: Node;
        let mut key;
        let default_distance = 10 * n as u32;

        for i in 0..n {
            for j in i + 1..n {
                ni = nodes[i];
                nj = nodes[j];
                if nj < ni {
                    key = (nj, ni)
                } else {
                    key = (ni, nj)
                }
                condensed.push(*distances.get(&key).unwrap_or(&default_distance));
            }
        }

        condensed
    }

    #[pyo3(signature = (smoothness=2.0, p=0.75, mu=0.5, neighbor_map=None))]
    fn simple_centrality(
        &self,
        smoothness: f64,
        p: f64,
        mu: f64,
        neighbor_map: Option<Dict<Node, Vec<Node>>>,
    ) -> Dict<Node, f64> {
        let n = self.get_num_edges();

        let neighbor_map = match neighbor_map {
            Some(x) => x,
            None => self.build_neighbor_map(),
        };

        // take a rough closeness as the starting point
        let mut c = self.simple_closeness(p, mu, Some(&neighbor_map));

        // take the propagation time as sqrt hypergraph size
        let r = 10.max((n as f64).powf(0.5) as usize);

        let mut ci: f64;
        let mut previous_c: Dict<Node, f64>;
        for _ in 0..r {
            // do a parallel update
            previous_c = c.clone();

            // spread the centrality of each node into its neighbors
            for &i in self.nodes.keys() {
                ci = previous_c[&i];
                for &j in &neighbor_map[&i] {
                    c.entry(j)
                        .and_modify(|v| *v += smoothness * ci / (r as f64));
                }
            }
            // then rescale all the values between 0.0 and 1.0
            c = affine_renorm(&c);
        }
        c
    }

    #[pyo3(signature= (region, p=2.0))]
    fn simple_distance(&self, region: Vec<Node>, p: f64) -> Dict<Node, f64> {
        let mut ball: FxHashSet<Node> = FxHashSet::default();
        let mut distances: Dict<Node, f64> = Dict::default();
        let mut surface: Dict<Node, u32> = Dict::default();
        let mut queue: VecDeque<Node> = VecDeque::new();
        for i in region {
            ball.insert(i);
            distances.insert(i, 0.0);
            queue.push_back(i);
        }
        let mut d = 0;
        while queue.len() > 0 {
            d += 1;
            for i in queue.drain(..) {
                for &j in &self.neighbors(i) {
                    if !ball.contains(&j) {
                        *surface.entry(j).or_insert(0) += 1;
                    }
                }
            }
            for (j, c) in surface.drain() {
                ball.insert(j);
                queue.push_back(j);
                distances.insert(j, d as f64 + (1.0 / c as f64).powf(p));
            }
        }
        affine_renorm(&distances)
    }

    fn compute_loops(
        &self,
        start: Option<Vec<Node>>,
        max_loop_length: Option<usize>,
    ) -> Vec<Vec<Node>> {
        let n = self.get_num_nodes();
        let (mut max_loop_length, mut auto_length) = match max_loop_length {
            Some(i) => (i, false),
            None => (n, true),
        };

        // let mut queue: VecDeque<Vec<Node>> = self.nodes.keys().map(|&i| vec![i]).collect();
        let mut queue: VecDeque<Vec<Node>> = match start {
            Some(nodes) => nodes.into_iter().map(|i| vec![i]).collect(),
            None => self.nodes.keys().map(|&i| vec![i]).collect(),
        };

        let mut loops = Vec::new();
        let mut seen: FxHashSet<Vec<Node>> = FxHashSet::default();

        let mut path: Vec<Node>;
        let mut key: Vec<Node>;
        let mut new_path: Vec<Node>;
        let mut path_length: usize;
        let mut node_start: Node;
        let mut node_last: Node;
        let mut neighbor_map: Dict<Node, Vec<Node>> = Dict::default();

        loop {
            match queue.pop_front() {
                Some(x) => {
                    path = x;
                }
                None => {
                    break;
                }
            }
            path_length = path.len();
            node_start = path[0];
            node_last = path[path_length - 1];
            // for &node_next in neighbor_map[&node_last].iter() {
            for &node_next in neighbor_map
                .entry(node_last)
                .or_insert(self.neighbors(node_last))
                .iter()
            {
                new_path = path.clone();
                if (path_length > 2) && (node_start == node_next) {
                    // path is a non-trivial closed loop
                    key = path.clone();
                    key.sort_unstable();
                    if !seen.contains(&key) {
                        // only add if not a permutation
                        seen.insert(key);
                        loops.push(new_path);
                        if auto_length {
                            auto_length = false;
                            max_loop_length = path_length + 1;
                        }
                    }
                } else if path.contains(&node_next) {
                    // loops back too early
                    continue;
                } else if auto_length || (path_length < max_loop_length) {
                    // valid to continue exploring path
                    new_path.push(node_next);
                    queue.push_back(new_path);
                }
            }
        }
        loops
    }
}

// ----------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (nodes, smoothness = 2.0, p = 0.75, mu = 0.5))]
fn nodes_to_centrality(
    nodes: Dict<Node, Vec<String>>,
    smoothness: f64,
    p: f64,
    mu: f64,
) -> PyResult<Dict<Node, f64>> {
    let hg = HyperGraph::from_nodes(nodes, None, None);
    let neighbor_map = hg.build_neighbor_map();
    Ok(hg.simple_centrality(smoothness, p, mu, Some(neighbor_map)))
}

#[pyfunction]
#[pyo3(signature = (edges, smoothness = 2.0, p = 0.75, mu = 0.5))]
fn edges_to_centrality(
    edges: Dict<String, Vec<Node>>,
    smoothness: f64,
    p: f64,
    mu: f64,
) -> PyResult<Dict<Node, f64>> {
    let hg = HyperGraph::from_edges(edges, None, None);
    let neighbor_map = hg.build_neighbor_map();
    Ok(hg.simple_centrality(smoothness, p, mu, Some(neighbor_map)))
}

#[pyfunction]
fn optimal_compressed_path(
    chi: u128,
    inputs: Vec<Vec<String>>,
    output: Vec<String>,
    size_dict: Dict<String, u128>,
) -> Vec<(u32, u32)> {
    let nodes: Dict<u32, Vec<String>> = inputs
        .into_iter()
        .enumerate()
        // .map(|(i, term)| (2_u128.pow(i as u32), term))
        .map(|(i, term)| (2_u32.pow(i as u32), term))
        .collect();

    let hg = HyperGraph::from_nodes(nodes, Some(output), Some(size_dict));
    let mut ssa = hg.get_num_nodes() as u32;

    let size0: u128 = hg.nodes.keys().map(|&n| hg.node_size(n)).sum();
    let mut counter = 1..;
    let mut c = counter.next().unwrap();
    let mut queue: BinaryHeap<(usize, i128, u128)> = BinaryHeap::new();
    queue.push((0, 0, c));
    let mut cands: Dict<u128, (HyperGraph, Vec<(u32, u32)>, u128, u128)> = Dict::default();
    cands.insert(c, (hg, Vec::new(), size0, size0));

    let mut best_score: u128 = u128::MAX;
    let mut best_ssa_path: Vec<(u32, u32)> = Vec::new();
    let mut seen: Dict<BTreeSet<Node>, u128> = Dict::default();

    let mut ij;
    let mut i;
    let mut j;
    let mut hg_next;
    let mut edges;
    let mut i_size;
    let mut j_size;
    let mut ij_size;
    let mut dsize;
    let mut new_size;
    let mut new_score;
    let mut new_ssa_path;
    let mut new_rank;

    'outer: while let Some((_, _, c_cur)) = queue.pop() {
        let (hg_cur, ssa_path, size, score) = cands.remove(&c_cur).unwrap();

        // reached a completely contracted graph - check if the best
        if hg_cur.get_num_nodes() == 1 {
            if score < best_score {
                best_score = score;
                best_ssa_path = ssa_path;
            }
            continue;
        }

        // if we've reached this graph before with better score then skip
        let key: BTreeSet<Node> = hg_cur.nodes.keys().cloned().collect();
        if let Some(&seen_size) = seen.get(&key) {
            if size >= seen_size {
                continue 'outer;
            }
        }
        seen.insert(key, size);

        'inner: for (_, nodes) in &hg_cur.edges {
            i = nodes[0];
            j = nodes[1];
            hg_next = hg_cur.clone();

            // compress around pair and then contract
            edges = hg_next
                .get_node(i)
                .iter()
                .chain(hg_next.get_node(j).iter())
                .cloned()
                .collect();
            hg_next.compress(chi, Some(edges));
            ij = hg_next.contract(i, j, Some(i | j));

            // compute sizes and scores
            i_size = hg_cur.node_size(i);
            j_size = hg_cur.node_size(j);
            ij_size = hg_next.node_size(ij);
            dsize = ij_size - i_size - j_size;
            new_size = size + dsize;
            new_score = score.max(new_size);

            if new_score >= best_score {
                continue 'inner;
            }

            // add the new hypergraph to the stack of potential condidates
            c = counter.next().unwrap();
            new_ssa_path = ssa_path.clone();
            new_ssa_path.push((i, j));
            new_rank = (new_ssa_path.len(), -(dsize as i128), c);
            queue.push(new_rank);
            cands.insert(c, (hg_next, new_ssa_path, new_size, new_score));
        }
    }

    // convert the best path to the ssa node identifiers
    let mut ssa_path: Vec<(u32, u32)> = Vec::new();
    let mut ssa_lookup: Dict<u32, u32> = (0..ssa).map(|i| (2_u32.pow(i), i)).collect();
    for (i, j) in best_ssa_path {
        ij = i | j;
        ssa_lookup.insert(ij, ssa);
        ssa = ssa + 1;
        ssa_path.push((ssa_lookup[&i], ssa_lookup[&j]));
    }
    ssa_path
}

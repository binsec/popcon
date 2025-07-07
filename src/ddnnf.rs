/**************************************************************************/
/*  This file is part of POPCON.                                          */
/*                                                                        */
/*  Copyright (C) 2025                                                    */
/*    CEA (Commissariat à l'énergie atomique et aux énergies              */
/*         alternatives)                                                  */
/*                                                                        */
/*  you can redistribute it and/or modify it under the terms of the GNU   */
/*  Lesser General Public License as published by the Free Software       */
/*  Foundation, version 2.1.                                              */
/*                                                                        */
/*  It is distributed in the hope that it will be useful,                 */
/*  but WITHOUT ANY WARRANTY; without even the implied warranty of        */
/*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         */
/*  GNU Lesser General Public License for more details.                   */
/*                                                                        */
/*  See the GNU Lesser General Public License version 2.1                 */
/*  for more details (enclosed in the file licenses/LGPLv2.1).            */
/*                                                                        */
/**************************************************************************/

//! Operations on d-DNNF: model counting, popularity contest

use crate::cnf::{CnfFormula, Lit, ModelCount, PopConBounds, Var};
use crate::d4::D4dDNNFOutput;
use crate::model::Model;
use crate::smt::VarValue;
use crate::utils::Input;
use anyhow::Context;
use fixedbitset::FixedBitSet;
use num_bigint::BigUint;
use num_traits::{One, Zero};
use petgraph::data::DataMap;
use petgraph::graph::{EdgeReference, NodeIndex};
use petgraph::visit::{Dfs, DfsPostOrder, EdgeRef, IntoNodeReferences, NodeRef, Reversed};
use petgraph::EdgeDirection::{Incoming, Outgoing};
use petgraph::Graph;
use std::collections::BTreeMap;
use std::convert::{TryFrom, TryInto};
use std::fmt::Debug;
use std::io::Read;
use std::io::Write;
use std::ops::{Add, Mul, MulAssign};
use std::path::Path;
use tracing::{trace, trace_span};
use varisat_formula::ExtendFormula;

#[derive(Debug, Clone, PartialEq, Eq)]
/// One node of the d-DNNF formula
pub enum Node {
    /// Literal
    L(Lit),
    /// And node, represented by the index of conjoined nodes
    A(Vec<usize>),
    /// Or node, represented by the index of conjoined nodes and the opposing variable
    O(Option<Var>, usize, usize),
    /// False node
    F,
}
#[derive(Clone)]
/// A model count and set of variables present
/// # Example
/// ```rust
/// use popcon::ddnnf::QualifiedModelCount;
/// use num_bigint::BigUint;
/// use fixedbitset::FixedBitSet;
/// use std::iter::FromIterator;
/// let a = QualifiedModelCount {
///     count: 1u32.into(),
///     variables: FixedBitSet::from_iter([0, 4].iter().cloned()),
/// };
/// let b = QualifiedModelCount {
///     count: 2u32.into(),
///     variables: FixedBitSet::from_iter([1, 2, 4].iter().cloned()),
/// };
/// let c = QualifiedModelCount {
///     count: 3u32.into(),
///     variables: FixedBitSet::from_iter([0, 3].iter().cloned()),
/// };
/// let result_add = QualifiedModelCount {
///     count: 8u32.into(),
///     variables: FixedBitSet::from_iter([0, 1, 2, 4].iter().cloned()),
/// };
/// let result_mul = QualifiedModelCount {
///     count: 6u32.into(),
///     variables: FixedBitSet::from_iter([0, 1, 2, 3, 4].iter().cloned()),
/// };
/// assert_eq!(result_add, &a+&b);
/// assert_eq!(result_mul, &b*&c);
/// ```
pub struct QualifiedModelCount {
    /// number of models
    pub count: BigUint,
    /// variables present
    pub variables: FixedBitSet,
}

impl QualifiedModelCount {
    /// maximum between two counts, normalized
    pub fn maximum(&self, other: &QualifiedModelCount) -> QualifiedModelCount {
        let mut variables = self.variables.clone();
        variables.union_with(&other.variables);
        let bits = &variables.count_ones(..);
        let count = std::cmp::max(
            &self.count * (BigUint::one() << (bits - self.variables.count_ones(..))),
            &other.count * (BigUint::one() << (bits - other.variables.count_ones(..))),
        );
        QualifiedModelCount { count, variables }
    }

    /// Add variables to self.variables, adjusting the model count.
    /// Panics if variables is not a superset of self.variables.
    pub fn upgrade_to(&mut self, variables: &FixedBitSet) {
        assert_eq!(variables.len(), self.variables.len());
        assert!(variables.is_superset(&self.variables));
        let current_bits = self.variables.count_ones(..);
        let new_bits = variables.count_ones(..);
        self.count *= BigUint::one() << new_bits - current_bits;
        self.variables.union_with(variables);
    }

    /// A qualified model count of 1 on 0 variables (true node)
    pub fn one(nvars: usize) -> Self {
        QualifiedModelCount {
            count: One::one(),
            variables: FixedBitSet::with_capacity(nvars),
        }
    }

    /// A qualified model count of 0 on 0 variables (false node)
    pub fn zero(nvars: usize) -> Self {
        QualifiedModelCount {
            count: Zero::zero(),
            variables: FixedBitSet::with_capacity(nvars),
        }
    }

    /// Equivalent to `self = Self::zero()`, but reuses the allocation for `self.variables`.
    pub fn make_zero(&mut self) {
        self.count = Zero::zero();
        self.variables.clear();
    }

    /// Equivalent to `self = Self::one()`, but reuses the allocation for `self.variables`.
    pub fn make_one(&mut self) {
        self.count = One::one();
        self.variables.clear();
    }

    /// checks that `count <= 2^#variables`
    pub fn check(&self) -> anyhow::Result<()> {
        if self.count.is_zero() {
            return Ok(());
        }
        let nvars = self.variables.count_ones(..) as u64;
        let bits = self.count.bits();
        anyhow::ensure!(
            nvars + 1 >= bits,
            "there cannot be {} models ({} bits) for a formula with {} variables",
            &self.count,
            bits,
            nvars
        );
        Ok(())
    }
}

impl From<QualifiedModelCount> for ModelCount {
    fn from(q: QualifiedModelCount) -> Self {
        let bits = q.variables.count_ones(..);
        ModelCount {
            model_count: q.count,
            bits,
        }
    }
}

impl Add for &QualifiedModelCount {
    type Output = QualifiedModelCount;

    fn add(self, rhs: Self) -> Self::Output {
        let mut variables = self.variables.clone();
        variables.union_with(&rhs.variables);
        let bits = &variables.count_ones(..);
        let count = &self.count * (BigUint::one() << (bits - self.variables.count_ones(..)))
            + &rhs.count * (BigUint::one() << (bits - rhs.variables.count_ones(..)));
        QualifiedModelCount { count, variables }
    }
}

impl Mul for &QualifiedModelCount {
    type Output = QualifiedModelCount;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut variables = self.variables.clone();
        debug_assert_eq!(
            self.variables.intersection(&rhs.variables).count(),
            0,
            "and on intersecting var set: {:?} * {:?}",
            &self,
            &rhs
        );
        variables.union_with(&rhs.variables);
        let count = &self.count * &rhs.count;
        QualifiedModelCount { count, variables }
    }
}

impl MulAssign<&QualifiedModelCount> for QualifiedModelCount {
    fn mul_assign(&mut self, rhs: &QualifiedModelCount) {
        debug_assert_eq!(
            self.variables.intersection(&rhs.variables).count(),
            0,
            "and on intersecting var set: {:?} * {:?}",
            &self,
            &rhs
        );
        self.variables.union_with(&rhs.variables);
        self.count *= &rhs.count;
    }
}

impl Debug for QualifiedModelCount {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}@{:?}vars", self.count, self.variables.count_ones(..))
    }
}

impl Ord for QualifiedModelCount {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if self.variables == other.variables {
            self.count.cmp(&other.count)
        } else if self.variables.is_superset(&other.variables) {
            let mut tmp = other.clone();
            tmp.upgrade_to(&self.variables);
            self.count.cmp(&tmp.count)
        } else if self.variables.is_subset(&other.variables) {
            let mut tmp = self.clone();
            tmp.upgrade_to(&other.variables);
            tmp.count.cmp(&other.count)
        } else {
            let mut union = self.variables.clone();
            union.union_with(&other.variables);
            let mut s2 = self.clone();
            s2.upgrade_to(&union);
            let mut o2 = other.clone();
            o2.upgrade_to(&union);
            s2.count.cmp(&o2.count)
        }
    }
}

impl PartialOrd for QualifiedModelCount {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for QualifiedModelCount {}

impl PartialEq for QualifiedModelCount {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == std::cmp::Ordering::Equal
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// One node of the d-DNNF formula
pub enum CountingNode {
    /// An uncontrolled formula with n models
    Count(QualifiedModelCount),
    /// And node
    And,
    /// Ite node, with opposing variable
    Ite(Var),
}

impl CountingNode {
    /// returns the variable labeling this node, or None
    fn var(&self) -> Option<Var> {
        match &self {
            CountingNode::Ite(v) => Some(*v),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Metadata for a d-DNNF formula
pub struct Header {
    /// Number of variables
    pub nvars: usize,
    /// Number of nodes
    pub nnodes: usize,
    /// Number of edges. Dsharp sometimes reports wrong numbers.
    pub nedges: usize,
}

/// Iterate through the nodes of the d-DNNF formula in topological order
pub trait Visitor {
    /// Will be called before visit node
    fn visit_header(&mut self, _header: &Header) -> anyhow::Result<()> {
        Ok(())
    }
    /// Will be called in topological order
    fn visit_node(&mut self, _node: &Node) -> anyhow::Result<()> {
        Ok(())
    }
}

impl<T: Visitor> Visitor for &mut T {
    /// Will be called before visit node
    fn visit_header(&mut self, header: &Header) -> anyhow::Result<()> {
        T::visit_header(self, header)
    }
    /// Will be called in topological order
    fn visit_node(&mut self, node: &Node) -> anyhow::Result<()> {
        T::visit_node(self, node)
    }
}

/// Something that can be interpreted as a d-DNNF
pub trait Source {
    /// Call the visitor methods according to the formula encoded by the source.
    fn visit<T: Visitor>(self, visitor: T) -> anyhow::Result<()>;
}

struct BoundInfos {
    infos: Vec<Option<QualifiedBounds>>,
    lower_bound_type: LowerBoundQuality,
}

/// A d-DNNF held in memory
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryDDNNF {
    /// Metadata for the formula
    pub header: Header,
    /// Nodes in topological order
    pub nodes: Vec<Node>,
}

/// How to compute a lower bound
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LowerBoundQuality {
    /// Like model counting on uncontrolled subtrees, max elsewhere
    Bad,
    /// Like `Bad`, but take the witness associated with it and take its popularity
    Fast,
    /// On controlled ites, take the branch whose witness has highest popularity. Quadratic.
    Precise,
}

/// A popcon d-DNNF where uncontrolled subtrees where summarized as a model count
#[derive(Clone)]
pub struct CountingDDNNF {
    /// number of variables of the original formula
    pub nvars: usize,
    /// controlled variables
    pub controlled: FixedBitSet,
    /// uncontrolled variables that are left
    pub relaxed: FixedBitSet,
    /// variables that are set to a constant value due to the oriented variables pass
    pub oriented: FixedBitSet,
    /// number of variables of the model counts
    pub nuncontrolled: usize,
    /// root node of the graph
    pub root: NodeIndex,
    /// graph. Edge labels are Some(true) and Some(false) for Ite children, and None for others
    pub graph: Graph<CountingNode, Option<bool>>,
}

impl Debug for CountingDDNNF {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        let config = &[petgraph::dot::Config::EdgeNoLabel];
        let node_attr = |_g: &Graph<CountingNode, Option<bool>>,
                         node: (NodeIndex, &CountingNode)| {
            let attr = match node.weight() {
                CountingNode::Count(_) => "color=blue fontcolor=white",
                CountingNode::And => "color=yellow",
                CountingNode::Ite(v) if self.relaxed[v.index()] => "color=red fontcolor=white",
                CountingNode::Ite(v) if self.controlled[v.index()] => "color=green fontcolor=white",
                CountingNode::Ite(_) => "color=black fontcolor=white",
            };
            format!("style=filled {}", attr)
        };
        let edge_attr = |_g: &Graph<CountingNode, Option<bool>>, edge: EdgeReference<_>| {
            let attr = match edge.weight() {
                None => "",
                Some(true) => "color=green",
                Some(false) => "color=red",
            };
            attr.to_owned()
        };
        let dot =
            petgraph::dot::Dot::with_attr_getters(&self.graph, config, &edge_attr, &node_attr);
        write!(fmt, "{:?}", dot)
    }
}

impl CountingDDNNF {
    /// Remove unreachable nodes
    fn gc(&mut self) -> anyhow::Result<()> {
        let span = trace_span!("gc").entered();
        let root = self.root;
        let mut dfs = Dfs::new(&self.graph, root);
        while let Some(_) = dfs.next(&self.graph) {}
        self.graph
            .retain_nodes(|_g, idx| dfs.discovered[idx.index()]);
        // root node has changed index
        let mut iter = self.graph.externals(petgraph::Direction::Incoming);
        match iter.next() {
            Some(idx) => {
                self.root = idx;
            }
            None => anyhow::bail!("root lost during gc"),
        };
        anyhow::ensure!(iter.next().is_none(), "several roots after gc");
        drop(span);
        Ok(())
    }

    /// write the graph to a file in dot format.
    #[allow(unused)]
    fn dump(&self, name: &Path) -> anyhow::Result<()> {
        let mut file = std::fs::File::create(name)
            .with_context(|| format!("opening dot dump file {}", name.display()))?;
        writeln!(file, "{:?}", self)
            .with_context(|| format!("writing dot dump file {}", name.display()))?;
        Ok(())
    }

    /// trace!() some stats about the size of the graph
    fn debug_stats(&self, name: &'static str) {
        tracing::trace!(
            stats = true,
            node_count = self.graph.node_count(),
            ite_count = self
                .graph
                .node_weights()
                .filter(|i| matches!(i, CountingNode::Ite(_)))
                .count(),
            non_trivial_and_count = self
                .graph
                .node_references()
                .filter(|n| n.weight() == &CountingNode::And
                    && self
                        .graph
                        .neighbors_directed(n.id(), Outgoing)
                        .filter(|&c| !matches!(&self.graph[c], CountingNode::Count(_)))
                        .count()
                        > 1)
                .count(),
            "graph stats at {}",
            name
        );
    }

    /// detect oriented vars, ie controlled vars such that there is always more models in one
    /// polarity than in the other. Currently does not remove them yet.
    /// Returns Ok(false) when there was nothing to simplify
    fn oriented_vars(&mut self, opts: SimplificationOptions) -> anyhow::Result<bool> {
        if !opts.oriented_vars {
            return Ok(false);
        }
        use std::cmp::Ordering::*;
        let span = trace_span!("oriented_vars").entered();
        self.debug_stats("before original_vars");
        let mut orientation = vec![Some(Equal); self.nvars];
        for n in self.graph.node_references() {
            match n.weight() {
                CountingNode::Ite(v) => {
                    if !self.controlled[v.index()] {
                        continue;
                    };
                    if self.oriented[v.index()] {
                        continue;
                    };
                    let mut true_count = None;
                    let mut false_count = None;
                    for edge in self.graph.edges_directed(n.id(), Outgoing) {
                        match &self.graph[edge.target()] {
                            CountingNode::Count(c) => match edge.weight() {
                                Some(true) => true_count = Some(c),
                                Some(false) => false_count = Some(c),
                                None => (),
                            },
                            _ => (),
                        }
                    }
                    let this_orientation = match (true_count, false_count) {
                        (Some(t), Some(f)) => Some(t.cmp(f)),
                        _ => None,
                    };
                    let next = match (orientation[v.index()], this_orientation) {
                        (_, None) | (None, _) => None,
                        (Some(Less), Some(Greater)) | (Some(Greater), Some(Less)) => None,
                        (Some(Greater), Some(Greater))
                        | (Some(Greater), Some(Equal))
                        | (Some(Equal), Some(Greater)) => Some(Greater),
                        (Some(Less), Some(Less))
                        | (Some(Less), Some(Equal))
                        | (Some(Equal), Some(Less)) => Some(Less),
                        (Some(Equal), Some(Equal)) => Some(Equal),
                    };
                    orientation[v.index()] = next;
                }
                _ => (),
            }
        }
        // resolve ites for oriented variables
        let mut edges_to_remove = FixedBitSet::with_capacity(self.graph.edge_count());
        let mut edges_to_add = Vec::new();
        for i in self.graph.node_indices() {
            match self.graph[i] {
                CountingNode::Ite(v) => {
                    if self.controlled[v.index()] && !self.oriented[v.index()] {
                        if let Some(orientation) = orientation[v.index()] {
                            let polarity = orientation == Greater;
                            let mut replace_by = None;
                            for edge in self.graph.edges_directed(i, Outgoing) {
                                if *edge.weight() == Some(polarity) {
                                    replace_by = Some(edge.target());
                                }
                            }
                            let replace_by = match replace_by {
                                Some(x) => x,
                                None => anyhow::bail!("unlabeled edges out of ite"),
                            };
                            for edge in self.graph.edges_directed(i, Incoming) {
                                edges_to_remove.insert(edge.id().index());
                                edges_to_add.push((edge.source(), replace_by, *edge.weight()));
                            }
                        }
                    }
                }
                _ => (),
            }
        }
        self.remove_edges(&edges_to_remove);
        for (from, to, w) in edges_to_add {
            self.graph.add_edge(from, to, w);
        }
        // add implied literals on top
        let mut n = 0;
        let mut false_node = None;
        let nvars = self.nvars;
        let mut get_false = |g: &mut Graph<CountingNode, _>| match false_node {
            Some(n) => n,
            None => {
                let c = QualifiedModelCount::zero(nvars);
                let n = g.add_node(CountingNode::Count(c));
                false_node = Some(n);
                n
            }
        };
        for (i, orientation) in orientation.drain(..).enumerate() {
            let v = Var::from_index(i);
            if self.controlled[i] && !self.oriented[i] {
                if let Some(orientation) = orientation {
                    n += 1;
                    let polarity = orientation == Greater;
                    let new_root = self.graph.add_node(CountingNode::Ite(v));
                    let f = get_false(&mut self.graph);
                    self.graph.add_edge(new_root, self.root, Some(polarity));
                    self.graph.add_edge(new_root, f, Some(!polarity));
                    self.root = new_root;
                    self.oriented.insert(i);
                }
            }
        }
        drop(span);
        trace!("removed {} oriented variables", n);
        if n > 0 {
            self.gc().context("gc")?;
            self.simplify(opts).context("simplify")?;
            self.debug_stats("after original_vars");
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn simplify(&mut self, opts: SimplificationOptions) -> anyhow::Result<()> {
        let span = trace_span!("simplify").entered();
        let mut dfs = DfsPostOrder::new(&self.graph, self.root);
        let mut removed_edges = FixedBitSet::with_capacity(self.graph.edge_count());
        while let Some(idx) = dfs.next(&self.graph) {
            match &self.graph[idx] {
                CountingNode::Count(_) => (),
                CountingNode::Ite(v) => {
                    let children: Vec<_> = self
                        .graph
                        .edges_directed(idx, Outgoing)
                        .filter(|e| !removed_edges[e.id().index()])
                        .collect();
                    let removed_relaxed = !(self.controlled[v.index()] || self.relaxed[v.index()]);
                    match children.as_slice() {
                        &[a, b] => {
                            match (&self.graph[a.target()], &self.graph[b.target()]) {
                                (CountingNode::Count(na), CountingNode::Count(nb)) => {
                                    if self.controlled[v.index()] && na == nb && opts.rewriting {
                                        let result = na.clone();
                                        let bid = b.id();
                                        let aid = a.id();
                                        // borrow of self.graph ends here
                                        self.graph[idx] = CountingNode::Count(result);
                                        removed_edges.insert(bid.index());
                                        removed_edges.insert(aid.index());
                                    } else if removed_relaxed {
                                        let mut result = na + nb;
                                        result.variables.insert(v.index());
                                        let bid = b.id();
                                        let aid = a.id();
                                        // borrow of self.graph ends here
                                        self.graph[idx] = CountingNode::Count(result);
                                        removed_edges.insert(bid.index());
                                        removed_edges.insert(aid.index());
                                    }
                                }
                                _ => {
                                    if removed_relaxed {
                                        anyhow::bail!(
                                            "ite(removed relaxed) has non count children {:?}",
                                            children
                                                .iter()
                                                .map(|e| (*e.weight(), &self.graph[e.target()]))
                                                .collect::<Vec<_>>()
                                        );
                                    }
                                }
                            }
                        }
                        _ => anyhow::bail!("ite node with {} != 2 children", children.len()),
                    }
                }
                CountingNode::And => {
                    // check if the and has no children or a single one
                    let children: Vec<_> = self
                        .graph
                        .edges_directed(idx, Outgoing)
                        .filter(|e| !removed_edges[e.id().index()])
                        .map(|e| (e.id(), e.target()))
                        .collect();
                    match children.as_slice() {
                        &[] => {
                            // and with 0 child, replace by Count(1)
                            self.graph[idx] = CountingNode::Count(QualifiedModelCount {
                                count: One::one(),
                                variables: FixedBitSet::with_capacity(self.nvars),
                            });
                        }
                        &[(edge, node)] => {
                            // and with 1 child, remove the and by replacing it with it's child
                            self.graph[idx] = self.graph[node].clone();
                            removed_edges.insert(edge.index());
                            let new_children: Vec<_> = self
                                .graph
                                .edges_directed(node, Outgoing)
                                .filter(|e| !removed_edges[e.id().index()])
                                .map(|e| (e.target(), *e.weight()))
                                .collect();
                            // w in case `node` is an ite node
                            for (i, w) in new_children {
                                self.graph.add_edge(idx, i, w);
                            }
                            removed_edges.grow(self.graph.edge_count());
                        }
                        _ if opts.rewriting => {
                            // First absorb the children of children And nodes
                            let mut edges_to_add = vec![];
                            for (edge, child) in children {
                                match &self.graph[child] {
                                    CountingNode::And => {
                                        for eref in self.graph.edges_directed(child, Outgoing) {
                                            if removed_edges.contains(eref.id().index()) {
                                                continue;
                                            }
                                            edges_to_add.push((idx, eref.target(), *eref.weight()));
                                            removed_edges.insert(eref.id().index());
                                            removed_edges.insert(edge.index());
                                        }
                                    }
                                    _ => (),
                                }
                            }
                            for (from, to, w) in edges_to_add {
                                self.graph.add_edge(from, to, w);
                            }
                            removed_edges.grow(self.graph.edge_count());
                            // coalesce all Count children
                            let mut children: Vec<_> = self
                                .graph
                                .edges_directed(idx, Outgoing)
                                .filter(|e| !removed_edges[e.id().index()])
                                .filter_map(|e| match &self.graph[e.target()] {
                                    CountingNode::Count(n) => Some((e.id(), n.clone())),
                                    _ => None,
                                })
                                .collect();
                            // if there is at least a count child
                            if let Some((edge, mut count)) = children.pop() {
                                for (oedge, n) in children.drain(..) {
                                    count *= &n;
                                    removed_edges.insert(oedge.index());
                                }
                                removed_edges.insert(edge.index());
                                if self
                                    .graph
                                    .edges_directed(idx, Outgoing)
                                    .filter(|e| !removed_edges[e.id().index()])
                                    .count()
                                    == 0
                                {
                                    // all children were count nodes, replace the and fully by a
                                    // count node
                                    self.graph[idx] = CountingNode::Count(count);
                                } else {
                                    let new = self.graph.add_node(CountingNode::Count(count));
                                    self.graph.add_edge(idx, new, None);
                                    removed_edges.grow(self.graph.edge_count());
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
        self.remove_edges(&removed_edges);
        drop(span);
        self.gc().context("gc")?;
        Ok(())
    }

    /// faster than graph.retain_edges when there are a lot of removed edges
    fn remove_edges(&mut self, removed_edges: &FixedBitSet) {
        let n = self.graph.node_count();
        let m = self.graph.edge_count();
        let re = removed_edges.count_ones(..);
        if re == 0 {
            return;
        }
        let average_degree = (m as f32) / (n as f32);
        if (re as f32) * 4. * average_degree < (n + m) as f32 {
            // normal retain_edges has an advantage
            self.graph.retain_edges(|_, i| !removed_edges[i.index()]);
        } else {
            // create a new graph and move everything there
            let mut new = Graph::with_capacity(
                self.graph.node_count(),
                self.graph.edge_count() - removed_edges.count_ones(..),
            );
            std::mem::swap(&mut new, &mut self.graph);
            let (nodes, edges) = new.into_nodes_edges();
            for node in nodes {
                self.graph.add_node(node.weight);
            }
            for (i, edge) in edges.into_iter().enumerate() {
                if !removed_edges[i] {
                    self.graph
                        .add_edge(edge.source(), edge.target(), edge.weight);
                }
            }
        }
    }

    /// Converts to a smt2 formula with the same counting graph
    fn to_smt2<W: Write>(&self, mut out: W) -> anyhow::Result<()> {
        writeln!(out, "(set-logic QF_BV)")?;
        for i in 1..=self.controlled.len() {
            writeln!(out, "(declare-fun var{} () (_ BitVec 1))", i)?;
        }
        for i in self.controlled.ones() {
            writeln!(out, "(set-info :controlled var{})", i + 1)?;
        }
        let mut dfs = DfsPostOrder::new(&self.graph, self.root);
        while let Some(node) = dfs.next(&self.graph) {
            match &self.graph[node] {
                CountingNode::Count(c) => {
                    let size = c.variables.count_ones(..);
                    if c.count.is_zero() {
                        writeln!(out, "(define-fun node{} () Bool false)", node.index(),)?;
                    } else if size == 0 {
                        writeln!(
                            out,
                            "(define-fun node{} () Bool {})",
                            node.index(),
                            if c.count.is_zero() { "false" } else { "true" }
                        )?;
                    } else {
                        if size == 1 {
                            write!(
                                out,
                                "(define-fun x{} () (_ BitVec 1) var{})",
                                node.index(),
                                c.variables.ones().next().unwrap() + 1
                            )?;
                        } else {
                            write!(
                                out,
                                "(define-fun x{} () (_ BitVec {}) (concat",
                                node.index(),
                                size
                            )?;
                            for i in c.variables.ones() {
                                write!(out, " var{}", i + 1)?;
                            }
                            writeln!(out, "))")?;
                        }
                        writeln!(
                            out,
                            "(define-fun node{} () Bool (bvule x{} (_ bv{} {})))",
                            node.index(),
                            node.index(),
                            &c.count - 1u32,
                            size
                        )?;
                    }
                }
                CountingNode::And => {
                    write!(out, "(define-fun node{} () Bool (and", node.index())?;
                    for child in self.graph.neighbors_directed(node, Outgoing) {
                        write!(out, " node{}", child.index())?;
                    }
                    writeln!(out, "))")?;
                }
                CountingNode::Ite(v) => {
                    let children: Vec<(Option<bool>, NodeIndex)> = self
                        .graph
                        .edges_directed(node, Outgoing)
                        .map(|e| (*e.weight(), e.target()))
                        .collect();
                    let (true_child, false_child) = match children.as_slice() {
                        &[(Some(true), t), (Some(false), f)]
                        | &[(Some(false), f), (Some(true), t)] => (t, f),
                        &[(a, _), (b, _)] => anyhow::bail!(
                            "ite is not opposing: outgoind edges with label {:?} {:?}",
                            a,
                            b
                        ),
                        _ => anyhow::bail!("ite has {} children", children.len()),
                    };
                    writeln!(
                        out,
                        "(define-fun node{} () Bool (ite (= var{} (_ bv1 1)) node{} node{}))",
                        node.index(),
                        v.to_dimacs(),
                        true_child.index(),
                        false_child.index()
                    )?;
                }
            }
        }
        writeln!(out, "(assert node{})", self.root.index())?;
        Ok(())
    }

    fn simplify_by_bounds(
        &mut self,
        bound_info: &BoundInfos,
        opts: SimplificationOptions,
    ) -> anyhow::Result<()> {
        if !opts.simplify_by_bounds {
            return Ok(());
        }
        fn only_parent(s: &CountingDDNNF, node: NodeIndex) -> anyhow::Result<NodeIndex> {
            let mut iter = s
                .graph
                .neighbors_directed(node, petgraph::EdgeDirection::Incoming);
            let res = iter
                .next()
                .ok_or_else(|| anyhow::anyhow!("node has no parent"))?;
            anyhow::ensure!(iter.next().is_none(), "node has >= 2 parents");
            Ok(res)
        }
        fn max_mc(
            s: &CountingDDNNF,
            bound_info: &[Option<QualifiedBounds>],
            leaf: NodeIndex,
        ) -> anyhow::Result<QualifiedModelCount> {
            let mut current = leaf;
            let mut approx = match &s.graph[leaf] {
                CountingNode::Count(c) => c.clone(),
                _ => anyhow::bail!("max_mc on not counting node"),
            };
            loop {
                let parent = only_parent(s, current)?;
                let new = match s.graph[parent] {
                    CountingNode::Count(_) => anyhow::bail!("count node cannot have children"),
                    CountingNode::Ite(v) if s.controlled[v.index()] => approx,
                    CountingNode::Ite(v) => {
                        let other_children: Vec<NodeIndex> = s
                            .graph
                            .neighbors_directed(parent, Outgoing)
                            .filter(|&n| n != current)
                            .collect();
                        anyhow::ensure!(other_children.len() == 1, "ite node has != 2 children");
                        let other_child = other_children[0];
                        let bound = bound_info[other_child.index()]
                            .as_ref()
                            .ok_or_else(|| anyhow::anyhow!("no bound info for node"))?;
                        let mut res = &approx + &bound.upper;
                        res.variables.insert(v.index());
                        res
                    }
                    CountingNode::And => {
                        let mut res = approx;
                        for child in s.graph.neighbors_directed(parent, Outgoing) {
                            if child != current {
                                let bound = bound_info[child.index()]
                                    .as_ref()
                                    .ok_or_else(|| anyhow::anyhow!("no bound info for node"))?;
                                res *= &bound.upper;
                            }
                        }
                        res
                    }
                };
                approx = new;
                if parent == s.root {
                    break;
                }
                current = parent;
            }
            Ok(approx)
        }
        let global_bounds = self.bounds(&bound_info)?;
        let mut i = 0;
        for leaf in self.graph.externals(Outgoing) {
            if let CountingNode::Count(c) = &self.graph[leaf] {
                if c.count.is_zero() {
                    continue;
                };
                if let Ok(max) = max_mc(self, &bound_info.infos, leaf) {
                    if (&max) < &global_bounds.lower {
                        i += 1;
                    }
                }
            }
        }
        trace!(removable_by_bounds = i, stats = true);
        if i > 0 {
            tracing::warn!("could remove {} nodes with simplify_by_bounds", i);
        }
        Ok(())
    }

    /// get bounds for the whole formula from the return value of bound_info.
    ///
    /// Must be called with the same value of lower bound type
    fn bounds(&self, bound_infos: &BoundInfos) -> anyhow::Result<QualifiedBounds> {
        let mut res = bound_infos.infos[self.root.index()]
            .clone()
            .context("unreachable root")?;
        if bound_infos.lower_bound_type == LowerBoundQuality::Fast {
            // in fast mode the witness is actually better than the lower bound
            let witness_pop = self
                .popularity(self.root, &res.witness)
                .context("popularity of root witness")?;
            debug_assert!(&witness_pop >= &res.lower);
            res.lower = witness_pop;
        }
        Ok(res)
    }

    /// remove relaxed vars which can form a layer below others
    /// Returns Ok(false) when there was nothing to simplify
    fn remove_useless_relaxed_vars(&mut self, opts: SimplificationOptions) -> anyhow::Result<bool> {
        if !opts.useless_relaxed_vars {
            return Ok(false);
        }
        let span = trace_span!("remove_useless_relaxed_vars").entered();
        let mut one_controlled = None;
        // relaxed var -> last visited node of this var
        let mut by_var = BTreeMap::new();
        let mut graph = self.graph.clone();
        for i in 0..graph.node_count() {
            let i = NodeIndex::new(i);
            if let Some(var) = graph[i].var() {
                if self.controlled[var.index()] {
                    // make all controlled nodes reachable from one_controlled
                    match one_controlled {
                        None => one_controlled = Some(i),
                        Some(n) => {
                            graph.add_edge(i, n, None);
                        }
                    }
                } else if self.relaxed[var.index()] {
                    // put all nodes of a same relaxed variable in the same strongly connected
                    // component
                    let e = by_var.entry(var);
                    match e {
                        std::collections::btree_map::Entry::Occupied(e) => {
                            graph.add_edge(*e.get(), i, None);
                            graph.add_edge(i, *e.get(), None);
                        }
                        std::collections::btree_map::Entry::Vacant(e) => {
                            e.insert(i);
                        }
                    };
                }
            }
        }
        let mut useless_relaxed_vars = self.relaxed.clone();
        let reversed = Reversed(&graph);
        if let Some(n) = one_controlled {
            let mut dfs = Dfs::new(&reversed, n);
            while let Some(node) = dfs.next(&reversed) {
                if let Some(var) = reversed
                    .node_weight(node)
                    .ok_or_else(|| anyhow::anyhow!("invalid reverse node id"))?
                    .var()
                {
                    useless_relaxed_vars.set(var.index(), false);
                }
            }
        }
        let n = useless_relaxed_vars.count_ones(..);
        drop(span);
        if n != 0 {
            self.debug_stats("before removing useless relaxed vars");
            self.relaxed.difference_with(&useless_relaxed_vars);
            trace!("removing {} useless relaxed vars", n);
            self.simplify(opts).context("simplification")?;
            self.gc().context("gc")?;
            self.debug_stats("after removing useless relaxed vars");
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// fast mode: get a linear time interval with bad lower bound
    fn bound_info(&self, lower_bound_type: LowerBoundQuality) -> anyhow::Result<BoundInfos> {
        let mut infos: Vec<Option<QualifiedBounds>> = vec![None; self.graph.node_count()];
        let mut dfs = DfsPostOrder::new(&self.graph, self.root);
        while let Some(node) = dfs.next(&self.graph) {
            let mut children = vec![];
            for n in self.graph.edges_directed(node, Outgoing) {
                let info = infos[n.target().index()]
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("no info for child {:?}", n))?;
                children.push((*n.weight(), info));
            }
            let new = match &self.graph[node] {
                CountingNode::Count(c) => QualifiedBounds::exactly(c, Model::empty(self.nvars)),
                CountingNode::And => {
                    let mut res: QualifiedBounds =
                        children.pop().map(|(_, x)| x.clone()).unwrap_or_else(|| {
                            QualifiedBounds::exactly(
                                &QualifiedModelCount::one(self.nvars),
                                Model::empty(self.nvars),
                            )
                        });
                    for (_edge_weight, weight) in children {
                        res *= &weight;
                    }
                    res
                }
                CountingNode::Ite(v) => {
                    let (true_child, false_child) = match children.as_slice() {
                        [(Some(true), wt), (Some(false), wf)]
                        | [(Some(false), wf), (Some(true), wt)] => (wt, wf),
                        _ => anyhow::bail!("ite with children {:?}", children),
                    };
                    if self.controlled[v.index()] {
                        let lower = true_child.lower.maximum(&false_child.lower);
                        let mut witness;
                        if lower == false_child.lower {
                            witness = false_child.witness.clone();
                        } else {
                            assert_eq!(lower, true_child.lower);
                            witness = true_child.witness.clone();
                            witness.0.insert(v.index());
                        };
                        QualifiedBounds {
                            witness,
                            lower,
                            upper: true_child.upper.maximum(&false_child.upper),
                        }
                    } else {
                        let true_lower = match lower_bound_type {
                            LowerBoundQuality::Fast | LowerBoundQuality::Bad => {
                                let mut p = true_child.lower.clone();
                                p.variables.insert(v.index());
                                p
                            }
                            LowerBoundQuality::Precise => self
                                .popularity(node, &true_child.witness)
                                .context("witness popularity")?,
                        };
                        let false_lower = match lower_bound_type {
                            LowerBoundQuality::Fast | LowerBoundQuality::Bad => {
                                let mut p = false_child.lower.clone();
                                p.variables.insert(v.index());
                                p
                            }
                            LowerBoundQuality::Precise => self
                                .popularity(node, &false_child.witness)
                                .context("witness popularity")?,
                        };
                        let (witness, lower) = if false_lower > true_lower {
                            (false_child.witness.clone(), false_lower)
                        } else {
                            (true_child.witness.clone(), true_lower)
                        };
                        let mut upper = &false_child.upper + &true_child.upper;
                        upper.variables.insert(v.index());
                        #[cfg(debug)]
                        {
                            // computed as before
                            let mut old_lower = true_child.lower.maximum(&false_child.lower);
                            old_lower.variables.insert(v.index());
                            assert!(old_lower <= lower);
                            assert!(lower <= upper);
                        }
                        QualifiedBounds {
                            witness,
                            lower,
                            upper,
                        }
                    }
                }
            };
            if lower_bound_type == LowerBoundQuality::Precise {
                debug_assert_eq!(&self.popularity(node, &new.witness)?, &new.lower);
            }
            infos[node.index()] = Some(new);
        }
        Ok(BoundInfos {
            infos,
            lower_bound_type,
        })
    }

    /// gets the true and false children of an ite, in this order
    fn ite_children(&self, ite: NodeIndex) -> anyhow::Result<(NodeIndex, NodeIndex)> {
        let mut true_node = None;
        let mut false_node = None;
        let mut n = 0;
        for edge in self.graph.edges_directed(ite, Outgoing) {
            match edge.weight() {
                Some(true) => true_node = Some(edge.target()),
                Some(false) => false_node = Some(edge.target()),
                None => anyhow::bail!("ite node has unlabeled outgoing edge"),
            };
            n += 1;
        }
        anyhow::ensure!(n == 2, "ite node has {} != 2 children", n);
        match (true_node, false_node) {
            (Some(t), Some(f)) => Ok((t, f)),
            _ => anyhow::bail!("ite node does not have one true child and one false child"),
        }
    }

    /// Computes the popularity of a controlled model on a subformula
    fn popularity(&self, node: NodeIndex, model: &Model) -> anyhow::Result<QualifiedModelCount> {
        let mut dfs = DfsPostOrder::new(&self.graph, node);
        let mut infos: Vec<Option<QualifiedModelCount>> = vec![None; self.graph.node_count()];
        while let Some(node) = dfs.next(&self.graph) {
            let new = match &self.graph[node] {
                CountingNode::Count(c) => c.clone(),
                CountingNode::And => {
                    let mut res = QualifiedModelCount {
                        count: BigUint::one(),
                        variables: FixedBitSet::with_capacity(self.nvars),
                    };
                    for child in self.graph.neighbors(node) {
                        res *= infos[child.index()]
                            .as_ref()
                            .ok_or_else(|| anyhow::anyhow!("wrong exploration order"))?;
                    }
                    res
                }
                CountingNode::Ite(v) => {
                    let (t, f) = self.ite_children(node).context("ite children")?;
                    if self.controlled.contains(v.index()) {
                        let child = if model[*v] { t } else { f };
                        infos[child.index()]
                            .as_ref()
                            .ok_or_else(|| anyhow::anyhow!("wrong exploration order"))?
                            .clone()
                    } else {
                        let mut res = infos[t.index()]
                            .as_ref()
                            .ok_or_else(|| anyhow::anyhow!("wrong exploration order"))?
                            + infos[f.index()]
                                .as_ref()
                                .ok_or_else(|| anyhow::anyhow!("wrong exploration order"))?;
                        res.variables.insert(v.index());
                        res
                    }
                }
            };
            infos[node.index()] = Some(new);
        }
        infos[node.index()]
            .take()
            .ok_or_else(|| anyhow::anyhow!("no info for node"))
    }
}

/// Used to fail on "ite(relaxed) has non count children"
#[test]
fn test_simplification_ite_relaxed_non_count_children() -> anyhow::Result<()> {
    let mut controlled = FixedBitSet::with_capacity(16);
    controlled.insert(2);
    let res = crate::d4::D4::run(
        Path::new("assets/simplify_regression.cnf"),
        Some(&controlled),
        None,
        10,
    )
    .context("running d4")?;
    most_popular_model_relaxed(
        res,
        controlled,
        false,
        SimplificationOptions::all(),
        LowerBoundQuality::Precise,
    )
    .context("most_popular_model_relaxed")?;
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Info to bound the popcon of a relaxed counting ddnnf
pub struct QualifiedBounds {
    /// Witness of the lower bound
    pub witness: Model,
    /// Lower bound of real popcon, included. variable set is over uncontrolled union relaxed variables.
    pub lower: QualifiedModelCount,
    /// Upper bound of real popcon, included. variable set is over uncontrolled union relaxed variables.
    pub upper: QualifiedModelCount,
}

impl QualifiedBounds {
    /// bounds for an uncontrolled subtree of exactly this model count
    /// witness must have this popularity.
    pub fn exactly(what: &QualifiedModelCount, witness: Model) -> QualifiedBounds {
        QualifiedBounds {
            witness,
            lower: what.clone(),
            upper: what.clone(),
        }
    }
}

impl TryFrom<&QualifiedBounds> for crate::cnf::PopConBounds {
    type Error = anyhow::Error;
    fn try_from(b: &QualifiedBounds) -> anyhow::Result<Self> {
        let mut low = b.lower.clone();
        let mut hi = b.upper.clone();
        low.upgrade_to(&hi.variables);
        hi.upgrade_to(&low.variables);
        crate::cnf::PopConBounds::from_range(low.into(), hi.into())
            .with_context(|| format!("converting {:?} to PopConBounds, range upside down?", b))
    }
}

impl Mul for &QualifiedBounds {
    type Output = QualifiedBounds;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut witness = self.witness.clone();
        witness.0.union_with(&rhs.witness.0);
        QualifiedBounds {
            witness,
            lower: &self.lower * &rhs.lower,
            upper: &self.upper * &rhs.upper,
        }
    }
}

impl MulAssign<&QualifiedBounds> for QualifiedBounds {
    fn mul_assign(&mut self, rhs: &QualifiedBounds) {
        self.witness.0.union_with(&rhs.witness.0);
        self.lower *= &rhs.lower;
        self.upper *= &rhs.upper;
    }
}

impl<D: std::borrow::Borrow<MemoryDDNNF>> Source for D {
    fn visit<T: Visitor>(self, mut visitor: T) -> anyhow::Result<()> {
        visitor.visit_header(&self.borrow().header)?;
        for i in self.borrow().nodes.iter() {
            visitor.visit_node(i)?;
        }
        Ok(())
    }
}

/// A newtype to print the c2d format of the formula to the corresponding stream
#[derive(Debug, Clone, Copy)]
struct PrintVisitor<T: Write>(T);
impl<T: Write> Visitor for PrintVisitor<T> {
    fn visit_header(&mut self, h: &Header) -> anyhow::Result<()> {
        write!(self.0, "nnf {} {} {}\n", h.nnodes, h.nedges, h.nvars)?;
        Ok(())
    }
    fn visit_node(&mut self, node: &Node) -> anyhow::Result<()> {
        match node {
            Node::L(l) => write!(self.0, "L {}\n", l.to_dimacs())?,
            Node::A(args) => {
                write!(self.0, "A {}", args.len())?;
                for i in args.iter() {
                    write!(self.0, " {}", i)?;
                }
                write!(self.0, "\n")?;
            }
            Node::O(opposing, arg1, arg2) => {
                let o = match opposing {
                    Some(x) => x.to_dimacs(),
                    None => 0,
                };
                write!(self.0, "O {} 2 {} {}\n", o, arg1, arg2)?;
            }
            Node::F => {
                write!(self.0, "O 0 0\n")?;
            }
        }
        Ok(())
    }
}

/// Write the c2d format of the formula `s` to `f`
pub fn write<S: Source, W: Write>(f: W, s: S) -> anyhow::Result<()> {
    s.visit(PrintVisitor(f))
}

/// A visitor to store the formula of a source to memory
#[derive(Debug)]
struct ToMemoryVisitor(MemoryDDNNF);
impl Visitor for ToMemoryVisitor {
    fn visit_header(&mut self, h: &Header) -> anyhow::Result<()> {
        self.0.header = h.clone();
        self.0.nodes.reserve(h.nnodes);
        Ok(())
    }
    fn visit_node(&mut self, node: &Node) -> anyhow::Result<()> {
        self.0.nodes.push(node.clone());
        Ok(())
    }
}

impl MemoryDDNNF {
    /// Store the formula expressed by a source to memory.
    pub fn from_source<T: Source>(source: T) -> anyhow::Result<Self> {
        let m = MemoryDDNNF {
            header: Header {
                nedges: 0,
                nvars: 0,
                nnodes: 0,
            },
            nodes: vec![],
        };
        let mut visitor = ToMemoryVisitor(m);
        source.visit(&mut visitor)?;
        Ok(visitor.0)
    }
}

pub(crate) mod parser {
    use super::*;

    use anyhow::Context;
    use nom::bytes::streaming::{tag, take_while1};
    use nom::character::{is_digit};
    use nom::character::streaming::multispace1;
    use nom::combinator::{map, map_res};
    use nom::IResult;
    use nom::Parser;

    pub(crate) fn unsigned<T: std::str::FromStr<Err = std::num::ParseIntError>>(
        s: &[u8],
    ) -> IResult<&[u8], T> {
        map_res(
            map_res(take_while1(is_digit), std::str::from_utf8),
            |x: &str| x.parse(),
        )(s)
    }

    pub(crate) fn bigunsigned<T: std::str::FromStr<Err = num_bigint::ParseBigIntError>>(
        s: &[u8],
    ) -> IResult<&[u8], T> {
        map_res(
            map_res(take_while1(is_digit), std::str::from_utf8),
            |x: &str| x.parse(),
        )(s)
    }

    pub(crate) fn signed<
        T: std::str::FromStr<Err = std::num::ParseIntError> + std::ops::Neg<Output = T>,
    >(
        s: &[u8],
    ) -> IResult<&[u8], T> {
        unsigned
            .or(tag(b"-").and(unsigned).map(|(_, n): (&[u8], T)| -n))
            .parse(s)
    }

    /// Parses a file in c2d format to d-DNNF. Streaming.
    pub struct FileSource<R: Read> {
        pub(crate) feeder: crate::input::ReadToParser<std::io::Chain<R, &'static [u8]>>,
    }

    impl<R: Read> FileSource<R> {
        /// Create a new parser from an open file.
        pub fn new(read: R) -> Self {
            // the file must end with some whitespace, otherwise ReadToParser will return an error "not enough bytes" when reading a number `\d*` at the end of file.
            let read2 = read.chain(b" " as &[u8]);
            FileSource {
                feeder: crate::input::ReadToParser::new(read2, 1024 * 1024),
            }
        }

        /// Eat at least one whitespace
        pub fn whitespace(&mut self) -> anyhow::Result<()> {
            self.feeder
                .parse_and_advance(|s| map(multispace1, |_| ())(s))
                .context("expecting whitespace")?;
            Ok(())
        }
    }

    impl<R: Read> Source for FileSource<R> {
        fn visit<T>(mut self, mut visitor: T) -> anyhow::Result<()>
        where
            T: Visitor,
        {
            self.feeder
                .parse_and_advance(|s| map(tag(b"nnf"), |_| ())(s))?;
            self.whitespace()?;
            let nnodes = self
                .feeder
                .parse_and_advance(unsigned)
                .context("number of nodes")?;
            self.whitespace()?;
            let mut nedges = self
                .feeder
                .parse_and_advance(unsigned)
                .context("number of edges")?;
            self.whitespace()?;
            let nvars = self
                .feeder
                .parse_and_advance(unsigned)
                .context("number of variables")?;
            let header = Header {
                nnodes,
                nvars,
                nedges,
            };
            visitor.visit_header(&header)?;
            for current_index in 0..nnodes {
                self.whitespace()?;
                let node_type = self
                    .feeder
                    .parse_and_advance(|s: &[u8]| nom::character::streaming::one_of("ALO")(s))
                    .context("node kind")?;
                self.whitespace()?;
                let node = match node_type {
                    'L' => {
                        let litn = self
                            .feeder
                            .parse_and_advance(signed)
                            .context("literal node")?;
                        let lit = Lit::from_dimacs(litn);
                        anyhow::ensure!(
                            lit.var().index() < nvars,
                            "Literal {} impossible with only {} variables",
                            litn,
                            nvars
                        );
                        Node::L(lit)
                    }
                    'A' => {
                        let nargs = self.feeder.parse_and_advance(unsigned)?;
                        let mut args = Vec::with_capacity(nargs);
                        for _ in 0..nargs {
                            self.whitespace()?;
                            let index = self.feeder.parse_and_advance(unsigned)?;
                            anyhow::ensure!(
                                index < current_index,
                                "File is not in topological order: {} >= {}",
                                index,
                                current_index
                            );
                            args.push(index);
                            anyhow::ensure!(nedges >= 1, "Too many edges in file");
                            nedges -= 1;
                        }
                        Node::A(args)
                    }
                    'O' => {
                        let vari = self
                            .feeder
                            .parse_and_advance(unsigned)
                            .context("opposing variable of OR node")?;
                        let var = if vari == 0 {
                            None
                        } else {
                            Some(Var::from_dimacs(vari))
                        };
                        self.whitespace()?;
                        let nargs: usize = self
                            .feeder
                            .parse_and_advance(unsigned)
                            .context("arity of OR node")?;
                        anyhow::ensure!(
                            nargs == 2usize || nargs == 0usize,
                            "Or node with {}!=2 or 0 arguments",
                            nargs
                        );
                        if nargs == 0usize {
                            Node::F
                        } else {
                            self.whitespace()?;
                            let index1 = self
                                .feeder
                                .parse_and_advance(unsigned)
                                .context("target index of edge in OR node")?;
                            anyhow::ensure!(
                                index1 < current_index,
                                "File is not in topological order: {} >= {}",
                                index1,
                                current_index
                            );
                            self.whitespace()?;
                            let index2 = self
                                .feeder
                                .parse_and_advance(unsigned)
                                .context("target index of edge in OR node")?;
                            anyhow::ensure!(
                                index2 < current_index,
                                "File is not in topological order: {} >= {}",
                                index2,
                                current_index
                            );
                            anyhow::ensure!(nedges >= 2, "Too many edges in file");
                            nedges -= 2;
                            Node::O(var, index1, index2)
                        }
                    }
                    _ => unreachable!(),
                };
                visitor.visit_node(&node)?;
            }
            self.feeder
                .parse_and_exhaust(|s: &[u8]| nom::character::streaming::one_of(" \r\n\t")(s))
                .context("checking for unused data")?;
            // FIXME: dsharp outputs wrong number of edges
            // anyhow::ensure!(nedges == 0, "{} missing edges from file", nedges);
            Ok(())
        }
    }
}

pub use parser::FileSource;

#[test]
fn parse_from_string() {
    let m = MemoryDDNNF::from_source(FileSource::new(b"nnf 1 0 1\nL 1" as &[u8])).unwrap();
    assert_eq!(
        m,
        MemoryDDNNF {
            header: Header {
                nnodes: 1,
                nedges: 0,
                nvars: 1,
            },
            nodes: vec![Node::L(Lit::from_dimacs(1))],
        }
    );
}
#[test]
fn parse_from_string_2() {
    let m = MemoryDDNNF::from_source(FileSource::new(
        b"nnf 5 4 3\nL -1\nL 2\nL 3\nA 2 0 1\nO 3 2 2 3" as &[u8],
    ))
    .unwrap();
    assert_eq!(
        m,
        MemoryDDNNF {
            header: Header {
                nnodes: 5,
                nedges: 4,
                nvars: 3,
            },
            nodes: vec![
                Node::L(Lit::from_index(0, false)),
                Node::L(Lit::from_index(1, true)),
                Node::L(Lit::from_index(2, true)),
                Node::A(vec![0, 1]),
                Node::O(Some(Var::from_index(2)), 2, 3)
            ],
        }
    );
}

struct CountingVisitor<'a> {
    /// number of variables of the original formula
    nvars: usize,
    /// projection variables
    projection_variables: Option<&'a FixedBitSet>,
    /// for each node, its model count and the set of the projected variables it contains. Empty
    /// set means false node if 0 models or only unprojected variables if 1 model.
    node_counts: Vec<(BigUint, FixedBitSet)>,
}

impl Visitor for CountingVisitor<'_> {
    fn visit_header(&mut self, h: &Header) -> anyhow::Result<()> {
        self.nvars = h.nvars;
        if let Some(proj) = self.projection_variables {
            anyhow::ensure!(proj.len() == self.nvars);
        }
        self.node_counts.reserve(h.nnodes);
        trace!(
            stats = true,
            mc_ddnnf_node_count = h.nnodes,
            "counting models on d-DNNF"
        );
        Ok(())
    }
    fn visit_node(&mut self, node: &Node) -> anyhow::Result<()> {
        let res = match node {
            Node::L(l) => {
                let mut just_me = FixedBitSet::with_capacity(self.nvars);
                match self.projection_variables {
                    // a non projection variable
                    Some(proj) if !proj.contains(l.var().index()) => (),
                    // a projection variable (or it's not projection)
                    _ => {
                        just_me.insert(l.var().index());
                    }
                }
                (One::one(), just_me)
            }
            Node::F => (Zero::zero(), FixedBitSet::with_capacity(self.nvars)),
            Node::A(args) => {
                let mut res = One::one();
                let mut vars = FixedBitSet::with_capacity(self.nvars);
                for &i in args {
                    let (ref child_count, ref child_vars) = &self.node_counts[i];
                    res *= child_count;
                    vars.union_with(child_vars);
                }
                (res, vars)
            }
            Node::O(_, arg1, arg2) => {
                let vars = &self.node_counts[*arg1].1 | &self.node_counts[*arg2].1;
                let mut res = Zero::zero();
                for &&i in [arg1, arg2].iter() {
                    let (ref child_count, ref child_vars) = &self.node_counts[i];
                    anyhow::ensure!(
                        child_vars.count_ones(..) as u64 + 1 >= child_count.bits(),
                        "model count has more bits {} than number of variables {}",
                        child_count.bits(),
                        child_vars.count_ones(..)
                    );
                    // this is to offset that the d-DNNF is not smooth
                    let missing_vars = vars.difference(child_vars).count();
                    if child_vars.count_ones(..) == 0 {
                        anyhow::ensure!(
                            child_count.is_zero() || child_count.is_one(),
                            "node on no projected variables with non 0 or 1 model count"
                        )
                    } else {
                        res += child_count * (BigUint::one() << missing_vars);
                    }
                }
                (res, vars)
            }
        };
        self.node_counts.push(res);
        Ok(())
    }
}

/// Returns the number of models of the formula.
/// If `projection_variables` is not `None` then it must be a set of variables on which the formula
/// is projected.
pub fn count_models<S: Source>(
    formula: S,
    projection_variables: Option<&FixedBitSet>,
) -> anyhow::Result<ModelCount> {
    let mut visitor = CountingVisitor {
        nvars: 0,
        projection_variables,
        node_counts: Vec::new(),
    };
    formula.visit(&mut visitor)?;
    let (raw_count, vars) = visitor.node_counts.pop().context("formula is empty")?;
    anyhow::ensure!(
        vars.count_ones(..) as u64 + 1 >= raw_count.bits(),
        "model count has more bits {} than number of variables {}",
        raw_count.bits(),
        vars.count_ones(..)
    );
    let target_nvars = match projection_variables {
        None => visitor.nvars,
        Some(p) => p.count_ones(..),
    };
    let model_count = raw_count * (BigUint::one() << (target_nvars - vars.count_ones(..)));
    Ok(ModelCount {
        model_count,
        bits: visitor.nvars,
    })
}

#[test]
fn count() {
    assert_eq!(
        ModelCount {
            model_count: BigUint::one(),
            bits: 1
        },
        count_models(FileSource::new(b"nnf 1 0 1 L 1" as &[u8]), None).unwrap()
    );
    assert_eq!(
        count_models(FileSource::new(b"nnf 1 0 3 L 1" as &[u8]), None).unwrap(),
        ModelCount {
            model_count: 4u32.into(),
            bits: 3
        },
    );
    assert_eq!(
        count_models(
            FileSource::new(
                b"nnf 7 6 3
L 3
L 1
L -1
L 2
A 2 2 3
O 1 2 1 4
A 2 0 5
                " as &[u8]
            ),
            None
        )
        .unwrap(),
        // obtained from c2d for 3 & ( 1|2)
        ModelCount {
            model_count: 3u32.into(),
            bits: 3
        },
    );
}

#[derive(Debug)]
struct PopularityInfo {
    /// uncontrolled variables appearing in this subformula
    variables: Model,
    /// The model for controlled variables that hash the most subordinate models
    most_popular: Model,
    /// corresponding number of subordinate models
    count: BigUint,
    /// whether this node is allowed to appear below an uncontrolled decision. Only for consistency checks
    uncontrolled: bool,
}

struct PopularityVisitor {
    controlled_vars: FixedBitSet,
    by_node: Vec<PopularityInfo>,
}

impl Visitor for PopularityVisitor {
    fn visit_header(&mut self, h: &Header) -> anyhow::Result<()> {
        anyhow::ensure!(
            h.nvars == self.controlled_vars.len(),
            "Visited dDNNF has {} variables but there are {} controlled variables",
            h.nvars,
            self.controlled_vars.len()
        );
        self.by_node.reserve(h.nnodes);
        trace!(
            stats = true,
            popcon_ddnnf_node_count = h.nnodes,
            "popcon on d-DNNF"
        );
        Ok(())
    }
    fn visit_node(&mut self, node: &Node) -> anyhow::Result<()> {
        let res = match node {
            Node::L(l) => {
                let just_me = Model::just(l.var(), self.controlled_vars.len());
                let empty = Model::empty(self.controlled_vars.len());
                let var_index = l.var().index();
                if self.controlled_vars.contains(var_index) {
                    let most_popular = if l.is_positive() {
                        just_me
                    } else {
                        empty.clone()
                    };
                    PopularityInfo {
                        variables: empty,
                        most_popular,
                        count: One::one(),
                        uncontrolled: false,
                    }
                } else {
                    PopularityInfo {
                        variables: just_me,
                        most_popular: empty,
                        count: One::one(),
                        uncontrolled: true,
                    }
                }
            }
            Node::F => {
                let empty = Model::empty(self.controlled_vars.len());
                PopularityInfo {
                    variables: empty.clone(),
                    most_popular: empty,
                    count: Zero::zero(),
                    uncontrolled: true,
                }
            }
            Node::A(args) => {
                let mut res = PopularityInfo {
                    variables: Model::empty(self.controlled_vars.len()),
                    most_popular: Model::empty(self.controlled_vars.len()),
                    count: One::one(),
                    uncontrolled: true,
                };
                for &i in args {
                    let info = &self.by_node[i];
                    res.count *= &info.count;
                    res.variables.0.union_with(&info.variables.0);
                    res.most_popular.0.union_with(&info.most_popular.0);
                    res.uncontrolled &= info.uncontrolled;
                }
                res
            }
            Node::O(opposing, arg1, arg2) => {
                let (first_info, second_info) = (&self.by_node[*arg1], &self.by_node[*arg2]);
                let mut variables = first_info.variables.clone();
                variables.0.union_with(&second_info.variables.0);
                let offset = |child_count, child_vars: &Model| {
                    let missing_vars = variables.0.difference(&child_vars.0).count();
                    child_count * (BigUint::one() << missing_vars)
                };
                let first_count = offset(first_info.count.clone(), &first_info.variables);
                let second_count = offset(second_info.count.clone(), &second_info.variables);
                // wether we do a sum or a max
                let uncontrolled = match opposing {
                    // decision on uncontrolled var
                    Some(v) => !self.controlled_vars.contains(v.index()),
                    None => anyhow::bail!("cannot process {:?} because it does not indicate on which variable a decision was taken", node),
                };
                let (count, most_popular) = if !uncontrolled {
                    // controlled decision
                    if first_count < second_count {
                        (second_count, second_info.most_popular.clone())
                    } else {
                        (first_count, first_info.most_popular.clone())
                    }
                } else {
                    // uncontrolled decision
                    anyhow::ensure!(
                        first_info.uncontrolled,
                        "uncontrolled decision {:?} has controlled node below",
                        node
                    );
                    anyhow::ensure!(
                        second_info.uncontrolled,
                        "uncontrolled decision {:?} has controlled node below",
                        node
                    );
                    anyhow::ensure!(
                        first_info.most_popular.is_empty(),
                        "uncontrolled decision {:?} has controlled literal below",
                        node
                    );
                    anyhow::ensure!(
                        second_info.most_popular.is_empty(),
                        "uncontrolled decision {:?} has controlled literals below",
                        node
                    );
                    (first_count + second_count, first_info.most_popular.clone())
                };
                PopularityInfo {
                    variables,
                    most_popular,
                    count,
                    uncontrolled,
                }
            }
        };
        self.by_node.push(res);
        Ok(())
    }
}

/// Computes the most popular model for givent controlled variables.
/// This assumes that all OR nodes are decision nodes, and that decision
/// nodes for controlled variables are above those for uncontrolled variables.
/// If this is not the case, result will be wrong or an error.
pub fn most_popular_model<S: Source>(
    formula: S,
    controlled_vars: FixedBitSet,
) -> anyhow::Result<(Model, ModelCount)> {
    let mut visitor = PopularityVisitor {
        controlled_vars,
        by_node: Vec::new(),
    };
    formula.visit(&mut visitor)?;
    let info = visitor.by_node.pop().context("formula is empty")?;
    let bits = visitor.controlled_vars.len() - visitor.controlled_vars.count_ones(..);
    let model_count = info.count * (BigUint::one() << (bits - info.variables.nvars_set()));
    trace!(%model_count, bits, "max popularity");
    Ok((info.most_popular, ModelCount { model_count, bits }))
}

struct PartialPopularityVisitor {
    controlled_vars: FixedBitSet,
    relaxed_vars: FixedBitSet,
    by_node: Vec<PopularityInfo>,
}

impl Visitor for PartialPopularityVisitor {
    fn visit_header(&mut self, h: &Header) -> anyhow::Result<()> {
        anyhow::ensure!(
            h.nvars == self.controlled_vars.len(),
            "Visited dDNNF has {} variables but there are {} controlled variables",
            h.nvars,
            self.controlled_vars.len()
        );
        anyhow::ensure!(
            h.nvars == self.relaxed_vars.len(),
            "Visited dDNNF has {} variables but there are {} relaxed variables",
            h.nvars,
            self.relaxed_vars.len()
        );
        self.by_node.reserve(h.nnodes);
        Ok(())
    }
    fn visit_node(&mut self, node: &Node) -> anyhow::Result<()> {
        let res = match node {
            Node::L(l) => {
                let just_me = Model::just(l.var(), self.controlled_vars.len());
                let empty = Model::empty(self.controlled_vars.len());
                let var_index = l.var().index();
                if self.controlled_vars.contains(var_index) || self.relaxed_vars.contains(var_index)
                {
                    let most_popular = if l.is_positive() {
                        just_me
                    } else {
                        empty.clone()
                    };
                    PopularityInfo {
                        variables: empty,
                        most_popular,
                        count: One::one(),
                        uncontrolled: false,
                    }
                } else {
                    PopularityInfo {
                        variables: just_me,
                        most_popular: empty,
                        count: One::one(),
                        uncontrolled: true,
                    }
                }
            }
            Node::F => {
                let empty = Model::empty(self.controlled_vars.len());
                PopularityInfo {
                    variables: empty.clone(),
                    most_popular: empty,
                    count: Zero::zero(),
                    uncontrolled: true,
                }
            }
            Node::A(args) => {
                let mut res = PopularityInfo {
                    variables: Model::empty(self.controlled_vars.len()),
                    most_popular: Model::empty(self.controlled_vars.len()),
                    count: One::one(),
                    uncontrolled: true,
                };
                for &i in args {
                    let info = &self.by_node[i];
                    res.count *= &info.count;
                    res.variables.0.union_with(&info.variables.0);
                    res.most_popular.0.union_with(&info.most_popular.0);
                    res.uncontrolled &= info.uncontrolled;
                }
                res
            }
            Node::O(opposing, arg1, arg2) => {
                let (first_info, second_info) = (&self.by_node[*arg1], &self.by_node[*arg2]);
                let mut variables = first_info.variables.clone();
                variables.0.union_with(&second_info.variables.0);
                let offset = |child_count, child_vars: &Model| {
                    let missing_vars = variables.0.difference(&child_vars.0).count();
                    child_count * (BigUint::one() << missing_vars)
                };
                let first_count = offset(first_info.count.clone(), &first_info.variables);
                let second_count = offset(second_info.count.clone(), &second_info.variables);
                // wether we do a sum or a max
                let (controlled, relaxed) = match opposing {
                    // decision on uncontrolled var
                    Some(v) => (
                        self.controlled_vars.contains(v.index()),
                        self.relaxed_vars.contains(v.index()),
                        ),
                    None => anyhow::bail!("cannot process {:?} because it does not indicate on which variable a decision was taken", node),
                };
                let uncontrolled = !(controlled || relaxed);
                let (count, most_popular) = if !uncontrolled {
                    // controlled decision
                    if first_count < second_count {
                        (second_count, second_info.most_popular.clone())
                    } else {
                        (first_count, first_info.most_popular.clone())
                    }
                } else {
                    // uncontrolled decision
                    anyhow::ensure!(
                        first_info.most_popular.is_empty(),
                        "uncontrolled decision {:?} has controlled literals below: {:?}",
                        node,
                        first_info.most_popular,
                    );
                    anyhow::ensure!(
                        second_info.most_popular.is_empty(),
                        "uncontrolled decision {:?} has controlled literal below: {:?}",
                        node,
                        second_info.most_popular,
                    );
                    anyhow::ensure!(
                        first_info.uncontrolled,
                        "uncontrolled decision {:?} has controlled nodes below",
                        node
                    );
                    anyhow::ensure!(
                        second_info.uncontrolled,
                        "uncontrolled decision {:?} has controlled nodes below",
                        node
                    );
                    (first_count + second_count, first_info.most_popular.clone())
                };
                PopularityInfo {
                    variables,
                    most_popular,
                    count,
                    uncontrolled,
                }
            }
        };
        self.by_node.push(res);
        Ok(())
    }
}

/// Options for simplification
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct SimplificationOptions {
    /// Simple rewriting rules
    pub rewriting: bool,
    /// Remove relaxed variables that are below all others
    pub useless_relaxed_vars: bool,
    /// Detect controlled variables that are always better with one polarity
    pub oriented_vars: bool,
    /// Does not work
    pub simplify_by_bounds: bool,
}

impl SimplificationOptions {
    /// Use all simplifications
    pub const fn all() -> Self {
        Self {
            rewriting: true,
            useless_relaxed_vars: true,
            oriented_vars: true,
            simplify_by_bounds: true,
        }
    }
}

/// Computes the most popular model for givent controlled variables.
/// This assumes that all OR nodes are decision nodes, and that decision
/// nodes for controlled variables are above those for uncontrolled variables.
/// If this is not the case, result will be wrong or an error.
/// If fast_mode is true, then after simplification a linear time algo is used instead of
/// quadratic. The lower bound is bad.
pub fn most_popular_model_relaxed(
    formula: D4dDNNFOutput,
    controlled_vars: FixedBitSet,
    relax_exact: bool,
    opts: SimplificationOptions,
    lower_bound_type: LowerBoundQuality,
) -> anyhow::Result<QualifiedBounds> {
    let conversion_span = tracing::trace_span!("converting output to counting graph").entered();
    let nvars = controlled_vars.len();
    let mut relaxed_vars = formula
        .controlled_vars
        .clone()
        .ok_or(anyhow::anyhow!("no controlled variables reported by d4"))?;
    let nuncontrolled = relaxed_vars.len() - relaxed_vars.count_ones(..);
    relaxed_vars.difference_with(&controlled_vars);
    let mut uncontrolled_or_relaxed_vars = controlled_vars.clone();
    uncontrolled_or_relaxed_vars.toggle_range(..);
    let mut visitor = PartialPopularityVisitor {
        controlled_vars,
        relaxed_vars,
        by_node: Vec::new(),
    };
    let ddnnf = MemoryDDNNF::from_source(&formula)?;
    (&ddnnf).visit(&mut visitor)?;
    let mut graph = Graph::new();
    let mut old2new = vec![NodeIndex::end(); visitor.by_node.len()];
    let get_or_insert = |graph: &mut Graph<_, _>, old2new: &mut Vec<NodeIndex>, oldindex| {
        let idx = old2new[oldindex];
        if idx == NodeIndex::end() {
            let info: &PopularityInfo = &visitor.by_node[oldindex];
            assert!(info.uncontrolled);
            let count = info.count.clone();
            let variables = info.variables.0.clone();
            let idx = graph.add_node(CountingNode::Count(QualifiedModelCount {
                count,
                variables,
            }));
            old2new[oldindex] = idx;
            idx
        } else {
            idx
        }
    };
    let mut edges_to_remove = FixedBitSet::with_capacity(ddnnf.header.nedges);
    let mut remove_opposing_variable = |g: &Graph<CountingNode, Option<bool>>, var, node| match &g
        [node]
    {
        CountingNode::Ite(v) if *v == var => {
            anyhow::bail!("unimplemented in conversion to counting ddnnf: Or(v, lit(v), ...)")
        }
        CountingNode::And => {
            let mut to_remove = None;
            for child in g.neighbors_directed(node, Outgoing) {
                match g[child] {
                    CountingNode::Ite(v) if v == var => {
                        let mut zero = false;
                        let mut one = false;
                        let mut polarity = false;
                        for edge in g.edges_directed(child, Outgoing) {
                            match &g[edge.target()] {
                                CountingNode::Count(c) => {
                                    anyhow::ensure!(c.variables.count_ones(..) == 0, "opposing variable ite under Or() has non true/false count {:?} as child", &c);
                                    if c.count.is_zero() {
                                        zero = true;
                                    } else {
                                        anyhow::ensure!(
                                            c.count.is_one(),
                                            "count with 0 variables but > 1 count {:?}",
                                            &c
                                        );
                                        one = true;
                                        polarity = edge.weight().clone().ok_or_else(|| {
                                            anyhow::anyhow!("ite with None edge label")
                                        })?;
                                    }
                                }
                                x => {
                                    anyhow::bail!("opposing variable ite under Or() has non count {:?} as child", x);
                                }
                            }
                        }
                        anyhow::ensure!(
                            zero && one,
                            "failed to find ite({}, true, false) under or({})",
                            var,
                            var
                        );
                        to_remove = Some((child, polarity));
                        break;
                    }
                    _ => (),
                }
            }
            if let Some((child, polarity)) = to_remove {
                let idx = g
                    .find_edge(node, child)
                    .ok_or(anyhow::anyhow!("unexpected: child has no edge from parent"))?;
                edges_to_remove.grow(idx.index() + 1);
                edges_to_remove.insert(idx.index());
                Ok(polarity)
            } else {
                anyhow::bail!(
                    "opposing or({}, and(...), ...) has no opposing lit under the and",
                    var
                )
            }
        }
        x => anyhow::bail!("opposing or({}) has child {:?}", var, x),
    };
    let mut true_node = None;
    let mut get_true = |g: &mut Graph<CountingNode, _>| match true_node {
        Some(n) => n,
        None => {
            let c = QualifiedModelCount {
                count: BigUint::one(),
                variables: FixedBitSet::with_capacity(nvars),
            };
            let n = g.add_node(CountingNode::Count(c));
            true_node = Some(n);
            n
        }
    };
    let mut false_node = None;
    let mut get_false = |g: &mut Graph<CountingNode, _>| match false_node {
        Some(n) => n,
        None => {
            let c = QualifiedModelCount::zero(nvars);
            let n = g.add_node(CountingNode::Count(c));
            false_node = Some(n);
            n
        }
    };
    for (i, info) in visitor.by_node.iter().enumerate() {
        if !info.uncontrolled {
            let node = &ddnnf.nodes[i];
            let (new, children) = match node {
                Node::L(l) => {
                    let t = get_true(&mut graph);
                    let f = get_false(&mut graph);
                    (
                        CountingNode::Ite(l.var()),
                        vec![(t, Some(l.is_positive())), (f, Some(l.is_negative()))],
                    )
                }
                Node::A(args) => (
                    CountingNode::And,
                    args.iter()
                        .map(|&idx| (get_or_insert(&mut graph, &mut old2new, idx), None))
                        .collect(),
                ),
                Node::O(opposing, left, right) => {
                    let var =
                        opposing.ok_or(anyhow::anyhow!("no opposing variable on node {}", i))?;
                    let new = CountingNode::Ite(var);
                    let left = get_or_insert(&mut graph, &mut old2new, *left);
                    let right = get_or_insert(&mut graph, &mut old2new, *right);
                    let left_label = remove_opposing_variable(&graph, var, left)?;
                    let right_label = remove_opposing_variable(&graph, var, right)?;
                    anyhow::ensure!(left_label == !right_label);
                    (
                        new,
                        vec![(left, Some(left_label)), (right, Some(right_label))],
                    )
                }
                Node::F => (
                    CountingNode::Count(QualifiedModelCount::zero(nvars)),
                    vec![],
                ),
            };
            let idx = graph.add_node(new);
            old2new[i] = idx;
            for (child, label) in children {
                graph.add_edge(idx, child, label);
            }
        }
    }
    let root = old2new[visitor.by_node.len() - 1];
    let mut simplified = CountingDDNNF {
        nvars,
        controlled: visitor.controlled_vars.clone(),
        relaxed: visitor.relaxed_vars,
        oriented: FixedBitSet::with_capacity(nvars),
        nuncontrolled,
        graph,
        root,
    };
    edges_to_remove.grow(simplified.graph.edge_count());
    simplified.remove_edges(&edges_to_remove);
    simplified.gc().context("gc")?;
    drop(conversion_span);
    if opts.rewriting || opts.oriented_vars || opts.useless_relaxed_vars {
        simplified.debug_stats("before simplification");
        simplified.simplify(opts).context("simplify")?;
        simplified.debug_stats("after first simplification");
        loop {
            let mut changed = false;
            changed |= simplified
                .remove_useless_relaxed_vars(opts)
                .context("remove_useless_relaxed_vars")?;
            changed |= simplified.oriented_vars(opts).context("oriented_vars")?;
            if !changed {
                break;
            }
        }
        // simplified
        //     .dump(Path::new("/tmp/foo.dot"))
        //     .context("dumping")?;
        simplified.debug_stats("after simplification");
    }
    let bound_info = simplified
        .bound_info(lower_bound_type)
        .context("bounding counting graph")?;
    let mut bounds = simplified
        .bounds(&bound_info)
        .context("bounding counting graph")?;
    simplified.simplify_by_bounds(&bound_info, opts)?;
    bounds.upper.upgrade_to(&uncontrolled_or_relaxed_vars);
    bounds.lower.upgrade_to(&uncontrolled_or_relaxed_vars);
    if relax_exact {
        let mut smt2 = Vec::new();
        simplified
            .to_smt2(&mut smt2)
            .context("writing smt2 output")?;
        let (best, count) =
            crate::smt::popularity_contest(&crate::d4::D4::direct(), smt2.as_slice())
                .context("second call to d4")?;
        let mut witness = Model::empty(nvars);
        for (k, v) in best.0.iter() {
            assert!(k.starts_with("var"));
            let varindex: isize = k[3..].parse().context("parsing smt model")?;
            let value = match v {
                VarValue::Bool(Some(true)) => true,
                VarValue::Bool(Some(false)) => false,
                VarValue::Bool(None) => false,
                VarValue::BitVector(v) => {
                    if v.len() != 1 {
                        anyhow::bail!("in smt model, {} is a bitvector value: {:?}", k, v);
                    } else {
                        v[0] == Some(true)
                    }
                }
            };
            witness.set(Var::from_dimacs(varindex), value);
        }
        let exact_count = count
            .to_exact()
            .map_err(|c| anyhow::anyhow!("exact count is not exact: {}", c))?
            .upgrade_to(uncontrolled_or_relaxed_vars.count_ones(..));
        Ok(QualifiedBounds::exactly(
            &QualifiedModelCount {
                count: exact_count.model_count,
                variables: uncontrolled_or_relaxed_vars,
            },
            witness,
        ))
    } else {
        tracing::trace!(
            "cnf bounds: {}",
            PopConBounds::up_to_low_ratio(&((&bounds).try_into().unwrap()))
        );
        Ok(bounds)
    }
}

#[test]
fn most_popular() {
    // A fixed bit set of this size with only the first bit set
    let just_first = |size| {
        let mut res = FixedBitSet::with_capacity(size);
        res.insert(0);
        res
    };

    let (model, count) =
        most_popular_model(FileSource::new(b"nnf 1 0 1 L 1" as &[u8]), just_first(1)).unwrap();
    assert_eq!(
        count,
        ModelCount {
            model_count: 1u8.into(),
            bits: 0
        }
    );
    assert_eq!(model[0], true);

    let (model, count) =
        most_popular_model(FileSource::new(b"nnf 1 0 3 L 1" as &[u8]), just_first(3)).unwrap();
    assert_eq!(
        count,
        ModelCount {
            model_count: 4u8.into(),
            bits: 2
        }
    );
    assert_eq!(model[0], true);

    let (model, count) = most_popular_model(
        FileSource::new(b"nnf 3 2 3 L 1 L 2 A 2 0 1" as &[u8]),
        just_first(3),
    )
    .unwrap();
    assert_eq!(
        count,
        ModelCount {
            model_count: 2u8.into(),
            bits: 2
        }
    );
    assert_eq!(model[0], true);

    let (model, count) = most_popular_model(
        FileSource::new(
            b"nnf 10 10 3
L -1
L 2
L -2
L 3
A 2 2 3
O 2 2 1 4
A 2 0 5
L 1
A 2 1 7
O 1 2 6 8

                " as &[u8],
        ),
        just_first(3),
    )
    .unwrap();
    assert_eq!(
        count,
        ModelCount {
            model_count: 3u8.into(),
            bits: 2
        }
    );
    assert_eq!(model[0], false);
}

#[test]
fn random_mc() -> anyhow::Result<()> {
    use crate::cnf;
    use crate::Tool;
    let size = 19;
    for _ in 0..30 {
        // about 20 models
        let f = cnf::generate_random_3sat(size, size * 4);
        println!("{:?}", f);
        // varisat_dimacs::write_dimacs(&mut std::io::stderr(), &f)?;
        println!();
        let naive_count = cnf::brute_force_model_count(&f);
        let mut cnf_string = Vec::new();
        varisat_dimacs::write_dimacs(&mut cnf_string, &f)?;
        println!("modelcount={}", naive_count);
        for tool in [
            Tool::Dsharp,
            Tool::C2D,
            Tool::D4 {
                proj: false,
                relax: 0,
                relax_exact: false,
                bfs: false,
                simplification_options: SimplificationOptions::all(),
                lower_bound_type: LowerBoundQuality::Precise,
            },
        ]
        .iter()
        {
            let smart_count = tool.count_models(&cnf_string as &[u8])?;
            assert_eq!(naive_count, smart_count, "wrong model count");
        }
        let ddnnf = crate::d4::D4::direct()
            .compile(&cnf_string as &[u8])
            .context("d4 ddnnf compilation")?;
        let mem = MemoryDDNNF::from_source(ddnnf).context("parsing ddnnf d4 output")?;
        let no_controlled_variable = FixedBitSet::with_capacity(size);
        let model = Model::empty(size);
        let mut mc = mem
            .popularity(&model, &no_controlled_variable)
            .context("popularity of empty model with no controlled variable")?;
        let mut uncontrolled_variables = no_controlled_variable.clone();
        uncontrolled_variables.toggle_range(..);
        mc.upgrade_to(&uncontrolled_variables);
        assert_eq!(&naive_count, &ModelCount::from(mc));
    }
    Ok(())
}

#[test]
fn random_popcon() -> anyhow::Result<()> {
    use crate::cnf;
    use crate::Tool;
    use std::iter::FromIterator;
    use varisat_formula::ExtendFormula;
    let size = 19;
    for _ in 0..30 {
        // about 20 models
        let formula = cnf::generate_random_3sat(size, size * 4);
        let mut controlled = FixedBitSet::from_iter(
            rand::seq::index::sample(&mut rand::thread_rng(), size, size / 2).iter(),
        );
        controlled.grow(size);
        println!(
            "#######\n######\npopcon on :\n {:?}\n{}\n controlled variables {:?}",
            &formula,
            cnf::to_dimacs_string(&formula)?,
            Model(controlled.clone())
        );
        let naive_count = cnf::brute_force_popcon(&formula, &controlled);
        let best = naive_count.max();
        let mut cnf_string = Vec::new();
        varisat_dimacs::write_dimacs(&mut cnf_string, &formula)?;
        for tool in [
            Tool::Dsharp,
            Tool::D4 {
                proj: false,
                relax: 0,
                relax_exact: false,
                bfs: false,
                simplification_options: SimplificationOptions::all(),
                lower_bound_type: LowerBoundQuality::Precise,
            },
            Tool::D4 {
                proj: false,
                relax: 10,
                relax_exact: true,
                bfs: false,
                simplification_options: SimplificationOptions::all(),
                lower_bound_type: LowerBoundQuality::Precise,
            },
            Tool::D4 {
                proj: false,
                relax: 10,
                relax_exact: true,
                bfs: true,
                simplification_options: SimplificationOptions::all(),
                lower_bound_type: LowerBoundQuality::Precise,
            },
            Tool::Unconstrained {
                tool: Box::new(Tool::D4 {
                    proj: false,
                    relax: 0,
                    relax_exact: false,
                    bfs: false,
                    simplification_options: SimplificationOptions::all(),
                    lower_bound_type: LowerBoundQuality::Precise,
                }),
                lower_bound_type: LowerBoundQuality::Precise,
                exact: true,
            },
        ]
        .iter()
        {
            let (best_model, itv) = tool.popcon(&cnf_string as &[u8], &controlled, &FixedBitSet::with_capacity(FixedBitSet::len(&controlled)))?;
            let best_count = itv
                .to_exact()
                .map_err(|c| anyhow::anyhow!("exact method yields approx result: {}", c))?;
            println!(
                "with {:?}: Best model {:?}, best count {}",
                tool, &best_model, &best_count
            );
            assert_eq!(best, best_count.model_count, "wrong best model count");
            // check that this best_model really reaches the best count
            let mut f = cnf::CnfFormula::from(formula.iter());
            for index in controlled.ones() {
                let l = Lit::from_index(index, best_model[index]);
                f.add_clause(&[l]);
            }
            let proj_mc = cnf::brute_force_model_count(&f);
            assert_eq!(
                proj_mc.model_count, best_count.model_count,
                "this model does not have the expected popularity"
            );
        }
        let d4output = crate::d4::D4::run(cnf_string.as_slice(), Some(&controlled), None, 10)
            .context("d4 relaxed 10")?;
        for &lower_bound_type in &[
            LowerBoundQuality::Precise,
            LowerBoundQuality::Fast,
            LowerBoundQuality::Bad,
        ] {
            let bounds = most_popular_model_relaxed(
                d4output.clone(),
                controlled.clone(),
                false,
                SimplificationOptions::all(),
                lower_bound_type,
            )
            .context("bounds")?;
            assert!(
                bounds.lower.count <= best,
                "d4 relaxed 10 fast_mode={:?} has bad lower bound",
                lower_bound_type
            );
            assert!(bounds.upper.count >= best);
            if lower_bound_type != LowerBoundQuality::Bad {
                assert_eq!(&naive_count[&bounds.witness], &bounds.lower.count);
            }

            let (model, bounds) = cnf::CNFPopularityContest::popularity_contest(
                &(crate::bb::UnconstrainedDdnnfBounds {
                    compiler: crate::d4::D4::direct(),
                    lower_bound_type,
                }),
                cnf_string.as_slice(),
                &controlled,
                &FixedBitSet::with_capacity(FixedBitSet::len(&controlled))
            )
            .context("UnconstrainedDdnnfBounds")?;
            assert!(
                &bounds.lower().model_count <= &best,
                "unconstrained d4 fast_mode={:?} has bad lower bound {:?} > {}",
                lower_bound_type,
                bounds.lower(),
                &best
            );
            assert!(
                &bounds.upper().model_count >= &best,
                "unconstrained d4 fast_mode={:?} has bad upper bound",
                lower_bound_type
            );
            if lower_bound_type != LowerBoundQuality::Bad {
                assert_eq!(&naive_count[&model], &bounds.lower().model_count, "unconstrained d4 fast_mode={:?} returns a witness with popularity distinct from lower bound", lower_bound_type);
            }
        }
    }
    Ok(())
}

/// A compiler from CNF to d-DNNF
pub trait Cnf2DdnnfCompiler {
    /// The converted formula. May only be read once.
    type O: Source;
    /// Do the conversion.
    fn compile<I: Input>(&self, input: I) -> anyhow::Result<Self::O>;
}

struct CnfConversionVisitor {
    formula: CnfFormula,
    false_var: Var,
    /// correspondance from nnf node to dimacs variable index
    node2var: Vec<Lit>,
}

impl Visitor for CnfConversionVisitor {
    fn visit_header(&mut self, header: &Header) -> anyhow::Result<()> {
        self.formula.set_var_count(header.nvars + 1);
        self.false_var = Var::from_dimacs(header.nvars as isize + 1);
        self.formula
            .add_clause(&[Lit::from_var(self.false_var, false)]);
        Ok(())
    }

    fn visit_node(&mut self, node: &Node) -> anyhow::Result<()> {
        let corresponding = match node {
            Node::L(lit) => *lit,
            Node::F => Lit::from_var(self.false_var, true),
            Node::A(nodes) => {
                anyhow::ensure!(nodes.len() > 0, "empty and");
                if nodes.len() == 1 {
                    self.node2var[nodes[0]]
                } else {
                    nodes[1..]
                        .iter()
                        .fold(self.node2var[nodes[0]], |acc, conjunct| {
                            let a = self.node2var[*conjunct];
                            let b = acc;
                            let c = Lit::from_var(self.formula.new_var(), true);
                            let tseytin: &[&[Lit]] = &[&[!a, !b, c] as &[_], &[a, !c], &[b, !c]];
                            for clause in tseytin {
                                self.formula.add_clause(clause);
                            }
                            c
                        })
                }
            }
            Node::O(_, left, right) => {
                let a = self.node2var[*left];
                let b = self.node2var[*right];
                let c = Lit::from_var(self.formula.new_var(), true);
                let tseytin: &[&[Lit]] = &[&[a, b, !c] as &[_], &[!a, c], &[!b, c]];
                for clause in tseytin {
                    self.formula.add_clause(clause);
                }
                c
            }
        };
        self.node2var.push(corresponding);
        Ok(())
    }
}

/// Converts a nnf formula to cnf. adds spurious variables with tseytin transformation
pub fn to_cnf<T: Source>(formula: T) -> anyhow::Result<CnfFormula> {
    let mut visitor = CnfConversionVisitor {
        formula: CnfFormula::new(),
        node2var: Vec::new(),
        false_var: Var::from_dimacs(1),
    };
    formula
        .visit(&mut visitor)
        .context("converting nnf to cnf")?;
    visitor.formula.add_clause(&[visitor
        .node2var
        .pop()
        .ok_or(anyhow::anyhow!("empty visitor"))?]);
    Ok(visitor.formula)
}

#[test]
fn random_nnf_to_cnf() -> anyhow::Result<()> {
    use crate::cnf;
    use std::iter::FromIterator;
    let size = 4;
    for _ in 0..30 {
        let formula = cnf::generate_random_3sat(size, size * 4);
        let mut cnf_string = Vec::new();
        varisat_dimacs::write_dimacs(&mut cnf_string, &formula)?;
        let nnf = crate::d4::D4::run(&cnf_string as &[u8], None, None, 0).context("running d4")?;
        let result = to_cnf(&nnf).context("to_cnf")?;
        let mut resized_original = CnfFormula::new();
        for clause in formula.iter() {
            resized_original.add_clause(clause);
        }
        for var in size..(dbg!(result.var_count())) {
            resized_original.add_clause(&[Lit::from_index(var, true)] as &[_]);
        }
        resized_original.set_var_count(result.var_count());
        let original_vars = FixedBitSet::from_iter(0..size);
        assert_eq!(
            cnf::brute_force_popcon(&resized_original, &original_vars),
            cnf::brute_force_popcon(&result, &original_vars)
        );
    }
    Ok(())
}

struct DotVisitor<W: Write> {
    output: W,
    index: usize,
}

impl<W: Write> Visitor for DotVisitor<W> {
    fn visit_header(&mut self, _header: &Header) -> anyhow::Result<()> {
        writeln!(self.output, "digraph {{")?;
        Ok(())
    }

    fn visit_node(&mut self, node: &Node) -> anyhow::Result<()> {
        write!(self.output, "{} [ style=filled ", self.index)?;
        match node {
            Node::L(lit) => {
                writeln!(
                    self.output,
                    "label=\"L({})\" color=blue fontcolor=white ]",
                    lit
                )?;
            }
            Node::F => {
                writeln!(self.output, "label=F color=black fontcolor=white ]")?;
            }
            Node::A(nodes) => {
                writeln!(self.output, "label=A color=yellow ]")?;
                for i in nodes {
                    writeln!(self.output, "{} -> {} []", self.index, i)?;
                }
            }
            Node::O(var, left, right) => {
                writeln!(
                    self.output,
                    "label=\"O({})\" color=green ]",
                    var.map(|v| v.to_dimacs()).unwrap_or(0)
                )?;
                for i in &[left, right] {
                    writeln!(self.output, "{} -> {} []", self.index, i)?;
                }
            }
        };
        self.index += 1;
        Ok(())
    }
}

/// Converts a nnf formula to a graphviz format graph
pub fn to_dot<T: Source, W: Write>(formula: T, output: W) -> anyhow::Result<()> {
    let mut visitor = DotVisitor { output, index: 0 };
    formula
        .visit(&mut visitor)
        .context("converting nnf to dot")?;
    writeln!(visitor.output, "}}").context("writing to dot output")?;
    Ok(())
}

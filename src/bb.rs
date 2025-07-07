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

//! reimplementation of Pipatsrisawat, Knot, et Adnan Darwiche. « A New d-DNNF-Based Bound Computation Algorithm for Functional E-MAJSAT ». In IJCAI, 2009.

use std::fmt::Debug;

use anyhow::Context;
use fixedbitset::FixedBitSet;
use num_traits::One;

use crate::cnf::{Lit, ModelCount, PopConBounds, Var};
use crate::ddnnf::{self, LowerBoundQuality};
use crate::{
    ddnnf::{MemoryDDNNF, Node, QualifiedModelCount, Source, Visitor},
    model::Model,
};

mod oval {
    use super::*;

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct OptionPair {
        positive: QualifiedModelCount,
        negative: QualifiedModelCount,
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct NodeInfo {
        /// options pairs (see paper) of this node
        option_pairs: OptionPairMap,
        /// upper bound of max popularity assuming ambient_assignment
        upper_bound: QualifiedModelCount,
    }

    #[derive(Clone, PartialEq, Eq)]
    struct OptionPairMap(Vec<Option<Box<OptionPair>>>);
    impl Debug for OptionPairMap {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_map()
                .entries(
                    self.0
                        .iter()
                        .enumerate()
                        .filter_map(|(i, p)| p.as_ref().map(|p| (Var::from_index(i), p))),
                )
                .finish()
        }
    }

    pub(crate) struct UpperBoundVisitor<'a> {
        ambient_assignment: Vec<Option<bool>>,
        nvars: usize,
        controlled_variables: &'a FixedBitSet,
        by_node: Vec<NodeInfo>,
        index: usize,
    }

    impl<'a> UpperBoundVisitor<'a> {
        pub fn new(
            ambient_assignment: Vec<Option<bool>>,
            controlled_variables: &'a FixedBitSet,
        ) -> UpperBoundVisitor<'a> {
            let nvars = controlled_variables.len();
            assert_eq!(ambient_assignment.len(), nvars);
            UpperBoundVisitor {
                ambient_assignment,
                nvars,
                controlled_variables,
                by_node: Vec::new(),
                index: 0,
            }
        }

        /// kappa in the paper
        ///
        /// kappa is computed relative to ambient_assignment plus assuming_lit
        ///
        /// var can be any controlled variable except the ones of ambient_assignment
        fn option_pair_bound<'b>(
            &'a self,
            var: Var,
            assuming_lit: Option<Lit>,
            option_pair: &'b OptionPair,
        ) -> &'b QualifiedModelCount {
            assert!(self.ambient_assignment[var.index()].is_none());
            assert!(self.controlled_variables[var.index()]);
            // value of the variable according the the assignment
            let value = match assuming_lit {
                Some(l) => {
                    if l.var() == var {
                        Some(l.is_positive())
                    } else {
                        None
                    }
                }
                None => None,
            };
            match value {
                Some(true) => &option_pair.positive,
                Some(false) => &option_pair.negative,
                None if option_pair.positive < option_pair.negative => &option_pair.negative,
                None => &option_pair.positive,
            }
        }

        /// computes oval of some node
        ///
        /// assignment for oval is self.ambient_assignment plus assuming_lit
        /// node is node_info
        fn oval<'b>(
            &self,
            node_info: &'b NodeInfo,
            assuming_lit: Option<Lit>,
        ) -> &'b QualifiedModelCount {
            let mut res = &node_info.upper_bound;
            for i in self.controlled_variables.ones() {
                let option_pair = match &node_info.option_pairs.0[i] {
                    None => continue,
                    Some(x) => x,
                };
                let var = Var::from_index(i);
                let new = self.option_pair_bound(var, assuming_lit, option_pair);
                res = if res > new { res } else { new };
            }
            res
        }
    }

    impl<'a> Visitor for UpperBoundVisitor<'a> {
        fn visit_header(&mut self, header: &ddnnf::Header) -> anyhow::Result<()> {
            self.index = 0;
            anyhow::ensure!(
                self.nvars == header.nvars,
                "wrong number of variables for formula"
            );
            let mut empty_option_pair_map = Vec::new();
            for _ in 0..self.nvars {
                empty_option_pair_map.push(None);
            }
            if self.by_node.is_empty() {
                self.by_node.reserve(header.nnodes);
                for _ in 0..header.nnodes {
                    self.by_node.push(NodeInfo {
                        upper_bound: QualifiedModelCount::zero(self.nvars),
                        option_pairs: OptionPairMap(empty_option_pair_map.clone()),
                    });
                }
            }
            Ok(())
        }

        fn visit_node(&mut self, node: &ddnnf::Node) -> anyhow::Result<()> {
            let mut upper_bound = QualifiedModelCount::zero(0);
            let mut option_pairs = Vec::new();
            std::mem::swap(
                &mut option_pairs,
                &mut self.by_node[self.index].option_pairs.0,
            );
            std::mem::swap(&mut upper_bound, &mut self.by_node[self.index].upper_bound);
            for i in self.controlled_variables.ones() {
                option_pairs[i] = None;
            }
            match node {
                Node::L(lit) => {
                    match self.ambient_assignment[lit.var().index()] {
                        Some(v) => {
                            if lit.is_positive() == v {
                                // literal it true under ambient_assignment
                                upper_bound.make_one();
                            } else {
                                // literal is false under ambient_assignment
                                upper_bound.make_zero();
                            }
                        }
                        None => {
                            if self.controlled_variables[lit.var().index()] {
                                let one = QualifiedModelCount::one(self.nvars);
                                let zero = QualifiedModelCount::zero(self.nvars);
                                upper_bound.make_one();
                                // has 1 option pair
                                option_pairs[lit.var().index()] =
                                    Some(Box::new(if lit.is_positive() {
                                        OptionPair {
                                            positive: one,
                                            negative: zero,
                                        }
                                    } else {
                                        OptionPair {
                                            positive: zero,
                                            negative: one,
                                        }
                                    }));
                            } else {
                                // has no option pair
                                upper_bound.make_one();
                                upper_bound.variables.insert(lit.var().index());
                            }
                        }
                    }
                }
                Node::A(children) => {
                    upper_bound.make_one();
                    let mut option_variables = Model::empty(self.nvars);
                    for &child in children {
                        let info = &self.by_node[child];
                        upper_bound *= &info.upper_bound;
                        for i in self.controlled_variables.ones() {
                            let child = &info.option_pairs.0[i];
                            if child.is_some() {
                                option_variables.0.insert(i)
                            }
                        }
                    }
                    for i in option_variables.0.ones() {
                        // compute the option pair for variable i
                        let var = Var::from_index(i);
                        let compute = |assuming_lit| {
                            let mut res = QualifiedModelCount {
                                count: One::one(),
                                variables: FixedBitSet::with_capacity(self.nvars),
                            };
                            for &child in children.iter() {
                                res *= &self.oval(&self.by_node[child], assuming_lit);
                            }
                            res
                        };
                        let option_pair = OptionPair {
                            positive: compute(Some(Lit::from_var(var, true))),
                            negative: compute(Some(Lit::from_var(var, false))),
                        };
                        option_pairs[i] = Some(Box::new(option_pair));
                    }
                }
                Node::O(None, _, _) => {
                    anyhow::bail!("node without opposing variable");
                }
                Node::O(Some(opposing_var), yes, no) => {
                    let op: fn(&QualifiedModelCount, &QualifiedModelCount) -> QualifiedModelCount =
                        if self.controlled_variables[opposing_var.index()] {
                            QualifiedModelCount::maximum
                        } else {
                            |x, y| x + y
                        };
                    let yes_info = &self.by_node[*yes];
                    let no_info = &self.by_node[*no];
                    upper_bound = op(&yes_info.upper_bound, &no_info.upper_bound);

                    for i in 0..self.nvars {
                        if self.ambient_assignment[i].is_some()
                            || !self.controlled_variables[i]
                            || !(yes_info.option_pairs.0[i].is_some()
                                || no_info.option_pairs.0[i].is_some())
                        {
                            continue;
                        }
                        // compute the option pair for variable i
                        let var = Var::from_index(i);
                        let compute = |assuming_lit| {
                            op(
                                &self.oval(yes_info, assuming_lit),
                                &self.oval(no_info, assuming_lit),
                            )
                        };
                        let option_pair = OptionPair {
                            positive: compute(Some(Lit::from_var(var, true))),
                            negative: compute(Some(Lit::from_var(var, false))),
                        };
                        option_pairs[i] = Some(Box::new(option_pair));
                    }
                }
                Node::F => upper_bound.make_zero(),
            }
            std::mem::swap(
                &mut option_pairs,
                &mut self.by_node[self.index].option_pairs.0,
            );
            self.by_node[self.index].upper_bound = upper_bound;
            self.index += 1;
            Ok(())
        }
    }

    struct BBStats {
        /// number of visited potential witnesses
        visited_nodes: u64,
        /// number of times oval is called
        oval_count: u64,
        /// number of times a better witness was found
        improved_witness: u64,
        /// number of nodes where at least one literal was deduced
        deduced_nodes: u64,
        /// number of nodes where a conflicting deduction was done
        backtrack_nodes: u64,
    }

    /// Computes popularity contest by branch and bound as proposed by Pipatsrisawat and Darwiche
    pub fn bb(
        formula: &MemoryDDNNF,
        controlled_variables: &FixedBitSet,
    ) -> anyhow::Result<(Model, QualifiedModelCount)> {
        fn assignment_complete(
            assignment: &[Option<bool>],
            controlled_variables: &FixedBitSet,
        ) -> Option<Model> {
            for v in controlled_variables.ones() {
                if assignment[v].is_none() {
                    return None;
                }
            }
            let mut res = Model::empty(controlled_variables.len());
            for (v, l) in assignment.iter().enumerate() {
                match l {
                    None => (),
                    Some(l) => res.set_index(v, *l),
                }
            }
            Some(res)
        }
        /// returns the largest popularity of this subtree, along with a witness, or the (witness,
        /// popularity) passed as argument if it is better.
        fn aux(
            formula: &MemoryDDNNF,
            controlled_variables: &FixedBitSet,
            mut visitor: &mut oval::UpperBoundVisitor,
            pop: QualifiedModelCount,
            witness: Model,
            stats: &mut BBStats,
        ) -> anyhow::Result<(Model, QualifiedModelCount)> {
            stats.visited_nodes += 1;
            if let Some(leaf) =
                assignment_complete(&visitor.ambient_assignment, controlled_variables)
            {
                let this_pop = formula.popularity(&leaf, controlled_variables)?;
                if this_pop > pop {
                    stats.improved_witness += 1;
                    return Ok((leaf, this_pop));
                } else {
                    return Ok((witness, pop));
                }
            }
            stats.oval_count += 1;
            formula.visit(&mut visitor)?;
            let info = visitor.by_node[formula.header.nnodes - 1].clone();
            if &info.upper_bound < &pop {
                stats.backtrack_nodes += 1;
                return Ok((witness, pop));
            }
            let mut undecided_var = None;
            let mut deduced_once = false;
            for (i, pair) in info.option_pairs.0.iter().enumerate() {
                match pair {
                    None => continue,
                    Some(pair) => {
                        undecided_var = Some(i);
                        match (&pair.negative < &pop, &pair.positive < &pop) {
                            (true, true) => {
                                // backtrack
                                stats.backtrack_nodes += 1;
                                return Ok((witness, pop));
                            }
                            (true, false) => {
                                deduced_once = true;
                                visitor.ambient_assignment[i] = Some(true);
                            }
                            (false, true) => {
                                deduced_once = true;
                                visitor.ambient_assignment[i] = Some(false);
                            }
                            (false, false) => (),
                        }
                    }
                }
            }
            if deduced_once {
                stats.deduced_nodes += 1;
                return aux(formula, controlled_variables, visitor, pop, witness, stats);
            }
            // no option pair is useful
            match undecided_var {
                None => {
                    // the formula does not contain any of the remaining undecided variables
                    #[cfg(debug)]
                    {
                        for node in &formula.nodes {
                            match node {
                                Node::L(l) => {
                                    let i = l.var().index();
                                    assert!(
                                        !controlled_variables[i]
                                            || visitor.ambient_assignment[i].is_some()
                                    )
                                }
                                _ => (),
                            }
                        }
                    }
                    for i in controlled_variables.ones() {
                        if visitor.ambient_assignment[i].is_none() {
                            visitor.ambient_assignment[i] = Some(false);
                        }
                    }
                    return aux(formula, controlled_variables, visitor, pop, witness, stats);
                }
                Some(undecided_var) => {
                    let mut bak = visitor.ambient_assignment.clone();
                    visitor.ambient_assignment[undecided_var] = Some(true);
                    let (witness, pop) =
                        aux(formula, controlled_variables, visitor, pop, witness, stats)?;
                    bak[undecided_var] = Some(false);
                    visitor.ambient_assignment = bak;
                    let (witness, pop) =
                        aux(formula, controlled_variables, visitor, pop, witness, stats)?;
                    return Ok((witness, pop));
                }
            }
        }
        let mut stats = BBStats {
            visited_nodes: 0,
            oval_count: 0,
            improved_witness: 0,
            deduced_nodes: 0,
            backtrack_nodes: 0,
        };
        let assignment = vec![None; formula.header.nvars];
        let lower_bound_span =
            tracing::trace_span!("branch_and_bound_initial_lower_bound", timing = true).entered();
        let (witness, pop) =
            lower_bound(&formula, controlled_variables, LowerBoundQuality::Precise)
                .context("initial lower bound")?;
        drop(lower_bound_span);
        tracing::trace!("starting lower bound: {:?}", pop);
        let bb_span = tracing::trace_span!("branch_and_bound", timing = true).entered();
        let mut visitor = oval::UpperBoundVisitor::new(assignment, controlled_variables);
        let result = aux(
            formula,
            controlled_variables,
            &mut visitor,
            pop,
            witness,
            &mut stats,
        );
        drop(bb_span);
        tracing::trace!(
            bb_visited_nodes = stats.visited_nodes,
            bb_oval_count = stats.oval_count,
            bb_improved_witness = stats.improved_witness,
            bb_deduced_nodes = stats.deduced_nodes,
            bb_backrack_nodes = stats.backtrack_nodes,
            stats = true
        );
        result
    }

    /// Computes an upper bound on the popularity contest of formula assuming a partial assignment of
    /// controlled variables
    ///
    /// Computation time linear in the product of the size of the formula and the number of
    /// controlled variables.
    pub fn upper_bound(
        formula: &MemoryDDNNF,
        ambient_assignment: Vec<Option<bool>>,
        controlled_variables: &FixedBitSet,
    ) -> anyhow::Result<QualifiedModelCount> {
        let mut visitor = UpperBoundVisitor::new(ambient_assignment, controlled_variables);
        formula.visit(&mut visitor)?;
        let info = visitor.by_node.pop().context("empty formula")?;
        Ok(visitor.oval(&info, None).clone())
    }
}
pub use oval::bb;
pub use oval::upper_bound;

/// From an unconstrained d-DNNF, compute an upper bound for popcon using Pipatsrisawat & Darwiche's oval algorithm
/// and our own lower bound.
///
/// The upper bound is linear in the product of the size of the formula and the number of
/// controlled variables.
///
/// If fast_mode is true, then the lower bound is bad and linear in the size of the formula
/// If fast_mode is false, it is better and quadratic in the size of the formula.
pub struct UnconstrainedDdnnfBounds<S: ddnnf::Cnf2DdnnfCompiler> {
    /// How to obtain ddnnf
    pub compiler: S,
    /// Trade-off quality/complexity for lower bound
    pub lower_bound_type: LowerBoundQuality,
}
impl<S: ddnnf::Cnf2DdnnfCompiler> crate::cnf::CNFPopularityContest for UnconstrainedDdnnfBounds<S> {
    fn popularity_contest(
        &self,
        input: impl crate::utils::Input,
        controlled_variables: &FixedBitSet,
        _: &FixedBitSet
    ) -> anyhow::Result<(Model, crate::cnf::PopConBounds)> {
        let ddnnf = self.compiler.compile(input).with_context(|| {
            format!(
                "compiling {} to ddnnf for unconstrained bounding",
                input.display()
            )
        })?;
        let mem = MemoryDDNNF::from_source(ddnnf).context("parsing ddnnf output")?;
        let mut b = upper_bound(
            &mem,
            vec![None; controlled_variables.len()],
            controlled_variables,
        )
        .with_context(|| format!("upper bounding popcon of {}", input.display()))?;
        let mut uncontrolled_variables = controlled_variables.clone();
        uncontrolled_variables.toggle_range(..);
        b.upgrade_to(&uncontrolled_variables);
        let (witness, mut l) = lower_bound(&mem, controlled_variables, self.lower_bound_type)
            .with_context(|| format!("lower bounding popcon of {}", input.display()))?;
        l.upgrade_to(&uncontrolled_variables);
        let range = PopConBounds::from_range(ModelCount::from(l), ModelCount::from(b))?;
        Ok((witness, range))
    }
}

impl MemoryDDNNF {
    fn node_popularity_with_cache<'a, 'b>(
        &self,
        node: usize,
        model: &'a Model,
        controlled_variables: &'a FixedBitSet,
        by_node: &'b mut Vec<Option<QualifiedModelCount>>,
    ) {
        match by_node[node].as_ref() {
            Some(_) => return,
            None => (),
        };
        let pop = match &self.nodes[node] {
            &Node::L(lit) => {
                if controlled_variables[lit.var().index()] {
                    if model[lit] {
                        QualifiedModelCount::one(self.header.nvars)
                    } else {
                        QualifiedModelCount::zero(self.header.nvars)
                    }
                } else {
                    let mut res = QualifiedModelCount::one(self.header.nvars);
                    res.variables.insert(lit.var().index());
                    res
                }
            }
            &Node::F => QualifiedModelCount::zero(self.header.nvars),
            &Node::A(ref children) => {
                let mut res = QualifiedModelCount::one(self.header.nvars);
                for &child in children {
                    self.node_popularity_with_cache(child, model, controlled_variables, by_node);
                    res *= by_node[child].as_ref().unwrap();
                }
                res
            }
            &Node::O(_, yes, no) => {
                self.node_popularity_with_cache(yes, model, controlled_variables, by_node);
                self.node_popularity_with_cache(no, model, controlled_variables, by_node);
                by_node[yes].as_ref().unwrap() + by_node[no].as_ref().unwrap()
            }
        };
        by_node[node] = Some(pop);
    }
    /// Computes the popularity of a model containing only controlled variable on a sub node of the
    /// formula
    pub fn node_popularity(
        &self,
        node: usize,
        model: &Model,
        controlled_variables: &FixedBitSet,
    ) -> anyhow::Result<QualifiedModelCount> {
        anyhow::ensure!(
            model.0.is_subset(controlled_variables),
            "model contains uncontrolled variables"
        );
        anyhow::ensure!(
            controlled_variables.len() == self.header.nvars,
            "formula has {} variables but controlled_variables has size {}",
            self.header.nvars,
            controlled_variables.len()
        );
        let mut cache = Vec::with_capacity(node + 1);
        for _ in 0..=node {
            cache.push(None)
        }
        self.node_popularity_with_cache(node, model, controlled_variables, &mut cache);
        assert_eq!(cache.len(), node + 1);
        Ok(cache.pop().unwrap().unwrap())
    }
    /// Computes the popularity of a model containing only controlled variables
    pub fn popularity(
        &self,
        model: &Model,
        controlled_variables: &FixedBitSet,
    ) -> anyhow::Result<QualifiedModelCount> {
        self.node_popularity(self.header.nnodes - 1, model, controlled_variables)
    }
}

mod lower {
    use super::*;

    #[derive(Debug, Clone)]
    struct NodeInfo {
        witness: Model,
        lower_bound: QualifiedModelCount,
    }

    /// Lower bound for the popularity contest of an unconstrained formula
    ///
    /// also returns a witness for the lower bound (not for LowerBoundQuality::Bad)
    pub fn lower_bound(
        formula: &MemoryDDNNF,
        controlled_variables: &FixedBitSet,
        lower_bound_type: LowerBoundQuality,
    ) -> anyhow::Result<(Model, QualifiedModelCount)> {
        anyhow::ensure!(
            controlled_variables.len() == formula.header.nvars,
            "formula has {} variables but controlled_variables has size {}",
            formula.header.nvars,
            controlled_variables.len()
        );
        let mut by_node: Vec<NodeInfo> = Vec::with_capacity(formula.nodes.len());
        for (index, node) in formula.nodes.iter().enumerate() {
            let mut witness;
            let mut lower_bound;
            let nvars = controlled_variables.len();
            match node {
                &Node::L(lit) => {
                    if controlled_variables[lit.var().index()] {
                        witness = Model::empty(nvars);
                        witness.set(lit.var(), lit.is_positive());
                        lower_bound = QualifiedModelCount::one(nvars);
                    } else {
                        witness = Model::empty(nvars);
                        lower_bound = QualifiedModelCount::one(nvars);
                        lower_bound.variables.insert(lit.var().index());
                    }
                }
                &Node::F => {
                    witness = Model::empty(nvars);
                    lower_bound = QualifiedModelCount::zero(nvars);
                }
                &Node::A(ref children) => {
                    witness = Model::empty(nvars);
                    lower_bound = QualifiedModelCount::one(nvars);
                    for &child in children {
                        let info = &by_node[child];
                        lower_bound *= &info.lower_bound;
                        witness.0.union_with(&info.witness.0);
                    }
                }
                &Node::O(_, yes, no) => {
                    let yes_info = &by_node[yes];
                    let no_info = &by_node[no];
                    match lower_bound_type {
                        LowerBoundQuality::Bad | LowerBoundQuality::Fast => {
                            if yes_info.lower_bound < no_info.lower_bound {
                                witness = no_info.witness.clone();
                                lower_bound = no_info.lower_bound.clone();
                            } else {
                                witness = yes_info.witness.clone();
                                lower_bound = yes_info.lower_bound.clone();
                            }
                        }
                        LowerBoundQuality::Precise => {
                            let yes_pop = formula
                                .node_popularity(index, &yes_info.witness, controlled_variables)
                                .context("popularity of witness")?;
                            let no_pop = formula
                                .node_popularity(index, &no_info.witness, controlled_variables)
                                .context("popularity of witness")?;
                            if yes_pop > no_pop {
                                witness = yes_info.witness.clone();
                                lower_bound = yes_pop;
                            } else {
                                witness = no_info.witness.clone();
                                lower_bound = no_pop;
                            }
                        }
                    }
                }
            }
            #[cfg(debug)]
            {
                lower_bound.check()?;
            }
            match lower_bound_type {
                LowerBoundQuality::Bad | LowerBoundQuality::Fast => {
                    debug_assert!(
                        &lower_bound
                            <= &formula
                                .node_popularity(index, &witness, controlled_variables)
                                .unwrap(),
                        "lower bound does not have the right witness, node={:?}, witness={:?}",
                        &node,
                        &witness
                    );
                }
                LowerBoundQuality::Precise => {
                    debug_assert_eq!(
                        &lower_bound,
                        &formula
                            .node_popularity(index, &witness, controlled_variables)
                            .unwrap(),
                        "lower bound does not have the right witness, node={:?}, witness={:?}",
                        &node,
                        &witness
                    );
                }
            };
            by_node.push(NodeInfo {
                witness,
                lower_bound,
            });
        }
        let info = by_node.pop().context("empty formula")?;
        if lower_bound_type == LowerBoundQuality::Fast {
            let pop = formula
                .node_popularity(by_node.len(), &info.witness, controlled_variables)
                .context("popularity of lower bound witness")?;
            debug_assert!(&info.lower_bound <= &pop);
            Ok((info.witness, pop))
        } else {
            Ok((info.witness, info.lower_bound))
        }
    }
}

pub use lower::lower_bound;

/// From an unconstrained d-DNNF, solve popcon using Pipatsrisawat & Darwiche's branch and bound
/// algo.
pub struct Darwiche<S: ddnnf::Cnf2DdnnfCompiler> {
    /// How to obtain ddnnf
    pub compiler: S,
}
impl<S: ddnnf::Cnf2DdnnfCompiler> crate::cnf::CNFPopularityContest for Darwiche<S> {
    fn popularity_contest(
        &self,
        input: impl crate::utils::Input,
        controlled_variables: &FixedBitSet,
        _: &FixedBitSet
    ) -> anyhow::Result<(Model, crate::cnf::PopConBounds)> {
        let ddnnf = self.compiler.compile(input).with_context(|| {
            format!(
                "compiling {} to ddnnf for unconstrained bounding",
                input.display()
            )
        })?;
        let mem = MemoryDDNNF::from_source(ddnnf).context("parsing ddnnf output")?;
        let (witness, mut value) = bb(&mem, controlled_variables)
            .with_context(|| format!("upper bounding popcon of {}", input.display()))?;
        let mut uncontrolled_variables = controlled_variables.clone();
        uncontrolled_variables.toggle_range(..);
        value.upgrade_to(&uncontrolled_variables);
        Ok((witness, PopConBounds::exactly(ModelCount::from(value))))
    }
}

#[test]
fn projection() {
    use crate::cnf::CNFPopularityContest;
    let vars: FixedBitSet = [0usize, 2].iter().cloned().collect();
    let (model, count) = Darwiche {
        compiler: crate::d4::D4::direct(),
    }
    .popularity_contest("assets/simple.cnf".as_ref() as &std::path::Path, &vars, &FixedBitSet::with_capacity(FixedBitSet::len(&vars)))
    .context("d4 darwich")
    .unwrap();
    assert_eq!(
        count,
        ModelCount {
            model_count: 2usize.into(),
            bits: 1
        }
    );
    assert_eq!(&model.0, &vars);
}

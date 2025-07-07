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

//! Convert CNF to d-DNNF with d4

use crate::cnf::{CNFCounter, CNFPopularityContest, CNFProjectedCounter, ModelCount, PopConBounds};
use crate::cnf::{Lit, Var};
use crate::ddnnf::{LowerBoundQuality, SimplificationOptions, Source};
use crate::model::Model;
use crate::utils::Input;
use crate::utils::LastLines;
use anyhow::Context;
use fixedbitset::FixedBitSet;
use nom::Parser;
use num_bigint::BigUint;
use num_traits::One;
use petgraph::graph::NodeIndex;
use petgraph::visit::Bfs;
use petgraph::Graph;
use std::collections::BTreeSet;
use std::fs::File;
use std::io::Read;
use std::iter::FromIterator;
use std::os::unix::io::FromRawFd;
use std::os::unix::process::CommandExt;
use std::os::unix::process::ExitStatusExt;
use std::path::Path;
use tracing::trace;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
/// A node as returned by d4
pub enum Node {
    /// True formula
    True,
    /// False formula
    False,
    /// Disjunction of descendents
    Or,
    /// Conjunction of descendents
    And,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
/// A node after preprocesssing. Matches `ddnnf::Node`, except does not integrate the children
/// indices.
pub enum SimpleNode {
    /// Literal
    Literal(Lit),
    /// Or node, with optional opposing variable.
    Or(Option<Var>),
    /// And node
    And,
    /// False node
    False,
}

/// d4 output graphs are labelled by literals that are chosen true descending the edge
type Edge = Vec<Lit>;

/// graph as returned by d4
#[derive(Clone, Debug)]
pub struct D4Graph {
    /// the graph itself
    graph: Graph<Node, Edge>,
    /// number of variables, including those not present in the graph anymore
    nvars: usize,
    /// controlled variables
    controlled_vars: Option<FixedBitSet>,
}

/// graph after simplication
type SimpleDNNFGraph = Graph<SimpleNode, ()>;

/// graph after simplication plus number of variables, to be read as a `ddnnf::Source`
#[derive(Clone)]
pub struct D4dDNNFOutput {
    /// original graph, with maybe less variables than the original formula
    pub graph: SimpleDNNFGraph,
    /// number of variables of the original formula
    pub var_count: usize,
    /// controlled variables as reported by d4
    pub controlled_vars: Option<FixedBitSet>,
    /// root of the graph
    pub root: NodeIndex,
}

impl D4dDNNFOutput {
    /// add up to count of the topmost decisions to the specified set of variables
    pub fn add_topmost_decisions(&self, variables: &mut FixedBitSet, mut count: usize) {
        let mut bfs = Bfs::new(&self.graph, self.root);
        while let Some(node) = bfs.next(&self.graph) {
            if count == 0 {
                break;
            };
            match &self.graph[node] {
                SimpleNode::Or(Some(v)) => {
                    if !variables.put(v.index()) {
                        // v was no previously in the set
                        count -= 1;
                    }
                }
                _ => {}
            }
        }
    }
}

impl Source for D4dDNNFOutput {
    fn visit<T: crate::ddnnf::Visitor>(self, visitor: T) -> anyhow::Result<()> {
        (&self).visit(visitor)
    }
}

impl<'a> Source for &'a D4dDNNFOutput {
    fn visit<T: crate::ddnnf::Visitor>(self, mut visitor: T) -> anyhow::Result<()> {
        visitor.visit_header(&crate::ddnnf::Header {
            nedges: self.graph.edge_count(),
            nnodes: self.graph.node_count(),
            nvars: self.var_count,
        })?;
        let toposort = petgraph::algo::toposort(petgraph::visit::Reversed(&self.graph), None)
            .map_err(|cycle| anyhow::anyhow!("cycle on node {:?}", cycle.node_id()))?;
        let mut indexmap = vec![0; self.graph.node_count()];
        for (i, &node) in toposort.iter().enumerate() {
            indexmap[node.index()] = i;
            let outgoing: Vec<usize> = self
                .graph
                .neighbors_directed(node, petgraph::Direction::Outgoing)
                .map(|n| indexmap[n.index()])
                .collect();
            let translation = match self.graph[node] {
                SimpleNode::Literal(l) => {
                    anyhow::ensure!(outgoing.is_empty(), "literal node with descendents");
                    crate::ddnnf::Node::L(l)
                }
                SimpleNode::And => crate::ddnnf::Node::A(outgoing),
                SimpleNode::False => crate::ddnnf::Node::F,
                SimpleNode::Or(opposing) => {
                    anyhow::ensure!(outgoing.len() == 2, "or node with !=2 descendents");
                    crate::ddnnf::Node::O(opposing, outgoing[0], outgoing[1])
                }
            };
            visitor.visit_node(&translation)?;
        }
        Ok(())
    }
}

/// Removes true and false nodes, and transforms edges labeled by literals into `AND(literal, target
/// node)` Actually, unlabeled edges to true may error out.
pub fn preprocess(mut output: D4Graph) -> anyhow::Result<D4dDNNFOutput> {
    use petgraph::visit::EdgeRef;
    let controlled_vars = match &output.controlled_vars {
        None => BTreeSet::new(),
        Some(c) => c.ones().map(Var::from_index).collect(),
    };
    let mut res = SimpleDNNFGraph::new();
    let mut reachable_nodes = 0;

    // Literal nodes: we create only one literal node per literal
    // This is ensured by the `get_literal` closure
    let not_a_node = petgraph::graph::NodeIndex::<_>::end();
    let mut added_negative_literals = vec![not_a_node; output.nvars];
    let mut added_positive_literals = vec![not_a_node; output.nvars];
    let mut get_literal = |res: &mut SimpleDNNFGraph, lit: Lit| -> petgraph::graph::NodeIndex {
        let index = lit.var().index();
        let storage = if lit.is_positive() {
            &mut added_positive_literals
        } else {
            &mut added_negative_literals
        };
        let before = storage[index];
        if before != not_a_node {
            before
        } else {
            let res = res.add_node(SimpleNode::Literal(lit));
            storage[index] = res;
            res
        }
    };
    let sources: Vec<_> = output
        .graph
        .externals(petgraph::Direction::Incoming)
        .collect();
    anyhow::ensure!(sources.len() == 1, "more than one root node");
    let root = sources[0];

    // indexmap: old index -> new index
    let mut indexmap = vec![not_a_node; output.graph.node_count()];
    // we need to visit a node after all its parents have been visited
    let toposort = petgraph::algo::toposort(petgraph::visit::Reversed(&output.graph), None)
        .map_err(|cycle| anyhow::anyhow!("cycle on node {:?}", cycle.node_id()))?;
    for &node in toposort.iter().rev() {
        reachable_nodes += 1;
        let n_children = output
            .graph
            .neighbors_directed(node, petgraph::Direction::Outgoing)
            .count();
        // the node to which children will attach
        let corresponding = match output.graph[node] {
            Node::False | Node::Or if n_children == 0 => Some(res.add_node(SimpleNode::False)),
            Node::False => anyhow::bail!("false node may not have descendents"),
            Node::True => {
                if node == root {
                    // root
                    let v = Var::from_index(0);
                    let o = res.add_node(SimpleNode::Or(Some(v)));
                    let l1 = get_literal(&mut res, Lit::negative(v));
                    res.add_edge(o, l1, ());
                    let l2 = get_literal(&mut res, Lit::positive(v));
                    res.add_edge(o, l2, ());
                }
                anyhow::ensure!(n_children == 0, "true node may not have descendents");
                None
            }
            Node::Or if n_children == 1 => Some(res.add_node(SimpleNode::And)),
            Node::Or if n_children == 2 => {
                /*
                 * if a controlled literal l is on both branches ->
                 * And(l, recurse)
                 * if a controlled variable is opposing, use it
                 * otherwise, use any opposing variable.
                 */
                let mut child_labels = output
                    .graph
                    .edges_directed(node, petgraph::Direction::Outgoing)
                    .map(|e| e.weight());
                let first: BTreeSet<Lit> = child_labels.next().unwrap().iter().cloned().collect();
                let second: BTreeSet<Lit> = child_labels.next().unwrap().iter().cloned().collect();
                let controlled_propagated: BTreeSet<Lit> = first
                    .intersection(&second)
                    .cloned()
                    .filter(|l| controlled_vars.contains(&l.var()))
                    .collect();
                if !controlled_propagated.is_empty() {
                    // move these to the edge above
                    let incoming: Vec<_> = output
                        .graph
                        .edges_directed(node, petgraph::Direction::Incoming)
                        .map(|i| i.id())
                        .collect();
                    let outgoing: Vec<_> = output
                        .graph
                        .edges_directed(node, petgraph::Direction::Outgoing)
                        .map(|i| i.id())
                        .collect();
                    for edge_index in incoming {
                        output.graph[edge_index].extend(&controlled_propagated);
                    }
                    for edge_index in outgoing {
                        output.graph[edge_index].retain(|l| !controlled_propagated.contains(l));
                    }
                }
                let not_first = first.iter().map(|&l| !l).collect();
                let opposing_variables: BTreeSet<Var> =
                    second.intersection(&not_first).map(|l| l.var()).collect();
                let controlled_opposing_variables =
                    opposing_variables.intersection(&controlled_vars);
                let opposing_variable = controlled_opposing_variables
                    .chain(opposing_variables.iter())
                    .next()
                    .cloned();
                Some(res.add_node(SimpleNode::Or(opposing_variable)))
            }
            Node::Or => anyhow::bail!("or node with more than 2 descendents"),
            Node::And => Some(res.add_node(SimpleNode::And)),
        };
        if let Some(n) = corresponding {
            indexmap[node.index()] = n;
        };
        for e in output
            .graph
            .edges_directed(node, petgraph::Direction::Incoming)
        {
            let parent = indexmap[e.source().index()];
            anyhow::ensure!(parent != not_a_node, "no parent for {:?}", e.source());
            if e.weight().is_empty() {
                match corresponding {
                    Some(n) => {
                        res.add_edge(parent, n, ());
                    }
                    None => {
                        anyhow::ensure!(
                            output.graph[node] == Node::True,
                            "untranslated node is not true but {:?}, logic error",
                            output.graph[node]
                        );
                        anyhow::ensure!(res[parent]==SimpleNode::And, "untranslated true node has non AND parent: parent is {:?} translated to {:?}", &output.graph[e.source()], &res[parent]);
                        // do nothing
                    }
                }
            } else {
                let a = res.add_node(SimpleNode::And);
                for &l in e.weight() {
                    anyhow::ensure!(
                        l.var().index() < output.nvars,
                        "literal {} may not appear in d4 output for formula with {} variables",
                        l,
                        output.nvars
                    );
                    let ln = get_literal(&mut res, l);
                    res.add_edge(a, ln, ());
                }
                if let Some(n) = corresponding {
                    res.add_edge(a, n, ());
                }
                res.add_edge(parent, a, ());
            }
        }
    }
    anyhow::ensure!(
        reachable_nodes == output.graph.node_count(),
        "unreachable nodes in d4 output"
    );
    let var_count = output.nvars;
    Ok(D4dDNNFOutput {
        graph: res,
        var_count,
        controlled_vars: output.controlled_vars,
        root: indexmap[root.index()],
    })
}

/// Parses space then newline
fn next(input: &str) -> nom::IResult<&str, ()> {
    nom::character::complete::space0
        .and(nom::character::complete::line_ending)
        .map(|_| ())
        .parse(input)
}

/// Parses a space separated list of signed integers
fn int_list<'a>(input: &'a str) -> nom::IResult<&'a str, Vec<isize>> {
    use nom::*;
    multi::separated_list1(
        character::complete::space1,
        combinator::map(
            combinator::consumed(
                combinator::opt(character::complete::char('-')).and(character::complete::digit1),
            ),
            |(s, _): (&str, _)| s.parse::<isize>().unwrap(),
        ),
    )(input)
}

/// Inserts the value at the specified index, filling missing elements with NodeIndex::end()
fn insert_at_index(
    vec: &mut Vec<petgraph::graph::NodeIndex>,
    index: usize,
    value: petgraph::graph::NodeIndex,
) {
    match vec.len().cmp(&index) {
        std::cmp::Ordering::Equal => vec.push(value),
        std::cmp::Ordering::Greater => {
            vec[index] = value;
        }
        std::cmp::Ordering::Less => {
            vec.resize(index + 1, petgraph::graph::NodeIndex::end());
            vec[index] = value;
        }
    }
}

/// Gets the value at the specified index, and return None if the index is invalid or the node
/// `NodeIndex::end()`.
fn get_at_index(
    vec: &mut Vec<petgraph::graph::NodeIndex>,
    index: usize,
) -> Option<petgraph::graph::NodeIndex> {
    match vec.get(index) {
        None => None,
        Some(x) if *x == petgraph::graph::NodeIndex::end() => None,
        Some(x) => Some(*x),
    }
}

/// Parses the output of d4 into a `D4RawGraph`, and a `FixedBitSet` representing the controlled
/// variables.
pub fn parse_d4_output(s: &str) -> anyhow::Result<D4Graph> {
    let mut res = D4Graph {
        graph: Graph::new(),
        controlled_vars: None,
        nvars: 0,
    };
    let mut indexmap = Vec::new();
    let mut rest = s;
    let mut lineno = 0;
    while rest != "" {
        if let Ok((r, _)) =
            nom::bytes::complete::tag::<&str, &str, nom::error::Error<&str>>("c")(rest)
        {
            if let Ok((r, _)) =
                nom::bytes::complete::tag::<&str, &str, nom::error::Error<&str>>(" controlled ")(r)
            {
                let (_, args) = int_list(r).map_err(|_| {
                    anyhow::Error::msg(format!(
                        "expected list of ints c controlled comment at line {}",
                        lineno
                    ))
                })?;
                anyhow::ensure!(args.len() >= 1, "expected >= 1 integers line {}", lineno);
                anyhow::ensure!(
                    args[args.len() - 1] == 0,
                    "line {} does not end with 0",
                    lineno
                );
                res.controlled_vars = Some(FixedBitSet::from_iter(
                    args[..(args.len() - 1)].iter().map(|&i| (i - 1) as usize),
                ));
            } else if let Ok((r, _)) =
                nom::bytes::complete::tag::<&str, &str, nom::error::Error<&str>>(" nvars ")(r)
            {
                let (_, args) = int_list(r).map_err(|_| {
                    anyhow::Error::msg(format!(
                        "expected list of ints c controlled comment at line {}",
                        lineno
                    ))
                })?;
                anyhow::ensure!(args.len() == 1, "expected 1 integer line {}", lineno);
                res.nvars = args[0] as usize;
            }
            let (r, _) =
                nom::bytes::complete::take_until::<&str, &str, nom::error::Error<&str>>("\n")(r)
                    .map_err(|_| {
                        anyhow::Error::msg(format!("unterminated comment line {}", lineno))
                    })?;
            rest = r;
        } else if let Ok((r, tag)) =
            nom::character::complete::one_of::<&str, &str, nom::error::Error<&str>>("aotf")(rest)
        {
            let (r, _) = nom::character::complete::space1::<&str, nom::error::Error<&str>>(r)
                .map_err(|_| {
                    anyhow::Error::msg(format!("expected space after node type at line {}", lineno))
                })?;
            let (r, args) = int_list(r).map_err(|_| {
                anyhow::Error::msg(format!(
                    "expected list of ints after node type at line {}",
                    lineno
                ))
            })?;
            rest = r;
            anyhow::ensure!(args.len() == 2, "expected 2 arguments line {}", lineno);
            anyhow::ensure!(args[1] == 0, "line {} does not end with 0", lineno);
            let index = args[0];
            let node = match tag {
                'a' => Node::And,
                'o' => Node::Or,
                't' => Node::True,
                'f' => Node::False,
                _ => unreachable!(),
            };
            let real_index = res.graph.add_node(node);
            insert_at_index(&mut indexmap, index as usize, real_index);
        } else {
            let (r, args) = int_list(rest).map_err(|_| {
                anyhow::Error::msg(format!("expected list of ints line {}", lineno))
            })?;
            rest = r;
            anyhow::ensure!(
                args.len() >= 3,
                "expected at least 3 arguments line {}",
                lineno
            );
            anyhow::ensure!(
                args[args.len() - 1] == 0,
                "line {} does not end with 0",
                lineno
            );
            let from = get_at_index(&mut indexmap, args[0] as usize)
                .with_context(|| format!("unexpected index {} line {}", args[0], lineno))?;
            let to = get_at_index(&mut indexmap, args[1] as usize)
                .with_context(|| format!("unexpected index {} line {}", args[1], lineno))?;
            let labels: Edge = args[2..(args.len() - 1)]
                .iter()
                .map(|&i| Lit::from_dimacs(i))
                .collect();
            res.graph.add_edge(from, to, labels);
        }
        rest = next(rest)
            .map_err(|_| {
                anyhow::Error::msg(format!("expected whitespace after 0 line {}", lineno))
            })?
            .0;
        lineno += 1;
    }
    let nvars = res.nvars;
    anyhow::ensure!(
        nvars != 0,
        "d4 is not patched to output the number of variables, or output 0 variables"
    );
    res.controlled_vars.as_mut().map(|s| s.grow(nvars));
    Ok(res)
}

#[test]
fn parsing() -> anyhow::Result<()> {
    let output = parse_d4_output(
        "c nvars 6
o 1 0
o 2 0
o 3 0
t 4 0
3 4 -2 3 0
3 4 2 0
2 3 -1 0
2 4 1 0
1 2 0
c commentary
c controlled 1 0
",
    )?;
    assert_eq!(output.graph.node_count(), 4);
    assert_eq!(output.graph.edge_count(), 5);
    let mut expected = FixedBitSet::with_capacity(6);
    expected.insert(0);
    assert_eq!(output.controlled_vars, Some(expected));
    assert_eq!(output.nvars, 6);
    Ok(())
}

#[test]
fn processing() -> anyhow::Result<()> {
    let input = "c nvars 7
o 1 0
t 2 0
1 2 0
";
    let output = parse_d4_output(input)?;
    let graph2 = preprocess(output)?;
    let mc = crate::ddnnf::count_models(&graph2, None)?;
    assert_eq!(mc.model_count, 128u32.into());
    Ok(())
}

/// Runs d4 converting a file in DIMACS format.
/// If `priority_variables` is not None, these variables will be in the topmost OR nodes.
/// The resulting graph may omit useless variables. It is necessary to offset results to take this
/// into account.
fn run_d4(
    cnf_formula: &Path,
    priority_variables: Option<&FixedBitSet>,
    projection_variables: Option<&FixedBitSet>,
    relax: usize,
) -> anyhow::Result<D4Graph> {
    // check taht the input file exists.
    let cnf_formula = if cnf_formula.is_absolute() {
        cnf_formula.to_owned()
    } else {
        std::fs::canonicalize(&cnf_formula).with_context(|| {
            format!(
                "canonicalizing {},the input file to d4",
                cnf_formula.display()
            )
        })?
    };
    // create a pipe for output
    let (raw_read, raw_write) = nix::unistd::pipe().context("creating a pipe for c2d output")?;
    // doing this immediately ensure these are close in case something happens
    let (mut read, write) = unsafe { (File::from_raw_fd(raw_read), File::from_raw_fd(raw_write)) };
    crate::utils::set_cloexec(&read, true).context("d4 input pipe cloexec")?;

    let fd_path = format!("/proc/self/fd/{}", raw_write);
    let exe = match (priority_variables, projection_variables) {
        (None, _) => "d4",
        (Some(_), None) => "d4prio",
        (Some(_), Some(_)) => {
            anyhow::bail!("cannot have both prio and projection variables with d4")
        }
    };
    // spawn the process
    let mut cmd = std::process::Command::new(exe);
    cmd.arg("-dDNNF").arg(format!("-out={}", fd_path));
    let fpv2;
    let fpv; // must not be dropped before d4 has completed
    if let Some(prio) = priority_variables {
        if prio.count_ones(..) > 0 {
            let variables: Vec<String> = prio.ones().map(|i| (i + 1).to_string()).collect();
            let content = variables.join(",");
            trace!("controlled variables: {}", &content);
            fpv2 = content.as_bytes().to_path()?;
            cmd.arg(format!("-fpv2={}", fpv2.as_ref().display()));
            if relax > 0 {
                cmd.arg(format!("-relax={}", relax));
            }
        }
    }
    if let Some(proj) = projection_variables {
        if proj.count_ones(..) > 0 {
            let variables: Vec<String> = proj.ones().map(|i| (i + 1).to_string()).collect();
            let content = variables.join(",");
            trace!("projection variables: {}", &content);
            fpv = content.as_bytes().to_path()?;
            cmd.arg(format!("-fpv={}", fpv.as_ref().display()));
            // https://github.com/crillab/d4/issues/4#issuecomment-853928676
            cmd.arg("-pv=NO");
        }
    }
    cmd.arg(&cnf_formula)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    unsafe {
        cmd.pre_exec(|| prctl::set_death_signal(9).map_err(std::io::Error::from_raw_os_error))
    };
    let process_span = tracing::trace_span!("d4_subprocess", timing = true).entered();
    trace!(?cmd, "starting d4");
    let mut process = cmd.spawn().context("failed to run d4")?;
    // close our copy of the write pipe so that if c2d dies, the read end returns EOF
    drop(write);
    let stdout = LastLines::new(process.stdout.take().unwrap());
    let stderr = LastLines::new(process.stderr.take().unwrap());

    let mut text = String::new();
    read.read_to_string(&mut text)
        .context("reading d4 output")?;
    if crate::utils::try_wait_timeout(&mut process, std::time::Duration::from_secs(2))
        .context("waiting for d4")?
        .is_none()
    {
        // d4 has not terminated
        process.kill().context("killing d4")?;
    }
    let status = process.wait().context("waiting for d4")?;
    drop(process_span);
    let (out, _) = stdout.get();
    let (err, _) = stderr.get();
    anyhow::ensure!(
        status.success(),
        "d4 failed with code {:?} signal {:?}: {} {}",
        status.code(),
        status.signal(),
        String::from_utf8_lossy(&out),
        String::from_utf8_lossy(&err)
    );
    let parse_span = tracing::trace_span!("parsing_d4_output", timing = true).entered();
    trace!("d4 exited successfully, parsing output");
    let output = parse_d4_output(&text).context("failed to parse d4 output")?;
    drop(parse_span);
    Ok(output)
}

#[test]
fn simple() {
    // 3 & (1|2)
    let models = D4::direct()
        .model_count("assets/simple.cnf".as_ref() as &Path, None)
        .context("creating d4")
        .unwrap();
    assert_eq!(
        models,
        ModelCount {
            model_count: 3usize.into(),
            bits: 3
        }
    );
}

#[test]
fn proj() {
    let mut projected_vars = FixedBitSet::with_capacity(3);
    projected_vars.insert(2);
    projected_vars.insert(1);
    // 3 & (1|2)
    // Models: [3,2] [3,-2]
    let models = D4::direct()
        .projected_model_count("assets/simple.cnf".as_ref() as &Path, &projected_vars)
        .context("creating d4")
        .unwrap();
    assert_eq!(
        models,
        ModelCount {
            model_count: 2usize.into(),
            bits: 2
        }
    );
}

#[test]
fn spurious_var() {
    // 1 & (1|2), 2 gets simplified out so we return a formula with just one bit.
    let models = D4::direct()
        .model_count("assets/d4_simplified_out_var.cnf".as_ref() as &Path, None)
        .context("creating d4")
        .unwrap();
    assert_eq!(
        models,
        ModelCount {
            model_count: 2usize.into(),
            bits: 2
        }
    );
}

#[test]
fn with_false_node() {
    // this file makes d4 output a false node
    let models = D4::direct()
        .model_count("assets/d4_false.cnf".as_ref() as &Path, None)
        .context("creating d4")
        .unwrap();
    assert_eq!(
        models,
        ModelCount {
            model_count: 2usize.into(),
            bits: 3
        }
    );
}

#[test]
fn unsat() {
    let models = D4::direct()
        .model_count("assets/unsat.cnf".as_ref() as &Path, None)
        .context("creating d4")
        .unwrap();
    assert_eq!(
        models,
        ModelCount {
            model_count: 0usize.into(),
            bits: 1
        }
    );
}

#[test]
fn regression_dsharp_issue_12() {
    // regression test for https://github.com/QuMuLab/dsharp/issues/12
    let models = D4::direct()
        .model_count("assets/bug_dsharp_mc.cnf".as_ref() as &Path, None)
        .context("creating d4")
        .unwrap();
    assert_eq!(
        models,
        ModelCount {
            model_count: BigUint::from(104310u32) << 372,
            bits: 400
        }
    );
}

#[test]
fn projection() {
    let vars: FixedBitSet = [0usize, 2].iter().cloned().collect();
    let (model, count) = D4::direct()
        .popularity_contest("assets/simple.cnf".as_ref() as &Path, &vars, &FixedBitSet::with_capacity(FixedBitSet::len(&vars)))
        .context("creating d4")
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

#[test]
fn regression_d4_issue_4() {
    // https://github.com/crillab/d4/issues/4
    let mut vars = FixedBitSet::with_capacity(9);
    vars.insert(0);
    let (model, count) = crate::cnf::Compacter(D4::direct())
        .popularity_contest("assets/d4_popconbug.cnf".as_ref() as &Path, &vars, &FixedBitSet::with_capacity(FixedBitSet::len(&vars)))
        .context("creating d4")
        .unwrap();
    assert_eq!(
        count,
        ModelCount {
            model_count: 96usize.into(),
            bits: 8
        }
    );
    assert!(model.is_empty());
}

#[test]
fn with_compaction() {
    let count = crate::cnf::Compacter(D4::direct())
        .model_count("assets/d4_popconbug.cnf".as_ref() as &Path, None)
        .context("creating d4")
        .unwrap();
    assert_eq!(
        count,
        ModelCount {
            model_count: 192usize.into(),
            bits: 9
        }
    );
}

/// Use d4 to count models
pub struct D4 {
    /// relax controlled decisions to up to n uncontrolled variables
    pub relax: usize,
    /// how to choose relaxation variables, bfs or dfs
    pub bfs: bool,
    /// repass the counting graph into D4
    pub relax_exact: bool,
    /// Simplfication options
    pub simplification_options: SimplificationOptions,
    /// How to compute the lower bound
    pub lower_bound_type: LowerBoundQuality,
}

impl D4 {
    /// Returns D4 in normal, non relaxed mode
    pub fn direct() -> D4 {
        D4 {
            relax: 0,
            bfs: false,
            relax_exact: false,
            simplification_options: SimplificationOptions::all(),
            lower_bound_type: LowerBoundQuality::Fast,
        }
    }
}

impl CNFCounter for D4 {
    fn model_count(
        &self,
        input: impl Input,
        projection_variables: Option<&FixedBitSet>,
    ) -> anyhow::Result<ModelCount> {
        let graph = D4::run(input, None, projection_variables, 0)
            .with_context(|| format!("running d4 on {}", input.display()))?;
        // crate::ddnnf::write(std::io::stdout(), &graph)?;
        let models = crate::ddnnf::count_models(&graph, projection_variables)
            .with_context(|| format!("counting models for output of d4 on {}", input.display()))?;
        Ok(models)
    }
}

impl CNFProjectedCounter for D4 {
    fn projected_model_count(
        &self,
        input: impl Input,
        projection_variables: &FixedBitSet,
    ) -> anyhow::Result<ModelCount> {
        let graph = D4::run(input, None, Some(projection_variables), 0)
            .with_context(|| format!("running d4 on {}", input.display()))?;
        // crate::ddnnf::write(std::io::stdout(), &graph)?;
        let mut models = crate::ddnnf::count_models(&graph, Some(projection_variables))
            .with_context(|| format!("counting models for output of d4 on {}", input.display()))?;
        let bits = projection_variables.count_ones(..);
        anyhow::ensure!(
            models.bits == projection_variables.len(),
            "Wrong number of bits {} in model count for {} variables",
            models.bits,
            projection_variables.len()
        );
        anyhow::ensure!(
            models.model_count <= (BigUint::one() << bits),
            "More models {} than projection variables count {} allows",
            models.model_count,
            bits
        );
        models.bits = bits;
        tracing::trace!(
            pmc_result_cnf_log2 = models.model_count.bits(),
            pmc_cnf_bits = models.bits,
            stats = true
        );
        Ok(models)
    }
}

impl CNFPopularityContest for D4 {
    fn popularity_contest(
        &self,
        input: impl Input,
        controlled_variables: &FixedBitSet,
        _: &FixedBitSet
    ) -> anyhow::Result<(Model, PopConBounds)> {
        let (model, count) = if self.relax == 0 {
            let graph = D4::run(input, Some(controlled_variables), None, self.relax)
                .with_context(|| format!("running d4 on {}", input.display()))?;
            let (model, count) =
                crate::ddnnf::most_popular_model(&graph, controlled_variables.clone())
                    .with_context(|| {
                        format!(
                            "solving popularity contest for output of d4 on {}",
                            input.display()
                        )
                    })?;
            (model, PopConBounds::exactly(count))
        } else {
            let graph = if self.bfs {
                let first_span =
                    tracing::trace_span!("preparation_in_mc_mode", timing = true).entered();
                let easy = D4::run(input, None, None, 0)
                    .with_context(|| format!("running d4 on {}", input.display()))?;
                drop(first_span);
                let mut relaxed_variables = controlled_variables.clone();
                easy.add_topmost_decisions(&mut relaxed_variables, self.relax);
                let second_span =
                    tracing::trace_span!("actual_relaxed_d4_popcon", timing = true).entered();
                let graph = D4::run(input, Some(&relaxed_variables), None, 0)
                    .with_context(|| format!("running d4 on {}", input.display()))?;
                drop(second_span);
                graph
            } else {
                D4::run(input, Some(&controlled_variables), None, self.relax)
                    .with_context(|| format!("running d4 on {}", input.display()))?
            };

            let bounds = crate::ddnnf::most_popular_model_relaxed(
                graph,
                controlled_variables.clone(),
                self.relax_exact,
                self.simplification_options,
                self.lower_bound_type,
            )
            .with_context(|| {
                format!(
                    "solving popularity contest for output of d4 on {}",
                    input.display()
                )
            })?;
            (
                bounds.witness,
                PopConBounds::from_range(bounds.lower.into(), bounds.upper.into())?,
            )
        };
        Ok((model, count))
    }
}

impl D4 {
    /// Runs d4 on the specified DIMACS file and returns the resulting DNNF.
    /// If controlled variables is not None, these variables should be in the topmost OR nodes.
    pub fn run(
        input: impl Input,
        controlled_variables: Option<&FixedBitSet>,
        projection_variables: Option<&FixedBitSet>,
        relax: usize,
    ) -> anyhow::Result<D4dDNNFOutput> {
        let tempfile = input.to_path()?;
        let d4_span = tracing::trace_span!("running_and_parsing_d4", timing = true).entered();
        let graph = run_d4(
            tempfile.as_ref(),
            controlled_variables,
            projection_variables,
            relax,
        )
        .with_context(|| format!("running d4 on {}", input.display()))?;
        trace!("parsed d4 output");
        drop(d4_span);
        let preprocess_span =
            tracing::trace_span!("d4_preprocessing_output", timing = true).entered();
        trace!("converting d4 output to d-DNNF...");
        let graph2 = preprocess(graph)
            .with_context(|| format!("processing output of d4 on {}", input.display()))?;
        trace!(ddnnf_nodes = graph2.graph.node_count(), ddnnf_edges = graph2.graph.edge_count(), stats=true, "done converting d4 output to d-DNNF...");
        drop(preprocess_span);
        Ok(graph2)
    }
}

impl crate::ddnnf::Cnf2DdnnfCompiler for D4 {
    type O = D4dDNNFOutput;
    fn compile<I: Input>(&self, input: I) -> anyhow::Result<D4dDNNFOutput> {
        D4::run(input, None, None, 0)
    }
}

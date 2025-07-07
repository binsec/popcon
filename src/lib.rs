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

#![warn(missing_docs)]

//! Popularity contest on d-DNNF, CNF and SMT(QF_BV)

pub mod aig2cnf;
pub mod bb;
pub mod bitblast;
pub mod c2d;
pub mod cnf;
pub mod d4;
pub mod dcssat;
pub mod ddnnf;
pub mod dsharp;
pub mod input;
pub mod maxcount;
pub mod model;
pub mod smt;
pub mod ssatabc;
pub mod stats;
pub mod utils;
pub mod baxmc;

use anyhow::Context;
use chrono::Duration;
use cnf::{
    CNFCounter, CNFPopularityContest, CNFProjectedCounter, Compacter, ModelCount, NoProjection,
    PopConBounds,
};
use ddnnf::{LowerBoundQuality, SimplificationOptions};
use fixedbitset::FixedBitSet;
use model::Model;
use std::cell::RefCell;
use std::fs::File;
use std::ops::DerefMut;
use std::path::PathBuf;
use std::time::Instant;
use structopt::clap::arg_enum;
use structopt::StructOpt;
use utils::Input;

use crate::bb::{Darwiche, UnconstrainedDdnnfBounds};

arg_enum! {
    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    enum ToolName {
        C2D,
        Dsharp,
        D4,
        BruteForce,
        Maxcount,
        SsatABC,
        DCSSAT,
        BAXMC
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
enum Tool {
    C2D,
    Dsharp,
    D4 {
        relax: usize,
        proj: bool,
        bfs: bool,
        relax_exact: bool,
        simplification_options: SimplificationOptions,
        lower_bound_type: LowerBoundQuality,
    },
    Maxcount {
        k: usize,
    },
    SsatABC,
    DCSSAT,
    BAXMC{opts: String},
    BruteForce,
    Unconstrained {
        tool: Box<Tool>,
        lower_bound_type: LowerBoundQuality,
        exact: bool,
    },
}

impl Tool {
    fn count_models(&self, input: impl Input) -> anyhow::Result<ModelCount> {
        match *self {
            Tool::C2D => Compacter(c2d::C2D).model_count(input, None),
            Tool::Dsharp => Compacter(dsharp::Dsharp).model_count(input, None),
            Tool::Maxcount { .. } => anyhow::bail!("maxcount does not support model counting"),
            Tool::SsatABC { .. } => anyhow::bail!("ssatabc does not support model counting"),
            Tool::DCSSAT { .. } => anyhow::bail!("dcssat does not support model counting"),
            Tool::BAXMC { .. } => anyhow::bail!("baxmc does not support model counting"),
            Tool::D4 {
                relax,
                bfs,
                relax_exact,
                simplification_options,
                lower_bound_type,
                proj: true,
            } => Compacter(d4::D4 {
                relax,
                bfs,
                relax_exact,
                simplification_options,
                lower_bound_type,
            })
            .model_count(input, None),
            Tool::D4 {
                relax,
                bfs,
                relax_exact,
                simplification_options,
                lower_bound_type,
                proj: false,
            } => NoProjection(Compacter(d4::D4 {
                relax,
                bfs,
                relax_exact,
                simplification_options,
                lower_bound_type,
            }))
            .model_count(input, None),
            Tool::BruteForce => Compacter(cnf::BruteForce).model_count(input, None),
            Tool::Unconstrained { ref tool, .. } => tool.count_models(input),
        }
    }
    fn projmc(
        &self,
        input: impl Input,
        projected_vars: &FixedBitSet,
    ) -> anyhow::Result<ModelCount> {
        match self {
            &Tool::D4 {
                relax,
                bfs,
                relax_exact,
                simplification_options,
                proj: false,
                lower_bound_type: relax_fast_bounds,
            } => (d4::D4 {
                relax,
                bfs,
                relax_exact,
                simplification_options,
                lower_bound_type: relax_fast_bounds,
            })
            .projected_model_count(input, projected_vars),
            Tool::BruteForce => cnf::BruteForce.projected_model_count(input, projected_vars),
            x => anyhow::bail!("{:?} does not support projected model counting", x),
        }
    }
    fn popcon(
        &self,
        input: impl Input,
        controlled: &FixedBitSet,
        uncontrolled: &FixedBitSet
    ) -> anyhow::Result<(Model, PopConBounds)> {
        match self {
            &Tool::C2D => anyhow::bail!("c2d cannot do popcon"),
            &Tool::Dsharp => Compacter(dsharp::Dsharp).popularity_contest(input, controlled, uncontrolled),
            &Tool::Maxcount { k } => {
                Compacter(maxcount::Maxcount { k }).popularity_contest(input, controlled, uncontrolled)
            }
            &Tool::SsatABC => Compacter(ssatabc::SsatABC).popularity_contest(input, controlled, uncontrolled),
            &Tool::DCSSAT => Compacter(dcssat::DCSSAT).popularity_contest(input, controlled, uncontrolled),
            &Tool::BAXMC{ref opts} => Compacter(baxmc::BAXMC{opts: baxmc::Options::create(opts.clone())}).popularity_contest(input, controlled, uncontrolled),
            &Tool::D4 {
                relax,
                bfs,
                relax_exact,
                simplification_options,
                proj: false,
                lower_bound_type: relax_fast_bounds,
            } => Compacter(d4::D4 {
                relax,
                bfs,
                relax_exact,
                simplification_options,
                lower_bound_type: relax_fast_bounds,
            })
            .popularity_contest(input, controlled, uncontrolled),
            &Tool::D4 {
                relax,
                bfs,
                relax_exact,
                simplification_options,
                lower_bound_type: relax_fast_bounds,
                proj: true,
            } => NoProjection(Compacter(d4::D4 {
                relax,
                bfs,
                relax_exact,
                simplification_options,
                lower_bound_type: relax_fast_bounds,
            }))
            .popularity_contest(input, controlled, uncontrolled),
            &Tool::BruteForce => Compacter(cnf::BruteForce).popularity_contest(input, controlled, uncontrolled),
            &Tool::Unconstrained {
                ref tool,
                lower_bound_type,
                exact,
            } => {
                if exact {
                    match tool.as_ref() {
                        Tool::C2D => {
                            Darwiche { compiler: c2d::C2D }.popularity_contest(input, controlled, uncontrolled)
                        }
                        Tool::Dsharp => Darwiche {
                            compiler: dsharp::Dsharp,
                        }
                        .popularity_contest(input, controlled, uncontrolled),
                        Tool::D4 { .. } => Darwiche {
                            compiler: d4::D4::direct(),
                        }
                        .popularity_contest(input, controlled, uncontrolled),
                        _ => anyhow::bail!(
                            "cannot obtain ddnnf from {:?} (implied by --unconstrained",
                            tool
                        ),
                    }
                } else {
                    match tool.as_ref() {
                        Tool::C2D => UnconstrainedDdnnfBounds {
                            compiler: crate::c2d::C2D,
                            lower_bound_type,
                        }
                        .popularity_contest(input, controlled, uncontrolled),
                        Tool::Dsharp => UnconstrainedDdnnfBounds {
                            compiler: crate::dsharp::Dsharp,
                            lower_bound_type,
                        }
                        .popularity_contest(input, controlled, uncontrolled),
                        Tool::D4 { .. } => UnconstrainedDdnnfBounds {
                            compiler: crate::d4::D4::direct(),
                            lower_bound_type,
                        }
                        .popularity_contest(input, controlled, uncontrolled),
                        _ => anyhow::bail!(
                            "cannot obtain ddnnf from {:?} (implied by --unconstrained",
                            tool
                        ),
                    }
                }
            }
        }
    }
}

arg_enum! {
    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    enum Goal {
        ModelCount,
        PopCon,
        ProjectedModelCount,
    }
}

fn parse_duration(txt: &str) -> anyhow::Result<Duration> {
    let n = txt.parse()?;
    Ok(Duration::milliseconds(n))
}

#[derive(Debug)]
/// Writes the result in json to a file.
pub struct ResultWriter {
    file: RefCell<File>,
    path: PathBuf,
}

impl ResultWriter {
    fn write<R: serde::Serialize>(&self, result: &R) -> anyhow::Result<()> {
        serde_json::to_writer_pretty(self.file.borrow_mut().deref_mut(), result)
            .with_context(|| format!("writing result to {}", self.path.display()))
    }
}

impl From<&std::ffi::OsStr> for ResultWriter {
    fn from(path: &std::ffi::OsStr) -> ResultWriter {
        let path: PathBuf = path.into();
        let file = match File::create(&path) {
            Ok(f) => RefCell::new(f),
            Err(e) => {
                tracing::error!(
                    "failed to open {} to write results (--json option): {}",
                    path.display(),
                    e
                );
                std::process::exit(1);
            }
        };
        ResultWriter { path, file }
    }
}

const POSSIBLE_LOWER_BOUNDS: &'static [&'static str] = &["bad", "fast", "precise"];

/// Configuration options
#[derive(Debug, StructOpt)]
#[structopt(
    name = "popcon",
    about = "Determines the most popular controlled input"
)]
pub struct Opt {
    #[structopt(possible_values = &ToolName::variants(), case_insensitive = true, default_value="d4", short, long)]
    /// What CNF to DNNF converter to use
    tool: ToolName,

    #[structopt(possible_values = POSSIBLE_LOWER_BOUNDS, case_insensitive = true, default_value="fast", short, long)]
    /// How to compute lower bounds.
    ///
    /// For bad, the reported assignment is not a witness of the
    /// reported popularity.
    /// For precise, bound is quadratic.
    lower_bound: String,

    #[structopt(possible_values = &Goal::variants(), case_insensitive = true, default_value="modelcount", short, long)]
    /// What to compute with the formula
    goal: Goal,

    #[structopt(short, long)]
    /// Controlled variables, only for CNF input.
    controlled: Vec<usize>,

    #[structopt(short, long)]
    /// Uncontrolled variables, only for CNF input with BaxMC.
    uncontrolled: Vec<usize>,

    #[structopt(short, long)]
    /// Projected variables, only for CNF input.
    projected: Vec<usize>,

    /// Input file, must end with .cnf or .nnf or .smt2
    #[structopt(parse(from_os_str))]
    input: PathBuf,

    /// JSON output to the specified file (models are omitted)
    #[structopt(short, long, parse(from_os_str))]
    json: Option<ResultWriter>,

    /// If true, project out tseytin variables. Only affects d4 on model counting
    #[structopt(long)]
    project_tseytin: bool,

    /// Relax controlled decisions to up to n uncontrolled variables. Default 0.
    #[structopt(long)]
    relax: Option<usize>,

    /// Choose relaxation variables by bfs instead of dfs
    #[structopt(long)]
    relax_bfs: bool,

    /// Rerun simplified relaxed formula trough d4 to get an exact result
    #[structopt(long)]
    relax_exact: bool,

    /// Use a branch and bound algo to obtain an exact result with --unconstrained
    #[structopt(long)]
    unconstrained_exact: bool,

    /// Timeout, in milliseconds. Return code is 42 on timeout.
    #[structopt(short="T", long, parse(try_from_str = parse_duration))]
    timeout: Option<Duration>,

    /// Enable debug output and persist temporary files for inspection.
    #[structopt(short, long)]
    debug: bool,

    /// Output some stats to this file
    #[structopt(long, parse(from_os_str))]
    stats: Option<PathBuf>,

    /// Disable simplification of relaxed d-DNNF by rewriting
    #[structopt(long)]
    simplify_no_rewriting: bool,

    /// Disable simplification of relaxed d-DNNF by oriented vars detection
    #[structopt(long)]
    simplify_no_oriented_vars: bool,

    /// Disable simplification of relaxed d-DNNF by useless relaxed vars detection
    #[structopt(long)]
    simplify_no_useless_relaxed_vars: bool,

    /// Disable simplification of relaxed d-DNNF by bounds
    #[structopt(long)]
    simplify_no_bounds: bool,

    /// Number of copies of formula. Required parameter when using maxcount.
    #[structopt(long)]
    maxcount_k: Option<usize>,

    /// Options for BaxMC.
    #[structopt(default_value = "", long)]
    baxmc_opts: String,

    /// Unconstrained ddnnf
    #[structopt(long)]
    unconstrained: bool,
}

impl Opt {
    fn tool(&self) -> anyhow::Result<Tool> {
        let lower_bound_type = match self.lower_bound.as_str() {
            "bad" => LowerBoundQuality::Bad,
            "precise" => LowerBoundQuality::Precise,
            "fast" => LowerBoundQuality::Fast,
            x => unreachable!("unknown lower bound {}", x),
        };
        let tool = match self.tool {
            ToolName::C2D => Tool::C2D,
            ToolName::Dsharp => Tool::Dsharp,
            ToolName::SsatABC => Tool::SsatABC,
            ToolName::DCSSAT => Tool::DCSSAT,
            ToolName::BAXMC => {
                Tool::BAXMC{opts: self.baxmc_opts.clone()}
            },
            ToolName::Maxcount => {
                let k = match self.maxcount_k {
                    Some(k) => k,
                    None => anyhow::bail!("--maxcount-k required when using --tool maxcount"),
                };
                Tool::Maxcount { k }
            }
            ToolName::D4 => Tool::D4 {
                relax: self.relax.unwrap_or(0),
                relax_exact: self.relax_exact,
                proj: self.project_tseytin,
                bfs: self.relax_bfs,
                simplification_options: SimplificationOptions {
                    rewriting: !self.simplify_no_rewriting,
                    oriented_vars: !self.simplify_no_oriented_vars,
                    simplify_by_bounds: !self.simplify_no_bounds,
                    useless_relaxed_vars: !self.simplify_no_useless_relaxed_vars,
                },
                lower_bound_type,
            },
            ToolName::BruteForce => Tool::BruteForce,
        };
        let tool = if self.unconstrained {
            anyhow::ensure!(
                matches!(tool, Tool::C2D | Tool::Dsharp | Tool::D4 { .. }),
                "tool {:?} cannot create unconstrained ddnnf",
                tool
            );
            Tool::Unconstrained {
                tool: Box::new(tool),
                lower_bound_type,
                exact: self.unconstrained_exact,
            }
        } else {
            tool
        };
        Ok(tool)
    }
}

/// Either does a model count or popularity contest
fn process_ddnnf<S: ddnnf::Source>(
    formula: S,
    opt: &Opt,
    controlled_vars: FixedBitSet,
    projected_vars: FixedBitSet,
) -> anyhow::Result<()> {
    match opt.goal {
        Goal::PopCon => anyhow::ensure!(
            opt.controlled.len() > 0,
            "no controlled variable for popularity contest"
        ),
        _ => anyhow::ensure!(
            opt.controlled.len() == 0,
            "controlled variables are only accepted in popularity contest mode."
        ),
    }
    match opt.goal {
        Goal::ProjectedModelCount => anyhow::ensure!(
            opt.projected.len() > 0,
            "no projected variable for projected model count"
        ),
        _ => anyhow::ensure!(
            opt.controlled.len() == 0,
            "projected variables are only accepted in projected model counting mode."
        ),
    };
    match opt.goal {
        Goal::ModelCount => {
            let count = ddnnf::count_models(formula, None)
                .with_context(|| format!("processing input formula {}", opt.input.display()))?;
            if let Some(writer) = &opt.json {
                writer.write(&count)?;
            } else {
                println!("Formula has {} models", count);
            }
        }
        Goal::ProjectedModelCount => {
            let count = ddnnf::count_models(formula, Some(&projected_vars))
                .with_context(|| format!("processing input formula {}", opt.input.display()))?;
            if let Some(writer) = &opt.json {
                writer.write(&count)?;
            } else {
                println!("Formula has {} projected models", count);
            }
        }
        Goal::PopCon => {
            if opt.tool()? == Tool::C2D {
                eprintln!("C2D does not support priority variables, result will be wrong.");
            }
            let (model, count) = ddnnf::most_popular_model(formula, controlled_vars)
                .with_context(|| format!("processing input formula {}", opt.input.display()))?;
            if let Some(writer) = &opt.json {
                writer.write(&count)?;
            } else {
                println!("Most popular model: {:?} with {} models", model, count);
            }
        }
    }
    Ok(())
}

fn setup_tracing(opt: &Opt) -> anyhow::Result<Option<stats::StatsLayer>> {
    use tracing::Level;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::prelude::*;
    let min_level = if opt.debug { Level::TRACE } else { Level::INFO };
    let fmt_layer = tracing_subscriber::fmt::layer().with_filter(
        tracing_subscriber::filter::filter_fn(move |metadata| *metadata.level() <= min_level),
    );
    let subscriber = tracing_subscriber::Registry::default().with(fmt_layer);
    #[cfg(feature = "tracy")]
    let subscriber = subscriber.with(tracing_tracy::TracyLayer::new());
    let subscriber_with_stats: Box<dyn tracing::Subscriber + 'static + Send + Sync>;
    let stats_layer = match opt.stats.as_ref() {
        Some(path) => {
            let file = File::create(path)
                .with_context(|| format!("cannot open {} for writing stats", path.display()))?;
            let stats_layer = stats::StatsLayer::new(file);
            subscriber_with_stats = Box::new(subscriber.with(stats_layer.clone()));
            Some(stats_layer)
        }
        None => {
            subscriber_with_stats = Box::new(subscriber);
            None
        }
    };
    tracing::subscriber::set_global_default(subscriber_with_stats)
        .context("setting default tracing collector")?;
    Ok(stats_layer)
}

/// entrypoint of the binary
pub fn run() -> anyhow::Result<()> {
    let opt = Opt::from_args();
    let start_time = Instant::now();
    let stats = setup_tracing(&opt)?;
    let timer_stats = stats.clone();
    let ext = opt.input.extension().unwrap_or(std::ffi::OsStr::new(""));
    let mut controlled_vars: FixedBitSet = opt.controlled.iter().map(|&x| x - 1).collect();
    let mut uncontrolled_vars : FixedBitSet = opt.uncontrolled.iter().map(|&x| x - 1).collect();
    let mut projected_vars: FixedBitSet = opt.projected.iter().map(|&x| x - 1).collect();
    let global_stats = move || {
        let mut usage = nc::rusage_t::default();
        nc::getrusage(nc::RUSAGE_SELF, &mut usage).unwrap_or_else(|e| {
            tracing::warn!("failed to get own memory usage: {}", e);
        });
        let max_rss_self_kb = usage.ru_maxrss;
        usage = nc::rusage_t::default();
        // max rss of the largest child, so presumably the solver...
        nc::getrusage(nc::RUSAGE_CHILDREN, &mut usage).unwrap_or_else(|e| {
            tracing::warn!("failed to get children memory usage: {}", e);
        });
        let max_rss_solver_kb = usage.ru_maxrss;
        tracing::trace!(
            wall_time_ms = start_time.elapsed().as_millis() as u64,
            max_rss_self_kb,
            max_rss_solver_kb,
            stats = true
        );
    };
    let timer = opt.timeout.map(|duration| {
        let timer = timer::Timer::new();
        let giveup = move || {
            global_stats();
            tracing::warn!(timeout = true, stats = true);
            timer_stats.clone().map(|s| s.dump());
            std::process::exit(42)
        };
        // the guard returned by schedule_with_delay must be ignored otherwise it is cancelled on
        // drop
        timer.schedule_with_delay(duration, giveup).ignore();
        // the timer must not be dropped, otherwise everything is cancelled
        timer
    });
    if ext == "nnf" {
        let read = File::open(&opt.input)
            .with_context(|| format!("opening input NNF file {}", opt.input.display()))?;
        let formula = ddnnf::FileSource::new(read);
        process_ddnnf(formula, &opt, controlled_vars, projected_vars)?;
    } else if ext == "cnf" {
        match opt.goal {
            Goal::ModelCount => {
                let count = opt.tool()?.count_models(opt.input.as_path())?;
                if let Some(writer) = opt.json {
                    writer.write(&count)?;
                } else {
                    println!("Formula has {} models", count);
                }
            }
            Goal::ProjectedModelCount => {
                let file = std::fs::File::open(&opt.input)
                    .with_context(|| format!("opening input file {}", opt.input.display()))?;
                let h = cnf::read_header_only(file)
                    .with_context(|| format!("parsing CNF header of {}", opt.input.display()))?;
                anyhow::ensure!(
                    projected_vars.count_ones(..) != 0,
                    "no projected variables declared with -p"
                );
                projected_vars.grow(h.var_count);
                let count = opt.tool()?.projmc(opt.input.as_path(), &projected_vars)?;
                if let Some(writer) = opt.json {
                    writer.write(&count)?;
                } else {
                    println!("Projected model count: {}", count);
                }
            }
            Goal::PopCon => {
                let file = std::fs::File::open(&opt.input)
                    .with_context(|| format!("opening input file {}", opt.input.display()))?;
                let h = cnf::read_header_only(file)
                    .with_context(|| format!("parsing CNF header of {}", opt.input.display()))?;
                anyhow::ensure!(
                    controlled_vars.count_ones(..) != 0,
                    "no controlled variables declared with -c"
                );
                controlled_vars.grow(h.var_count);
                uncontrolled_vars.grow(h.var_count);
                let (model, count) = opt.tool()?.popcon(opt.input.as_path(), &controlled_vars, &uncontrolled_vars)?;
                tracing::trace!(
                    up_to_low_ratio_percent = (100. * count.up_to_low_ratio()) as u64,
                    stats = true,
                    "popcoun bounds: {:?}",
                    &count,
                );
                if let Some(writer) = opt.json {
                    writer.write(&count)?;
                } else {
                    println!("Most popular model: {:?} with {} models", model, count);
                }
            }
        };
    } else if ext == "smt2" {
        anyhow::ensure!(controlled_vars.count_ones(..)==0, "don't use -c with smt2 input. Label controlled variables with `(set-info :controlled varname)` inside the file instead.");
        anyhow::ensure!(projected_vars.count_ones(..)==0, "don't use -p with smt2 input. Label projected variables with `(set-info :projected varname)` inside the file instead.");
        match opt.goal {
            Goal::PopCon => {
                let res = match opt.tool()? {
                    Tool::C2D => anyhow::bail!("cannot use c2d for popcon"),
                    Tool::Dsharp => {
                        smt::popularity_contest(&Compacter(dsharp::Dsharp), opt.input.as_path())
                    }
                    Tool::SsatABC => {
                        smt::popularity_contest(&Compacter(ssatabc::SsatABC), opt.input.as_path())
                    }
                    Tool::DCSSAT => {
                        smt::popularity_contest(&Compacter(dcssat::DCSSAT), opt.input.as_path())
                    }
                    Tool::BAXMC{opts} => {
                        smt::popularity_contest(&Compacter(baxmc::BAXMC{opts: baxmc::Options::create(opts)}), opt.input.as_path())
                    }
                    Tool::Maxcount { k } => smt::popularity_contest(
                        &Compacter(maxcount::Maxcount { k }),
                        opt.input.as_path(),
                    ),
                    Tool::D4 {
                        relax,
                        bfs,
                        relax_exact,
                        simplification_options,
                        proj: true,
                        lower_bound_type: relax_fast_bounds,
                    } => smt::popularity_contest(
                        &Compacter(d4::D4 {
                            relax,
                            bfs,
                            simplification_options,
                            relax_exact,
                            lower_bound_type: relax_fast_bounds,
                        }),
                        opt.input.as_path(),
                    ),
                    Tool::D4 {
                        relax,
                        bfs,
                        relax_exact,
                        lower_bound_type: relax_fast_bounds,
                        simplification_options,
                        proj: false,
                    } => smt::popularity_contest(
                        &NoProjection(Compacter(d4::D4 {
                            relax,
                            bfs,
                            relax_exact,
                            lower_bound_type: relax_fast_bounds,
                            simplification_options,
                        })),
                        opt.input.as_path(),
                    ),
                    Tool::BruteForce => {
                        smt::popularity_contest(&Compacter(cnf::BruteForce), opt.input.as_path())
                    }
                    Tool::Unconstrained {
                        ref tool,
                        lower_bound_type,
                        exact,
                    } => {
                        if exact {
                            match tool.as_ref() {
                                Tool::C2D => smt::popularity_contest(
                                    &Darwiche { compiler: c2d::C2D },
                                    opt.input.as_path(),
                                ),
                                Tool::Dsharp => smt::popularity_contest(
                                    &Darwiche {
                                        compiler: dsharp::Dsharp,
                                    },
                                    opt.input.as_path(),
                                ),
                                Tool::D4 { .. } => smt::popularity_contest(
                                    &Darwiche {
                                        compiler: d4::D4::direct(),
                                    },
                                    opt.input.as_path(),
                                ),
                                _ => {
                                    anyhow::bail!("{:?} cannot be used with --unconstrained", tool)
                                }
                            }
                        } else {
                            match tool.as_ref() {
                                Tool::C2D => smt::popularity_contest(
                                    &UnconstrainedDdnnfBounds {
                                        compiler: c2d::C2D,
                                        lower_bound_type,
                                    },
                                    opt.input.as_path(),
                                ),
                                Tool::Dsharp => smt::popularity_contest(
                                    &UnconstrainedDdnnfBounds {
                                        compiler: dsharp::Dsharp,
                                        lower_bound_type,
                                    },
                                    opt.input.as_path(),
                                ),
                                Tool::D4 { .. } => smt::popularity_contest(
                                    &UnconstrainedDdnnfBounds {
                                        compiler: d4::D4::direct(),
                                        lower_bound_type,
                                    },
                                    opt.input.as_path(),
                                ),
                                _ => {
                                    anyhow::bail!("{:?} cannot be used with --unconstrained", tool)
                                }
                            }
                        }
                    }
                };
                let (model, count) =
                    res.with_context(|| format!("popularity contest of {}", opt.input.display()))?;
                tracing::trace!(
                    up_to_low_ratio_percent = (100. * count.up_to_low_ratio()) as u64,
                    stats = true,
                    "popcoun bounds: {:?}",
                    &count,
                );
                if let Some(writer) = opt.json {
                    writer.write(&count)?;
                } else {
                    println!("Max popularity: {}", count);
                    if count.lower() == count.upper() {
                        println!("Best model: {}", model);
                    } else {
                        println!("Lower bound reached by: {}", model);
                    }
                }
            }
            Goal::ProjectedModelCount => {
                let res = match opt.tool()? {
                    Tool::D4 {
                        relax,
                        bfs,
                        relax_exact,
                        simplification_options,
                        lower_bound_type: relax_fast_bounds,
                        proj: _,
                    } => smt::projected_model_count(
                        &d4::D4 {
                            relax,
                            bfs,
                            relax_exact,
                            simplification_options,
                            lower_bound_type: relax_fast_bounds,
                        },
                        opt.input.as_path(),
                    ),
                    Tool::BruteForce => {
                        smt::projected_model_count(&cnf::BruteForce, opt.input.as_path())
                    }
                    t => anyhow::bail!("{:?} does not support projected model counting", t),
                };
                let res = res.with_context(|| {
                    format!("counting projected models of {}", opt.input.display())
                })?;
                anyhow::ensure!(
                    res.model_count.bits() <= 1 + res.bits as u64,
                    "model count has more bits than variable size {}",
                    res
                );
                if let Some(writer) = opt.json {
                    writer.write(&res)?;
                } else {
                    println!("{}", res);
                }
            }
            Goal::ModelCount => {
                let res = match opt.tool()? {
                    Tool::C2D => smt::model_count(&Compacter(c2d::C2D), opt.input.as_path()),
                    Tool::Maxcount { .. } => {
                        anyhow::bail!("maxcount cannot be used for model counting")
                    }
                    Tool::SsatABC => {
                        anyhow::bail!("ssatabc cannot be used for model counting")
                    }
                    Tool::DCSSAT => {
                        anyhow::bail!("dcssat cannot be used for model counting")
                    }
                    Tool::BAXMC {..} => {
                        anyhow::bail!("baxmc cannot be used for model counting")
                    }
                    Tool::Unconstrained { .. } => {
                        anyhow::bail!("--unconstrained cannot be used with model counting")
                    }
                    Tool::Dsharp => {
                        smt::model_count(&Compacter(dsharp::Dsharp), opt.input.as_path())
                    }
                    Tool::D4 {
                        relax,
                        bfs,
                        relax_exact,
                        simplification_options,
                        lower_bound_type: relax_fast_bounds,
                        proj: true,
                    } => smt::model_count(
                        &Compacter(d4::D4 {
                            relax,
                            bfs,
                            relax_exact,
                            simplification_options,
                            lower_bound_type: relax_fast_bounds,
                        }),
                        opt.input.as_path(),
                    ),
                    Tool::D4 {
                        relax,
                        bfs,
                        relax_exact,
                        simplification_options,
                        lower_bound_type: relax_fast_bounds,
                        proj: false,
                    } => smt::model_count(
                        &NoProjection(Compacter(d4::D4 {
                            relax,
                            bfs,
                            relax_exact,
                            simplification_options,
                            lower_bound_type: relax_fast_bounds,
                        })),
                        opt.input.as_path(),
                    ),
                    Tool::BruteForce => {
                        smt::model_count(&Compacter(cnf::BruteForce), opt.input.as_path())
                    }
                };
                let res =
                    res.with_context(|| format!("counting models of {}", opt.input.display()))?;
                anyhow::ensure!(
                    res.model_count.bits() <= 1 + res.bits as u64,
                    "model count has more bits than variable size {}",
                    res
                );
                if let Some(writer) = opt.json {
                    writer.write(&res)?;
                } else {
                    println!("{}", res);
                }
            }
        }
    } else {
        anyhow::bail!("don't know what to do with {}", opt.input.display())
    }
    drop(timer);
    global_stats();
    tracing::trace!(timeout = false, stats = true);
    stats.map(|s| s.dump());
    Ok(())
}

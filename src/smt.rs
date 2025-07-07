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

//! Processing of SMT files

use crate::aig2cnf::BitSort;
use crate::cnf::{CNFCounter, CNFPopularityContest, CNFProjectedCounter, ModelCount, PopConBounds};
use crate::utils::Input;
use anyhow::Context;
use fixedbitset::FixedBitSet;
use num_bigint::BigUint;
use num_traits::One;
use num_traits::ToPrimitive;
use smt2parser::concrete::{Command, Sort};
use smt2parser::visitors::{AttributeValue, Identifier, Index};
use smt2parser::CommandStream;
use std::collections::BTreeMap;
use std::iter::FromIterator;
use std::path::Path;
use tracing::{span, trace, trace_span, Level};

/// Sorts for supported variables
#[derive(Debug, Clone, Copy, PartialEq, Eq, Ord, PartialOrd)]
pub enum VarSort {
    /// boolean variable
    Bool,
    /// Bitvector variable, with its size in bits
    BitVector {
        /// size of the bitvector in bits
        size: usize,
    },
}

impl VarSort {
    /// number of bits in this sort
    fn size(&self) -> usize {
        match self {
            VarSort::Bool => 1,
            VarSort::BitVector { size } => *size,
        }
    }
}

/// tracks infos about a SMT variable
#[derive(Debug, Clone, Copy, PartialEq, Eq, Ord, PartialOrd)]
pub struct VarInfo {
    /// Sort of the variable
    pub sort: VarSort,
    /// Whether the variable is controlled
    pub controlled: bool,
    /// Whether the variable is to be projected against
    pub projected: bool,
}

/// What we know about a variable while parsing.
///
/// Since it is allowed for a variable to be declared or set-info :controlled in any order,
/// the sort is an option.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Ord, PartialOrd, Default)]
struct PartialVarInfo {
    sort: Option<VarSort>,
    controlled: bool,
    projected: bool,
}

impl PartialVarInfo {
    /// To be called when the file is fully parsed. At this point it is an error
    /// to have a variable without a sort (undeclared) but set-info :controlled
    fn unwrap(self) -> anyhow::Result<VarInfo> {
        let PartialVarInfo {
            sort,
            controlled,
            projected,
        } = self;
        match sort {
            Some(sort) => Ok(VarInfo {
                sort,
                controlled,
                projected,
            }),
            None => {
                anyhow::bail!("variable is labelled as controlled or projected but not declared.")
            }
        }
    }
}

/// Parses the SMT2 file to list its variables
pub fn variables(input: impl Input) -> anyhow::Result<BTreeMap<String, VarInfo>> {
    let read = input.to_read()?;
    let bufread = std::io::BufReader::new(read);
    let commands = CommandStream::new(bufread, smt2parser::concrete::SyntaxBuilder);
    let mut res: BTreeMap<String, PartialVarInfo> = BTreeMap::new();
    for command in commands {
        let command = match command {
            Ok(c) => c,
            Err(position) => {
                let p = std::path::PathBuf::from(input.display().to_string());
                anyhow::bail!("SMT2 parsing failure at {}", position.location_in_file(&p))
            }
        };
        match command {
            Command::SetInfo { keyword, value } if keyword.0 == "controlled" || keyword.0 == "projected" => {
                match value {
                    AttributeValue::Symbol(s) => {
                        let entry = res.entry(s.0).or_default();
                        match keyword.0.as_str() {
                            "controlled" => entry.controlled = true,
                            "projected" => entry.projected = true,
                            _ => unreachable!()
                        };
                    },
                    x => anyhow::bail!("invalid (set-info :{keyword} {:?}), expected (set-info :{keyword} thevariablename)", x, keyword=&keyword.0)
                }
            },
            Command::DeclareFun {
                symbol,
                parameters,
                sort,
            } if !parameters.is_empty() => {
                anyhow::bail!(
                    "{} symbol {} is a function, this is unsupported",
                    sort,
                    symbol
                )
            }
            Command::DeclareFun { symbol, sort, .. } | Command::DeclareConst { symbol, sort } => {
                match sort.clone() {
                    Sort::Simple {
                        identifier: Identifier::Simple { symbol: sort_name },
                    } if sort_name.0 == "Bool" => {
                        res.entry(symbol.0).or_default().sort = Some(VarSort::Bool);
                    }
                    Sort::Simple {
                        identifier:
                            Identifier::Indexed {
                                symbol: sort_name,
                                indices,
                            },
                    } if sort_name.0 == "BitVec" => {
                        anyhow::ensure!(
                            indices.len() == 1,
                            "bitvector with several sizes: {} {}",
                            symbol,
                            sort
                        );
                        let size = match &indices[0] {
                            Index::Numeral(s) => {
                                s.to_usize().context("bitvector size is absurdly high")?
                            }
                            _ => anyhow::bail!("bitvector with symbolic size: {} {}", symbol, sort),
                        };
                        res.entry(symbol.0).or_default().sort = Some(VarSort::BitVector { size });
                    }
                    _ => {
                        anyhow::bail!(
                            "Symbol {} has unexpected sort {}, only bv and bool are supported",
                            symbol,
                            sort
                        );
                    }
                }
            }
            _ => {}
        }
    }
    let mut real_res = BTreeMap::new();
    for (symbol, info) in res.into_iter() {
        let info = info
            .unwrap()
            .with_context(|| format!("parsing variable {}", &symbol))?;
        real_res.insert(symbol, info);
    }
    Ok(real_res)
}

#[test]
fn test_parse_controlled_variables() {
    let v = variables("assets/choose_interval.smt2".as_ref() as &Path).expect("variables");
    assert_eq!(v.len(), 2, "file has 2 variables");
    assert_eq!(
        v["thevar"],
        VarInfo {
            sort: VarSort::BitVector { size: 8 },
            controlled: false,
            projected: false,
        }
    );
    assert_eq!(
        v["thechoice"],
        VarInfo {
            sort: VarSort::Bool,
            controlled: true,
            projected: false,
        }
    );
}

#[test]
fn test_parse_projected_variables() {
    let v =
        variables("assets/choose_interval_projection.smt2".as_ref() as &Path).expect("variables");
    assert_eq!(v.len(), 2, "file has 2 variables");
    assert_eq!(
        v["thevar"],
        VarInfo {
            sort: VarSort::BitVector { size: 8 },
            controlled: false,
            projected: true,
        }
    );
    assert_eq!(
        v["thechoice"],
        VarInfo {
            sort: VarSort::Bool,
            controlled: false,
            projected: false,
        }
    );
}

/// Counts the number of models of a SMT2 file, parametrized by how to count the bitblasted CNF
/// formula.
/// # Example
/// ```
/// use popcon::dsharp::Dsharp;
/// use popcon::smt::model_count;
/// use popcon::cnf::ModelCount;
/// use std::path::Path;
/// assert_eq!(model_count(&Dsharp, Path::new("assets/interval.smt2")).unwrap(), ModelCount {
/// model_count: (0x234 - 0x123 -
/// 1u32).into(), bits: 12})
/// ```
pub fn model_count<T: CNFCounter, I: Input>(method: &T, input: I) -> anyhow::Result<ModelCount> {
    let variables = variables(input)
        .with_context(|| format!("enumerating variables of {}", input.display()))?;
    let mut write = Vec::new();
    let span = span!(Level::TRACE, "bitblasting", file = %input.display(), timing=true).entered();
    let (header, symbol_table) = crate::bitblast::smt2cnf(input, &mut write)
        .with_context(|| format!("bit blasting {}", input.display()))?;
    drop(span);
    let span =
        span!( Level::TRACE, "cnf_model_counting", file = %input.display(), timing=true).entered();
    let mut projection_variables =
        FixedBitSet::from_iter(symbol_table.keys().map(|v| v.index() as usize));
    projection_variables.grow(header.var_count);
    let mc = method
        .model_count(&write as &[u8], Some(&projection_variables))
        .with_context(|| format!("counting models on bit blasting of {}", input.display()))?;
    drop(span);
    let cnf_bits = symbol_table.len();
    let smt_bits: usize = variables.values().map(|info| info.sort.size()).sum();
    trace!(?mc, cnf_bits, smt_bits, "result of cnf model counting");
    anyhow::ensure!(
        mc.model_count.bits() <= 1 + cnf_bits as u64,
        "cnf model count has more bits {} than original aiger variables {}",
        mc.model_count.bits(),
        cnf_bits
    );
    let model_count = mc.model_count * (BigUint::one() << (smt_bits - cnf_bits));
    Ok(ModelCount {
        model_count,
        bits: smt_bits,
    })
}

#[test]
fn slice_elimination() {
    let mc = model_count(
        &crate::dsharp::Dsharp,
        "assets/slice_elimination.smt2".as_ref() as &Path,
    )
    .unwrap();
    assert_eq!(
        mc,
        ModelCount {
            model_count: 8u32.into(),
            bits: 4
        }
    )
}

#[test]
fn fails_with_rwl2() {
    let (_best, mc) = popularity_contest(
        &crate::d4::D4::direct(),
        "assets/fails_with_rwl2.smt2".as_ref() as &Path,
    )
    .unwrap();
    assert_eq!(
        mc,
        ModelCount {
            model_count: 1u32.into(),
            bits: 1
        }
    )
}

// #[test]
#[allow(unused)]
fn regression_dsharp_issue_12() {
    // yields Err` value: cnf model count has more bits 46 than original aiger variables 41'
    // because of https://github.com/QuMuLab/dsharp/issues/12
    let mc = model_count(
        &crate::dsharp::Dsharp,
        "assets/bug_more_models_search_space.smt2".as_ref() as &Path,
    )
    .unwrap();
    assert_eq!(
        mc,
        ModelCount {
            // obtained with c2d
            model_count: "16100732351324662208915539871377876262465542473293280954177915888073619654202350531388184199168".parse().unwrap(),
            bits: 313
        }
    )
}

/// Counts the number of models of the projected variables of a SMT2 file, parametrized by how to count the bitblasted CNF
/// formula. In other words, counts how many distinct assignments of the variables labelled as
/// projected leave the formula satisfiable. Variables are labelled as projected by the command
/// `(set-info :projected variablename)`
/// # Example
/// ```
/// use popcon::d4::D4;
/// use popcon::smt::projected_model_count;
/// use popcon::cnf::ModelCount;
/// use std::path::Path;
/// assert_eq!(projected_model_count(&D4::direct(), Path::new("assets/choose_interval_projection.smt2")).unwrap(), ModelCount {
/// model_count: 0x45u32.into(), bits: 8})
/// ```
pub fn projected_model_count<T: CNFProjectedCounter, I: Input>(
    method: &T,
    input: I,
) -> anyhow::Result<ModelCount> {
    let variables = variables(input)
        .with_context(|| format!("enumerating variables of {}", input.display()))?;
    anyhow::ensure!(
        variables.values().any(|v| v.projected),
        "No projected variable in {}",
        input.display()
    );
    let mut write = Vec::new();
    let span = span!(Level::TRACE, "bitblasting", file = %input.display(), timing=true).entered();
    let (header, symbol_table) = crate::bitblast::smt2cnf(input, &mut write)
        .with_context(|| format!("bit blasting {}", input.display()))?;
    drop(span);
    let span =
        span!( Level::TRACE, "cnf_projected_model_counting", file = %input.display(), timing=true)
            .entered();
    let mut projection_variables = FixedBitSet::with_capacity(header.var_count);
    for (cnf_var, correspondance) in symbol_table.iter() {
        let infos: anyhow::Result<&VarInfo> =
            variables.get(&correspondance.name).ok_or_else(|| {
                anyhow::anyhow!(
                    "no declaration info for bitblasted variable {}",
                    correspondance.name
                )
            });
        if infos?.projected {
            projection_variables.insert(cnf_var.index())
        }
    }
    let mc = method
        .projected_model_count(&write as &[u8], &projection_variables)
        .with_context(|| format!("counting models on bit blasting of {}", input.display()))?;
    drop(span);
    let cnf_bits = projection_variables.count_ones(..);
    let smt_bits: usize = variables
        .values()
        .filter(|info| info.projected)
        .map(|info| info.sort.size())
        .sum();
    trace!(
        ?mc,
        cnf_bits,
        smt_bits,
        "result of cnf model counting with projection"
    );
    anyhow::ensure!(
        mc.model_count.bits() <= 1 + cnf_bits as u64,
        "cnf model count has more bits {} than original aiger variables {}",
        mc.model_count.bits(),
        cnf_bits
    );
    let model_count = mc.model_count * (BigUint::one() << (smt_bits - cnf_bits));
    Ok(ModelCount {
        model_count,
        bits: smt_bits,
    })
}

/// The value of a variable for a model
///
/// Implements partial equality with bool and pairs (bitvector value as integer, size as usize).
/// Equality evaluates to true only when the bitvector only contains Some(...) bits.
/// ```rust
/// use popcon::smt::VarValue;
/// assert_eq!(VarValue::Bool(Some(true)), true);
/// assert_ne!(VarValue::Bool(Some(true)), false);
/// assert_ne!(VarValue::Bool(None), false);
/// assert_eq!(VarValue::BitVector(vec![Some(false), Some(false), Some(true)]), (4u8, 3));
/// assert_ne!(VarValue::BitVector(vec![Some(false), Some(false), Some(true)]), (4u8, 4));
/// assert_ne!(VarValue::BitVector(vec![None, Some(false), Some(true)]), (4u8, 3));
/// assert_eq!(VarValue::BitVector(vec![Some(true); 128]), (u128::max_value(), 128));
/// assert_eq!(VarValue::BitVector(vec![Some(true), Some(false), None]).to_string(), "0bX01");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Ord, PartialOrd)]
pub enum VarValue {
    /// A boolean. None if irrelevant to the model.
    Bool(Option<bool>),
    /// A bitvector. None bits are irrelevant to the model. Most significant bit last.
    BitVector(Vec<Option<bool>>),
}

impl std::fmt::Display for VarValue {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            VarValue::Bool(None) => write!(fmt, "X"),
            VarValue::Bool(Some(x)) => write!(fmt, "{}", *x as usize),
            VarValue::BitVector(v) => {
                write!(fmt, "0b")?;
                for x in v.iter().rev() {
                    match x {
                        None => write!(fmt, "X")?,
                        Some(x) => write!(fmt, "{}", *x as usize)?,
                    }
                }
                Ok(())
            }
        }
    }
}

impl PartialEq<bool> for VarValue {
    fn eq(&self, rhs: &bool) -> bool {
        match self {
            VarValue::Bool(Some(x)) => x == rhs,
            _ => false,
        }
    }
}

impl<T: Clone> PartialEq<(T, usize)> for VarValue
where
    T: Into<u128>,
{
    fn eq(&self, (value, size): &(T, usize)) -> bool {
        let rhs: u128 = value.clone().into();
        let rhs = rhs.to_le();
        match self {
            VarValue::BitVector(v) => {
                if v.len() != *size {
                    return false;
                }
                for i in 0..*size {
                    let bit = (rhs & 1 << i) == 1 << i;
                    if v[i] != Some(bit) {
                        return false;
                    }
                }
                true
            }
            _ => false,
        }
    }
}

/// A model for a SMT formula, mapping variable name to value
#[derive(Debug, Clone)]
pub struct SmtModel(pub BTreeMap<String, VarValue>);

impl std::fmt::Display for SmtModel {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        for (name, value) in self.0.iter() {
            write!(fmt, "{}: {},", name, value)?
        }
        Ok(())
    }
}

/// Finds a most popular controlled model of a SMT2 file, parametrized by how to count the bitblasted CNF
/// formula.
/// What variables are controlled must be specified in the smt2 file by the command `(set-info
/// :controlled thevar)`
/// # Example
/// ```
/// use popcon::dsharp::Dsharp;
/// use popcon::smt::popularity_contest;
/// use popcon::cnf::ModelCount;
/// use std::path::Path;
/// assert_eq!(popularity_contest(&Dsharp, Path::new("assets/choose_interval.smt2")).unwrap().1, ModelCount {
/// model_count: 0x45u32.into(), bits: 8})
/// ```
pub fn popularity_contest<T: CNFPopularityContest, I: Input>(
    method: &T,
    input: I,
) -> anyhow::Result<(SmtModel, PopConBounds)> {
    let variables = variables(input)
        .with_context(|| format!("enumerating variables of {}", input.display()))?;
    anyhow::ensure!(
        variables.values().any(|x| x.controlled),
        "{} contains no controlled variable.",
        input.display()
    );
    let mut bitblasted = Vec::new();
    let span = trace_span!("bitblasting", file = %input.display(), timing=true).entered();
    let (cnf_header, mut symbol_table) = crate::bitblast::smt2cnf(input, &mut bitblasted)
        .with_context(|| format!("bit blasting {}", input.display()))?;
    drop(span);
    let mut controlled_cnf_vars = FixedBitSet::with_capacity(cnf_header.var_count);
    let mut uncontrolled_cnf_vars = controlled_cnf_vars.clone();
    let mut bitblasted_uncontrolled_bits = 0;
    let mut smt_model: BTreeMap<String, VarValue> = BTreeMap::new();
    for (cnf_var, correspondance) in symbol_table.iter_mut() {
        let info = variables.get(&correspondance.name).with_context(|| {
            format!(
                "CNF variable {} corresponds to non existent SMT variable {} in file {}",
                cnf_var,
                &correspondance.name,
                input.display()
            )
        })?;
        correspondance
            .sort
            .unify_with_real_sort(info.sort)
            .with_context(|| format!("checking type of {}", &correspondance.name))?;
        if info.controlled {
            controlled_cnf_vars.insert(cnf_var.index());
            let empty_value = match info.sort {
                VarSort::Bool => VarValue::Bool(None),
                VarSort::BitVector { size } => VarValue::BitVector(vec![None; size]),
            };
            smt_model.insert(correspondance.name.to_owned(), empty_value);
        } else {
            uncontrolled_cnf_vars.insert(cnf_var.index());
            bitblasted_uncontrolled_bits += 1;
        }
    }
    let span = trace_span!("cnf_popcon", file = %input.display(), timing=true).entered();
    let (cnf_model, mut mc) = method
        .popularity_contest(&bitblasted as &[u8], &controlled_cnf_vars, &uncontrolled_cnf_vars)
        .with_context(|| format!("popularity_contest on bit blasting of {}", input.display()))?;
    drop(span);
    let smt_uncontrolled_bits: usize = variables
        .values()
        .map(|info| if info.controlled { 0 } else { info.sort.size() })
        .sum();
    let current_bits = mc.bits();
    mc = mc.upgrade_to(smt_uncontrolled_bits - bitblasted_uncontrolled_bits + current_bits);
    mc.set_bits(smt_uncontrolled_bits)
        .with_context(|| format!("removing aiger vars"))?;
    for (&cnf_var, correspondance) in symbol_table.iter() {
        if variables[&correspondance.name].controlled {
            match correspondance.sort {
                BitSort::Bool => {
                    *smt_model.get_mut(&correspondance.name[..]).unwrap() =
                        VarValue::Bool(Some(cnf_model[cnf_var]))
                }
                BitSort::Bit { index } => {
                    match smt_model.get_mut(&correspondance.name[..]).unwrap() {
                        VarValue::BitVector(v) => v[index] = Some(cnf_model[cnf_var]),
                        VarValue::Bool(_) => unreachable!(),
                    }
                }
            };
        }
    }
    Ok((SmtModel(smt_model), mc))
}

#[test]
fn ok_syntax_unparseable() {
    assert!(popularity_contest(
        &crate::dsharp::Dsharp,
        "assets/missing_var.smt2".as_ref() as &Path
    )
    .is_err());
    assert!(popularity_contest(
        &crate::dsharp::Dsharp,
        "assets/wrong_sort.smt2".as_ref() as &Path
    )
    .is_err());
}

#[test]
fn choose_interval_bv1() {
    let (model, count) = popularity_contest(
        &crate::dsharp::Dsharp,
        "assets/choose_interval_bv1.smt2".as_ref() as &Path,
    )
    .unwrap();
    dbg!(&model);
    assert_eq!(
        count,
        ModelCount {
            model_count: 0x45u32.into(),
            bits: 8
        }
    );
    assert_eq!(model.0.len(), 1);
    assert_eq!(model.0["thechoice"], (1u8, 1));
}
#[test]
fn boolector_rewrite_level_2() {
    let (model, count) = popularity_contest(
        &crate::dsharp::Dsharp,
        "assets/simplified_vars.smt2".as_ref() as &Path,
    )
    .unwrap();
    dbg!(&model);
    assert_eq!(
        count,
        ModelCount {
            model_count: 69u32.into(),
            bits: 26
        }
    );
    assert_eq!(model.0.len(), 1);
    assert_eq!(model.0["var7"], (1u8, 1));
}

#[test]
fn choose_interval_mc() {
    // regression test for forgetting -pv=NO
    let count = model_count(
        &crate::d4::D4::direct(),
        "assets/choose_interval.smt2".as_ref() as &Path,
    )
    .unwrap();
    assert_eq!(
        count,
        ModelCount {
            model_count: 121u32.into(),
            bits: 9
        }
    );
}

#[test]
fn random_smt_popcon() {
    // Creates formulas of the form
    // match choice
    // 1 => assert value \in [some interval]
    // 2 => assert value \in [some interval]
    // ...
    // 5 => assert value \in [some interval]
    // _ => assert false
    // and checks that when choice is controlled, max popularity is the max
    // interval size.
    use num_traits::Zero;
    use rand::Rng;
    use std::io::Write;
    let mut rng = rand::thread_rng();
    for _ in 0..30 {
        let mut max_length: u8 = 0;
        let mut best: Vec<u8> = Vec::new();
        let mut formula = Vec::new();
        writeln!(formula, "(declare-fun choice () (_ BitVec 8))").unwrap();
        writeln!(formula, "(set-info :controlled choice)").unwrap();
        writeln!(formula, "(declare-fun value () (_ BitVec 8))").unwrap();
        let size = 5;
        writeln!(formula, "(assert (bvult choice #x{:02x}))", size).unwrap();
        for i in 0..size {
            let (min, max): (u8, u8) = rng.gen();
            let range = min..max;
            if range.len() >= max_length as usize {
                if range.len() > max_length as usize {
                    best.clear();
                }
                best.push(i);
                max_length = range.len() as u8;
            }
            writeln!(
            formula,
            "(assert (=> (= choice #x{:02x}) (and (bvule #x{:02x} value) (bvult value #x{:02x}))))",
            i, min, max
        )
            .unwrap();
        }
        println!("considering \n{}\n", String::from_utf8_lossy(&formula));
        let (model, mc) = popularity_contest(&crate::d4::D4::direct(), &formula as &[u8]).unwrap();
        println!(
            "Model {} count {}, expected best {:?}, pop {}",
            model, mc, &best, max_length
        );
        assert_eq!(
            mc,
            ModelCount {
                model_count: max_length.into(),
                bits: 8
            }
        );
        assert_eq!(model.0.len(), 1);
        // if mc is 0, then all models are ok, but best will be empty
        assert!(
            mc.upper().model_count.is_zero() || best.iter().any(|&b| model.0["choice"] == (b, 8))
        );
    }
}

#[test]
fn random_smt_pmc() {
    // Creates formulas of the form
    // match choice
    // 1 => assert value \in [some interval]
    // 2 => assert value \in [some interval]
    // ...
    // 5 => assert value \in [some interval]
    // _ => assert false
    // and checks that the model count projected on value is the size of the union of the intervals
    use rand::Rng;
    use std::io::Write;
    let mut rng = rand::thread_rng();
    for _ in 0..30 {
        let mut possible = [false; 256];
        let mut formula = Vec::new();
        writeln!(formula, "(declare-fun choice () (_ BitVec 8))").unwrap();
        writeln!(formula, "(declare-fun value () (_ BitVec 8))").unwrap();
        writeln!(formula, "(set-info :projected value)").unwrap();
        let size = 5;
        writeln!(formula, "(assert (bvult choice #x{:02x}))", size).unwrap();
        for i in 0..size {
            let (min, max): (u8, u8) = rng.gen();
            for i in min..max {
                possible[i as usize] = true;
            }
            writeln!(
            formula,
            "(assert (=> (= choice #x{:02x}) (and (bvule #x{:02x} value) (bvult value #x{:02x}))))",
            i, min, max
            )
            .unwrap();
        }
        let expected = possible.iter().filter(|&&x| x).count();
        println!("considering \n{}\n", String::from_utf8_lossy(&formula));
        let mc = projected_model_count(&crate::d4::D4::direct(), &formula as &[u8]).unwrap();
        println!("Model count {}, expected {}", mc, expected);
        assert_eq!(mc.model_count, expected.into());
    }
}
#[test]
fn random_smt_mc() {
    // Creates formulas of the form
    // match choice
    // 1 => assert value \in [some interval]
    // 2 => assert value \in [some interval]
    // ...
    // 5 => assert value \in [some interval]
    // _ => assert false
    // and checks that the model count is the sum of the sizes of the intervals
    use rand::Rng;
    use std::io::Write;
    let mut rng = rand::thread_rng();
    for _ in 0..30 {
        let mut total_len: usize = 0;
        let mut formula = Vec::new();
        writeln!(formula, "(declare-fun choice () (_ BitVec 8))").unwrap();
        writeln!(formula, "(declare-fun value () (_ BitVec 8))").unwrap();
        let size = 5;
        writeln!(formula, "(assert (bvult choice #x{:02x}))", size).unwrap();
        for i in 0..size {
            let (min, max): (u8, u8) = rng.gen();
            let range = min..max;
            total_len += range.len();
            writeln!(
            formula,
            "(assert (=> (= choice #x{:02x}) (and (bvule #x{:02x} value) (bvult value #x{:02x}))))",
            i, min, max
        )
            .unwrap();
        }
        println!("considering \n{}\n", String::from_utf8_lossy(&formula));
        let mc = model_count(&crate::d4::D4::direct(), &formula as &[u8]).unwrap();
        println!("Model count {}, expected {}", mc, total_len);
        assert_eq!(mc.model_count, total_len.into());
    }
}

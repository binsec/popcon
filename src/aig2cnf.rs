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

//! Converting boolector AIG dump to CNF

use crate::cnf;
use crate::smt::VarSort;
use aiger::{Aiger, Reader};
use anyhow::Context;
use std::collections::BTreeMap;
use std::io::{Read, Write};
use varisat_dimacs::DimacsHeader;

/// What a bitblasted boolean variable refers to
#[derive(Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub enum BitSort {
    /// a SMT boolean variable
    Bool,
    /// one bit of a SMT bitvector variable
    Bit {
        /// 0-based index of the bit
        index: usize,
    },
}

#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
/// this boolean variable corresponds to an original SMT variable whose name is given.
pub struct BitCorrespondance {
    /// name of the smt variable
    pub name: String,
    /// what bit of this variable it is
    pub sort: BitSort,
}

impl BitSort {
    /// returns false if this bit cannot belong to a variable of sort `sort`.
    /// # Example
    /// ```rust
    /// use popcon::smt::VarSort;
    /// use popcon::aig2cnf::BitSort;
    /// assert!(BitSort::Bool.unify_with_real_sort(VarSort::Bool).is_ok());
    /// assert!(BitSort::Bool.unify_with_real_sort(VarSort::BitVector { size: 12 }).is_err());
    /// assert!(BitSort::Bit { index: 3 }.unify_with_real_sort(VarSort::Bool).is_err());
    /// assert!(BitSort::Bit { index: 3 }.unify_with_real_sort(VarSort::BitVector { size: 1
    /// }).is_err());
    /// assert!(BitSort::Bit { index: 3 }.unify_with_real_sort(VarSort::BitVector { size: 12
    /// }).is_ok());
    /// // Case where boolector bitblasts a boolean to a bitvector of size 1
    /// let mut c = BitSort::Bit { index: 0 };
    /// assert!(c.unify_with_real_sort(VarSort::Bool).is_ok());
    /// assert_eq!(c, BitSort::Bool);
    /// ```
    pub fn unify_with_real_sort(&mut self, sort: VarSort) -> anyhow::Result<()> {
        match (*self, sort) {
            (BitSort::Bool, VarSort::Bool) => Ok(()),
            (BitSort::Bit { index }, VarSort::Bool) => {
                if index == 0 {
                    *self = BitSort::Bool;
                    Ok(())
                } else {
                    anyhow::bail!("a {:?} was bitblasted to {:?}, it can only happen with bit 0 (for bitvectors of size 1)", sort, self)
                }
            }
            (BitSort::Bit { index }, VarSort::BitVector { size }) => {
                anyhow::ensure!(
                    index < size,
                    "a {:?} was bitblasted to {:?}: incompatible bit index with bitvector size",
                    sort,
                    self
                );
                Ok(())
            }
            (BitSort::Bool, VarSort::BitVector { size }) => {
                if size == 1 {
                    *self = BitSort::Bit { index: 0 };
                    Ok(())
                } else {
                    anyhow::bail!("a {:?} was bitblasted to {:?}, it can only happen with bitvectors of size 1", sort, self)
                }
            }
        }
    }
}

/// Mapping from cnf variable to original variable
pub type SymbolTable = BTreeMap<cnf::Var, BitCorrespondance>;

/// converts an AIGER literal to a CNF literal
/// CNF variable with index 0 must mean false.
/// # Example
/// ```
/// use popcon::aig2cnf::aig_lit_to_cnf_lit;
/// use popcon::cnf::Lit;
/// assert_eq!(aig_lit_to_cnf_lit(aiger::Literal(0)), Lit::from_dimacs(1));
/// assert_eq!(aig_lit_to_cnf_lit(aiger::Literal(2)), Lit::from_dimacs(2));
/// assert_eq!(aig_lit_to_cnf_lit(aiger::Literal(3)), Lit::from_dimacs(-2));
/// ```
pub fn aig_lit_to_cnf_lit(x: aiger::Literal) -> cnf::Lit {
    cnf::Lit::from_var(cnf::Var::from_index(x.variable() as _), !x.is_inverted())
}

/// converts AigerError to something that is good enough for anyhow
fn dbg_to_string(x: aiger::AigerError) -> anyhow::Error {
    anyhow::Error::msg(format!("{:?}", x))
}

fn is_identifier(x: char) -> bool {
    x.is_ascii_alphanumeric() || x == '-' || x == '_'
}

/// parses a symbol name as output by boolector
/// # Example
/// ```
/// use popcon::aig2cnf::{BitCorrespondance,parse_symbol,BitSort};
/// assert_eq!(parse_symbol("foo_4").unwrap(), BitCorrespondance { name: "foo_4".to_owned(), sort: BitSort::Bool });
/// assert_eq!(parse_symbol("bar[3]").unwrap(), BitCorrespondance { name: "bar".to_owned(), sort: BitSort::Bit { index: 3 } });
/// ```
pub fn parse_symbol(symbol: &str) -> anyhow::Result<BitCorrespondance> {
    use nom::bytes::complete::take_while1;
    use nom::character::complete as character;
    use nom::combinator::{eof, opt, recognize};
    use nom::sequence::Tuple;
    use nom::Parser;
    let (_rest, (name, bit, _eof)): (_, (&str, Option<&str>, _)) = (
        recognize::<_, _, nom::error::Error<_>, _>(take_while1(is_identifier)),
        opt((character::char('[')
            .and(character::digit1)
            .and(character::char(']')))
        .map(|((_, x), _): ((_, &str), _)| x)),
        eof,
    )
        .parse(&symbol)
        .map_err(|_| anyhow::Error::msg(format!("parsing aiger symbol {}", &symbol)))?;
    let name = name.to_owned();
    Ok(match bit {
        None => BitCorrespondance {
            name,
            sort: BitSort::Bool,
        },
        Some(index) => {
            let index = index
                .parse()
                .with_context(|| format!("bad bit index {} in symbol {}", index, &symbol))?;
            BitCorrespondance {
                name,
                sort: BitSort::Bit { index },
            }
        }
    })
}

/// write a CNF formula to `out` equivalent to "all outputs of `input` are true".
/// `input` is meant to be the output of `boolector -dai -vs 0` and does a few
/// assumptions (same order constraints as binary format, symbol table is present, covers all inputs and
/// has a specific syntax).
/// Based on Tseitin transformation.
/// Returns the dimacs header of the output formula, and the symbol table.
pub fn aig2cnf<R: Read, W: Write>(
    input: R,
    mut out: W,
) -> anyhow::Result<(DimacsHeader, SymbolTable)> {
    let reader = Reader::from_reader(input)
        .map_err(dbg_to_string)
        .context("parsing aiger stream")?;
    let h = reader.header();
    anyhow::ensure!(h.l == 0, "aiger stream has {} latches", h.l);
    let mut symbol_table = BTreeMap::new();
    let mut last_input = 0;
    let cnf_header = varisat_dimacs::DimacsHeader {
        clause_count: 1 + h.o + 3 * h.a,
        var_count: 1 + h.i + h.a,
    };
    varisat_dimacs::write_dimacs_header(&mut out, cnf_header)
        .context("writing cnf header for aig conversion")?;
    for record in reader.records() {
        match record.map_err(dbg_to_string)? {
            Aiger::Input(l) => {
                anyhow::ensure!(!l.is_inverted(), "aiger inputs must be unnegated");
                anyhow::ensure!(
                    l.0 == last_input + 2,
                    "aiger literals must come in sequential order"
                );
                last_input = l.0;
            }
            Aiger::Latch { .. } => unreachable!("aiger stream cannot contain latches"),
            Aiger::AndGate { output, inputs } => {
                anyhow::ensure!(!output.is_inverted(), "and gate outputs to {:?}", output);
                anyhow::ensure!(
                    output.0 == last_input + 2,
                    "aiger literals must come in sequential order"
                );
                last_input = output.0;
                let c = aig_lit_to_cnf_lit(output);
                let a = aig_lit_to_cnf_lit(inputs[0]);
                let b = aig_lit_to_cnf_lit(inputs[1]);
                let tseitin: &[&[cnf::Lit]] = &[&[!a, !b, c], &[a, !c], &[b, !c]];
                varisat_dimacs::write_dimacs_clauses(&mut out, tseitin.iter().copied())
                    .context("writing cnf conversion of aig")?;
            }
            Aiger::Output(l) => {
                varisat_dimacs::write_dimacs_clauses(
                    &mut out,
                    [&[aig_lit_to_cnf_lit(l)] as &[_]].iter().copied(),
                )
                .context("writing cnf conversion of aig")?;
            }
            Aiger::Symbol {
                type_spec,
                position,
                symbol,
            } => {
                match type_spec {
                    aiger::Symbol::Input => {
                        // we check that inputs are literal 2 to 2*i, so we can map position 0 to variable
                        // 1 and so on.
                        let var = cnf::Var::from_index(position + 1);
                        let interpretation = parse_symbol(&symbol)
                            .with_context(|| format!("parsing symbol for variable {}", var))?;
                        symbol_table.insert(var, interpretation);
                    }
                    _ => {}
                }
            }
        }
    }
    varisat_dimacs::write_dimacs_clauses(
        &mut out,
        [&[cnf::Lit::from_dimacs(-1)] as &[_]].iter().copied(),
    )
    .context("writing output clause")?;
    if symbol_table.len() != h.i {
        for i in 1..=h.i {
            let v = cnf::Var::from_index(i);
            if !symbol_table.contains_key(&v) {
                anyhow::bail!("missing aiger symbol for variable {}", v)
            }
        }
    }
    Ok((cnf_header, symbol_table))
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::model::Model;
    type MemoryAiger = Vec<Aiger>;

    /// check that m satisfies a
    fn satisfies(m: &Model, a: &MemoryAiger) -> bool {
        // the value of variables
        let mut value = Model::empty(a.len());
        // which values already have been computed
        let mut computed = Model::empty(a.len());
        value.set_index(0, false);
        computed.set_index(0, true);
        // to redo a pass untill all outputs have been checked to be true
        let mut done = false;
        while !done {
            done = true;
            for record in a {
                match record {
                    Aiger::Input(l) => {
                        let l = aig_lit_to_cnf_lit(*l);
                        value.set(l.var(), m[l.index() - 1]);
                        computed.set(l.var(), true);
                    }
                    Aiger::Output(l) => {
                        let l = aig_lit_to_cnf_lit(*l);
                        if computed[l.var()] {
                            if !value[l] {
                                return false;
                            }
                        } else {
                            done = false;
                        }
                    }
                    Aiger::AndGate { output, inputs } => {
                        let output = aig_lit_to_cnf_lit(*output);
                        let i1 = aig_lit_to_cnf_lit(inputs[0]);
                        let i2 = aig_lit_to_cnf_lit(inputs[1]);
                        if computed[i1.var()] && computed[i2.var()] {
                            value.set(output.var(), value[i1] & value[i2]);
                            computed.set(output.var(), true);
                        }
                    }
                    _ => {}
                }
            }
        }
        true
    }

    /// parses an aig file to a MemoryAiger
    fn parse<R: Read>(read: R) -> (aiger::Header, MemoryAiger) {
        let read = Reader::from_reader(read).expect("reader");
        let h = read.header();
        let m = read.records().map(|x| x.unwrap()).collect();
        (h, m)
    }

    #[test]
    #[ignore]
    fn large() {
        test_file("assets/interval.aag");
    }

    #[test]
    fn or() {
        test_file("assets/or.aag");
    }

    #[test]
    fn unsat() {
        test_file("assets/halfadder.aag");
    }

    fn test_file(name: &'static str) {
        println!("reading aiger from {}", name);
        let mut file = std::fs::File::open(name).expect("opening aiger file");
        let mut content = Vec::new();
        file.read_to_end(&mut content).expect("reading file");
        test_string(&content[..]);
    }

    fn test_string(content: &[u8]) {
        println!("got aiger {}", String::from_utf8_lossy(&content[..]));
        let (header, aiger) = parse(&content[..]);
        let mut cnf = Vec::new();
        aig2cnf(&content[..], &mut cnf).expect("conversion");
        println!("got cnf {}", String::from_utf8_lossy(&cnf));
        let formula = varisat_dimacs::DimacsParser::parse(&cnf[..]).expect("parsing cnf");
        // checking that formula and aiger have models in bijection
        // every model of cnf maps to a model of aiger, and they have the same number
        // of models
        let mut cnf_model = Model::first_model(formula.var_count());
        let mut aiger_model = Model::empty(header.i);
        let mut cnf_model_count = 0;
        loop {
            if cnf_model.satisfies(&formula) {
                cnf_model_count += 1;
                aiger_model.0.clear();
                for i in 0..aiger_model.nvars() {
                    aiger_model.set_index(i, cnf_model[i + 1]);
                }
                assert!(
                    satisfies(&aiger_model, &aiger),
                    "{:?} does not satisfy aiger",
                    &aiger_model
                );
            }
            if !cnf_model.next_model() {
                break;
            }
        }
        aiger_model = Model::first_model(header.i);
        let mut aiger_model_count = 0;
        loop {
            if satisfies(&aiger_model, &aiger) {
                aiger_model_count += 1;
            }
            if !aiger_model.next_model() {
                break;
            }
        }
        assert_eq!(aiger_model_count, cnf_model_count);
    }
}

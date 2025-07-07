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

//! Solve popcon on CNF with `maxcount`

use crate::cnf::{CNFPopularityContest, ModelCount, PopConBounds};
use crate::model::Model;
use crate::utils::LastLines;
use crate::utils::{Input, MaybePersistentTempFile};
use anyhow::Context;
use fixedbitset::FixedBitSet;
use itertools::Itertools;
use nom::Parser;
use num_rational::BigRational;
use num_traits::One;
use std::io::BufWriter;
use std::io::{Read, Write};
use std::os::unix::prelude::{CommandExt, ExitStatusExt};

/// Runs `maxcount formula k`
pub fn run_maxcount(
    cnf_formula: impl Input,
    controlled_variables: &FixedBitSet,
    k: usize,
) -> anyhow::Result<(Model, PopConBounds)> {
    let mut input = cnf_formula.to_read()?;
    let mut formula = MaybePersistentTempFile::new(".maxcount")
        .context("creating temp file for maxcount input")?;
    let mut fmt = BufWriter::new(&mut formula);
    writeln!(
        &mut fmt,
        "c max {} 0",
        controlled_variables.ones().map(|x| x + 1).format(" ")
    )
    .context("writing maxcount input")?;
    let mut complement = controlled_variables.clone();
    complement.toggle_range(..);
    writeln!(
        &mut fmt,
        "c ind {} 0",
        complement.ones().map(|x| x + 1).format(" ")
    )
    .context("writing maxcount input")?;
    let mut buf = [0; 1024];
    loop {
        let n = input
            .read(&mut buf)
            .with_context(|| format!("reading {}", cnf_formula.display()))?;
        if n == 0 {
            break;
        }
        fmt.write_all(&buf[..n]).context("writing to temp file")?;
    }
    fmt.flush().context("flushing temp file")?;
    drop(fmt);
    let mut cmd = std::process::Command::new("maxcount");
    cmd.arg(formula.as_ref())
        .arg(k.to_string())
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    unsafe {
        cmd.pre_exec(|| prctl::set_death_signal(9).map_err(std::io::Error::from_raw_os_error))
    };
    let process_span = tracing::trace_span!("running maxcount", timing = true).entered();
    tracing::trace!("maxcount command: {:?}", &cmd);
    let mut process = cmd.spawn().context("running maxcount")?;
    let mut reader = crate::ddnnf::parser::FileSource::new(process.stdout.take().unwrap());
    let stderr = LastLines::new(process.stderr.take().unwrap());
    let mut model = Model::empty(controlled_variables.len());
    let mut lo = None;
    let mut hi = None;
    const MODEL: &'static [u8] = b"v";
    const LO: &'static [u8] = b"c Max-count is >=";
    const HI: &'static [u8] = b"c Max-count is <=";
    let (low, high) = loop {
        use crate::ddnnf::parser::{signed, unsigned};
        /// Like nom::bytes::streaming::tag but does not borrow the parsed string.
        fn static_tag<'a>(
            tag: &'static [u8],
        ) -> impl FnMut(&'a [u8]) -> nom::IResult<&'a [u8], &'static [u8]> {
            nom::combinator::value(tag, nom::bytes::streaming::tag(tag))
        }
        let typ = reader.feeder.parse_and_advance(|s: &[u8]| {
            static_tag(MODEL)
                .or(static_tag(LO))
                .or(static_tag(HI))
                .parse(s)
        });
        match typ {
            Ok(MODEL) => loop {
                reader.whitespace()?;
                let lit = reader.feeder.parse_and_advance(signed).context("literal")?;
                if lit == 0 {
                    break;
                }
                let lit = crate::cnf::Lit::from_dimacs(lit);
                model.set(lit.var(), lit.is_positive());
            },
            Ok(LO) | Ok(HI) => {
                reader.whitespace()?;
                let float = reader
                    .feeder
                    .parse_and_advance(|s| nom::number::streaming::double(s))
                    .context("bound")?;
                reader.feeder.parse_and_advance(|s| {
                    nom::combinator::map(nom::bytes::streaming::tag(" x 2^"), |_| ())(s)
                })?;
                let shift: u32 = reader.feeder.parse_and_advance(unsigned)?;

                let (storage, name, rounding): (_, _, fn(&BigRational) -> BigRational) = match typ {
                    Ok(LO) => (&mut lo, "lower", BigRational::floor),
                    Ok(HI) => (&mut hi, "upper", BigRational::ceil),
                    _ => unreachable!(),
                };
                let mut value = BigRational::from_float(float).context("parsed nan")?;
                value *= &BigRational::from_integer(num_bigint::BigInt::one() << shift);
                let value = rounding(&value)
                    .to_integer()
                    .to_biguint()
                    .context("negative bound parsed")?;
                tracing::trace!(
                    "parsing {} bound {} x 2^{} => {}",
                    name,
                    float,
                    shift,
                    &value
                );
                anyhow::ensure!(
                    storage.is_none(),
                    "parsed two times the same bound in maxcount output"
                );
                *storage = Some(value);
            }
            Ok(_) => unreachable!(),
            Err(_) => {}
        };
        reader
            .feeder
            .parse_and_advance(|s| {
                nom::combinator::map(nom::bytes::streaming::take_until(b"\n" as &[u8]), |_| ())(s)
            })
            .context("consuming rest of line")?;
        match (lo, hi) {
            (Some(x), Some(y)) => break (x, y),
            (x, y) => {
                lo = x;
                hi = y;
            }
        }
        reader.whitespace()?;
    };
    let status = process.wait().context("waiting for maxcount")?;
    drop(process_span);
    let (err, _) = stderr.get();
    anyhow::ensure!(
        status.success(),
        "maxcount failed with code {:?} signal {:?}: {}",
        status.code(),
        status.signal(),
        String::from_utf8_lossy(&err)
    );
    let bits = controlled_variables.len();
    let uncontrolled_bits = bits - controlled_variables.count_ones(..);
    let bounds = PopConBounds::from_range(
        ModelCount {
            model_count: low,
            bits: uncontrolled_bits,
        },
        ModelCount {
            model_count: high,
            bits: uncontrolled_bits,
        },
    )
    .context("range is upside down")?;
    Ok((model, bounds))
}

/// Use maxcount to solve popcon probabilistically
pub struct Maxcount {
    /// number of clones of the formula.
    ///
    /// Should be 2n/log2(1/epsilon) where n is the number of
    /// controlled variables and epsilon indicated the size of the interval (times of divided by
    /// 1+epsilon)
    pub k: usize,
}

#[test]
fn simple() {
    let mut what = FixedBitSet::with_capacity(3);
    what.insert(0);
    // 3 & (1|2)
    (Maxcount { k: 1 })
        .popularity_contest("assets/simple.cnf".as_ref() as &std::path::Path, &what, &FixedBitSet::with_capacity(FixedBitSet::len(&what)))
        .context("counting models")
        .unwrap();
}

#[test]
fn unsat() {
    let mut what = FixedBitSet::with_capacity(1);
    what.insert(0);
    (Maxcount { k: 1 })
        .popularity_contest("assets/unsat.cnf".as_ref() as &std::path::Path, &what, &FixedBitSet::with_capacity(FixedBitSet::len(&what)))
        .context("popcon")
        .unwrap();
}

impl CNFPopularityContest for Maxcount {
    fn popularity_contest(
        &self,
        input: impl Input,
        controlled_variables: &FixedBitSet,
        _: &FixedBitSet
    ) -> anyhow::Result<(Model, PopConBounds)> {
        let (model, count) = run_maxcount(input, controlled_variables, self.k)
            .with_context(|| format!("running maxcount on {}", input.display()))?;
        Ok((model, count))
    }
}

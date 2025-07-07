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

//! Solve popcon on CNF with `ssatABC`

use crate::cnf::{CNFPopularityContest, ModelCount, PopConBounds};
use crate::model::Model;
use crate::utils::LastLines;
use crate::utils::{Input, MaybePersistentTempFile};
use anyhow::Context;
use fixedbitset::FixedBitSet;
use itertools::Itertools;
use nom::Parser;
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Zero};
use std::io::BufWriter;
use std::io::{Read, Write};
use std::os::unix::prelude::{CommandExt, ExitStatusExt};

/// Runs `abc -c ssat ./formula`
pub fn run_ssatabc(
    cnf_formula: impl Input,
    controlled_variables: &FixedBitSet,
) -> anyhow::Result<(Model, PopConBounds)> {
    anyhow::ensure!(controlled_variables.count_ones(..) != controlled_variables.len(), "ssatABC does not support full existential quantification: https://github.com/NTU-ALComLab/ssatABC/issues/22");
    let mut input = cnf_formula.to_read()?;
    let mut formula =
        MaybePersistentTempFile::new(".sdimacs").context("creating temp file for ssatABC input")?;
    let mut fmt = BufWriter::new(&mut formula);
    writeln!(
        &mut fmt,
        "e {} 0",
        controlled_variables.ones().map(|x| x + 1).format(" ")
    )
    .context("writing ssatABC input")?;
    let mut complement = controlled_variables.clone();
    complement.toggle_range(..);
    writeln!(
        &mut fmt,
        "r 0.5 {} 0",
        complement.ones().map(|x| x + 1).format(" ")
    )
    .context("writing ssatABC input")?;
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
    let mut cmd = std::process::Command::new("abc");
    cmd.arg("-c")
        .arg("ssat -v")
        .arg(formula.as_ref())
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    unsafe {
        cmd.pre_exec(|| prctl::set_death_signal(9).map_err(std::io::Error::from_raw_os_error))
    };
    let process_span = tracing::trace_span!("running ssatABC", timing = true).entered();
    tracing::trace!("ssatABC command: {:?}", &cmd);
    let mut process = cmd.spawn().context("running ssatABC")?;
    let mut reader = crate::ddnnf::parser::FileSource::new(process.stdout.take().unwrap());
    let stderr = LastLines::new(process.stderr.take().unwrap());
    let bits = controlled_variables.len();
    let uncontrolled_bits = bits - controlled_variables.count_ones(..);
    let offset = BigInt::one() << uncontrolled_bits;
    let offset = BigRational::from_integer(offset);
    let mut lo = None;
    let mut hi = None;
    let mut improvement_count = 0u64;
    let mut iteration_count = None;
    const LO: &'static [u8] = b"> Best lower bound:";
    const HI: &'static [u8] = b"> Best upper bound:";
    const EXACT: &'static [u8] = b"> Satisfying probability:";
    const IMPROVE: &'static [u8] = b"> Found a better solution";
    const ITERATIONS: &'static [u8] = b"> main loop iterations =";
    let (low, high) = loop {
        /// Like nom::bytes::streaming::tag but does not borrow the parsed string.
        fn static_tag<'a>(
            tag: &'static [u8],
        ) -> impl FnMut(&'a [u8]) -> nom::IResult<&'a [u8], &'static [u8]> {
            nom::combinator::value(tag, nom::bytes::streaming::tag(tag))
        }
        fn rational<'a>(what: &'a [u8]) -> nom::IResult<&'a [u8], BigRational> {
            (nom::character::streaming::digit1
                .and(nom::bytes::streaming::tag("/"))
                .and(nom::character::streaming::digit1)
                .map(|((num, _), den)| {
                    let num = BigInt::parse_bytes(num, 10).unwrap();
                    let den = BigInt::parse_bytes(den, 10).unwrap();
                    BigRational::new(num, den)
                }))
            .or(
                nom::combinator::value(BigRational::one(), nom::bytes::streaming::tag("1")).or(
                    nom::combinator::value(BigRational::zero(), nom::bytes::streaming::tag("0")),
                ),
            )
            .parse(what)
        }
        let typ = reader.feeder.parse_and_advance(|s: &[u8]| {
            static_tag(LO)
                .or(static_tag(HI).or(static_tag(EXACT)))
                .or(static_tag(IMPROVE))
                .or(static_tag(ITERATIONS))
                .parse(s)
        });
        match typ {
            Ok(LO) | Ok(HI) | Ok(EXACT) => {
                reader.whitespace()?;
                let mut value = reader.feeder.parse_and_advance(rational).context("bound")?;
                value *= &offset;
                anyhow::ensure!(
                    value.is_integer(),
                    "found non integral model count: {}",
                    &value
                );
                let value = value
                    .to_integer()
                    .to_biguint()
                    .context("negative model count")?;
                match typ {
                    Ok(LO) => {
                        lo = Some(value);
                    }
                    Ok(HI) => {
                        hi = Some(value);
                    }
                    Ok(EXACT) => {
                        lo = Some(value.clone());
                        hi = Some(value);
                    }
                    _ => unreachable!(),
                }
            }
            Ok(ITERATIONS) => {
                reader.whitespace()?;
                let value = reader
                    .feeder
                    .parse_and_advance(|s| nom::character::streaming::u64(s));
                iteration_count = Some(value);
            }
            Ok(IMPROVE) => {
                improvement_count += 1;
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
        reader.whitespace().context("next line")?;
    };
    let status = process.wait().context("waiting for ssatABC")?;
    drop(process_span);
    let (err, _) = stderr.get();
    anyhow::ensure!(
        status.success(),
        "ssatABC failed with code {:?} signal {:?}: {}",
        status.code(),
        status.signal(),
        String::from_utf8_lossy(&err)
    );
    tracing::trace!(ssatabc_improvement_count = improvement_count, stats = true);
    if let Some(Ok(iterations)) = iteration_count {
        tracing::trace!(ssatabc_step_count = iterations, stats = true);
    }
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
    let model = Model::empty(bits);
    Ok((model, bounds))
}

/// Use ssatABC to solve popcon
///
/// Model is not parsed and always 0.
pub struct SsatABC;

#[test]
fn simple() {
    let mut what = FixedBitSet::with_capacity(3);
    what.insert(0);
    // 3 & (1|2)
    let (_, bounds) = SsatABC
        .popularity_contest("assets/simple.cnf".as_ref() as &std::path::Path, &what, &FixedBitSet::with_capacity(FixedBitSet::len(&what)))
        .context("counting models")
        .unwrap();
    assert_eq!(
        bounds,
        PopConBounds::exactly(ModelCount {
            model_count: 2u8.into(),
            bits: 2
        })
    );
}

// https://github.com/NTU-ALComLab/ssatABC/issues/22
#[test]
fn unsat() {
    let mut what = FixedBitSet::with_capacity(1);
    what.insert(0);
    assert!(SsatABC
        .popularity_contest("assets/unsat.cnf".as_ref() as &std::path::Path, &what, &FixedBitSet::with_capacity(FixedBitSet::len(&what)))
        .is_err());
}

#[test]
fn unsat2() {
    let mut what = FixedBitSet::with_capacity(2);
    what.insert(0);
    let (_, bounds) = SsatABC
        .popularity_contest(
            "
p cnf 2 3
1 0
-1 0
2 0
"
            .as_bytes(),
            &what,
            &FixedBitSet::with_capacity(FixedBitSet::len(&what))
        )
        .context("running ssat")
        .unwrap();
    assert_eq!(
        bounds,
        PopConBounds::exactly(ModelCount {
            model_count: 0u8.into(),
            bits: 1
        })
    );
}

#[test]
fn robust() {
    let mut what = FixedBitSet::with_capacity(2);
    what.insert(0);
    let (_, bounds) = SsatABC
        .popularity_contest(
            "
p cnf 2 1
1 2 0
"
            .as_bytes(),
            &what,
            &FixedBitSet::with_capacity(FixedBitSet::len(&what))
        )
        .context("running ssat")
        .unwrap();
    assert_eq!(
        bounds,
        PopConBounds::exactly(ModelCount {
            model_count: 2u8.into(),
            bits: 1
        })
    );
}

impl CNFPopularityContest for SsatABC {
    fn popularity_contest(
        &self,
        input: impl Input,
        controlled_variables: &FixedBitSet,
        _: &FixedBitSet
    ) -> anyhow::Result<(Model, PopConBounds)> {
        let (model, count) = run_ssatabc(input, controlled_variables)
            .with_context(|| format!("running ssatABC on {}", input.display()))?;
        Ok((model, count))
    }
}

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

//! Solve popcon on CNF with `baxmc`

use crate::cnf::{CNFPopularityContest, ModelCount, PopConBounds};
use crate::model::Model;
use crate::utils::LastLines;
use crate::utils::{Input, MaybePersistentTempFile};
use anyhow::Context;
use fixedbitset::FixedBitSet;
use itertools::Itertools;
use nom::Parser;
use std::io::{BufRead, BufWriter};
use std::io::{Read, Write};
use std::os::unix::prelude::{CommandExt, ExitStatusExt};
use num_traits::Zero;
use nom::number::complete::float;
use num_bigint::ToBigUint;

/// Runs `baxmc ./formula`
pub fn run_baxmc(
    opts: &Vec<String>,
    cnf_formula: impl Input,
    controlled_variables: &FixedBitSet,
    uncontrolled_variables: &FixedBitSet,
) -> anyhow::Result<(Model, PopConBounds)> {
    let mut input = std::io::BufReader::new(cnf_formula.to_read()?);
    let mut header = String::new();
    input
        .read_line(&mut header)
        .context("reading formula header")?;
    let mut formula =
        MaybePersistentTempFile::new(".cnf").context("creating temp file for baxmc input")?;
    let mut fmt = BufWriter::new(&mut formula);
    writeln!(
        &mut fmt,
        "{}c target {} 0",
        header,
        controlled_variables.ones().map(|x| x + 1).format(" ")
    )
    .context("writing baxmc input")?;
    writeln!(
        &mut fmt,
        "c proj {} 0",
        uncontrolled_variables.ones().map(|x| x + 1).format(" ")
    )
    .context("writing baxmc input")?;
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
    let mut cmd = std::process::Command::new("baxmc");
    for opt in opts.iter() {
        cmd.arg(opt);
    }
    cmd.arg(formula.as_ref())
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    unsafe {
        cmd.pre_exec(|| prctl::set_death_signal(9).map_err(std::io::Error::from_raw_os_error))
    };
    let process_span = tracing::trace_span!("running baxmc", timing = true).entered();
    tracing::trace!("baxmc command: {:?}", &cmd);
    let mut process = cmd.spawn().context("running baxmc")?;
    let mut reader = crate::ddnnf::parser::FileSource::new(process.stdout.take().unwrap());
    let stderr = LastLines::new(process.stderr.take().unwrap());
    let bits = controlled_variables.len();
    let uncontrolled_bits = bits - controlled_variables.count_ones(..);
    let mut model = Model::empty(controlled_variables.len());
    const EXACT: &'static [u8] = b"v";
    const MODEL: &'static [u8] = b"s";
    const ENDL: &'static [u8] = b"\n";
    const SPC: &'static [u8] = b" ";
    let mut mc = None;
    loop {
        use crate::ddnnf::parser::{signed, bigunsigned};
        fn static_tag<'a>(
            tag: &'static [u8],
        ) -> impl FnMut(&'a [u8]) -> nom::IResult<&'a [u8], &'static [u8]> {
            nom::combinator::value(tag, nom::bytes::streaming::tag(tag))
        }
        let typ = reader
            .feeder
            .parse_and_advance(|s: &[u8]| static_tag(EXACT).or(static_tag(MODEL)).parse(s));
        match typ {
            Ok(MODEL) => { 
                loop {
                    match reader.feeder.parse_and_advance(|s: &[u8]| static_tag(ENDL).or(static_tag(SPC)).parse(s)) {
                        Ok(ENDL) => {break;}
                        Ok(SPC) => {}
                        _ => assert!(false)
                    }
                    let lit = reader.feeder.parse_and_advance(signed).context("literal")?;
                    let lit = crate::cnf::Lit::from_dimacs(lit);
                    model.set(lit.var(), lit.is_positive());
                };
                break;
            },
            Ok(EXACT) => {
                reader.whitespace()?;
                let cnt : num_bigint::BigUint = reader
                    .feeder
                    .parse_and_advance(|s| match float(s) {Ok((s, f)) => Ok((s, match f.to_biguint() {Some(b) => b, None => num_bigint::BigUint::new(vec![0])})), Err(e) => Err(e)})?;
                mc = Some(cnt);
            },
            Ok(_) => unreachable!(),
            Err(_) => {}
        };
        reader
            .feeder
            .parse_and_advance(|s| {
                nom::combinator::map(nom::bytes::streaming::take_until(b"\n" as &[u8]), |_| ())(s)
            })
            .context("consuming rest of line")?;
        reader.whitespace().context("next line")?;
    };
    let status = process.wait().context("waiting for baxmc")?;
    drop(process_span);
    let (err, _) = stderr.get();
    anyhow::ensure!(
        status.success(),
        "baxmc failed with code {:?} signal {:?}: {}",
        status.code(),
        status.signal(),
        String::from_utf8_lossy(&err)
    );
    let model_count = match mc {Some(mc) => mc, None => {assert!(false); num_bigint::BigUint::zero()}};
    let bounds = PopConBounds::exactly(ModelCount {
        model_count,
        bits: uncontrolled_bits,
    });
    Ok((model, bounds))
}

/// BaxMC options
pub struct Options {
    /// command line options
    pub opts: Vec<String>
}

impl Options {
    /// create options
    pub fn create(s: String) -> Self {
        let mut opts = vec![];
        if s != "" {
            for opt in s.split(" ").collect::<Vec<&str>>().iter() {
                opts.push(opt.to_string());
            }
        }
        Self{opts}
    }
}

/// Use baxmc to solve popcon approximately.
pub struct BAXMC {
    ///command line options
    pub opts: Options
}

impl CNFPopularityContest for BAXMC {
    fn popularity_contest(
        &self,
        input: impl Input,
        controlled_variables: &FixedBitSet,
        uncontrolled_variables: &FixedBitSet
    ) -> anyhow::Result<(Model, PopConBounds)> {
        let (model, count) = run_baxmc(&self.opts.opts, input, controlled_variables, uncontrolled_variables)
            .with_context(|| format!("running baxmc on {}", input.display()))?;
        Ok((model, count))
    }
}

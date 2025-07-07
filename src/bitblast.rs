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

//! Convert a SMT(QF_BV) formula to CNF

use anyhow::Context;
use std::io::Write;
use std::os::unix::process::CommandExt;
use std::os::unix::process::ExitStatusExt;
use std::process::{Command, Stdio};
use tracing::{error, trace};
use varisat_dimacs::DimacsHeader;

use crate::aig2cnf;
use crate::utils::{try_wait_timeout, LastLines};

/// Convert a SMT(QF_BV) formula to CNF with boolector.
pub fn smt2cnf<W: Write>(
    input: impl crate::utils::Input,
    output: W,
) -> anyhow::Result<(DimacsHeader, aig2cnf::SymbolTable)> {
    let input_path = input.to_path()?;
    let mut cmd = Command::new("boolector");
    /*
     * -vs 0: disables variable substitution, necessary to get a non empty symbol table.
     * This apparently makes the formula larger, I don't know why.
     * -es 0: introduces new variables for bitvector slices (bvextract).
     * In practice this makes operators bvsge and bvsgt introduce extra variables in aig,
     * which are not labeled.
     * -uc also does that but it's off by default...
     * --rewrite-level 2 because some bits that are necessarily 0 are simplified out otherwise in
     * ../assets/simplified_vars.smt2
     * --rewrite-level 1 because some variables simplified out otherwise in
     * ../assets/fails_with_rwl2.smt2
     * Could be detected based on missing variables ????
     **/
    cmd.arg("-m")
        .arg("-vs")
        .arg("0")
        .arg("-daa")
        .arg("-es")
        .arg("0")
        .arg("--rewrite-level")
        .arg("1")
        .arg(input_path.as_ref());
    cmd.stderr(Stdio::piped()).stdout(Stdio::piped());
    unsafe {
        cmd.pre_exec(|| prctl::set_death_signal(9).map_err(std::io::Error::from_raw_os_error))
    };
    trace!(?cmd, file=%input.display(), "starting boolector");
    let mut child = cmd
        .spawn()
        .with_context(|| format!("running boolector on {}", input.display()))?;
    let stderr = LastLines::new(child.stderr.take().unwrap());

    let res = aig2cnf::aig2cnf(child.stdout.take().unwrap(), output);
    let status = match try_wait_timeout(&mut child, std::time::Duration::from_secs(2))
        .context("waiting for boolector")?
    {
        Some(x) => x,
        None => {
            trace!("killing boolector");
            child.kill().context("killing boolector")?;
            child.wait().context("waiting for boolector after kill")?
        }
    };
    if !status.success() {
        let (txt, _) = stderr.get();
        let txt = String::from_utf8_lossy(&txt[..]);
        let details = format!(
            "boolector failed with code {:?} signal {:?} output {}",
            status.code(),
            status.signal(),
            &txt
        );
        error!(code=?status.code(), signal=?status.signal(), output=txt.as_ref(), "boolector failed");
        match res {
            Ok(_) => anyhow::bail!("{}", details),
            Err(e) => Err(e).context(details),
        }
    } else {
        match &res {
            Ok((header, table)) => trace!(
                biblasted_var_count = header.var_count,
                bitblasted_clause_count = header.clause_count,
                aiger_vars = table.len(),
                stats = true,
                "bitblasting result"
            ),
            _ => (),
        };
        res
    }
}

#[test]
#[ignore]
fn interval() {
    let mut out = Vec::new();
    let (_, table) = smt2cnf(
        "assets/interval.smt2".as_ref() as &std::path::Path,
        &mut out,
    )
    .expect("smt2cnf");
    let formula = varisat_dimacs::DimacsParser::parse(&out[..]).expect("parsing cnf");
    let mc = crate::cnf::brute_force_model_count(&formula);
    // real answer is 272 = 0x234 - 0x123 - 1
    // some bits are irrelevant and ommited in the answer
    let real = 0x234 - 0x123 - 1u32;
    let compensated = (real >> (12 - table.len())).into();
    assert_eq!(mc.model_count, compensated);
}

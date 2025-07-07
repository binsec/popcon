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

//! Convert CNF to d-DNNF with dsharp

use crate::cnf::{CNFCounter, CNFPopularityContest, ModelCount, PopConBounds};
use crate::ddnnf::FileSource;
use crate::ddnnf::Source;
use crate::model::Model;
use crate::utils::Input;
use crate::utils::LastLines;
use anyhow::Context;
use fixedbitset::FixedBitSet;
use std::fmt::Write;
use std::fs::File;
use std::os::unix::io::FromRawFd;
use std::os::unix::process::CommandExt;
use std::os::unix::process::ExitStatusExt;
use std::path::Path;
use std::process::Child;
use tracing::trace;

/// Run dsharp to transform a CNF formula to a d-DNNF formula.
pub struct DsharpProcess {
    // option to be able to take it
    process: Option<Child>,
    // option to be able to take it
    output: Option<FileSource<File>>,
    stdout: LastLines,
    stderr: LastLines,
}

impl DsharpProcess {
    /// Spawns a dsharp process converting a file in DIMACS format.
    /// The child process is terminated when the result is visited, or dropped,
    /// or the current *thread* dies.
    /// If `priority_variables` is not None, these variables will be in the topmost OR nodes.
    pub fn new(
        cnf_formula: &Path,
        priority_variables: Option<&FixedBitSet>,
    ) -> anyhow::Result<Self> {
        // check taht the input file exists.
        let cnf_formula = if cnf_formula.is_absolute() {
            cnf_formula.to_owned()
        } else {
            std::fs::canonicalize(&cnf_formula).with_context(|| {
                format!(
                    "canonicalizing {},the input file to dsharp",
                    cnf_formula.display()
                )
            })?
        };
        // create a pipe for output
        let (raw_read, raw_write) =
            nix::unistd::pipe().context("creating a pipe for c2d output")?;
        // doing this immediately ensure these are close in case something happens
        let (read, write) = unsafe { (File::from_raw_fd(raw_read), File::from_raw_fd(raw_write)) };
        crate::utils::set_cloexec(&read, true).context("dsharp input pipe cloexec")?;

        let fd_path = format!("/proc/self/fd/{}", raw_write);
        // spawn the process
        let mut cmd = std::process::Command::new("dsharp");
        cmd.arg("-Fnnf").arg(&fd_path);
        if let Some(prio) = priority_variables {
            if prio.count_ones(..) > 0 {
                cmd.arg("-priority");
                let mut prio_arg = String::new();
                for i in prio.ones() {
                    write!(prio_arg, ",{}", i + 1).context("writing to string !!!")?;
                }
                cmd.arg(&prio_arg[1..]);
            }
        }
        cmd.arg(&cnf_formula)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());
        unsafe {
            cmd.pre_exec(|| prctl::set_death_signal(9).map_err(std::io::Error::from_raw_os_error))
        };
        trace!(?cmd, "starting dsharp");
        let mut process = cmd.spawn().context("failed to run d4")?;
        // close our copy of the write pipe so that if c2d dies, the read end returns EOF
        drop(write);
        let stdout = LastLines::new(process.stdout.take().unwrap());
        let stderr = LastLines::new(process.stderr.take().unwrap());
        Ok(Self {
            process: Some(process),
            output: Some(FileSource::new(read)),
            stdout,
            stderr,
        })
    }
}

impl Source for DsharpProcess {
    fn visit<T: crate::ddnnf::Visitor>(mut self, visitor: T) -> anyhow::Result<()> {
        let res = self.output.take().unwrap().visit(visitor);
        if crate::utils::try_wait_timeout(
            self.process.as_mut().unwrap(),
            std::time::Duration::from_secs(2),
        )
        .context("waiting for dsharp")?
        .is_none()
        {
            // dsharp has not terminated
            self.process
                .as_mut()
                .unwrap()
                .kill()
                .context("killing dsharp")?;
        }
        let status = self
            .process
            .take()
            .unwrap()
            .wait()
            .context("waiting for dsharp")?;
        let (out, _) = self.stdout.get();
        let (err, _) = self.stderr.get();
        anyhow::ensure!(
            status.success(),
            "dsharp failed with code {:?} signal {:?}: {} {}",
            status.code(),
            status.signal(),
            String::from_utf8_lossy(&out),
            String::from_utf8_lossy(&err)
        );
        res
    }
}

impl Drop for DsharpProcess {
    fn drop(&mut self) {
        if let Some(mut p) = self.process.take() {
            let _ignore = p.kill();
        }
    }
}

/// Use dsharp as a backend solver.
pub struct Dsharp;

#[test]
fn simple() {
    // 3 & (1|2)
    let models = Dsharp
        .model_count("assets/simple.cnf".as_ref() as &Path, None)
        .context("counting models")
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
fn unsat() {
    let models = Dsharp
        .model_count("assets/unsat.cnf".as_ref() as &Path, None)
        .context("counting models")
        .unwrap();
    assert_eq!(
        models,
        ModelCount {
            model_count: 0usize.into(),
            bits: 1
        }
    );
}

// #[test]
#[allow(unused)]
fn regression_dsharp_issue_12() {
    // regression test for https://github.com/QuMuLab/dsharp/issues/12
    let models = crate::cnf::Compacter(Dsharp)
        .model_count("assets/bug_dsharp_mc.cnf".as_ref() as &Path, None)
        .context("counting models")
        .unwrap();
    assert_eq!(
        models,
        ModelCount {
            model_count: num_bigint::BigUint::from(104310u32) << 372,
            bits: 400
        }
    );
}

#[test]
fn projection() {
    let vars: FixedBitSet = [0usize, 2].iter().cloned().collect();
    let (model, count) = Dsharp
        .popularity_contest("assets/simple.cnf".as_ref() as &Path, &vars, &FixedBitSet::with_capacity(FixedBitSet::len(&vars)))
        .context("counting models")
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
fn projection_compaction() {
    let vars: FixedBitSet = [0usize, 2].iter().cloned().collect();
    let (model, count) = crate::cnf::Compacter(Dsharp)
        .popularity_contest("assets/simple.cnf".as_ref() as &Path, &vars, &FixedBitSet::with_capacity(FixedBitSet::len(&vars)))
        .context("counting models")
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

impl CNFCounter for Dsharp {
    fn model_count(
        &self,
        input: impl Input,
        _: Option<&FixedBitSet>,
    ) -> anyhow::Result<ModelCount> {
        let tempfile = input.to_path()?;
        let dsharp = DsharpProcess::new(tempfile.as_ref(), None)
            .with_context(|| format!("creating dsharp on {}", input.display()))?;
        let models = crate::ddnnf::count_models(dsharp, None).with_context(|| {
            format!(
                "counting models for output of dsharp on {}",
                input.display()
            )
        })?;
        Ok(models)
    }
}

impl CNFPopularityContest for Dsharp {
    fn popularity_contest(
        &self,
        input: impl Input,
        controlled_variables: &FixedBitSet,
        _: &FixedBitSet
    ) -> anyhow::Result<(Model, PopConBounds)> {
        let tempfile = input.to_path()?;
        let dsharp = DsharpProcess::new(tempfile.as_ref(), Some(controlled_variables))
            .with_context(|| format!("creating dsharp on {}", input.display()))?;
        let (model, count) = crate::ddnnf::most_popular_model(dsharp, controlled_variables.clone())
            .with_context(|| {
                format!(
                    "solving popularity contest for output of dsharp on {}",
                    input.display()
                )
            })?;
        Ok((model, PopConBounds::exactly(count)))
    }
}

impl crate::ddnnf::Cnf2DdnnfCompiler for Dsharp {
    type O = DsharpProcess;
    fn compile<I: Input>(&self, input: I) -> anyhow::Result<Self::O> {
        let path = input
            .to_path()
            .context("input.to_path() as input to dsharp")?;
        DsharpProcess::new(path.as_ref(), None)
    }
}

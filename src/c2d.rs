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

//! Use c2d to convert a CNF to a d-DNNF

use crate::cnf::{CNFCounter, ModelCount};
use crate::ddnnf::FileSource;
use crate::ddnnf::Source;
use crate::utils::Input;
use crate::utils::LastLines;
use anyhow::Context;
use fixedbitset::FixedBitSet;
use std::fs::File;
use std::os::unix::fs::symlink;
use std::os::unix::io::FromRawFd;
use std::os::unix::process::CommandExt;
use std::os::unix::process::ExitStatusExt;
use std::path::Path;
use std::process::Child;
use tempfile::TempDir;
use tracing::trace;

/// Use c2d as a backend
pub struct C2D;

/// Run c2d to transform a CNF formula to a d-DNNF formula.
/// The subprocess is killed on drop.
pub struct C2DProcess {
    // option to be able to take it
    process: Option<Child>,
    // option to be able to take it
    output: Option<FileSource<File>>,
    // only to keep ownership to prevent the tempdir to be destroyed too early
    // option to be able to take it
    tempdir: Option<TempDir>,
    stdout: LastLines,
    stderr: LastLines,
}

impl C2DProcess {
    /// Spawns a C2D process converting a file in DIMACS format.
    /// The child process is terminated when the resuld is visited, or dropped,
    /// or the current *thread* dies.
    pub fn new(cnf_formula: &Path) -> anyhow::Result<Self> {
        // check taht the input file exists.
        let cnf_formula = if !cnf_formula.is_absolute() {
            std::fs::canonicalize(&cnf_formula).with_context(|| {
                format!(
                    "canonicalizing {},the input file to c2d",
                    cnf_formula.display()
                )
            })?
        } else {
            cnf_formula.to_path_buf()
        };
        // c2d must absolutely write the output to $input.nnf
        // we do this in a temporary directory, where input and output are symlinks
        let dir = tempfile::tempdir().context("creating a temporary directory to run c2d")?;
        let input = dir.path().join("input.cnf");
        let output = dir.path().join("input.cnf.nnf");
        // input.cnf point to real input
        symlink(&cnf_formula, &input).with_context(|| {
            format!(
                "symlinking {} to {} for input to c2d",
                cnf_formula.display(),
                input.display()
            )
        })?;
        // input.cnf.nnf (the output) points to a pipe
        let (raw_read, raw_write) =
            nix::unistd::pipe().context("creating a pipe for c2d output")?;
        // doing this immediately ensure these are close in case something happens
        let (read, write) = unsafe { (File::from_raw_fd(raw_read), File::from_raw_fd(raw_write)) };

        let fd_path = format!("/proc/self/fd/{}", raw_write);
        symlink(&fd_path, &output).with_context(|| {
            format!(
                "symlinking {} to {} for input to c2d",
                &fd_path,
                output.display()
            )
        })?;
        // spawn the process
        let mut cmd = std::process::Command::new("c2d");
        cmd.arg("-in")
            .arg(&input)
            .arg("-silent")
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());
        unsafe {
            cmd.pre_exec(|| prctl::set_death_signal(9).map_err(std::io::Error::from_raw_os_error))
        };
        trace!(?cmd, filename=%cnf_formula.display(), "starting c2d");
        let mut process = cmd.spawn().context("failed to run c2d")?;
        // close our copy of the write pipe so that if c2d dies, the read end returns EOF
        drop(write);
        let stdout = LastLines::new(process.stdout.take().unwrap());
        let stderr = LastLines::new(process.stderr.take().unwrap());
        Ok(C2DProcess {
            process: Some(process),
            output: Some(FileSource::new(read)),
            tempdir: Some(dir),
            stdout,
            stderr,
        })
    }
}

impl Source for C2DProcess {
    fn visit<T: crate::ddnnf::Visitor>(mut self, visitor: T) -> anyhow::Result<()> {
        let res = self.output.take().unwrap().visit(visitor);
        if crate::utils::try_wait_timeout(
            self.process.as_mut().unwrap(),
            std::time::Duration::from_secs(2),
        )
        .context("waiting for c2d")?
        .is_none()
        {
            // c2d has not terminated
            self.process
                .as_mut()
                .unwrap()
                .kill()
                .context("killing c2d")?;
        }
        // remove the temporary directory
        if let Some(t) = self.tempdir.take() {
            t.close().context("removing temporary directory")?;
        }
        let status = self
            .process
            .take()
            .unwrap()
            .wait()
            .context("reading c2d stderr")?;
        if !status.success() {
            let (stdout, _) = self.stdout.get();
            let (stderr, _) = self.stderr.get();
            anyhow::bail!(
                "c2d failed with code {:?} signal {:?}: {} {}",
                status.code(),
                status.signal(),
                String::from_utf8_lossy(&stdout),
                String::from_utf8_lossy(&stderr)
            );
        }
        res
    }
}

impl Drop for C2DProcess {
    fn drop(&mut self) {
        if let Some(mut p) = self.process.take() {
            let _ignore = p.kill();
        }
        if let Some(t) = self.tempdir.take() {
            let _ignore = t.close();
        }
        if let Some(s) = self.output.take() {
            drop(s)
        }
    }
}

#[test]
fn simple() {
    let models = C2D
        .model_count("assets/simple.cnf".as_ref() as &Path, None)
        .context("counting models")
        .unwrap();
    assert_eq!(models.model_count, 3usize.into());
}

impl CNFCounter for C2D {
    fn model_count(
        &self,
        input: impl Input,
        _: Option<&FixedBitSet>,
    ) -> anyhow::Result<ModelCount> {
        let file = input.to_path()?;
        let c2d = C2DProcess::new(file.as_ref())
            .with_context(|| format!("creating c2d on {}", input.display()))?;
        let models = crate::ddnnf::count_models(c2d, None)
            .with_context(|| format!("counting models for output of c2d on {}", input.display()))?;
        Ok(models)
    }
}

impl crate::ddnnf::Cnf2DdnnfCompiler for C2D {
    type O = C2DProcess;
    fn compile<I: Input>(&self, input: I) -> anyhow::Result<Self::O> {
        let path = input
            .to_path()
            .context("input.to_path() as input to dsharp")?;
        C2DProcess::new(path.as_ref())
    }
}

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

use anyhow::Context;
use popcon::ddnnf;
use std::path::PathBuf;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(
    name = "nnf2cnf",
    about = "Converts a d-DNNF formula to a CNF formula in dimacs format"
)]
struct Opt {
    /// Input file
    #[structopt(parse(from_os_str))]
    input: PathBuf,

    /// Output file
    #[structopt(parse(from_os_str))]
    output: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let opts = Opt::from_args();
    let f = std::fs::File::open(&opts.input).context("opening input file")?;
    let input = ddnnf::FileSource::new(std::io::BufReader::new(f));
    let res = ddnnf::to_cnf(input).context("converting")?;
    let mut out =
        std::fs::File::create(&opts.output).context("failed to open ouput for writing")?;
    varisat_dimacs::write_dimacs(&mut out, &res).context("writing output")?;
    Ok(())
}

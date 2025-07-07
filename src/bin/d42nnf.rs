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
use popcon::d4;
use std::path::PathBuf;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(name = "d42nnf", about = "Converts d4 output to a nnf file")]
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
    let input = std::fs::read_to_string(&opts.input).context("reading input")?;
    let output = d4::parse_d4_output(&input).context("failed to parse input")?;
    let nnf = d4::preprocess(output).context("failed to process input")?;
    let out = std::fs::File::create(&opts.output).context("failed to open ouput for writing")?;
    popcon::ddnnf::write(out, &nnf).context("writing to output")?;
    Ok(())
}

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

//! A type for a model, or a set of variables
use fixedbitset::FixedBitSet;

use crate::cnf::CnfFormula;
use crate::cnf::Lit;
use crate::cnf::Var;
use std::ops::Index;

#[derive(Clone, PartialOrd, Ord, PartialEq, Eq)]
/// Represents a model by the set of true variables
/// Can be indexed by `usize` (0-based), `cnf::Var` or `cnf::Lit`. In the last case,
/// returns `false` true the negated literal corresponding to a true variable.
pub struct Model(pub FixedBitSet);

impl std::fmt::Debug for Model {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        fmt.debug_set()
            .entries(self.0.ones().map(Var::from_index))
            .finish()
    }
}

impl Model {
    /// creates a model for a formula of `nvars` variables with all variables false
    pub fn empty(nvars: usize) -> Model {
        Self(FixedBitSet::with_capacity(nvars))
    }

    /// creates a model for a formula of `nvars` variables with all variables false except `var`.
    pub fn just(var: Var, nvars: usize) -> Model {
        let mut x = Model::empty(nvars);
        x.0.insert(var.index());
        x
    }

    /// number of variables for the whole formula
    pub fn nvars(&self) -> usize {
        self.0.len()
    }

    /// number of variables set to true in the formula
    pub fn nvars_set(&self) -> usize {
        self.0.count_ones(..)
    }

    /// returns true if the variables are all set to false
    pub fn is_empty(&self) -> bool {
        self.nvars_set() == 0
    }

    /// set the specified variable to the specified value
    pub fn set_index(&mut self, index: usize, value: bool) {
        self.0.set(index, value)
    }

    /// set the specified variable to the specified value
    pub fn set(&mut self, var: Var, value: bool) {
        self.0.set(var.index(), value)
    }
}

impl Index<usize> for Model {
    type Output = bool;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl Index<Var> for Model {
    type Output = bool;

    fn index(&self, var: Var) -> &Self::Output {
        &self.0[var.index()]
    }
}

static TRUE: bool = true;
static FALSE: bool = false;
impl Index<Lit> for Model {
    type Output = bool;

    fn index(&self, literal: Lit) -> &Self::Output {
        let v = self[literal.var()];
        let res = v ^ literal.is_negative();
        if res {
            &TRUE
        } else {
            &FALSE
        }
    }
}

impl Model {
    /// returns the first model to iterate over all possible models with `next_model`
    pub fn first_model(nvars: usize) -> Model {
        Model::empty(nvars)
    }

    /// modifies a model in place to get to the next one, returns wether there are further models
    /// # Example
    /// ```
    /// use popcon::model::Model;
    /// use popcon::cnf::Var;
    /// // iterate on and collect all models of 1 variable
    /// let mut model = Model::first_model(1);
    /// let mut all_models = Vec::new();
    /// loop {
    ///     // use the model
    ///     all_models.push(model.clone());
    ///     // iterate to next
    ///     if !model.next_model() {
    ///         break
    ///     }
    /// }
    /// assert_eq!(all_models, vec![Model::empty(1), Model::just(Var::from_index(0), 1)]);
    /// ```
    pub fn next_model(&mut self) -> bool {
        let l = self.0.len();
        let s = self.0.as_mut_slice();
        let extra = l % 32;
        let blocks = l / 32 + if extra == 0 { 0 } else { 1 };
        let sentinel_for_last_block = if extra == 0 { 0 } else { 1 << extra };
        for (i, block) in s.iter_mut().enumerate() {
            *block = block.wrapping_add(1);
            let value = *block;
            if i == blocks - 1 {
                if value == sentinel_for_last_block {
                    *block = 0;
                    return false;
                } else {
                    return true;
                }
            } else {
                if value != 0 {
                    return true;
                }
            }
        }
        unreachable!()
    }

    /// Checks wether the model satisfies a formula.
    pub fn satisfies(&self, formula: &CnfFormula) -> bool {
        'outer: for clause in formula.iter() {
            for lit in clause {
                if self[*lit] {
                    continue 'outer;
                }
            }
            return false;
        }
        return true;
    }
}

#[test]
fn test_satisfies() -> anyhow::Result<()> {
    let f = b"p cnf 3 3
-1 3 -2 0
-2 1 3 0
3 -1 -2 0
";
    let formula = varisat_dimacs::DimacsParser::parse(f as &[u8])?;
    let mut model = crate::model::Model::empty(3);
    assert!(model.satisfies(&formula));
    model.0.insert(0);
    model.0.insert(1);
    assert!(!model.satisfies(&formula));
    Ok(())
}

// takes 1 min
// cargo test --release -- --include-ignored to run
#[test]
#[ignore]
fn test_next_model() {
    fn to_u64(m: &Model) -> u64 {
        let s = m.0.as_slice();
        if s.len() == 2 {
            s[0] as u64 | (s[1] as u64) << 32
        } else if s.len() == 1 {
            s[0] as u64
        } else {
            unreachable!()
        }
    }
    for size in 31..=33 {
        let mut m = Model::first_model(size);
        let mut last = true;
        println!("start with m.len()={}", m.nvars());
        for i in 0..(1u64 << size) {
            assert!(last, "size: {}, return value after {} is false", size, i);
            assert_eq!(to_u64(&m), i, "size: {}", size);
            last = m.next_model();
        }
        assert!(
            !last,
            "size: {}, return value after {} is true",
            size,
            1u64 << size
        );
    }
}

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

//! CNF formulas

use crate::model::Model;
use crate::utils::Input;
use anyhow::Context;
use fixedbitset::FixedBitSet;
use num_bigint::{BigUint, ToBigInt};
use num_rational::BigRational;
use num_traits::{One, ToPrimitive, Zero};
use serde::{Serialize, Serializer};
use std::fmt::Display;
use std::io::Read;
use std::ops::{Div, Sub};
#[cfg(test)]
use std::path::Path;
use std::{collections::BTreeMap, ops::Add};
use tracing::trace;
pub use varisat_dimacs::{write_dimacs, DimacsHeader, DimacsParser};
pub use varisat_formula::{CnfFormula, ExtendFormula, Lit, Var};

/// Converts a string to dimacs in a string, for debugging.
pub fn to_dimacs_string(f: &CnfFormula) -> anyhow::Result<String> {
    let mut buf = Vec::new();
    write_dimacs(&mut buf, f)?;
    Ok(String::from_utf8(buf)?)
}

#[derive(Debug, Clone, Copy)]
struct FoundHeader(DimacsHeader);

impl std::fmt::Display for FoundHeader {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for FoundHeader {}

/// stops parsing on first time there is a header
fn found_header(p: &mut DimacsParser) -> anyhow::Result<()> {
    match p.header() {
        Some(h) => Err(FoundHeader(h).into()),
        None => Ok(()),
    }
}

/// Returns the header of the CNF formula in argument, but stops parsing as soon as possible.
pub fn read_header_only(read: impl Read) -> anyhow::Result<DimacsHeader> {
    let res = DimacsParser::parse_incremental(read, found_header);
    match res {
        Ok(finished_parser) => Ok(finished_parser
            .header()
            .context("getting header of parsed formula")?),
        Err(e) => match e.downcast_ref::<FoundHeader>() {
            Some(FoundHeader(h)) => Ok(*h),
            None => Err(e),
        },
    }
}

/// returns a random 3sat cnf formula with specified number of variables and clauses
#[cfg(test)]
pub fn generate_random_3sat(nvars: usize, nclauses: usize) -> CnfFormula {
    use std::iter::FromIterator;
    let mut f = CnfFormula::new();
    let mut rng = rand::thread_rng();
    f.set_var_count(nvars);
    for _ in 0..nclauses {
        let chosen = rand::seq::index::sample(&mut rng, nvars, 3);
        let clause = Vec::from_iter(chosen.iter().map(|v| {
            let v = Var::from_index(v);
            Lit::from_var(v, rand::random())
        }));
        f.add_clause(&clause);
    }
    f
}

#[cfg(test)]
mod test {
    use super::*;
    /// implements Read, returns some data over and over
    struct ReadLoop {
        data: &'static [u8],
        index: usize,
    }
    impl Read for ReadLoop {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            let size = std::cmp::min(buf.len(), self.data.len() - self.index);
            buf[0..size].copy_from_slice(&self.data[self.index..(self.index + size)]);
            self.index += size;
            self.index = self.index % self.data.len();
            Ok(size)
        }
    }
    impl ReadLoop {
        fn new(data: &'static [u8]) -> Self {
            Self { data, index: 0 }
        }
    }
    #[test]
    fn read_only_the_header() {
        let h = read_header_only(ReadLoop::new(b"p cnf 12 13\n 0")).unwrap();
        assert_eq!(h.var_count, 12);
        assert_eq!(h.clause_count, 13);
    }
}

/// returns the model count of the formula by testing all models sequentially.
pub fn brute_force_model_count(f: &CnfFormula) -> ModelCount {
    let mut m = Model::first_model(f.var_count());
    let mut n = BigUint::zero();
    loop {
        if m.satisfies(&f) {
            n += BigUint::one();
        }
        if !m.next_model() {
            break;
        }
    }
    ModelCount {
        model_count: n,
        bits: f.var_count(),
    }
}

#[test]
fn test_brute_force_mc() -> anyhow::Result<()> {
    let f = b"p cnf 3 3
-1 3 -2 0
-2 1 3 0
3 -1 -2 0
";
    let formula = varisat_dimacs::DimacsParser::parse(f as &[u8])?;
    let mc = brute_force_model_count(&formula);
    // FIXME, does not make sense
    assert_eq!(
        mc,
        ModelCount {
            model_count: 6u32.into(),
            bits: 3
        }
    );
    Ok(())
}

/// returns what controlled models have how many uncontrolled models
pub fn brute_force_popcon(
    f: &CnfFormula,
    controlled: &fixedbitset::FixedBitSet,
) -> crate::utils::CountingMap<crate::model::Model> {
    let mut m = Model::first_model(f.var_count());
    let mut counts = crate::utils::CountingMap::new();
    loop {
        if m.satisfies(&f) {
            let mut controlled_part = m.clone();
            controlled_part.0.intersect_with(&controlled);
            counts[&controlled_part] += BigUint::one();
        }
        if !m.next_model() {
            break;
        }
    }
    counts
}

/// Model counting by brute force
pub struct BruteForce;

impl CNFCounter for BruteForce {
    fn model_count(
        &self,
        input: impl Input,
        _: Option<&FixedBitSet>,
    ) -> anyhow::Result<ModelCount> {
        let file = input.to_read().with_context(|| {
            format!("opening {} for brute force model counting", input.display())
        })?;
        let formula = DimacsParser::parse(file).with_context(|| {
            format!("parsing {} for brute force model counting", input.display())
        })?;
        Ok(brute_force_model_count(&formula))
    }
}

impl CNFPopularityContest for BruteForce {
    fn popularity_contest(
        &self,
        input: impl Input,
        controlled: &FixedBitSet,
        _: &FixedBitSet
    ) -> anyhow::Result<(Model, PopConBounds)> {
        let formula = DimacsParser::parse(input.to_read()?)
            .with_context(|| format!("parsing {} for brute force popcon", input.display()))?;
        let n_uncontrolled = controlled.len() - controlled.count_ones(..);
        let nvars = formula.var_count();
        anyhow::ensure!(
            controlled.len() == nvars,
            "{} variables in control information whereas formula {} has {} variables",
            controlled.len(),
            input.display(),
            nvars
        );
        let map = brute_force_popcon(&formula, controlled);
        let (model, count) = map.argmax();
        let model = model.cloned().unwrap_or_else(|| Model::empty(nvars));
        let count = PopConBounds::exactly(ModelCount {
            model_count: count.clone(),
            bits: n_uncontrolled,
        });
        Ok((model, count))
    }
}

impl CNFProjectedCounter for BruteForce {
    fn projected_model_count(
        &self,
        input: impl Input,
        projection_variables: &FixedBitSet,
    ) -> anyhow::Result<ModelCount> {
        let formula = DimacsParser::parse(input.to_read()?)
            .with_context(|| format!("parsing {} for brute force popcon", input.display()))?;
        let nvars = formula.var_count();
        anyhow::ensure!(
            projection_variables.len() == nvars,
            "{} variables in projection information whereas formula {} has {} variables",
            projection_variables.len(),
            input.display(),
            nvars
        );
        let map = brute_force_popcon(&formula, projection_variables);
        let count = ModelCount {
            model_count: map.count_non_zero().into(),
            bits: projection_variables.count_ones(..),
        };
        Ok(count)
    }
}

fn biguint_to_string<S: Serializer>(n: &BigUint, s: S) -> Result<S::Ok, S::Error> {
    s.serialize_str(&n.to_string())
}

/// The result of counting models on a formula
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ModelCount {
    /// Number of models of the formula
    #[serde(serialize_with = "biguint_to_string")]
    pub model_count: BigUint,
    /// Number of bits of the full model space
    pub bits: usize,
}

impl PartialOrd for ModelCount {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self.bits != other.bits {
            None
        } else {
            self.model_count.partial_cmp(&other.model_count)
        }
    }
}

impl ModelCount {
    /// 1 model in 3 variables to 4 models in 5 variables
    #[must_use]
    pub fn upgrade_to(&self, bits: usize) -> ModelCount {
        assert!(bits >= self.bits, "{:?}.upgrade_to({})", &self, bits);
        ModelCount {
            model_count: &self.model_count << (bits - self.bits),
            bits,
        }
    }

    /// Sets the number of bits without changing model_count.
    ///
    /// Errors out when model_count becomes too large for the number of bits.
    pub fn set_bits(&mut self, bits: usize) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.model_count.bits() <= 1 + bits as u64,
            "model count {} has more bits than variable size {}",
            self,
            bits
        );
        self.bits = bits;
        Ok(())
    }
}

impl Display for ModelCount {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let influence = crate::utils::log2(&self.model_count);
        write!(
            f,
            "Models: {}, Influence {}, Total bits: {}, Incidence {}",
            &self.model_count,
            influence,
            self.bits,
            influence - self.bits as f32
        )
    }
}

/// Bounds for popcon computation
///
/// Invariant: self.lower.bits == self.upper.bits and self.lower <= self.upper
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PopConBounds {
    /// Lowerbound, included
    lower: ModelCount,
    /// Upper bound, included
    upper: ModelCount,
}

impl PopConBounds {
    /// Exact popcon result
    pub fn exactly(m: ModelCount) -> PopConBounds {
        let lower = m.clone();
        let upper = m;
        PopConBounds { lower, upper }
    }

    /// Creates a PopConBounds from an interval of possible popularity.
    ///
    /// Errors when lower.bits != upper.bits or upper < lower
    pub fn from_range(lower: ModelCount, upper: ModelCount) -> anyhow::Result<PopConBounds> {
        anyhow::ensure!(
            lower.bits == upper.bits,
            "mismatch lower.bits = {}, upper.bits = {}",
            lower.bits,
            upper.bits
        );
        anyhow::ensure!(
            upper.model_count >= lower.model_count,
            "mismatch lower.model_count = {} > upper.model_count = {}",
            &lower.model_count,
            &upper.model_count
        );
        Ok(PopConBounds { lower, upper })
    }

    /// Lifts ModelCount::upgrade_to
    #[must_use]
    pub fn upgrade_to(self, bits: usize) -> PopConBounds {
        PopConBounds {
            lower: self.lower.upgrade_to(bits),
            upper: self.upper.upgrade_to(bits),
        }
    }

    /// Gets the number of bits.
    pub fn bits(&self) -> usize {
        self.lower.bits
    }

    /// Lower bound of the interval for popcon, included.
    pub fn lower(&self) -> &ModelCount {
        &self.lower
    }

    /// Upper bound of the interval for popcon, included.
    pub fn upper(&self) -> &ModelCount {
        &self.upper
    }

    /// If the interval is a singleton, return its value. Else, return `Err(self)`.
    pub fn to_exact(self) -> Result<ModelCount, PopConBounds> {
        if self.lower == self.upper {
            Ok(self.lower)
        } else {
            Err(self)
        }
    }

    /// Change the number of bits without affecting the model_count.
    /// Errors if lower.model_count > 2^bits
    /// Caps the upper bound to 2^bits.
    pub fn set_bits(&mut self, bits: usize) -> anyhow::Result<()> {
        if self.upper.set_bits(bits).is_err() {
            self.upper = ModelCount {
                model_count: BigUint::one() << bits,
                bits,
            };
        };
        self.lower.set_bits(bits)
    }

    /// Gives a float approximation of upper/lower, can be +inf.
    pub fn up_to_low_ratio(&self) -> f32 {
        if self.lower.model_count.is_zero() {
            if self.upper.model_count.is_zero() {
                1.
            } else {
                f32::INFINITY
            }
        } else {
            let exact = BigRational::new(
                self.upper.model_count.to_bigint().unwrap(),
                self.lower.model_count.to_bigint().unwrap(),
            );
            exact.to_f32().unwrap_or(f32::INFINITY)
        }
    }
}

impl PartialEq<ModelCount> for PopConBounds {
    fn eq(&self, other: &ModelCount) -> bool {
        self.lower == self.upper && &self.lower == other
    }
}

/// returns a representation of min..max in middle±error
fn error_bars<T>(min: &T, max: &T) -> (T, T)
where
    for<'a> &'a T: Add<Output = T> + Sub<Output = T> + Div<Output = T>,
    T: PartialOrd + From<u8>,
{
    let middle = &(min + max) / &2u8.into();
    let ea = max - &middle;
    let eb = &middle - min;
    let error = if ea > eb { ea } else { eb };
    (middle, error)
}

impl Display for PopConBounds {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let (cm, ce) = error_bars(&self.lower.model_count, &self.upper.model_count);
        let (im, ie) = error_bars(
            &crate::utils::log2(&self.lower.model_count),
            &crate::utils::log2(&self.upper.model_count),
        );
        if self.lower.bits != self.upper.bits {
            write!(
                f,
                "Inconsistent !!!!! {} != {} ",
                self.lower.bits, self.upper.bits
            )?;
        }
        let relative_error = if cm.is_zero() {
            f64::NAN
        } else {
            (&ce * 100u32 / &cm).to_f64().unwrap_or(f64::NAN)
        };
        write!(
            f,
            "Models: {}±{} ({}%), Influence {:.1}±{:.1}, Total bits: {}, Incidence {:.1}±{:.1}",
            cm,
            ce,
            relative_error,
            im,
            ie,
            self.lower.bits,
            im - self.lower.bits as f32,
            ie,
        )
    }
}

/// A trait for methods to count models
pub trait CNFCounter {
    /// returns the number of models of the CNF formula in this file
    /// If projection_variables is not None it must be an independent set: a set of variables such
    /// that when a valuation affects these variables, there is exactly one or zero model for the
    /// other variables.
    fn model_count(
        &self,
        input: impl Input,
        projection_variables: Option<&FixedBitSet>,
    ) -> anyhow::Result<ModelCount>;
}

/// A trait for methods to count models with projection
pub trait CNFProjectedCounter {
    /// returns the number of models of the CNF formula in this file with projection over the
    /// specified set of variables.
    ///
    /// In other words, returns the number of assignments of this variables that leave the formula
    /// satisfiable.
    fn projected_model_count(
        &self,
        input: impl Input,
        projection_variables: &FixedBitSet,
    ) -> anyhow::Result<ModelCount>;
}

/// A trait for methods to solve popularity contest on CNF formulas
pub trait CNFPopularityContest {
    /// Returns an interval for the maximum popularity of this CNF formula.
    /// Additionally returns a model reaching the lower bound. In case the interval is a
    /// singleton, this model is actually the model with maximal popularity.
    /// Returns an error is `controlled_variables` does not have the same size as the number
    /// of variables in the file.
    fn popularity_contest(
        &self,
        input: impl Input,
        controlled_variables: &FixedBitSet,
        uncontrolled_variables: &FixedBitSet
    ) -> anyhow::Result<(Model, PopConBounds)>;
}

/// Wraps a solver to disable the optimisation by projection
pub struct NoProjection<T>(pub T);

impl<T: CNFCounter> CNFCounter for NoProjection<T> {
    fn model_count(
        &self,
        input: impl Input,
        _projection_variables: Option<&FixedBitSet>,
    ) -> anyhow::Result<ModelCount> {
        self.0.model_count(input, None)
    }
}

impl<T: CNFPopularityContest> CNFPopularityContest for NoProjection<T> {
    fn popularity_contest(
        &self,
        input: impl Input,
        controlled_variables: &FixedBitSet,
        uncontrolled_variables: &FixedBitSet
    ) -> anyhow::Result<(Model, PopConBounds)> {
        self.0.popularity_contest(input, controlled_variables, uncontrolled_variables)
    }
}

/// Wraps a solver by removing unused variable indices from the formula
pub struct Compacter<T>(pub T);

impl<T: CNFCounter> CNFCounter for Compacter<T> {
    fn model_count(
        &self,
        input: impl Input,
        projection_variables: Option<&FixedBitSet>,
    ) -> anyhow::Result<ModelCount> {
        let res =
            compact(input, None, None, projection_variables).context("compacting for model counting")?;
        let res = match res {
            None => {
                // original formula was already compact
                return self.0.model_count(input, projection_variables);
            }
            Some(res) => res,
        };
        let mut string = Vec::new();
        varisat_dimacs::write_dimacs(&mut string, &res.formula)
            .context("writing compacted formula to buffer")?;
        let mut raw = self
            .0
            .model_count(&string as &[u8], res.projection_variables.as_ref())
            .context("counting models on compacted formula")?;
        trace!(result = ?raw, "model count of compacted formula");
        anyhow::ensure!(
            raw.bits == res.nvars,
            "model count on {} vars has only {} bits",
            res.nvars,
            raw.bits
        );
        raw.model_count <<= res.original_nvars - res.nvars;
        raw.bits = res.original_nvars;
        trace!(result = ?raw, "model count of uncompacted formula");
        Ok(raw)
    }
}

impl<T: CNFPopularityContest> CNFPopularityContest for Compacter<T> {
    fn popularity_contest(
        &self,
        input: impl Input,
        controlled_variables: &FixedBitSet,
        uncontrolled_variables: &FixedBitSet
    ) -> anyhow::Result<(Model, PopConBounds)> {
        let res = compact(input, Some(controlled_variables), Some(uncontrolled_variables), None)
            .context("compacting for popularity_contest")?;
        let res = match res {
            None => {
                // original formula was already compact
                return self.0.popularity_contest(input, controlled_variables, uncontrolled_variables);
            }
            Some(res) => res,
        };
        let compact_controlled_variables = match res.controlled_variables {
            None => anyhow::bail!("compaction did not return controlled variables"),
            Some(c) => c,
        };
        let compact_uncontrolled_variables = match res.uncontrolled_variables {
            None => anyhow::bail!("compaction did not return uncontrolled variables"),
            Some(c) => c,
        };
        let mut compacted_string = Vec::new();
        varisat_dimacs::write_dimacs(&mut compacted_string, &res.formula)
            .context("writing compacted formula to temp file")?;
        let (model, count) = self
            .0
            .popularity_contest(&compacted_string as &[u8], &compact_controlled_variables, &compact_uncontrolled_variables)
            .context("popcon on compacted formula")?;
        trace!(result = ?count, "model count of compacted formula");
        anyhow::ensure!(
            count.bits() == res.nvars - res.ncontrolled,
            "popcon on {} vars - {} controlled vars = {} uncontrolled vars has only {} bits",
            res.nvars,
            res.ncontrolled,
            res.nvars - res.ncontrolled,
            count.bits()
        );
        let count = count.upgrade_to(res.original_nvars - res.original_ncontrolled);
        trace!(result = ?count, "model count of uncompacted formula");
        let mut translated_model = Model::empty(res.original_nvars);
        for (new, &old) in res.indexmap.iter().enumerate() {
            translated_model.set(old, model[new])
        }
        Ok((translated_model, count))
    }
}

struct CompactionInfo {
    formula: CnfFormula,
    controlled_variables: Option<FixedBitSet>,
    uncontrolled_variables: Option<FixedBitSet>,
    projection_variables: Option<FixedBitSet>,
    original_nvars: usize,
    nvars: usize,
    original_ncontrolled: usize,
    ncontrolled: usize,
    /// var v in the new formula corresponds to var `indexmap[v.index()]` in the old one.
    indexmap: Vec<Var>,
}

/// Rewrites a new formula, equivalent formula but with contiguous variables used.
/// Example: `1 & (4|3)` -> `1 & (2&3)`. If there is no variable to spare, then `None`
/// is returned.
fn compact(
    input: impl Input,
    controlled_variables: Option<&FixedBitSet>,
    uncontrolled_variables: Option<&FixedBitSet>,
    projection_variables: Option<&FixedBitSet>,
) -> anyhow::Result<Option<CompactionInfo>> {
    let orig = varisat_dimacs::DimacsParser::parse(input.to_read()?)
        .with_context(|| format!("parsing cnf {} for compaction", input.display()))?;
    let mut old_to_new = BTreeMap::new();
    let mut formula = CnfFormula::new();
    let mut new = Vec::new();
    for clause in orig.iter() {
        new.clear();
        new.reserve(clause.len());
        for &lit in clause {
            let v = lit.var();
            let next_free = old_to_new.len();
            let v2 = *old_to_new.entry(v).or_insert(Var::from_index(next_free));
            new.push(lit.map_var(|_| v2));
        }
        formula.add_clause(&new);
    }
    let nvars = old_to_new.len();
    if nvars == orig.var_count() {
        return Ok(None);
    }
    let mut indexmap = vec![Var::from_index(0); nvars];
    for (old, new) in old_to_new.iter() {
        indexmap[new.index()] = *old;
    }
    let (controlled_variables, original_ncontrolled, ncontrolled) = match controlled_variables {
        None => (None, 0, 0),
        Some(o) => {
            let mut v = FixedBitSet::with_capacity(nvars);
            for old in o.ones() {
                if let Some(new) = old_to_new.get(&Var::from_index(old)) {
                    v.insert(new.index())
                }
            }
            let n = v.count_ones(..);
            (Some(v), o.count_ones(..), n)
        }
    };
    let uncontrolled_variables = match uncontrolled_variables {
        None => None,
        Some(o) => {
            let mut v = FixedBitSet::with_capacity(nvars);
            for old in o.ones() {
                if let Some(new) = old_to_new.get(&Var::from_index(old)) {
                    v.insert(new.index())
                }
            }
            Some(v)
        }
    };
    let projection_variables = match projection_variables {
        None => None,
        Some(o) => {
            let mut v = FixedBitSet::with_capacity(nvars);
            for old in o.ones() {
                if let Some(new) = old_to_new.get(&Var::from_index(old)) {
                    v.insert(new.index())
                }
            }
            Some(v)
        }
    };
    trace!(
        compaction_before_vars = orig.var_count(),
        compaction_before_clauses = orig.len(),
        removed_vars_by_compaction = orig.var_count() - nvars,
        stats = true,
    );
    Ok(Some(CompactionInfo {
        formula,
        indexmap,
        controlled_variables,
        uncontrolled_variables,
        original_nvars: orig.var_count(),
        nvars,
        original_ncontrolled,
        ncontrolled,
        projection_variables,
    }))
}

#[test]
fn compaction() -> anyhow::Result<()> {
    let mut controlled = FixedBitSet::with_capacity(9);
    controlled.insert(0);
    controlled.insert(3);
    let mut proj = FixedBitSet::with_capacity(9);
    proj.insert(8);
    let res = compact(
        "assets/d4_popconbug.cnf".as_ref() as &Path,
        Some(&controlled),
        None,
        Some(&proj),
    )?;
    let res = res.expect("formula was already compact !!!");
    let expected_formula = varisat_dimacs::DimacsParser::parse(
        b"p cnf 3 2
1 0
2 3 0
" as &[u8],
    )?;
    let mut expected_controlled = FixedBitSet::with_capacity(3);
    expected_controlled.insert(1);
    let mut expected_proj = FixedBitSet::with_capacity(3);
    expected_proj.insert(2);
    assert_eq!(res.formula, expected_formula);
    assert_eq!(res.projection_variables, Some(expected_proj));
    assert_eq!(
        res.indexmap,
        vec![
            Var::from_dimacs(6),
            Var::from_dimacs(4),
            Var::from_dimacs(9)
        ]
    );
    assert_eq!(res.controlled_variables, Some(expected_controlled));
    assert_eq!(res.nvars, 3);
    assert_eq!(res.original_nvars, 9);
    assert_eq!(res.original_ncontrolled, 2);
    assert_eq!(res.ncontrolled, 1);
    Ok(())
}

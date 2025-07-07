# Popcon

Popcon is a front-end for various counting and functional E-MAJSAT solvers. It also has some built-in solving algorithms based on d-DNNF compilation.

## Terminology

Note that because of its eventful history, popcon (and much of the documentation and the code) designates E-MAJSAT under the name "popularity contest".
Given a boolean formula `f(a, x)`:
- for a given value of `a`, the model count of `f(a, x)` is called popularity of `a`
- the popularity contest of `f` is the value of `a` with the maximal popularity, along with its popularity.

Variables in `a` are usually called choice variables in the literature, and
`x` chance variables, but we call them controlled and uncontrolled variables
respectively.

Popularity contest is abbreviated into popcon, hence the name of the tool.

## Compiling

For full functionality, you need two patched versions of d4 and a patched version of ssatABC, etc. As this is quite complex we only offer compilation through [nix](https://nixos.org/download.html#nix-install-linux):

#### For a release build
`nix-build` creates a symlink `result` in the current directory such that `result/bin/popcon` is the built executable.

#### For development
Run `nix-shell`. In the shell that opens, you can develop as usual for a rust project: `cargo build`, `cargo test`, etc. It's also a good way to use the underlying solvers.

## Quickstart

Popcon can compute the model count of a formula (the number of satisfying models), its popularity contest (explained previously), and projected model counts. The projected model count for a formula `f(a,x)` with `a` as a projection variable is the number of values of `a` for which `f(a,x)` is still satisfiable.

For example, let us compute the popularity contest for `x < n` with single-byte variables and `n` controlled, i.e., the following formula ([8.smt2](bench/compare/8.smt2)):
```
(set-logic QF_BV)
(declare-fun n () (_ BitVec 8))
(declare-fun x () (_ BitVec 8))
(set-info :controlled n)
(assert (bvult x n))
```
The controlled status of `n` is set using `set-info`.
If we wanted to do projected model counting with `n` as a projection, we would use `set-info :projected n`.

We would run the solver with the following command:
```
popcon -g popcon bench/compare/8.smt2
```
`-g popcon` selects popularity contest as the problem to solve.
You can also use `modelcount` or `projectedmodelcount` for those other problems.

In the output, we get the maximum popularity (`Models: 255`) and the best model for `n` (`0b11111111`), as well as other metrics (read the output format section for more information):
```
Max popularity: Models: 255±0 (0%), Influence 7.9±0.0, Total bits: 8, Incidence -0.1±0.0
Best model: n: 0b11111111,
```

## Detailed input format
The extension of the input file is used to determine the input format:

#### d-DNNF
`file.nnf` is for a formula in d-DNNF format as output by `c2d`. The format is full ASCII:
The first line is `nnf 12 14 15` where 12 is the number of variables, 14 is the number of nodes and 15 is the number of edges.
Nodes are then spelled one per line in topological order; one refers to a node as its index in this list starting from 1:
- `A` for an and node followed its number of children and the by space separated indices of the children
- `L` for a literal node followed by the literal (variable number starting from 1 or minus variable number for a negated literal)
- `O 2 v a b` for an or node between node indices `a` and `b`. `v` can be 0 or an opposing variable (a variable that is implied to be false by one child and true by the other).
Not all solving algorithms support `v = 0`.
- `O 0 0` a false node.

d-DNNF input is only supported by a subset of solving algorithms.

#### DIMACS
`file.cnf` is for a formula in DIMACS format.
Lines starting with `c` are comments and ignored.
The first non-comment line must be `p cnf a b` where `a` is the number of variables and `b` the number of clauses.
Then all other lines must be clauses given as a space separated list of literals (variable index starting from 1, with a minus if negated) ended by 0.

In the case of popcon and projected model count you need to indicate the partition of variables into `a` and `x`:
* `popcon --goal popcon foo.cnf -c 1 2 3 4 5`
* `popcon --goal projectedmodelcount -p 1 2 3 4`

#### SMTLib2
`file.smt2` is a file in SMTLib2 formal in the theory of quantifier free bitvectors (`QF_BV`).
In the case of popcon and projected model count, you must indicate the partition of variables into `a` and `x`. For a bitvector variable `foo` symbol:
- for popcon, `(set-info :controlled foo)` marks `foo` as controlled, unmarked variables are uncontrolled
- for projected model count, `(set-info :projected foo)` marks `foo` as projected.

## Output format
```commands
$ popcon --goal modelcount assets/choose_interval.smt2 
Models: 121, Influence 6.9, Total bits: 9, Incidence -2.1
```
This formula has 121 models. The influence is the logarithm of base 2 of the number of models. "Total bits" indicates that the corresponding true formula would have `2^9` models. Incidence is (total bits - influence), giving an idea of how many assignments are not satisfying this formula.

A machine parsable output is available with `--json outputfile`:
```commands
$ popcon --goal modelcount assets/choose_interval.smt2 --json /dev/stdout
{
  "model_count": "121",
  "bits": 9
}
```

For popcon on smt2 input, one value of controlled variables reaching the maximal popularity is reported. Additionally, the value of popcon is reported as an interval (possibly a singleton interval) depending on the algorithm used.
```commands
$ popcon --goal popcon assets/choose_interval.smt2             
Max popularity: Models: 69±0 (0%), Influence 6.1±0.0, Total bits: 8, Incidence -1.9±0.0
Best model: thechoice: 0,
```
`0%` indicates the ratio `upper bound of the interval/lower bound of the interval - 1`. Here it means the result is exact. When the result is not exact, the model indicated reaches the lower bounds (except for some backends where it is not implemented).
Models for bitvectors are printed as `0b01011010XXXX1001` where `X` means that this SMT level bit was stripped at bit-blasting time, and therefore that the value of the formula does not depend on this bit.

Reporting the model is not implemented in JSON output:
```commands
$ popcon --goal popcon assets/choose_interval.smt2 --json /dev/stdout
{
  "lower": {
    "model_count": "69",
    "bits": 8
  },
  "upper": {
    "model_count": "69",
    "bits": 8
  }
}
```

Debug level is enabled by `--debug` and a lot of statistics are output by `--stats file.json`
Stats are key values where keys are strings and values are integers. As some code paths outputting key = value can be called several times, a key `foo` will yield several keys in the JSON stats:
- `key_min`: min value
- `key_max`: max value
- `key_count`: number of times the value was output
- `key_last`: last value
- `key_first`: first value
- `key_sum`: sum


## Supported resolution algorithms

### Model counting
- with c2d `--tool c2d`
- with [dsharp](https://github.com/QuMuLab/dsharp) `--tool dsharp`
- with [d4](https://github.com/crillab/d4) `--tool d4`, default

### Projected model counting
- with [dsharp](https://github.com/QuMuLab/dsharp) `--tool dsharp`
- with [d4](https://github.com/crillab/d4) `--tool d4`, default

### Popularity contest
- with [ssatABC](https://github.com/NTU-ALComLab/ssatABC) `--tool ssatabc`, does not return the correct model (always full 0)
- with [maxcount](https://github.com/dfremont/maxcount) `--tool maxcount --maxcount-k 1`. `--maxcount-k` is a parameter that can be increased to improve precision at the expense of performance.
- with dc-ssat: `--tool dc-ssat`, does not return the correct model (always full 0)
- with the exact d-DNNF constrained algorithm: `--tool d4` or `--tool dsharp`
- with complan+: `--tool d4|c2d|dsharp --unconstrained --unconstrained-exact`
- with just the upper bound of complan+: `--tool c2d --unconstrained`. The lower bound can be chosen with `--lower-bound bad|fast|precise`
- with [BaxMC](https://gricad-gitlab.univ-grenoble-alpes.fr/tava/baxmc): `-tool BAXMC`. BaxMC is a CEGAR-based approximate solver. Additional specific options can be added with `-baxmc-opts`.
- with relaxation, there are many options. The common one is `--relax n` where n is an integer. 0 (the default) is equivalent to not specifying the option. Only `--tool d4` will work, relaxation related options are ignored by other tools.
  * `--relax-bfs` use BFS to choose relaxed variables instead of DFS
  * `--relax-exact` convert the relaxed and simplified d-DNNF to CNF back and start with the exact method again. tends to always timeout.
  * `--simplify-no-*` disables some simplifications which have high complexity and rarely bring better intervals (but it happens!)
  * `--lower-bound bad|fast|precise`, default `fast`. `bad` is really imprecise and not less expensive than `fast`, and `precise` is only sometimes more precise, but quadratic instead of linear.


## Timeout

You can pass a timeout to popcon with `--timeout`, in milliseconds. We patched some of the backends so that there is no left-over process after the timeout is reached.
There is no support for reporting an imprecise result when this occurs.

## Publication

- [Quantitative Robustness for Vulnerability Assessment](https://doi.org/10.1145/3656407) (PLDI 2024)

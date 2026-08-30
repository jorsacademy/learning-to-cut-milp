# Learning to Cut for MILP

A reproducible research sandbox for **machine-learning-guided cutting-plane selection** in mixed-integer linear programming.

The project deliberately separates two questions that are easy to conflate:

1. **Is a cut mathematically valid?**
2. **Among already valid cuts, which cut should be added next?**

The machine-learning component only answers the second question. Candidate inequalities are generated as valid minimal cover cuts from multidimensional 0-1 knapsack constraints. A learned scorer ranks those candidates; it never invents arbitrary inequalities and therefore cannot make the MILP invalid by itself.

The project is intended as a transparent bridge between classical integer programming and learning-for-optimization. It is not a reimplementation of the cut-management internals of SCIP, Gurobi, CPLEX, or Xpress.

## Mathematical problem

The synthetic instances are multidimensional binary knapsack problems:

```text
maximize      c^T x
subject to    A x <= b
              x in {0,1}^n
```

The root LP relaxation replaces binary restrictions with:

```text
0 <= x_i <= 1
```

For a resource row `r`, a set of items `C` is a cover if:

```text
sum(i in C) A[r,i] > b[r]
```

All items in a cover cannot simultaneously equal one. Therefore the inequality

```text
sum(i in C) x_i <= |C| - 1
```

is valid for every integer-feasible solution. The generator further reduces covers until they are minimal: removing any one item destroys the cover property.

Every cut is checked against its declared source row before it can enter either the LP or MILP model.

## Learning-to-cut pipeline

```text
multidimensional 0-1 knapsack MILP
                 |
                 v
          root LP relaxation
                 |
                 v
   violated minimal cover candidates
                 |
                 v
        transparent cut features
                 |
        +--------+---------+-------------+----------+
        |                  |             |          |
     random            efficacy       learned     oracle
        |                  |             |          |
        +--------+---------+-------------+----------+
                 |
                 v
           select one cut
                 |
                 v
              re-solve LP
                 |
                 v
        repeat for a cut budget
```

The separator is intentionally heuristic. It creates a diverse pool by combining deterministic orderings and reproducible randomized orderings for each knapsack resource row. This isolates **selection quality** from the much larger problem of implementing a production-grade separator library.

## Expert labels

The supervised expert uses one-step LP lookahead. If the current maximization LP upper bound is `z_LP` and candidate cut `C` would produce bound `z_LP(C)`, its expert score is:

```text
Delta(C) = z_LP - z_LP(C)
```

A larger value means the candidate tightens the current relaxation more strongly in that one step.

This is intentionally expensive: scoring `k` candidates requires `k` extra LP solves. The oracle selector is therefore a training/evaluation reference rather than a deployable policy.

## Sequential training states

Training only on root-node candidates creates a distribution-shift problem: after the first selected cut, the LP solution changes, which changes later candidate pools and features.

`collect_cut_dataset` therefore gathers states over several cut rounds. Rollouts alternate among:

- oracle selection;
- efficacy selection;
- random selection.

Every encountered candidate pool is labeled by the same one-step expert. The mixed rollouts expose the scorer to states induced by both strong and imperfect previous decisions.

## Learned scorer

The model is a small PyTorch MLP that scores each candidate cut. Features are standardized using statistics from the training set. Training uses groupwise cross-entropy: for each LP state, the target is the candidate with the largest expert one-step bound improvement.

Current cut features are:

- violation;
- efficacy;
- cut density;
- normalized cover excess;
- objective parallelism;
- mean fractionality on the cut support;
- mean LP value on the cut support;
- source-row activity ratio;
- share of total objective profit on the cut support.

The feature set is deliberately inspectable. Modern solver cut selectors can use richer context such as directed cutoff distance, integer support, pseudo-costs, locks, dynamism, and pairwise parallelism filtering. Those mechanisms are outside this repository's scope.

## Selection policies

Five policies are implemented:

| Policy | Selection rule | Extra LP solves for selection |
| --- | --- | ---: |
| `random` | uniform candidate choice | 0 |
| `efficacy` | largest distance-style efficacy | 0 |
| `hybrid` | transparent weighted combination of efficacy, violation, objective parallelism, and density | 0 |
| `learned` | MLP candidate score | 0 |
| `oracle` | largest one-step LP-bound improvement | one per candidate |

The `hybrid` score is a teaching baseline. It is **not** claimed to reproduce SCIP's hybrid or ensemble cut selector.

## Correct evaluation: integrality-gap closure

Absolute LP-bound improvement is scale dependent, so the benchmark also solves the small binary MILP with `scipy.optimize.milp` and reports integrality-gap closure.

For a maximization problem:

```text
UB_0 = root LP upper bound
UB_k = LP upper bound after k selected cuts
z_IP = exact small-instance integer optimum
```

The reported metric is:

```text
gap closure = (UB_0 - UB_k) / (UB_0 - z_IP)
```

This distinguishes genuine progress toward the integer hull from merely reporting a raw change in LP objective.

## Development benchmark

The following development run was executed with disjoint training, validation, and benchmark seed ranges:

```bash
python scripts/train.py \
  --instances 120 \
  --validation-instances 40 \
  --items 20 \
  --constraints 4 \
  --rounds 3 \
  --epochs 160 \
  --hidden-dim 48 \
  --learning-rate 0.002 \
  --seed 2026 \
  --checkpoint checkpoints/cut_scorer.pt

python scripts/benchmark.py \
  --checkpoint checkpoints/cut_scorer.pt \
  --instances 60 \
  --items 20 \
  --constraints 4 \
  --rounds 5 \
  --seed-start 202600
```

Training produced 287 candidate groups and 1,725 candidate cuts. On the separate validation set, top-1 expert agreement was approximately `70.1%` and mean normalized one-step regret was approximately `0.0966`.

The 60-instance unseen benchmark produced the following mean integrality-gap closures:

```text
random      0.1213
efficacy    0.1361
hybrid      0.1365
learned     0.1366
oracle      0.1377
```

The oracle used about `15.6` additional LP solves per instance for candidate lookahead. The learned, efficacy, hybrid, and random selectors used no extra LP solves for selection.

The paired mean difference in gap closure between `learned` and `efficacy` was approximately `+0.00053` with standard error `0.00123`. Therefore this development experiment does **not** support a claim that the learned model is superior to the strong handcrafted efficacy baseline. In this deliberately simple cover-cut environment, efficacy already captures most of the useful signal. The value of the repository is the correctly controlled learning-to-rank pipeline, not a cherry-picked speedup claim.

These results are reproducible for the stated synthetic generator, seeds, software stack, and cut budget. They are not claims about industrial MILP benchmarks or production solvers.

## Installation

```bash
python -m pip install -e ".[dev]"
```

Python 3.11+ is required.

## Run a single instance

Without a learned checkpoint:

```bash
python -m learning_to_cut --items 20 --constraints 4 --rounds 5 --seed 7
```

With a trained model:

```bash
python -m learning_to_cut \
  --items 20 \
  --constraints 4 \
  --rounds 5 \
  --seed 7 \
  --checkpoint checkpoints/cut_scorer.pt
```

The CLI prints JSON containing the exact small-instance integer optimum, LP bounds, gap closure, number of cuts, and LP-solve accounting for each policy.

## Train

```bash
python scripts/train.py --checkpoint checkpoints/cut_scorer.pt
```

The training script uses a disjoint validation seed range and reports:

- number of candidate groups;
- number of candidate cuts;
- initial and final ranking loss;
- training top-1 accuracy and normalized regret;
- validation top-1 accuracy and normalized regret.

## Benchmark

```bash
python scripts/benchmark.py \
  --checkpoint checkpoints/cut_scorer.pt \
  --instances 30 \
  --rounds 6
```

Machine-readable output:

```bash
python scripts/benchmark.py \
  --checkpoint checkpoints/cut_scorer.pt \
  --instances 30 \
  --rounds 6 \
  --json
```

## Tests

```bash
pytest -q
```

The regression suite checks, among other things:

- reproducible MILP generation;
- LP relaxation upper-bound validity;
- minimal-cover validity and violation;
- preservation of the exact integer optimum after adding generated cuts;
- deterministic candidate generation under fixed seeds;
- integrality-gap closure arithmetic;
- sequential dataset group consistency;
- finite ranking training and evaluation;
- checkpoint round-trip equivalence;
- monotone LP-bound tightening;
- the one-round oracle dominating efficacy within the same candidate pool;
- benchmark summarization and paired comparison logic.

GitHub Actions runs linting, compilation, the full test suite, and a tiny end-to-end train/benchmark smoke experiment on Python 3.11 and 3.12.

## Repository structure

```text
.
├── .github/workflows/ci.yml
├── examples/run_demo.py
├── scripts/
│   ├── benchmark.py
│   └── train.py
├── src/learning_to_cut/
│   ├── __init__.py
│   ├── __main__.py
│   ├── benchmark.py
│   ├── core.py
│   ├── loop.py
│   └── model.py
├── tests/
│   ├── test_benchmark.py
│   ├── test_core.py
│   ├── test_loop.py
│   └── test_model.py
├── LICENSE
├── README.md
└── pyproject.toml
```

## Methodological boundaries

This project intentionally does **not** claim to provide:

- a general-purpose branch-and-cut solver;
- integration with a solver callback or separator plugin API;
- Gomory, MIR, flow-cover, clique, conflict, or disjunctive separation;
- learned cut generation;
- branch-node cut management;
- cut aging or deletion;
- local-versus-global validity handling;
- pairwise parallelism filtering after batch cut selection;
- solver-runtime improvements on MIPLIB;
- generalization outside the synthetic multidimensional-knapsack distribution.

The expert is also myopic: it optimizes one-step LP-bound improvement, not total branch-and-bound tree size or end-to-end solve time. A candidate that looks best in one LP resolve need not be the best cut for the eventual tree.

A production research extension would connect the selector to SCIP, generate heterogeneous cut families, use solver-state and graph features, and evaluate end-to-end primal-dual progress, node count, and solve time on train/validation/test instance families.

## Research grounding

The repository is most closely related to work on learned **cut selection**, not learned cut generation.

- Z. Huang, K. Wang, F. Liu, H.-L. Zhen, W. Zhang, M. Yuan, J. Hao, Y. Yu, and J. Wang, *Learning to Select Cuts for Efficient Mixed-Integer Programming*, Pattern Recognition 123 (2022), 108353. DOI: `10.1016/j.patcog.2021.108353`.
- SCIP cut-selector documentation describes classical scoring signals including efficacy, directed cutoff distance, objective parallelism, integer support, and parallelism filtering: `https://scipopt.org/doc/html/group__CUTSELECTORS.php`.
- A. Deza, E. B. Khalil, Z. Fan, Z. Zhou, and Y. Zhang, *Learn2Aggregate: Supervised Generation of Chvatal-Gomory Cuts Using Graph Neural Networks*, AAAI 2025. DOI: `10.1609/aaai.v39i25.34900`. This is a useful contrast because it learns part of **cut generation**, whereas this repository learns only **selection**.
- G. L. Nemhauser and L. A. Wolsey, *Integer and Combinatorial Optimization*, Wiley, 1988, for classical valid inequalities and integer-programming foundations.

The Huang et al. work formulates cut selection as a learning/ranking problem, while SCIP's documented selectors show why efficacy and objective-oriented geometric quantities are natural handcrafted baselines. Learn2Aggregate demonstrates a different learning location in the cutting-plane pipeline: learning which constraints to aggregate when generating Chvatal-Gomory cuts. The distinction is kept explicit here.

## License

PolyForm Noncommercial License 1.0.0. Commercial use is not permitted.

# Extension guide

## A Python function

```python
# my_experiment.py
import random

def evaluate(parameters, context):
    rng = random.Random(context.seed)
    # Call your simulator here. Use context.case for the scenario and
    # context.fidelity for simulation length (0 must mean full in your adapter).
    loss = (parameters['gain'] - 2.0)**2 + 1.0
    return {'loss': loss, 'max_residual': 0.0}
```

Keep the module importable and pass `PythonFunction('my_experiment:evaluate')`.
Use the `revision` and `inputs` arguments to fingerprint external dependencies
or data files not contained in the function's own source file. The library
cannot discover every dependency of arbitrary user code. No credentials should
be stored in experiment configuration or command environment overrides.

## A custom experiment class

```python
from tuning import Experiment, RunResult

class MySimulation(Experiment):
    def fingerprint(self):
        return {'simulation_version': 'v3', 'data_sha256': 'YOUR_ACTUAL_HASH'}

    def run(self, values, context):
        # Must be thread-safe when jobs > 1. Do not modify values or shared files.
        # Write artifacts only under context.directory.
        # Return a RunResult(error='...') on an expected simulator failure.
        return RunResult({'loss': 1.0, 'max_residual': 0.0})
```

Override `baseline(space)` to return the exact current configuration, and
`export(values, directory)` to write a reusable selected configuration. A returned
value map represents active explicitly set parameters; your adapter must define
what an absent optional parameter means (for example, delete it from a candidate configuration file).

Run results must be finite numeric scalar metrics. Store arrays/time series in
`context.directory`, then expose summaries such as maximum constraint violation,
mean runtime, p95 runtime or failed-solve fraction. Quality gates are applied to
these summaries in every run; choose maximum/p95 metrics where a mean would hide
unacceptable behavior.

Set `isolated_timing = True` on timing experiments. The controller then requires
`jobs=1` and `cache=False`. This only controls this study; it cannot stop unrelated
processes on the machine. Python functions are executed in threads; cancellation
or timeouts of uncooperative functions need process isolation in your adapter.

## An external process

A request file contains:

```json
{"parameters":{"gain":2.0},"case":"nominal","fidelity":200,"seed":12,"baseline":false}
```

The process writes:

```json
{"metrics":{"loss":1.2,"max_residual":0.00001}}
```

Use `CommandExperiment([executable, '{request}', '{result}'], inputs=[...])`.
Arguments are passed as an array, without a shell. Provide absolute paths for
scripts, executables and other command arguments; trials run in their own
artifact directory. The YAML `inputs` entries are resolved relative to the study
file. Include script/data paths under `inputs`, or advance `revision` when they
change. Only files declared there and the executable itself are fingerprinted.
Subprocess stdout/stderr are preserved even on failure. Timeouts kill the process
group on POSIX; Windows cleanup only guarantees termination of the immediate child.

## Domains and conditional options

```python
from tuning import Space, Cat, Log, Int, Relation

space = Space([
    Cat('strategy', ['adaptive', 'fixed'], default='adaptive'),
    Log('initial_weight', 1e-4, 100, default=0.1,
        active_if={'strategy': ['fixed']}),
    Int('min_steps', 0, 3, default=1),
    Int('max_steps', 1, 10, default=5),
    Log('optional_floor', 1e-10, 1e-4, optional=True),
], relations=[Relation('min_steps', 'max_steps')])
```

Conditions reference earlier parameters, making dependency order explicit.
Relations are checked before any simulation. Optional/conditional absent operands
make their relation inactive. Invalid combinations are recorded as rejected;
the search learns about rejection instead of crashing. Defaults describe search
proposal behavior; baseline measurements are not silently clipped to those domains.

## A custom strategy

Implement this small interface:

```python
from tuning.strategies import Proposal

class MyStrategy:
    batch_limit = 1  # or None, to use the global jobs budget

    def __init__(self, space, seed, **options):
        self.space, self.seed = space, seed

    def ask(self, index, history, start):
        # index is the durable issued-attempt index; history has completed
        # search evaluations. Use these to reconstruct state after resume.
        import random
        return Proposal(self.space.sample(random.Random(self.seed + index)))

    def tell(self, proposal, result):
        # Optional feedback. result.accepted includes all quality gates.
        pass
```

Pass `strategy_factory=MyStrategy`, `method='my_strategy'` to `Study` from Python.
Its source hash and optional `strategy_revision` become part of the resume
manifest. A custom strategy's data dependencies must be represented in its revision.
Proposal values must be JSON-compatible and lie in the declared space; relational
failures are returned as rejected evaluations. Do not mutate history evaluations.
Do not use the suggested configuration's prediction as an experimental result.

## New objectives

The built-in scalar `Objective` supports raw or baseline-relative mean metrics,
minimization/maximization and weighted penalty metrics. For another aggregation,
return the desired scalar from your experiment as a metric and optimize that.
Constraints stay independent of the objective, so a low score cannot waive a hard
quality requirement. Genuine multiobjective/Pareto search would need an extension
of the evaluation/strategy contract; it is outside this version's scope.

"""Run: PYTHONPATH=examples python examples/python_api.py"""

from tuning import Space, Float, Log, Cat, Constraint, Objective, Settings, Study
from tuning.adapters.function import PythonFunction

space = Space(
    [
        Float("x", -4, 4, default=0),
        Log("rate", 0.001, 10, default=1),
        Cat("mode", ["plain"], default="plain"),
    ]
)
study = Study(
    space,
    PythonFunction("toy_problem:evaluate", revision="1"),
    Objective("loss"),
    [Constraint("violation", 0)],
    directory="api-study",
    method="random",
    settings=Settings(cases=(0, 0.2), validation_cases=(-0.2,), jobs=2),
)
winner = study.run(trials=20)
print("No valid configuration" if winner is None else winner.values)

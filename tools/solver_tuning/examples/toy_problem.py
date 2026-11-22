"""Run from this directory, or put it on PYTHONPATH."""

import math


def evaluate(parameters, context):
    x = parameters["x"]
    rate = parameters["rate"]
    shift = float(context.case)
    loss = 1 + (x - 1.2 + shift) ** 2 + (math.log10(rate) + 1) ** 2
    if parameters["mode"] == "corrected":
        loss += 0.2 * (parameters["correction"] - 0.4) ** 2
    else:
        loss += 0.3
    return {"loss": loss, "violation": max(0, abs(x) - 3.0)}

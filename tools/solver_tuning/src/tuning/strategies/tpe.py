"""Optuna 4.x TPE via public ask/tell API; storage remains core-owned.

Model is rebuilt from completed search trials on resume. Each ask receives its
own deterministic seed, avoiding unsafe pickled sampler state. Rejected trials
are constrained and have infinite objective, not an arbitrary finite penalty.
"""

from __future__ import annotations
import math
import warnings
from .base import Proposal


class TPE:
    batch_limit = None

    def __init__(self, space, seed, startup=20, **kwargs):
        try:
            import optuna
        except ImportError as e:
            raise RuntimeError('TPE requires: pip install "solver-tuning[bo]"') from e
        if not optuna.__version__.startswith("4."):
            raise RuntimeError(
                "this release supports Optuna 4.x; install solver-tuning[bo]"
            )
        if startup < 1:
            raise ValueError("startup must be positive")
        self.optuna, self.space, self.seed, self.startup = optuna, space, seed, startup
        self.storage = optuna.storages.InMemoryStorage()
        self.study = None
        self.loaded = False
        self.pending = {}
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def _study(self, index):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", self.optuna.exceptions.ExperimentalWarning)
            sampler = self.optuna.samplers.TPESampler(
                seed=(self.seed + index * 104729) % (2**32),
                n_startup_trials=self.startup,
                multivariate=True,
                group=True,
                constant_liar=True,
                constraints_func=lambda t: t.user_attrs.get("constraints", [1.0]),
            )
        return self.optuna.create_study(
            study_name="search",
            storage=self.storage,
            direction="minimize",
            sampler=sampler,
            load_if_exists=True,
        )

    def ask(self, index, history, start):
        self.study = self._study(index)
        if not self.loaded:
            for e in history:
                # Distributions are reconstructed from actual active values.
                params, dists = self._record(e.values)
                self.study.enqueue_trial(
                    params, user_attrs={"constraints": [0.0 if e.accepted else 1.0]}
                )
                replay = self.study.ask(fixed_distributions=dists)
                self.study.tell(replay, e.score if e.accepted else math.inf)
            self.loaded = True
            self.study = self._study(index)
        trial = self.study.ask()
        values = {}
        try:
            for p in self.space:
                if not p.active(values):
                    continue
                if p.optional and not trial.suggest_categorical(
                    "__tuning__set:" + p.key, [False, True]
                ):
                    continue
                if p.kind == "cat":
                    v = trial.suggest_categorical(p.key, list(p.choices))
                elif p.kind == "int":
                    v = trial.suggest_int(p.key, int(p.low), int(p.high))
                else:
                    v = trial.suggest_float(
                        p.key, p.low, p.high, log=p.kind in ("log", "one_minus_log")
                    )
                    if p.kind == "one_minus_log":
                        v = 1 - v
                values[p.key] = v
        except BaseException:
            self.study.tell(trial, state=self.optuna.trial.TrialState.FAIL)
            raise
        self.pending[trial.number] = trial
        return Proposal(values, {"number": trial.number})

    def _record(self, values):
        from optuna.distributions import (
            FloatDistribution,
            IntDistribution,
            CategoricalDistribution,
        )

        params, dists = {}, {}
        for p in self.space:
            if not p.active(values):
                continue
            if p.optional:
                key = "__tuning__set:" + p.key
                params[key] = p.key in values
                dists[key] = CategoricalDistribution([False, True])
            if p.key not in values:
                continue
            params[p.key] = (
                1 - values[p.key] if p.kind == "one_minus_log" else values[p.key]
            )
            if p.kind == "cat":
                dists[p.key] = CategoricalDistribution(p.choices)
            elif p.kind == "int":
                dists[p.key] = IntDistribution(int(p.low), int(p.high))
            else:
                dists[p.key] = FloatDistribution(
                    p.low, p.high, log=p.kind in ("log", "one_minus_log")
                )
        return params, dists

    def tell(self, proposal, result):
        trial = self.pending.pop(proposal.metadata["number"])
        trial.set_user_attr("constraints", [0.0 if result.accepted else 1.0])
        self.study.tell(trial, result.score if result.accepted else math.inf)

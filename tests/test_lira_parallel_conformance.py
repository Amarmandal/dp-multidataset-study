"""Numerical conformance of LiRA's sequential and process-pool target paths.

See ``tests/README_lira_parallel_conformance.md`` for rationale and scope.
"""

from argparse import Namespace
from pathlib import Path
import sys

import numpy as np


REPO = Path(__file__).resolve().parents[1]
LIRA_DIR = REPO / "Attack" / "LiRA"
sys.path.insert(0, str(LIRA_DIR))

import run_lira as driver  # noqa: E402


def _args(workers):
    return Namespace(
        balance=False,
        epsilons=driver.DEFAULT_EPSILONS,
        n_shadow=32,
        offline=False,
        workers=workers,
    )


def _assert_result_equal(sequential, parallel):
    assert sequential.keys() == parallel.keys()
    for key in sequential:
        left, right = sequential[key], parallel[key]
        if key.startswith("_roc"):
            np.testing.assert_array_equal(left, right)
        elif isinstance(left, float):
            assert left == right, key
        else:
            assert left == right, key


def test_bcp_svm_parallel_matches_sequential():
    """BCP is small enough to prove both paths with real exported targets."""
    ds, fam, seeds = "BCP", "SVM", tuple(range(10))
    data, n_classes, low, high = driver.load_dataset(ds)
    args = _args(workers=2)

    sequential = [
        driver.attack_target(
            fam,
            "standard",
            data,
            n_classes,
            low,
            high,
            args,
            ds,
            seeds=seeds,
        ),
    ]
    sequential.extend(
        driver.attack_target(
            fam,
            "dp",
            data,
            n_classes,
            low,
            high,
            args,
            ds,
            epsilon=epsilon,
            seeds=seeds,
        )
        for epsilon in args.epsilons
    )
    parallel = driver._parallel_targets(ds, fam, args, seeds)

    assert [(r["variant"], r["epsilon"]) for r in parallel] == [
        ("standard", None),
        *(("dp", epsilon) for epsilon in args.epsilons),
    ]
    for expected, actual in zip(sequential, parallel, strict=True):
        _assert_result_equal(expected, actual)

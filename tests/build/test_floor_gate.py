"""The build floor gate must not depend on the arithmetic accident that
fsum-then-divide happens to round back to the floor for this corpus's paper
sizes."""
import math

import pytest

from locus.build.pipeline import floor_ok
from locus.core.bootstrap import paper_mean

FLOORS = {
    10: sum(1.0 / i for i in range(1, 11)) / 10,
    9: sum(1.0 / i for i in range(1, 10)) / 9,
    5: sum(1.0 / i for i in range(1, 6)) / 5,
}


@pytest.mark.parametrize("k", sorted(FLOORS))
def test_gate_holds_for_every_group_size(k):
    """A constant scorer gives every pool exactly the floor. The gate must
    accept that for any number of pools per paper, not merely for sizes where
    the division happens to round back."""
    floor = FLOORS[k]
    bad = [n for n in range(1, 2001) if math.fsum([floor] * n) / n != floor]
    assert bad, f"no drifting sizes at k={k}; the test would be vacuous"

    for n in bad:
        values = [floor] * n
        groups = ["p0"] * n
        assert floor_ok(values, paper_mean(values, groups), floor), (
            f"gate rejected a perfectly constant scorer at n={n}, k={k}"
        )


@pytest.mark.parametrize("k", sorted(FLOORS))
def test_gate_rejects_a_scorer_that_is_not_on_the_floor(k):
    """The relaxation must not make the gate unfalsifiable."""
    floor = FLOORS[k]
    values = [floor] * 99 + [floor + 1e-9]
    groups = ["p0"] * 100
    assert not floor_ok(values, paper_mean(values, groups), floor)


@pytest.mark.parametrize("k", sorted(FLOORS))
def test_gate_rejects_a_varying_scorer_whose_mean_lands_on_the_floor(k):
    """The brief's original negative test perturbs one value by 1e-9 -- millions
    of ULPs away from the floor -- so the *tolerance* conjunct alone already
    rejects it and the *input* conjunct (`set(values) != {floor}`) is never
    exercised. This case isolates that conjunct: two values are pushed off the
    floor in opposite directions by exactly the same amount, so `paper_mean`
    cancels them and lands EXACTLY on the floor (0 ULPs away, verified below).
    The tolerance conjunct alone would therefore ACCEPT this input. Only the
    exact per-item check catches it -- and it must, because a context-dependent
    scorer that merely averages to the chance floor is not a constant scorer
    tied at every pool, and must not be reported as though it were.
    """
    floor = FLOORS[k]
    values = [floor - 1e-9, floor + 1e-9] + [floor] * 98
    groups = ["p0"] * 100
    aggregate = paper_mean(values, groups)
    assert aggregate == floor, (
        f"test setup assumption violated: aggregate {aggregate!r} != floor "
        f"{floor!r} (0 ULPs expected) -- this case no longer isolates the "
        "input conjunct"
    )
    assert not floor_ok(values, aggregate, floor)

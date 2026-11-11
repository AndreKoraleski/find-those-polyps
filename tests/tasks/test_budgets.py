import numpy as np
import pytest
from pydantic import ValidationError

from find_those_polyps.tasks import Budget, budget_indices


@pytest.mark.parametrize("text", ["full", "frac:0.1", "shots:5", "n:50"])
def test_budget_round_trip(text: str) -> None:
    assert str(Budget.model_validate(text)) == text


@pytest.mark.parametrize("text", ["frac:1.5", "shots:0", "n:2.5", "half", "full:3"])
def test_invalid_budget(text: str) -> None:
    with pytest.raises(ValidationError):
        Budget.model_validate(text)


def test_fraction_subsets_are_nested_and_cover_classes() -> None:
    labels = np.repeat(np.arange(4), [100, 50, 10, 2])
    previous: set[int] = set()
    for text in ("frac:0.01", "frac:0.1", "frac:0.5"):
        chosen = set(budget_indices(Budget.model_validate(text), 162, labels, 3))
        assert previous <= chosen
        assert set(labels[list(chosen)]) == {0, 1, 2, 3}
        previous = chosen


def test_shots_take_k_per_class_and_nest() -> None:
    labels = np.repeat(np.arange(3), 20)
    one = budget_indices(Budget.model_validate("shots:1"), 60, labels, 0)
    five = budget_indices(Budget.model_validate("shots:5"), 60, labels, 0)
    assert np.bincount(labels[five]).tolist() == [5, 5, 5]
    assert set(one) <= set(five)


def test_unlabelled_budget_and_seed_dependence() -> None:
    budget = Budget.model_validate("n:10")
    first = budget_indices(budget, 100, None, 0)
    assert len(first) == 10
    assert not np.array_equal(first, budget_indices(budget, 100, None, 1))


def test_slugs_and_order() -> None:
    texts = ["shots:1", "n:100", "frac:0.01", "full", "shots:10", "frac:0.1"]
    budgets = sorted((Budget.model_validate(t) for t in texts), key=lambda b: b.order)
    assert [b.slug for b in budgets] == ["full", "10pct", "1pct", "n100", "10shot", "1shot"]

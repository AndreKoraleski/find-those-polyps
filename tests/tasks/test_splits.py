import pytest

from find_those_polyps.tasks import SplitFile, make_split
from find_those_polyps.tasks.splits import merge_duplicates, merge_near_duplicates
from find_those_polyps.utils.perceptual import Signature


def _groups(split_ids: tuple[str, ...], groups: dict[str, str]) -> set[str]:
    return {groups[i] for i in split_ids}


def test_benchmark_split_is_group_disjoint_and_stratified() -> None:
    ids = [f"s{i}" for i in range(300)]
    groups = {i: f"p{n // 3}" for n, i in enumerate(ids)}
    labels = [n % 3 for n in range(300)]
    split = make_split(ids, list(groups.values()), labels, test=0.2, val=0.1)
    partitions = [split.train, split.val, split.test]
    assert sum(map(len, partitions)) == 300
    group_sets = [_groups(p, groups) for p in partitions]
    assert not group_sets[0] & group_sets[1]
    assert not group_sets[0] & group_sets[2]
    assert 45 <= len(split.test) <= 75
    assert 20 <= len(split.val) <= 45


def test_official_test_is_kept_verbatim() -> None:
    ids = [f"s{i}" for i in range(100)]
    official = ids[:25]
    split = make_split(ids, ids, test=official, test_source="official")
    assert set(split.test) == set(official)
    assert split.test_source == "official"


def test_official_test_requires_source() -> None:
    with pytest.raises(ValueError, match="source"):
        make_split(["a", "b", "c"], ["a", "b", "c"], test=["a"])


def test_training_keeps_rare_classes_held_in_one_group() -> None:
    ids = [f"s{i}" for i in range(200)]
    labels = [i % 4 for i in range(196)] + [4] * 4
    groups = [f"g{i // 2}" for i in range(196)] + ["rare"] * 4
    split = make_split(ids, groups, labels, test=0.2, val=0.1, seed=0)
    by_id = dict(zip(ids, labels, strict=True))
    assert {by_id[i] for i in split.train} == {0, 1, 2, 3, 4}
    assert {by_id[i] for i in split.test} >= {0, 1, 2, 3}


def test_pool_samples_sharing_a_test_group_are_excluded() -> None:
    ids = [f"s{i}" for i in range(60)]
    groups = [f"g{i // 3}" for i in range(60)]
    split = make_split(ids, groups, test=["s0", "s3"], test_source="official")
    assert set(split.excluded) == {"s1", "s2", "s4", "s5"}
    assert not set(split.excluded) & (set(split.train) | set(split.val))
    with pytest.raises(ValueError, match="disjoint"):
        SplitFile(test_source="t", train=("a",), val=("b",), test=("c",), excluded=("a",))


def test_duplicates_merge_groups_and_conflicts_are_reported() -> None:
    ids = ["a", "b", "c", "d", "e"]
    groups = ["p1", "p2", "p3", "p4", "p5"]
    prints = ["x", "x", "y", "y", "z"]
    merged, conflicts = merge_duplicates(ids, groups, prints, [0, 0, 1, 2, 1])
    assert merged[0] == merged[1]
    assert merged[2] == merged[3]
    assert merged[4] == "p5"
    assert conflicts == {"c", "d"}
    merged, conflicts = merge_duplicates(ids, groups, prints)
    assert conflicts == set()


def test_near_duplicates_share_a_group_whatever_their_labels() -> None:
    coarse, fine = 0x0F0F_0F0F_0F0F_0F0F, (1 << 200) - 1
    signatures = [
        Signature(coarse, fine),
        Signature(coarse ^ 0b111, fine ^ 0b1111),
        Signature(coarse ^ 0xFFFF_0000, fine),
        None,
        Signature(coarse ^ 0b1, fine ^ ((1 << 40) - 1)),
        Signature(coarse ^ 0b11, fine ^ 0b1),
    ]
    groups = merge_near_duplicates(["a", "b", "c", "d", "e", "f"], signatures)
    assert groups[0] == groups[1] == groups[5]
    assert len({groups[0], groups[2], groups[3], groups[4]}) == 4


def test_rejected_candidates_and_pairs_within_a_group_never_merge() -> None:
    same = Signature(0x0F0F_0F0F_0F0F_0F0F, (1 << 200) - 1)
    asked: list[tuple[int, int]] = []

    def reject(first: int, second: int) -> bool:
        asked.append((first, second))
        return False

    assert merge_near_duplicates(["a", "b", "b"], [same, same, same], reject) == ["a", "b", "b"]
    assert asked == [(0, 1), (0, 2)]

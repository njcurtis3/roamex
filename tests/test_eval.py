"""The scorer has to be right before any number it prints means anything."""

from src.eval.score import PRF, score_extraction, score_resolution
from src.models import Provenance, Triple

PROV = Provenance(
    block_uid="b1", page_title="P", origin="llm", extracted_at="2026-01-01T00:00:00Z"
)


def triple(s, p, o):
    return Triple(
        subject=s, subject_type="person", predicate=p, object=o,
        object_type="org", provenance=PROV,
    )


def test_prf_on_a_known_case():
    prf = PRF.compute({"a", "b", "c"}, {"b", "c", "d"})
    assert (prf.true_positives, prf.false_positives, prf.false_negatives) == (2, 1, 1)
    assert prf.precision == prf.recall == round(2 / 3, 4)


def test_prf_handles_empty_predictions_without_dividing_by_zero():
    prf = PRF.compute(set(), {"a"})
    assert prf.precision == 0.0 and prf.recall == 0.0 and prf.f1 == 0.0


def test_extraction_scoring_is_normalization_insensitive():
    triples = [triple("dana  whitfield", "works_at", "Rivet Labs")]
    gold = [{"subject": "Dana Whitfield", "predicate": "works_at", "object": "rivet labs"}]
    assert score_extraction(triples, gold).f1 == 1.0


def test_ignoring_predicate_separates_wording_from_wrongness():
    """Right entities, different verb: a wording miss, not a factual one."""
    triples = [triple("Dana", "employed_by", "Rivet Labs")]
    gold = [{"subject": "Dana", "predicate": "works_at", "object": "Rivet Labs"}]
    assert score_extraction(triples, gold).f1 == 0.0
    assert score_extraction(triples, gold, ignore_predicate=True).f1 == 1.0


def test_resolution_scoring_counts_a_correct_merge():
    groups = [{"canonical": "Dana Whitfield", "members": ["Dana", "Dana Whitfield"]}]
    result = score_resolution(groups, [["Dana", "Dana Whitfield"]])
    assert result["over_merge_rate"] == 0.0
    assert result["missed_merge_rate"] == 0.0


def test_resolution_scoring_catches_an_over_merge():
    """Two different people welded into one — the expensive error."""
    groups = [{"canonical": "Dana", "members": ["Dana Whitfield", "Dana Reyes"]}]
    result = score_resolution(groups, [["Dana Whitfield"], ["Dana Reyes"]])
    assert result["over_merge_rate"] == 1.0


def test_resolution_scoring_catches_a_missed_merge():
    groups = [
        {"canonical": "Dana", "members": ["Dana"]},
        {"canonical": "Dana Whitfield", "members": ["Dana Whitfield"]},
    ]
    result = score_resolution(groups, [["Dana", "Dana Whitfield"]])
    assert result["missed_merge_rate"] == 1.0
    assert result["over_merge_rate"] == 0.0
    assert result["singleton_groups"] == 2

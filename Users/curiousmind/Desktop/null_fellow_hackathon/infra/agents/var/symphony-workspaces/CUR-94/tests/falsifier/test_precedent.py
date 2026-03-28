from falsifier.precedent import graph_aware_precedent
from falsifier.types import TheoryHistoryRecord


def test_graph_aware_precedent_uses_theory_text_overlap_for_near_duplicate() -> None:
    text = "Shift capacity into a local gating path that specializes early residual routing."
    novelty, reason, evidence = graph_aware_precedent(
        text,
        theory_history=[
            TheoryHistoryRecord(
                theory_id="theory-001",
                verdict="refuted",
                theory_text=text,
            )
        ],
    )
    assert novelty < 0.55
    assert evidence.matched_theory_id == "theory-001"
    assert "theory_text" in evidence.matched_fields
    assert "graph-aware precedent overlap" in reason


def test_graph_aware_precedent_uses_failure_context_and_graph_neighbors() -> None:
    novelty, _, evidence = graph_aware_precedent(
        "Stabilize qk gain saturation with a softer logit clamp in the attention path.",
        theory_history=[
            TheoryHistoryRecord(
                theory_id="theory-010",
                verdict="refuted",
                theory_text="Tune the attention path to improve optimization stability.",
                failure_context="Refuted after qk gain saturation caused unstable logits during training.",
                related_theory_ids=["theory-011"],
            ),
            TheoryHistoryRecord(
                theory_id="theory-011",
                verdict="surviving",
                theory_text="Attention softcap changes can contain extreme logits.",
                mechanism_tags=["logit clamp", "qk gain", "attention saturation"],
            ),
        ],
    )
    assert novelty < 0.55
    assert evidence.matched_theory_id == "theory-010"
    assert "failure_context" in evidence.matched_fields
    assert "graph_context" in evidence.matched_fields
    assert evidence.supporting_theory_ids == ["theory-011"]


def test_graph_aware_precedent_keeps_materially_distinct_theory_novel() -> None:
    novelty, _, evidence = graph_aware_precedent(
        "Compress feed-forward width while widening the token mixing path in later layers.",
        theory_history=[
            TheoryHistoryRecord(
                theory_id="theory-020",
                verdict="refuted",
                theory_text="Stabilize qk gain saturation in attention heads.",
                failure_context="Attention logits overflowed under aggressive softcaps.",
                mechanism_tags=["attention", "softcap", "qk gain"],
            )
        ],
    )
    assert novelty >= 0.55
    assert evidence.matched_theory_id == "theory-020"
    assert evidence.overlap_score < 0.45

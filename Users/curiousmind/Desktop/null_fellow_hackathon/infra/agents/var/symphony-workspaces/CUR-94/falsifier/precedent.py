from __future__ import annotations

import re
from typing import Iterable

from .types import PrecedentEvidence, TheoryHistoryRecord


_WORD_RE = re.compile(r"[a-z0-9]+")


def _normalize_words(text: str) -> set[str]:
    return {token for token in _WORD_RE.findall(text.lower()) if len(token) > 2}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _legacy_reference_records(reference_texts: Iterable[str]) -> list[TheoryHistoryRecord]:
    records: list[TheoryHistoryRecord] = []
    for index, reference in enumerate(reference_texts, start=1):
        if not reference.strip():
            continue
        records.append(
            TheoryHistoryRecord(
                theory_id=f"reference-{index}",
                verdict="reference",
                theory_text=reference,
            )
        )
    return records


def graph_aware_precedent(
    candidate_text: str,
    theory_history: Iterable[TheoryHistoryRecord],
    reference_texts: Iterable[str] = (),
) -> tuple[float, str, PrecedentEvidence]:
    candidate_words = _normalize_words(candidate_text)
    if not candidate_words:
        evidence = PrecedentEvidence(
            query_mode="graph_aware_precedent",
            matched_theory_id=None,
            matched_verdict=None,
            explanation="candidate text is empty or too short",
        )
        return 0.0, evidence.explanation, evidence

    records = [record for record in theory_history if record.theory_id.strip()]
    if not records:
        records = _legacy_reference_records(reference_texts)
    if not records:
        evidence = PrecedentEvidence(
            query_mode="graph_aware_precedent",
            matched_theory_id=None,
            matched_verdict=None,
            explanation="no theory-history records provided",
        )
        return 1.0, evidence.explanation, evidence

    by_id = {record.theory_id: record for record in records}
    best_record: TheoryHistoryRecord | None = None
    best_overlap = 0.0
    best_fields: list[str] = []
    best_supporting_ids: list[str] = []

    for record in records:
        field_word_sets = {
            "theory_text": _normalize_words(record.theory_text),
            "failure_context": _normalize_words(record.failure_context),
            "mechanism_tags": _normalize_words(" ".join(record.mechanism_tags)),
        }

        supporting_ids = [theory_id for theory_id in record.related_theory_ids if theory_id in by_id]
        graph_words: set[str] = set()
        for theory_id in supporting_ids:
            neighbor = by_id[theory_id]
            graph_words |= _normalize_words(neighbor.theory_text)
            graph_words |= _normalize_words(neighbor.failure_context)
            graph_words |= _normalize_words(" ".join(neighbor.mechanism_tags))
        field_word_sets["graph_context"] = graph_words

        field_overlaps = {
            field_name: _jaccard(candidate_words, words) for field_name, words in field_word_sets.items() if words
        }
        matched_fields = [field_name for field_name, overlap in field_overlaps.items() if overlap > 0.0]
        if field_overlaps:
            overlap = max(field_overlaps.values())
            overlap += 0.08 * max(0, len(matched_fields) - 1)
            if record.verdict == "refuted" and {"failure_context", "graph_context"} & set(matched_fields):
                overlap += 0.05
            if supporting_ids and "graph_context" in matched_fields:
                overlap += 0.10
            overlap = min(overlap, 1.0)
        else:
            overlap = 0.0

        if overlap >= best_overlap:
            best_record = record
            best_overlap = overlap
            best_fields = matched_fields
            best_supporting_ids = supporting_ids

    novelty = max(0.0, 1.0 - best_overlap)
    if best_record is None:
        evidence = PrecedentEvidence(
            query_mode="graph_aware_precedent",
            matched_theory_id=None,
            matched_verdict=None,
            explanation="no usable theory-history records provided",
        )
        return novelty, evidence.explanation, evidence

    if best_fields:
        joined_fields = ", ".join(best_fields)
        reason = (
            f"graph-aware precedent overlap {best_overlap:.2f} against {best_record.theory_id} "
            f"via {joined_fields}"
        )
    else:
        reason = f"no graph-aware precedent overlap against {best_record.theory_id}"

    evidence = PrecedentEvidence(
        query_mode="graph_aware_precedent",
        matched_theory_id=best_record.theory_id,
        matched_verdict=best_record.verdict,
        matched_fields=best_fields,
        supporting_theory_ids=best_supporting_ids,
        overlap_score=best_overlap,
        explanation=reason,
    )
    return novelty, reason, evidence

# Execution Admission Gate PRD

## Goal

Prevent any execution-project theory work from starting unless the deterministic falsifier has emitted a valid promote artifact.

## Scope

- define the promote artifact path and schema
- define the mapping from local artifact -> Linear execution issue eligibility
- keep the admission logic outside the buildout falsifier-core internals

## Required behavior

- no execution issue can move to `Todo` without a valid promote verdict
- invalid or missing verdict artifacts keep work blocked
- the gate should be checkable locally without launching Symphony
- buildout and execution project slugs must differ in repo-local env
- execution eligibility must be derivable from a repo-local verdict artifact alone

## Verdict artifact contract

Execution admission accepts only verdict artifacts that satisfy all of:

- `theory_id` matches the requested theory
- `outcome == "survived"`
- `decision == "promote"`
- `created_at` is present
- `supporting_results` exists and is non-empty

Default lookup path:

- `research/falsification/<theory_id>/verdict_*.json`
- latest matching verdict wins unless an explicit path is provided

## Validation

- contract test: missing verdict blocks promotion
- contract test: invalid verdict blocks promotion
- contract test: valid promote artifact allows execution issue creation

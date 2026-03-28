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

## Validation

- contract test: missing verdict blocks promotion
- contract test: invalid verdict blocks promotion
- contract test: valid promote artifact allows execution issue creation

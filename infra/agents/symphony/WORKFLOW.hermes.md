# Parameter Golf Symphony Workflow

Operate this repo as a falsifier-first research loop for the OpenAI Parameter Golf challenge.

## Execution Rules

- Prefer small, explicit issues with concrete delivery paths.
- Treat checked-in handoffs under `infra/agents/handoffs/` as the repo-local source of truth.
- Release only dependency-free `Todo` issues.
- Do not retry a failed issue without a material issue mutation.
- Do not promote engineering or training work unless a falsifier verdict artifact exists when required.
- Require a declared validation target before implementation starts.
- Do not complete an issue if its regression checks or declared validation commands are missing or failing.
- Do not allow Linear state transitions that are not justified by the local handoff file and local repo artifacts.

## Recommended Issue Types

- environment-bootstrap
- baseline-profile
- theory-proposal
- falsification-batch
- engineering-change
- training-eval
- distillation-update

## Validation Bias

- Run cheap, non-training checks first.
- Prefer local execution for profiling and falsification.
- Escalate to remote GPU only for training or when the validation command explicitly requires it.
- When possible, make the first validation step a focused regression check that can fail fast before broader runs.

## Delivery Bias

- Repo artifacts are authoritative.
- A Linear issue is not complete unless the files listed under `Delivery` exist in the main repo working tree.
- Local handoff files and local validation results are authoritative for scope and completion; Linear mirrors that state.

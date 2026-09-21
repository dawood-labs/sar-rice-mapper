# Working rules for AI coding assistants

This file is read automatically by Claude Code. It holds only generic, public working rules.
Project-specific context (AOIs, storage paths, run ids, decisions, next steps) lives in
`CLAUDE.local.md`, which is gitignored and never committed. Read it first if it exists.

## Talking to the user

- Reply in Roman Urdu: simple, structured, nothing hidden.
- Areas always in **acres**, never m² or hectares. 1 pixel (10 m) = 0.025 acre; the 5×5 window = 0.62 acre.
- Give critical feedback: point out flaws, offer alternatives with their tradeoffs, ask clarifying questions.
- Keep the user updated during long work.

## Before acting

- Confirm first: Earth Engine exports, bulk downloads, cloud-storage uploads, `git push`, deleting or
  overwriting data.
- KISS / YAGNI: build the simplest thing that meets today's need; ask before adding complexity.
- Time every long step, log per-stage timings, find the slow stage and optimise it proactively.
- Never hard-code CPU, RAM or threads; use `sar_pipeline.resources` (container-aware).

## Code and documentation

- **No throwaway scripts.** Any code used for any task — however small, including a one-off
  cross-check — belongs in this repository, in a sensible place, with a test. Do not run an ad-hoc
  snippet in a terminal and move on.
- Every module and step documents **why** it exists, not only what it does.
- The README must carry a detailed, easy-to-understand description and a "how to use" section, with
  each step's inputs, outputs and the exact command that runs it.
- Code and docs in **English**, written so a junior can follow from the basics. This project will be
  handed over; assume the reader has no radar background and has never seen the codebase.

## Git

- Commit only with the repo-local identity (check `git config user.name` before committing).
- No AI attribution in commits or PRs: no `Co-Authored-By`, no session links, no mention of any
  assistant in commit messages, PR descriptions or the collaborator list.
- This repository is **public**: no client, region, country, bucket, project or track names, no
  coordinates and no email addresses in tracked files — test fixtures included. Use a neutral
  location (and a UTM zone that actually contains it) in any geometry fixture.
  `tests/test_repo_hygiene.py` checks this against `secrets/private_terms.txt`.
- Never commit `secrets/`, `data/`, `processed/`, `reference/`, non-example configs, or local changes
  to `notebooks/04_pixel_explorer.ipynb`.

## Where things are

- Prep CLI: `python -m sar_pipeline.prep <step> --config <yaml>` (AOI QC, S1 availability; read-only).
- Pipeline CLI: `python -m sar_pipeline --config <yaml> <step>`; analysis CLI:
  `python -m sar_pipeline.analysis <step> --config <yaml>`.
- Docs: `docs/00_step_by_step_guide.md` (start here), `docs/04_runbook.md` (pipeline),
  `docs/08_analysis.md` (ground truth, model, field labels).
- Tests: `pytest` (no network); `pytest -m gee` for live read-only Earth Engine tests.

## Crop context

The target crop is **paddy rice**. The defining radar feature is the flooding minimum in VH before
transplanting, followed by a steep growth rise and a sharp harvest drop. The season window must
contain the flooding minimum, the harvest drop, and a dry baseline before flooding — otherwise the
minimum/maximum/variance features stop separating rice from other classes. See the README and
`docs/01_sar_basics.md`.

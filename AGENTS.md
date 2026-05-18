# AGENTS.md

## Public Repository Hygiene

- Treat this repository as a front-facing public GitHub project.
- Do not add internal planning, execution prompts, agent task ledgers, lessons, or private audit notes to Git.
- Keep repo-local agent work under ignored paths such as `tasks/`, `docs/prompts/`, or `docs/plans/`.
- Do not stage or commit `docs/code-audit-report.md`, `docs/code-review-report.md`, swap files, or other local-only review artifacts.
- Public documentation should live in `README.md`, `BENCHMARKS.md`, `Sources/MetalANNS/MetalANNS.docc/`, `locales/`, or intentionally public files under `docs/`.

## Before Committing

- Run `git status --short` and verify that only source, tests, benchmarks, public docs, or release assets are staged.
- Use `git check-ignore -v` for any uncertain planning or agent-generated file before adding it.

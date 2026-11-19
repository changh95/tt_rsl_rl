# Owner preferences (rsl_rl TT port)

- **Language:** reports to the owner in Korean, as 3-line summary → numbers → decisions the owner must make. Code, comments, commit messages and docs in English. Keep reports short; do not paste documents back.
- **Autonomy (owner, 2026-11-11):** never end a turn with "다음 단계 진행할까요?", "계속할까요?", "이렇게 해도 될까요?". When a phase's completion criteria are met: commit, merge to `main`, tag `phase-N-done`, update PROGRESS.md, give the short Korean report, and continue into the next phase in the same turn. Decide small design questions yourself and record them in DECISIONS.md.
- **Still stop and report** (not ask-to-continue, but stop): anything touching the device-isolation rules of BRIEF section 2 / CLAUDE.md, an ADR change contradicting an accepted decision, a hang or device error.
- **Order of work (owner):** all post-v1 items in order (8.1 … 8.7), then the 2026-11-19 requirements in BRIEF 4-b. Merging to `main` and tags at each phase completion (BRIEF 2-10).
- **Git:** commit author = the identity of the existing commits in this repo (check `git log -1`); commits end with the Co-Authored-By trailer. The repo is **public**: before each push check the diff for anything machine-identifying (see README of this folder).
- **CI of the fork stays off** (workflows are `workflow_dispatch` only); do not re-enable when syncing upstream; `gh workflow` / repo settings are off limits.

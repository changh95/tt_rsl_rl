# Incident 2026-11-11: a debug job hit its timeout (exit 124) and chip 16 can no longer be opened (BRIEF 2-6)

**What happened.** While debugging a failing test of the CNN work (Phase 8.6, branch `tt/phase-8-6-cnn`), I ran one scratch script through the wrapper:

    scripts/tt_run.sh --chips 16 --timeout 600 --tag p8cdbg -- python <scratchpad>/cnn_load_dbg.py        (pid 2354078, tray 3, node 16 only, one job, flock held)

It opened the mesh (with a 256 MiB trace region), printed the Metal warning "Allocating device buffers is potentially unsafe due to the existence of an active trace" about one second later, and
then produced no further output until `timeout` ended it (exit 124 after 600 s). The same scenario had run to completion (with a wrong result, not a hang) a few minutes earlier as the pytest test
`tests/tt/test_tt_cnn_trace.py::test_traced_cnn_path_survives_load`.

**What I found afterwards.** A harmless follow-up job (`scripts/tt_run.sh --chips 16 --timeout 300 --tag p8chealth -- python -m pytest -m tt tests/tt/test_tt_cnn.py`, pid 2603656) could not open
the device: every test errors with

    RuntimeError: MMIO per-op timeout: 4B load took 55484 us (budget=2 ms), 4 of 4 bytes remaining.     (ttnn/device.py:38)
    UMD warning: Setting power state failed on device 16: Invalid argument (pci_device.cpp:1164)

No process of mine is left (checked `pgrep`, `.run/pids`). So the node 16 itself is in a state in which MMIO reads time out.

**What I did NOT do.** No reset, no `tt-smi`, no other chip was touched, no further device job was started after the failed health check. Logs are in `.run/incidents/` (git-ignored):
`2026-11-11_cnn_dbg_hang.log`, `2026-11-11_chip16_health_check_after_hang.log`.

**Cause.** Not established. Hypotheses, none verified: (1) the scratch script left a trace capture open or hung inside a capture (the earlier incident `2026-11-11_trace_capture_left_open.md` had the same
signature, and `TTPPO._capture` ends captures on exceptions, but the CNN path is new: its gather indices, the `ttnn.embedding` / `ttnn.pad` / `ttnn.permute` ops and the very large number of ops inside
one mini-batch trace are untested inside a capture on a fresh mesh); (2) the job was killed by `timeout` while a command was in flight on the device (SIGINT, then SIGKILL after 60 s), which left the
chip in a state that needs a reset.

**State of the work.** The CNN encoder and its tests were green before the incident (7 component tests, 5 model / update parity tests, the eager-vs-traced bit-identity test); the test
`test_traced_cnn_path_survives_load` fails (the second algorithm's act after `load()` returns garbage values), which is what the debug job was meant to explain. Everything is committed on
`tt/phase-8-6-cnn` (not merged); Phase 8.6 is NOT done. Phases 8.1 - 8.5 are done, merged to `main` and tagged.

**What is needed from the owner.** Check / reset chip 16 (I must not). Nodes 17 - 23 of tray 3 are untouched and may still open; I will not use another node for the work that belongs on node 16 without
being told to. After the node opens again, the first job should be `tests/tt/test_smoke_device.py` on chip 16; then the debug of `test_traced_cnn_path_survives_load` continues WITHOUT the scratch script
(inside pytest, which ends a failed capture and releases traces), and a timeout of 120 s instead of 600 s for the debug jobs.

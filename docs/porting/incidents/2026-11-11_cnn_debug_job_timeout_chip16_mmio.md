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

## Resolution (2026-11-12)

The owner replied "do the reset yourself, continue" (explicit approval of BRIEF 2-4 for this one action). I started it as the narrowest possible job:

    scripts/tt_run.sh --chips 16 --timeout 300 --tag p8creset -- tt-smi -r 16        (pid 2785066)

Why this form: `tt-smi -r N` takes the /dev/tenstorrent/N number (tt_umd `enumerate_devices_info()` is a `dict[int, ...]` keyed by pci device id, `warm_reset(pci_device_ids)`), and inside the
`tt_run.sh --chips 16` namespace only node 16 exists, so no other chip could be touched.

The session ended during that call: the log holds only the wrapper's two header lines (`.run/incidents/2026-11-12_chip16_reset_by_owner_request.log`), no tt-smi output, and the process was gone
afterwards. **It is unknown whether tt-smi ran to completion, was interrupted, or never started.** A smoke test on chip 16 then passed (`tests/tt/test_smoke_device.py`, 1 passed in 0.99 s,
`.run/incidents/2026-11-12_chip16_smoke_after_reset.log`), so no second reset was issued. Whether the chip recovered because of the reset or on its own is not established.
Work on Phase 8.6 resumes; debug jobs use pytest with a 120 s timeout.

## Second timeout (2026-11-12, about 08:55 UTC): the same scenario alone in a fresh process hangs again (BRIEF 2-6, stopped)

After the smoke test passed, two pytest jobs on chip 16 (120 s timeout, as planned):

    scripts/tt_run.sh --chips 16 --timeout 120 --tag p8ctrace -- python -m pytest -m tt tests/tt/test_tt_cnn_trace.py -q -p no:cacheprovider -x
        -> completed in 6.7 s: 1 passed (eager vs traced bit identity), 1 FAILED (`survives_load`, assertion)
    scripts/tt_run.sh --chips 16 --timeout 120 --tag p8ctrace -- python -m pytest -m tt tests/tt/test_tt_cnn_trace.py -q -p no:cacheprovider -x -k survives_load --tb=short
        -> exit 124 (pid 9894)

So the file as a whole completes (the first test runs first on the mesh and releases its traces, then `survives_load` fails its assertion), but `survives_load` ALONE as the first test on a fresh mesh
produces no output until `timeout` ends it. The first hang (scratch script, 600 s) had the same shape: the scenario first on a fresh mesh, the Metal warning "Allocating device buffers is potentially
unsafe due to the existence of an active trace" about 1 s after the open, then silence.

Hypothesis (unverified): two traced `TTPPO` objects (`a`, `b`) built in one process, and the allocator / capture state they find differs from what the preceding test leaves behind (T-15 family: a buffer
allocated after a capture, e.g. by `b`'s construction or by the first `load()`, aliases memory a replayed trace uses, and a replay never finishes). Host-only finding: the encoder's `sync_from_torch`
writes IN PLACE (`upload_into`) like the MLP layers, so "stale encoder weights after load" is not the explanation of the failing assertion.

No log of the hung process exists: my redirect sent the job's stdout to a file that stayed empty (pytest `-q` prints at the end) and the wrapper's stderr only had its two header lines.
No process of mine is left. **The chip state after this second timeout is UNKNOWN**: I ran no follow-up job (BRIEF 2-6) and reset nothing (the reset approval of 2026-11-12 covered the first incident only).

**Needed from the owner**: a go for ONE smoke test on chip 16 (to see whether the node still opens) and, if it does not open, a reset (by the owner, or approved again).
Rules for the debug jobs after that: never run `survives_load` alone, run the whole file; capture the output with `tee` into `.run/incidents/` and use `-s`; 120 s timeout.

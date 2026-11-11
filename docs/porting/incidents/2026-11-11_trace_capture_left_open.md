# Incident 2026-11-11: device job hit its timeout after a TT_FATAL during trace capture (BRIEF 2-6)

**What happened.** `scripts/tt_run.sh --chips 16 --timeout 580 --tag p5t -- python -m pytest -m tt tests/tt/test_tt_trace.py ...` (first run of the traced
path, tray 3 node 16 only, one job, flock held) printed

    TT_FATAL: Cannot load new binaries during trace capture. This program is not yet in program cache. Warm up before capturing a trace.
    TT_FATAL: Reads are not supported during trace capture.   (x2)

about 2 s after the mesh was opened, the test failed (`F`), and the process then made no progress until `timeout` ended it (exit 124 after 580 s; pytest never
printed its failure report). Last pid recorded: 2020437 (tag p5t), gone afterwards. No other process was touched, no reset or any hardware command was issued.

**Root cause (host software).** (1) The first capture of the act step ran `ttnn.copy` (the in-place write into persistent output buffers) for the first time INSIDE the
capture: the persist-path had only been run once eagerly, and that first pass adopts the outputs without copying, so the copy program was not in the program cache
(ttnn refuses to compile during capture). (2) The resulting exception propagated out of `begin_trace_capture ... end_trace_capture` without ending the capture, so every
later device read failed ("Reads are not supported during trace capture") and the interpreter stalled in the pytest teardown / mesh close with the capture still open.

**Fix (host only, no device involved).** `TTPPO._capture` always ends (and releases) the capture, also on exceptions; the act step is warmed up twice before it is captured
so every program, including the in-place copies, is cached; captured traces are released by a close hook right before the mesh is closed
(`TrayMesh.add_close_hook`); the traced test releases its traces in `finally`.

**State of the hardware.** Not verified: per BRIEF 2-6 no device job was started after the timeout. The next device job should be a plain open/close of chip 16
(`tests/tt/test_smoke_device.py`) to confirm that the node opens normally before the traced tests are repeated.

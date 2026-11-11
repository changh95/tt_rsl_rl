# SPDX-License-Identifier: BSD-3-Clause
"""Trace capture / replay shared by the TT algorithms (`TTPPO`, `TTDistillation`): the mixin expects `self.ctx`, `self.tray`, `self._traces` (name -> trace id) and `self.timer`.

The persistent-state rule of the traced path (TTNN_ISSUES T-15): a trace's intermediates are freed when its capture ends and rewritten by every replay, so whatever must keep its value
across replays is allocated BEFORE the first capture, and an algorithm refuses to be constructed on a mesh that has a live captured trace of another algorithm.
"""

from __future__ import annotations

import sys

CAPTURED_TRACES: dict[int, int] = {}  # id(TrayMesh) -> number of live captured traces on that mesh (guard against the T-15 overwrite hazard)


class TraceMixin:
    """`_capture(name, fn)`, `_replay(name)` and `release_traces()` on top of `self.ctx` / `self.tray` / `self._traces` / `self.timer`."""
    def _capture(self, name: str, fn) -> None:
        """Record `fn()` (device ops only, persistent tensors, every program already in the program cache) as a trace; the ops are recorded, not executed.

        Whatever happens inside, the capture is ALWAYS ended (an exception that left it open made every later read fail with "Reads are not supported
        during trace capture" and hung the process until the job timeout: docs/porting/incidents/2026-11-11_trace_capture_left_open.md)."""
        ttnn = self.ctx.ttnn
        trace_id = ttnn.begin_trace_capture(self.ctx.mesh, cq_id=0)
        self.timer.paused = True  # the profiling timer synchronises the device, which is not allowed inside a capture
        try:
            fn()
        except BaseException:
            try:
                ttnn.end_trace_capture(self.ctx.mesh, trace_id, cq_id=0)
                ttnn.release_trace(self.ctx.mesh, trace_id)
            except BaseException as exc:
                print(f"rsl_rl_tt.ppo: could not end the failed trace capture cleanly: {exc!r}", file=sys.stderr)
            raise
        finally:
            self.timer.paused = False
        ttnn.end_trace_capture(self.ctx.mesh, trace_id, cq_id=0)
        self._traces[name] = trace_id
        CAPTURED_TRACES[id(self.tray)] = CAPTURED_TRACES.get(id(self.tray), 0) + 1

    def _replay(self, name: str) -> None:
        self.ctx.ttnn.execute_trace(self.ctx.mesh, self._traces[name], cq_id=0, blocking=False)

    def release_traces(self) -> None:
        for trace_id in self._traces.values():
            self.ctx.ttnn.release_trace(self.ctx.mesh, trace_id)
        CAPTURED_TRACES[id(self.tray)] = CAPTURED_TRACES.get(id(self.tray), 0) - len(self._traces)
        self._traces.clear()

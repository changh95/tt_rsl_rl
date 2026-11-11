# SPDX-License-Identifier: BSD-3-Clause
"""Fixtures for the device tests (marker `tt`). They only run under `scripts/tt_run.sh` (the hook and `pytest.ini` keep them out of host runs).

`tt_mesh` opens the tray mesh ONCE per process through the single guarded entry point; `tt_ctx` wraps it in a `TTContext`. The mesh shape comes from the environment
variable `TT_TEST_MESH` ("1x1" by default; "1x2" with `--chips 16,17`; "2x4" with all eight chips) and must match the chips selected for the job.
One process, one open mesh: select the files that share the fixture in one pytest invocation, e.g.

    scripts/tt_run.sh --chips 16 --timeout 900 --tag p2 -- python -m pytest -m tt tests/tt/test_tt_modules.py tests/tt/test_tt_rollout.py -q
    scripts/tt_run.sh --chips 16,17 --timeout 900 --tag p6 -- env TT_TEST_MESH=1x2 python -m pytest -m tt tests/tt/test_tt_multichip.py -q
"""

import pytest

from rsl_rl_tt.testing.reference import test_mesh_shape


SINGLE_CHIP_SEMANTICS = ("test_tt_backward.py", "test_tt_update.py", "test_tt_hetero_update.py", "test_tt_rnd_update.py")


def pytest_collection_modifyitems(config, items):
    """On a multi-chip mesh the whole-batch references of the Phase 3 suites do not apply: the loss is a chip-LOCAL mean, advantages are normalised per chip and the
    gradients are averaged over the chips (ADR-0004). Their multi-chip counterparts are in test_tt_multichip.py; the forward / rollout / checkpoint / normalizer tests
    of the other files are chip-count independent and DO run on every mesh."""
    rows, cols = test_mesh_shape()
    if rows * cols == 1:
        return
    skip = pytest.mark.skip(reason="single-chip semantics (whole-batch reference); see tests/tt/test_tt_multichip.py for the multi-chip equivalents")
    for item in items:
        if item.fspath.basename in SINGLE_CHIP_SEMANTICS:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def tt_mesh():
    from rsl_rl_tt.device import open_tray_mesh

    rows, cols = test_mesh_shape()
    multi = rows * cols > 1
    # a multi-chip mesh also gets a trace region so that the traced multi-chip tests can capture on it
    tm = open_tray_mesh((rows, cols), fabric_config="FABRIC_1D" if multi else None, **({"trace_region_size": 256 * 1024 * 1024} if multi else {}))
    yield tm
    tm.close()


@pytest.fixture(scope="session")
def tt_ctx(tt_mesh):
    from rsl_rl_tt.tt_common import TTContext

    return TTContext.create(tt_mesh.mesh_device, mesh_shape=tt_mesh.mesh_shape)

# SPDX-License-Identifier: BSD-3-Clause
"""Fixtures for the device tests (marker `tt`). They only run under `scripts/tt_run.sh` (the hook and `pytest.ini` keep them out of host runs).

`tt_mesh` opens the 1x1 tray mesh ONCE per process through the single guarded entry point; `tt_ctx` wraps it in a `TTContext`.
One process, one open mesh: select the files that share the fixture in one pytest invocation, e.g.

    scripts/tt_run.sh --chips 16 --timeout 900 --tag p2 -- python -m pytest -m tt tests/tt/test_tt_modules.py tests/tt/test_tt_rollout.py -q
"""

import pytest


@pytest.fixture(scope="session")
def tt_mesh():
    from rsl_rl_tt.device import open_tray_mesh

    tm = open_tray_mesh((1, 1))
    yield tm
    tm.close()


@pytest.fixture(scope="session")
def tt_ctx(tt_mesh):
    from rsl_rl_tt.tt_common import TTContext

    return TTContext.create(tt_mesh.mesh_device)

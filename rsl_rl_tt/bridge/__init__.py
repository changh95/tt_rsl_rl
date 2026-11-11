# SPDX-License-Identifier: BSD-3-Clause
"""Environment bridge (Phase 7, ADR-0003): `RemoteVecEnv` (TT host, rsl_rl `VecEnv`) <-> `serve(env, endpoint)` (environment host, e.g. Isaac Lab on a GPU machine).

`protocol` and `server` need only numpy / pyzmq (and torch tensors by duck typing); `client` needs rsl_rl. Import the parts you need:

    from rsl_rl_tt.bridge.server import serve          # environment host
    from rsl_rl_tt.bridge.client import RemoteVecEnv   # TT host
"""

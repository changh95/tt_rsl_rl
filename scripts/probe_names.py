# SPDX-License-Identifier: BSD-3-Clause
"""List ttnn attribute names matching the given substrings (import only; opens no device).

    scripts/tt_run.sh --chips 16 --timeout 120 --tag names -- python scripts/probe_names.py gather index embedding
"""

import sys

import ttnn

pats = [p.lower() for p in sys.argv[1:]]
print(sorted(n for n in dir(ttnn) if any(p in n.lower() for p in pats)))

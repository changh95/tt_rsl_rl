# SPDX-License-Identifier: BSD-3-Clause
"""Import ttnn and print the real signatures/docstrings of the APIs the smoke tests use (BRIEF: never invent TTNN APIs).

Opens NO device. Run it only through the wrapper (the import then happens inside the chip-masked namespace):

    scripts/tt_run.sh --chips 16 --timeout 300 --tag probe -- python scripts/probe_ttnn_api.py [dotted.name ...]

With no argument a default list is printed; with arguments (names relative to the ttnn module, e.g. `all_reduce`) only those
objects are described.
"""

import inspect
import sys

import ttnn

DEFAULT = ["WormholeComputeKernelConfig", "synchronize_device", "all_reduce", "ttnn.Tensor"]


def resolve(dotted: str):
    obj = ttnn
    for part in dotted.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def describe(dotted: str, max_doc_lines: int = 60) -> None:
    obj = resolve(dotted)
    if obj is None:
        print(f"\n=== {dotted}: MISSING")
        return
    print(f"\n=== {dotted}: {type(obj).__name__}")
    try:
        print("signature:", inspect.signature(obj))
    except (TypeError, ValueError):
        pass
    doc = inspect.getdoc(obj) or ""
    print("\n".join(doc.splitlines()[:max_doc_lines]))
    if inspect.isclass(obj):
        names = [n for n in dir(obj) if not n.startswith("_")]
        print("public attributes:", names)


print("ttnn file:", ttnn.__file__)
for name in sys.argv[1:] or DEFAULT:
    describe(name)

# Hosts

- **Galaxy (2026-11-10 … 11-12, no longer used):** the project was confined to one 8-chip tray of a shared Galaxy; resets were not allowed without the owner. The guards and hooks still encode that tray: they must be rewritten per host (BRIEF 4-b, Phase 0-b) before any device is opened. Keep the guards; change only the allow-list.
- **Desktop (RTX 5090 + one p150a, PCIe x16):** 1-chip work and Phase 7 with Isaac Lab on the same host.
- **QuietBox2 (multi-chip Blackhole; confirm chip count and topology at bring-up):** multi-chip data parallel and the G1 demo on several chips.
- Reset policy on the new hosts: confirm with the owner once per host and record the answer here (without machine identifiers).
- Environment to rebuild per host: `.venv` with ttnn 0.81.1 (PyPI wheel) + sfpi, `scripts/fetch_g1_model.sh`, golden data are in git.

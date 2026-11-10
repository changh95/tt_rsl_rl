# Parity tolerances

Status: **empty on purpose.** Tolerances are derived from the measured fp32 noise floor of TTNN matmul and the main
element-wise ops (Phase 0 step 6, on-device probes) and are filled in here with their evidence. Per BRIEF 2-9, a failing
test never loosens a tolerance by itself: the root cause is analysed and the change is recorded in `DECISIONS.md` with
the evidence.

| Component | Metric | Tolerance | Evidence (noise-floor measurement / reasoning) |
|-----------|--------|-----------|-----------------------------------------------|
| *(to be filled)* | | | |

# vendor_02 — ported sim from 02-memristive-crossbar @95868e7

These are the **transitive closure** of `02`'s noisy-training read-noise sweep
(`readnoise` + `coupled` + `manhattan`), copied **byte-for-byte** (only a one-line
provenance header prepended) so `08` runs the H5 sim hermetically — the portfolio is
**port, not import** (each project is a separate repo).

Files: `readnoise coupled manhattan crossbar solver jacobian memristor task ceilings`.
`ceilings.py` is included **only** so `manhattan.py` stays unmodified; the H5 path never
calls it.

**Do not edit these to diverge.** On an upstream `02` change, re-vendor with a clean copy:
re-run the copy + header step from `docs/superpowers/plans/2026-06-24-phase0-h5-sweep-runner.md`,
bumping the sha.

# Plan — Okika OTC2902K pre-purchase checklist and first-week bench plan

**Date:** 2026-11-08 · **Status:** checklist; nothing bought, nothing sent.
**Gate state:** `classify_access` → BUY (characterization), `audit_ceiling=(cpu, physical)`,
`in_fabric_path=mpw`. Budget `ACCESS_BUDGET_USD = 5000` awaits the user's ratification.

## A. Before money moves (user)

1. **Ratify the budget** in `verdict.py` (`ACCESS_BUDGET_USD`) or override with `python fg_access.py --budget N`.
2. **Ask Okika** (draft below): academic price + lead time; endurance (program/erase) rating; confirm the
   open toolset release exposes *individual-FG* target programming with read-back (the Georgia Tech
   programming routines) rather than only compiled circuits; confirm the VM/tools ship with the board.
3. **Order the SoC board only.** The FlexAnalog / Pi-HAT products are the Anadigm AN231E04 switched-cap
   part — not floating gates — and are not a substitute at any price.

### Draft inquiry (to Okika sales / support — not sent)

> Subject: OTC2902K for floating-gate characterization — academic pricing and three technical questions
>
> We are evaluating the OTC2902K (OTC2902E04 SoC FPAA) as a floating-gate characterization platform for a
> small analog-learning research project. Before ordering, could you confirm:
>
> 1. **Academic pricing and lead time** for one OTC2902K, shipped to [country].
> 2. **Individual floating-gate programming:** does the released toolset (Scilab/Xcos + x2c VM) expose the
>    Georgia Tech floating-gate programming routines so a user can program a *chosen* FG element to an
>    arbitrary target current and read it back, independent of a compiled Xcos design?
> 3. **Program/erase endurance** rating (cycles) for the floating-gate elements, and whether tunneling erase
>    is global to the array (as in the published RASP 3.0 design).
>
> We understand floating-gate writes occur in program mode under the on-chip programming infrastructure; if
> there is any supported mechanism to trigger injection from a CAB-local signal during run mode, we would be
> very interested to know.
>
> Thank you — [name, institution]

## B. First week with the board (the protocol, in order)

Run `fg_measured.CHARACTERIZATION_PROTOCOL` top to bottom, filling `measurements_template.json`:

| day | rows | output |
|---|---|---|
| 1 | toolchain bring-up; `span` (pick the current window); `run-mode local write` bench confirmation (expect: no) | `I_min`, `I_max`, `run_mode_local_write` |
| 2 | `read noise`; `open-loop potentiation ladder` (the **raw ν** — the number that decides the open-loop learner) | `sigma_read`, `nonlin_raw`, `dG0` |
| 3 | `per-cell depression test` (expect: injection-only → `differential_pair=True`); `differential-pair ladder`; `cycle-to-cycle` | `asym=1.0`, `sigma_write` |
| 4 | `closed-loop landing error` over a ≥32-target ladder; `global erase + re-inject cycle` | `prog_rms_error`, `nonlin_effective`, `renorm_sigma` |
| 5 | `retention` over one training run's wall-clock; endurance from datasheet | `drift`, `endurance_cycles` |

Then:

```bash
python transfer_sweep.py --measured measurements.json --differential --write-mode closed_loop --out transfer_measured_closed.json
```

```bash
python transfer_sweep.py --measured measurements.json --differential --write-mode open_loop --out transfer_measured_open.json
```

Both verdicts carry `params_confirmed=True`. Write the Phase-1c/1d findings (Observed vs Interpretation),
then and only then freeze `classify_build` for the Phase-2a inference-grade learner.

## C. What would end the project honestly at this point

- Measured **raw ν ≥ 3 and the differential-pair sim (H6) refuted** → no open-loop learner on this family;
  closed-loop program-verify (host-mediated) is the only learner → `08`'s in-fabric thesis forecloses on
  this device family. Publishable as a clean "no".
- Measured **closed-loop landing error worse than ~7 bits in-span** → even the inference-grade learner
  forecloses (Phase-1 precision cliff).

# Findings — Phase 1b: device access — Okika sells the Georgia Tech floating-gate SoC FPAA

**Date:** 2026-11-05
**Scope:** the access half of Phase 1 of `08-floating-gate`. The user reported that the floating-gate FPAA
boards from the academic papers are now commercially available from Okika; this note records what was
verified, runs the **frozen** `classify_access` (`verdict.py`, committed before any purchase) on the observed
inputs, and states what it does and does not change. **Artifact:** `access_okika.json` (committed).
Reproduce with `python fg_access.py`.

Observed and Interpretation are kept separate. **Nothing was bought.** Every input carries an evidence
grade (`fg_access.py`): *primary* = datasheet/paper read; *secondary* = vendor page, press, or search
summaries of a primary; *unknown* = not established.

> **Verification limits of this session.** The cloud session's network policy denied direct reads of
> `okikadevices.com`, `hasler.ece.gatech.edu`, `businesswire.com`, `semanticscholar.org` and the author
> mirrors, so the facts below come from search-engine summaries of those pages and papers (grade
> *secondary*). Nothing below is primary-read. Re-verify prices and the two vendor questions (§Interpretation
> 4) before money moves.

---

## Observed

### Availability and price (secondary)

| item | observed | source |
|---|---|---|
| Okika OTC2902K dev board | one OTC2902E04 SoC FPAA; 56 CABs (≈194 amplifiers/transistors), 42 CLBs (≈10k gates), 16 DACs, MSP430; 350 nm CMOS; **$2,499 list** | okikadevices.com product/collection pages; "What is SoC FPAA" page |
| distributor pricing | DigiKey single-unit ≈ £3,798 / €4,275 (OTC2902K) | DigiKey listings |
| academic pricing | "eligible students, professors, graduate researchers and institutions may qualify for discounted pricing" — unquantified | okikadevices.com |
| FlexAnalog boards / Pi HAT | AN231K04-SING1 $156–199, DUAL2 $219–249, QUAD4 $299, OTC2310K04-PIKA HAT $156–299 — all **Anadigm AN231E04** switched-capacitor FPAAs | okikadevices.com; DigiKey |
| Okika ← Anadigm | Okika Devices completed the acquisition of Anadigm (Mesa, AZ) in April 2025; AN231E04 now listed under Okika at DigiKey | press release (Business Wire / silicon.co.uk) |
| announcement | "Floating-Gate FPAA Now Commercially Available!" (video); Embedded World NA 2025 Best-in-Show for the OTC2902A SoC FPAA | YouTube; Embedded Computing Design |

### Lineage and toolchain (secondary)

- The SoC FPAA is the Georgia Tech / Hasler **RASP 3.0** line: George, Kim, Shah, Hasler et al., *A
  Programmable and Configurable Mixed-Mode FPAA SoC*, IEEE TVLSI 24(6), 2016 (FG-programmed analog + digital
  blocks with an open-source 16-bit MSP430). The toolchain is the open **Scilab/Xcos + x2c** flow (VPR-based
  place-and-route) that emits a single programming file for **floating-gate programming and SRAM setup**;
  Okika states FG programming runs "entirely on the device as an input data stream" and distributes the
  tools as an open-source VM.
- Published FG programming on this family is **closed-loop** hot-electron injection: measure channel current
  (14-bit), compute the next injection pulse (7-bit DAC), iterate; tunneling is the (less selective) erase.

### Write-side numbers published for the family (secondary; not in-span)

| quantity | published | paper |
|---|---|---|
| FG charge programming accuracy | ~10.2 bit | Kim, Shah, Hasler, *Calibration of Floating-Gate SoC FPAA System*, TVLSI 2017 |
| current-source error after calibration | 2.2% RMS over 25 dB | same |
| routine programming range / accuracy | 30 pA – 30 µA (six decades), better than 1% | Hasler & Natarajan, WOSET 2020 (citing TVLSI 2017) |
| raw open-loop update nonlinearity (ν in this model's sense) | **not found in usable form** | — |

### Learning on this chip (secondary)

- Shah & Hasler, *SoC FPAA Hardware Implementation of a VMM+WTA Embedded Learning Classifier*, IEEE JETCAS
  8(1), 2018: a VMM + winner-take-all classifier (XOR → sound classification) with "the digital computation
  of weight updates and floating-gate programming" on the mixed-mode system; a fast-injection small-update
  mode; 14-bit weight-update value. Companion algorithm paper: Hasler & Shah, JETCAS 8(1), 2018.
- Programming infrastructure on the RASP line: in **program mode** the FG pFET is disconnected from the
  circuit and tied to the programming drain line; **run mode** restores the circuit; generalized FG
  programming "transforms the Run-mode configuration into a crossbar" at the cost of extra run/program
  switches (Twigg PhD 2006; Hasler, VTS 2022).

### The frozen gate on the observed inputs (`access_okika.json`)

| candidate | inputs (FG? / price / FG-prog exposed? / run-mode local write?) | status | audit ceiling | in-fabric path |
|---|---|---|---|---|
| **Okika OTC2902K** | True / $2,499 / True (secondary) / **None** | **BUY** | `(cpu, physical)` | unknown → vendor query |
| Okika FlexAnalog AN231K04 | False / $156 / — / — | FORECLOSE (not a floating gate) | — | — |

`budget_usd = 5000` (pre-registered; see spec §2). With `--budget 2000` the OTC2902K forecloses on price
(test coverage), i.e. the BUY is budget-sensitive by construction.

---

## Interpretation

**1. The "no accessible floating-gate array" foreclosure path is closed.** A real FG array in the exact
academic lineage the charter named is purchasable at ~half the pre-registered budget and at a fraction of an
MPW characterization run. Phase 1's named next milestone — *a characterization tapeout to confirm the
provisional FG parameters* — becomes a **board purchase plus the bench protocol** in `fg_measured.py`. The
board's job is to flip `params_confirmed` on the *unchanged, frozen* `classify_transfer`.

**2. The board as sold does not advance the in-fabric thesis; it hosts the inference-grade object.** Every
published write on this family goes through program mode under the on-chip programmer, and the only
published learning on the chip computes updates on the MSP430. That is `(cpu, physical)`: the half-co-located
learner Phase 1 already PASSed, i.e. an FG **inference accelerator with non-volatile weights**. The
`(physical, physical)` object needs a fabric-local signal to trigger injection in run mode. Nothing reachable
here says that exists on the OTC2902E04; the infrastructure papers suggest it does not. Recorded as
`run_mode_local_write=None`, audit ceiling `(cpu, physical)`, `in_fabric_path="unknown"`.

**3. Even with run-mode writes, the in-fabric path faces the raw-ν wall Phase 1 found.** A fabric-triggered
write is an *open-loop* pulse; the learner then sees the **raw** FG nonlinearity, and Phase 1 showed ν ≥ 3
forecloses even at continuous precision. So the single most valuable bench number is the **open-loop
potentiation ladder** (raw ν), which the literature does not supply in this model's form. If it comes back
≥ 3, 08's in-fabric thesis forecloses *on this device family* independently of the vendor answer, unless
pulse co-design toward ν→0 is realizable on the part. That would be a clean, device-grounded result.

**4. The purchase hinges on two vendor facts, not on the physics.** (a) An **individual-FG target API** in
the Okika toolchain release (evidence: the open tools emit FG programming files and the calibration papers
program arbitrary targets — grade secondary); (b) **run-mode local write** (unknown). (a) gates the BUY; (b)
only sets the in-fabric path. Both are one e-mail to the vendor. Academic pricing should be asked for in the
same message.

**5. The FlexAnalog line is a trap and the gate names it.** Fifteen times cheaper, same vendor, same word
"FPAA" — and it is the Anadigm AN231E04 that `fpaa-neural-net` already classified inference-only. Buying it
would re-run that project.

**6. Published precision is favorable but not yet a parameter.** ~10-bit charge programming would clear the
8-bit `n_levels` cliff with margin *if it holds as a fraction of the learner's span*; the published figures
are relative to current over decades. Measure in-span (protocol row "closed-loop landing error").

**Caveats.** All facts are grade *secondary* (network limits above). Prices vary by page and distributor.
The JETCAS 2018 full text was not read; "updates computed on the MSP430" is from its abstract and summaries.
`ACCESS_BUDGET_USD` is a pre-registered modeling choice the user must ratify.

---

## Verdict (recorded, not asserted)

- **Frozen `classify_access` on the OTC2902K: BUY** — floating gate, FG programming exposed (secondary),
  $2,499 ≤ $5,000. `replaces_characterization_tapeout=True`.
- **Elevated:** `audit_ceiling=(cpu, physical)`, `in_fabric_path=unknown`, `vendor_query_needed=True`. The
  board, as sold, is an inference-grade host; in-fabric learning is **not** established by this purchase.
- **Negative control:** FlexAnalog AN231K04 → FORECLOSE (not a floating gate).
- **Who computes the update?** On this board as published: the on-chip MSP430 → `(cpu, physical)`.

## What this licenses next

1. **User actions (no code):** ratify `ACCESS_BUDGET_USD`; ask Okika the two questions in §4 plus academic
   price and endurance rating; re-run `python fg_access.py` with the answers (edit `OKIKA_OTC2902K`).
2. **On a confirmed BUY:** purchase; run `CHARACTERIZATION_PROTOCOL` on the bench → `measurements.json` →
   `python transfer_sweep.py --measured measurements.json` (and `--write-mode open_loop`) → the Phase-1
   verdict with `params_confirmed=True`; write the Phase-1c findings.
3. **Then, and only then,** freeze `classify_build` for Phase 2a (the inference-grade learner on the board)
   and decide the in-fabric path: `device` (if run-mode write confirmed **and** raw ν learns) or `mpw`
   (custom cell) or FORECLOSE (raw ν ≥ 3 on this family).
4. **Commons handoff (pending; harness not in this checkout):** candidate lessons — *"a commercial FG-FPAA
   is an inference-grade host unless the write can be fabric-triggered in run mode"*; *"measure raw
   open-loop ν before buying the in-fabric story on any FG part"*; *"same vendor, same 'FPAA' word, different
   physics: audit the write mechanism, not the product line."*

## Methodological notes (reusable)

- **Freeze the purchase decision like a verdict.** Device availability is a measurement with a budget; a
  pre-registered `classify_access` prevents "it's finally buyable" from becoming "buy it".
- **Separate the characterization job from the thesis.** The board earns a BUY on the cheap job (confirm the
  operating point) while the thesis-bearing question (in-fabric write) is recorded as elevated and unknown.
- **Record an evidence grade per input** when the inputs come from the web rather than the bench.
- **Name the trap product.** A negative control in the access record costs one dict and prevents a
  category error.

## Sources (all accessed via search summaries, 2026-11-05)

- Okika OTC2902K: https://okikadevices.com/products/otc2902k-development-board-for-polymorphic-fpaa-with-56-cabs-42-clbs-and-msp430-microcontroller ·
  collection: https://okikadevices.com/collections/otc2902k-soc-fpaa-development-board ·
  SoC tools: https://okikadevices.com/collections/soc-software · FlexAnalog boards:
  https://okikadevices.com/collections/flexanalog-development-boards · DigiKey OTC2902K:
  https://www.digikey.co.uk/en/products/detail/okika-devices/OTC2902K/28167253
- Okika–Anadigm acquisition: https://www.silicon.co.uk/press-release/okika-devices-completes-acquisition-of-anadigm
- "Floating-Gate FPAA Now Commercially Available!": https://www.youtube.com/watch?v=TWW0Q8sEGEA ·
  Embedded World NA 2025 award: https://embeddedcomputing.com/application/misc/embedded-world-north-america-2025-best-in-show-winners
- George et al., TVLSI 2016, DOI 10.1109/TVLSI.2015.2504119; PDF https://hasler.ece.gatech.edu/SoCFPAA/RASP30_SoCFPAA.pdf
- Kim, Shah, Hasler, *Calibration of Floating-Gate SoC FPAA System*, TVLSI 2017:
  https://www.researchgate.net/publication/318279213_Calibration_of_Floating-Gate_SoC_FPAA_System ·
  FG programming: https://hasler.ece.gatech.edu/SoCFPAA/FGProgramming_TVLSI.pdf
- Hasler & Natarajan, *An Open-Source ToolSet for FPAA Design*, WOSET 2020: https://woset-workshop.github.io/PDFs/2020/a18.pdf
- Shah & Hasler, JETCAS 2018, DOI 10.1109/JETCAS.2017.2777784; Hasler & Shah, JETCAS 2018, DOI 10.1109/JETCAS.2017.2771392
- Hasler, VTS 2022 special session: https://hasler.ece.gatech.edu/SoCFPAA/VTS2022_Hasler.pdf ·
  Twigg PhD 2006: https://hasler.ece.gatech.edu/Courses/ECE6435/theses/twigg_christopher_m_200608_phd.pdf
- Okika GitHub (naming): https://github.com/OkikaTechnologies/FPAASonar

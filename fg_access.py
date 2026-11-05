"""Phase-1b access record: the OBSERVED inputs for the frozen classify_access, with an evidence grade
per input, for the two Okika Devices product lines (2026-11-05). Outcomes recorded, not asserted.
Sources + verification status live in docs/superpowers/findings/2026-11-05-phase1b-okika-access-findings.md.

Evidence grades: "primary" (datasheet / primary paper read), "secondary" (vendor page, press, search
summaries of the primary), "unknown" (not established; the input is None)."""
import argparse
import json

from verdict import classify_access, ACCESS_BUDGET_USD

# Okika OTC2902K: dev board carrying one OTC2902E04 SoC FPAA (Georgia Tech / Hasler RASP 3.0 lineage,
# 350 nm, 56 CABs + 42 CLBs + MSP430, floating-gate programmed). List price $2,499 (vendor page);
# distributor single-unit prices run higher (DigiKey EU/UK listings ~EUR 4.3k / GBP 3.8k).
OKIKA_OTC2902K = {
    "device": "Okika OTC2902K (OTC2902E04 SoC FPAA)",
    "inputs": {
        "is_floating_gate": True,
        "price_usd": 2499.0,
        "fg_programming_user_exposed": True,
        "run_mode_local_write": None,
    },
    "evidence": {
        "is_floating_gate": "secondary: vendor describes FG analog memory; chip matches George et al. "
                            "TVLSI 2016 SoC FPAA (FG-programmed, MSP430, 350 nm)",
        "price_usd": "secondary: okikadevices.com list price; distributor prices higher; academic "
                     "pricing offered on request (unquantified)",
        "fg_programming_user_exposed": "secondary: open Scilab/Xcos + x2c toolchain emits an FG "
                                       "programming file; FG programming 'entirely on the device as an "
                                       "input data stream'; Hasler calibration papers program individual "
                                       "FG current sources to arbitrary targets over 6 decades. "
                                       "CONFIRM an individual-target API exists in the Okika release.",
        "run_mode_local_write": "unknown: on the published RASP infrastructure an FG is switched OUT of "
                                "the circuit onto a programming drain line for injection (program mode) "
                                "and the only published learning on the SoC FPAA computes weight updates "
                                "digitally on the on-chip MSP430 (Shah & Hasler, JETCAS 2018). Evidence "
                                "points to False; not confirmed from the datasheet.",
    },
}

# Negative control -- the same vendor's FlexAnalog line is the Anadigm AN231E04 switched-capacitor FPAA
# (Okika acquired Anadigm, April 2025): SRAM-held capacitor ratios, host-written. Not a floating gate.
# This is the part fpaa-neural-net already classified inference-only.
OKIKA_FLEXANALOG_AN231K04 = {
    "device": "Okika FlexAnalog AN231K04-SING1 (Anadigm AN231E04, switched-capacitor)",
    "inputs": {
        "is_floating_gate": False,
        "price_usd": 156.0,
        "fg_programming_user_exposed": False,
        "run_mode_local_write": False,
    },
    "evidence": {
        "is_floating_gate": "primary: AN231E04 datasheet -- switched-capacitor, SRAM configuration",
        "price_usd": "secondary: vendor product page ($156-$199 across pages)",
        "fg_programming_user_exposed": "n/a: no floating gates",
        "run_mode_local_write": "n/a: no floating gates",
    },
}

CANDIDATES = (OKIKA_OTC2902K, OKIKA_FLEXANALOG_AN231K04)


def access_record(candidate, budget_usd=ACCESS_BUDGET_USD):
    """Run the frozen gate on one candidate's observed inputs; attach device name + evidence grades."""
    v = classify_access(budget_usd=budget_usd, **candidate["inputs"])
    return {"device": candidate["device"], "verdict": v, "evidence": candidate["evidence"]}


def run_access(budget_usd=ACCESS_BUDGET_USD, candidates=CANDIDATES):
    """All candidates through the gate; the headline is the first BUY (if any)."""
    records = [access_record(c, budget_usd) for c in candidates]
    buys = [r for r in records if r["verdict"]["status"] == "BUY"]
    return {"budget_usd": float(budget_usd), "records": records,
            "headline": buys[0]["device"] if buys else None,
            "n_buy": len(buys)}


def _print_summary(res):
    print("=" * 78)
    print("PHASE-1b device-access gate (frozen classify_access) -- Okika candidates")
    print("=" * 78)
    print(f"budget_usd={res['budget_usd']:.0f}  (pre-registered; ratify before any purchase)")
    for r in res["records"]:
        v = r["verdict"]
        print(f"\n{r['device']}")
        print(f"  {v['status']}: {v['reason']}")
        print(f"  price={v['price_usd']:.0f} within_budget={v['within_budget']}  "
              f"audit_ceiling={v['audit_ceiling']}  in_fabric_path={v['in_fabric_path']}  "
              f"vendor_query_needed={v['vendor_query_needed']}")
    print(f"\nheadline: {res['headline']}  (n_buy={res['n_buy']})")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Phase-1b device-access gate on the observed Okika inputs.")
    ap.add_argument("--out", default="access_okika.json")
    ap.add_argument("--budget", type=float, default=ACCESS_BUDGET_USD)
    args = ap.parse_args(argv)
    res = run_access(budget_usd=args.budget)
    _print_summary(res)
    with open(args.out, "w") as fp:
        json.dump(res, fp, indent=2)
    print(f"\nraw record -> {args.out}")
    return res


if __name__ == "__main__":
    main()

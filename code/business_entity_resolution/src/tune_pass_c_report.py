"""Merge the per-country pass-C tuning results into one table, project full-test runtime, pick a config.

Calibration: one machine factor = Kaggle v1 base pass-C s/query (India) / local base s/query (India),
applied to every configuration and country (the India base ran alone; the US base overlapped another
run). Fit time is scaled the same way from the India base fit. Projection per country (test S1 and
query counts from Kaggle v1): fit_A + fit_C (per S1 row) + queries x (pass A + pass C per query);
France uses India's rates. The US base may be scored on a subset (see base_sample).

    python -m src.tune_pass_c_report
"""
import json

from .config import LOG_DIR, REPO_ROOT

RESULTS = [LOG_DIR / "pass_c_tuning.json", LOG_DIR / "tune_us" / "pass_c_tuning.json"]
KAGGLE_BENCH = REPO_ROOT / "kaggle" / "runs" / "20260926-v1" / "results" / "blocking_bench.json"
TARGET_MIN = 180
RECALL_TOL = 0.01


def load() -> dict:
    """Return {country: results} merged from every tuning results file."""
    out = {}
    for p in RESULTS:
        if p.exists():
            out.update(json.loads(p.read_text(encoding="utf-8")).get("results", {}))
    return out


def build_table(res: dict, kb: dict) -> tuple:
    """Return (rows, calibration factors) for configs measured in both US and India."""
    ind = res["India"]
    base_i = next(r for r in ind["configs"] if r["config"].startswith("base"))
    q_factor = (kb["bench"]["India"]["c_per_100k_s"] / 1e5) / (base_i["c_query_s"] / ind["queries"])
    fit_factor = kb["bench"]["India"]["c_fit_s"] / base_i["c_fit_s"]
    test = kb["projection"]["test"]["by_country"]
    names = [r["config"] for r in ind["configs"]]
    us_names = {r["config"] for r in res["US"]["configs"]}
    rows = []
    for name in [n for n in names if n in us_names]:
        per = {}
        for c in ("US", "India"):
            r = next(x for x in res[c]["configs"] if x["config"] == name)
            k = kb["bench"][c]
            per[c] = {"r": r, "c_s_q": q_factor * r["c_query_s"] / res[c]["queries"],
                      "fit_s_row": fit_factor * r["c_fit_s"] / res[c]["s1_rows"],
                      "a_s_q": k["a_per_100k_s"] / 1e5, "a_fit_row": k["a_index_s"] / k["s1_rows"]}
        minutes = {}
        for c, d in test.items():
            p = per.get(c, per["India"])
            minutes[c] = (d["s1"] * (p["fit_s_row"] + p["a_fit_row"]) + d["queries"] * (p["c_s_q"] + p["a_s_q"])) / 60
        row = {"config": name, "test_min": round(sum(minutes.values()), 1),
               "test_min_max_country": round(max(minutes.values()), 1),
               "by_country": {c: round(m, 1) for c, m in minutes.items()}}
        for c in ("US", "India"):
            r = per[c]["r"]
            row[c] = {"qps_kaggle": round(1 / per[c]["c_s_q"]), "pct_gated": r["pct_gated"],
                      "recall_C": r["recall_C"], "recall_union": r["recall_union"],
                      "cpq": r["cands_per_query"], "base_sample": r.get("base_sample", 0)}
        row["recall_union_mean"] = round((row["US"]["recall_union"] + row["India"]["recall_union"]) / 2, 4)
        rows.append(row)
    return rows, {"query": round(q_factor, 3), "fit": round(fit_factor, 3)}


def pick(rows: list) -> tuple:
    """Return (pick or None, best mean union recall): fastest row within RECALL_TOL of the best and <= TARGET_MIN."""
    best = max(r["recall_union_mean"] for r in rows)
    ok = [r for r in rows if r["recall_union_mean"] >= best - RECALL_TOL and r["test_min"] <= TARGET_MIN]
    return (min(ok, key=lambda r: r["test_min"]) if ok else None), best


def markdown(rows: list, factors: dict, chosen, best: float) -> str:
    """Return the Markdown table + pick."""
    lines = [f"Calibration (Kaggle / local, from the India base): query x{factors['query']}, fit x{factors['fit']}. "
             f"Best mean union recall {best:.4f}; pick rule: >= {best - RECALL_TOL:.4f} and <= {TARGET_MIN} min.", "",
             "| config | C q/s Kaggle US / IN | % gated US / IN | recall C US / IN | recall union US / IN | "
             "cands/q US / IN | test min (sum) | test min max country |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        u, i = r["US"], r["India"]
        sub = " (US base on 8k)" if u["base_sample"] else ""
        lines.append(f"| {r['config']}{sub} | {u['qps_kaggle']:,} / {i['qps_kaggle']:,} | {u['pct_gated']:.0f} / "
                     f"{i['pct_gated']:.0f} | {u['recall_C']:.4f} / {i['recall_C']:.4f} | {u['recall_union']:.4f} / "
                     f"{i['recall_union']:.4f} | {u['cpq']} / {i['cpq']} | {r['test_min']:,} | {r['test_min_max_country']:,} |")
    lines += ["", f"Pick: {chosen['config'] if chosen else 'NONE satisfies the rule'}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    """Build the table, print it and write logs/pass_c_tuning_table.md + .json."""
    res = load()
    kb = json.loads(KAGGLE_BENCH.read_text(encoding="utf-8"))
    rows, factors = build_table(res, kb)
    chosen, best = pick(rows)
    md = markdown(rows, factors, chosen, best)
    print(md)
    with open(LOG_DIR / "pass_c_tuning_table.md", "w", encoding="utf-8", newline="") as f:
        f.write(md)
    with open(LOG_DIR / "pass_c_tuning_table.json", "w", encoding="utf-8", newline="") as f:
        json.dump({"factors": factors, "rows": rows, "pick": chosen and chosen["config"]}, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()

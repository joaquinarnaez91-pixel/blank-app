"""Research registry: one JSON line per experiment in tools/research/registry.jsonl.

  from registry import log
  log(family="overnight", name="3x trend filter", hypothesis="...", data="...", method="...",
      variants=4, metrics={"sharpe": 0.95, "t": 2.9, ...}, verdict="adopted|promising|rejected", notes="...",
      code="tools/research/x.py", author="main")
  python3 tools/research/registry.py            # summary table + trial counts per family
"""
import json, sys, time, datetime
from pathlib import Path
PATH = Path(__file__).with_name("registry.jsonl")


def log(family, name, hypothesis, data, method, variants, metrics, verdict, notes="", code="", author="main"):
    rec = {"ts": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "author": author,
           "family": family, "name": name, "hypothesis": hypothesis, "data": data, "method": method,
           "variants": int(variants), "metrics": metrics, "verdict": verdict, "notes": notes, "code": code}
    with open(PATH, "a") as f:
        f.write(json.dumps(rec, default=float) + "\n")
    return rec


def load():
    if not PATH.exists():
        return []
    return [json.loads(l) for l in open(PATH) if l.strip()]


def trials(family):
    return sum(r["variants"] for r in load() if r["family"] == family)


if __name__ == "__main__":
    R = load()
    fams = {}
    for r in R:
        fams.setdefault(r["family"], []).append(r)
    for fam, rs in fams.items():
        print(f"\n[{fam}] experiments {len(rs)}, variants tried {sum(r['variants'] for r in rs)}")
        for r in rs:
            m = r["metrics"]
            key = ", ".join(f"{k} {v:.3g}" if isinstance(v, (int, float)) else f"{k} {v}" for k, v in list(m.items())[:5])
            print(f"  {r['ts'][:10]} {r['verdict']:<10} {r['name'][:60]:<60} | {key}")

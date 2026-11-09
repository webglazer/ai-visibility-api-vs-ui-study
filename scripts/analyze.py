"""Deterministic analysis: API vs website (UI) answers, with a within-website noise baseline.

For each engine and prompt:
  noise = mean Jaccard between pairs of the 5 website runs   (how much the website disagrees with itself)
  gap   = mean Jaccard between the API answer and each website run
  delta = noise - gap   (> 0 means the API differs from the website more than the website differs from itself)
computed on brand sets (from entities.jsonl.gz) and on cited-domain sets. 95% CIs by bootstrap over prompts
(prompt-clustered, 2,000 resamples). Google AI Mode has no API; it is compared with the Gemini API as a
stand-in, labelled as such. Writes results/*.csv, results/summary.json and results/summary.md.
"""
import csv, itertools, json, os, random, re, statistics, sys
from collections import defaultdict
from urllib.parse import urlparse
sys.path.insert(0, os.path.dirname(__file__))
from common import DATA, STUDY, load_prompts, read_jsonl_gz

RES = os.path.join(STUDY, "results")
os.makedirs(RES, exist_ok=True)
PAIRS = {"chatgpt": ("chatgpt", "chatgpt_api"), "gemini": ("gemini", "gemini_api"),
         "google_ai_mode": ("google", "gemini_api"),
         # the ChatGPT and Gemini websites decide themselves whether to search (ChatGPT searched in ~44% of
         # runs even with Bright Data's web_search=true); these rows keep only website runs that searched
         "chatgpt_searched_only": ("chatgpt", "chatgpt_api"), "gemini_searched_only": ("gemini", "gemini_api")}
LEGAL = r"\b(inc|ltd|llc|plc|gmbh|ag|sa|as|sia|uab|ou|oü|ab|co|corp|corporation|company|group|limited)\b\.?"


def norm_brand(b):
    b = b.lower().replace("&", " and ")
    b = re.sub(LEGAL, " ", b)
    b = re.sub(r"[^\w\s]", " ", b)
    b = re.sub(r"^the\s+", "", b.strip())
    return re.sub(r"\s+", " ", b).strip()


def host(u):
    if not u:
        return None
    u = str(u).strip()
    if "://" not in u:
        u = "http://" + u
    try:
        h = urlparse(u).hostname or ""
    except Exception:
        return None
    h = h.lower()
    h = re.sub(r"^(www\d?|m)\.", "", h)
    return h if "." in h and "vertexaisearch" not in h else None


def domains_of(r):
    out = []
    for key in ("citations", "links_attached"):
        for c in r.get(key) or []:
            if isinstance(c, dict):
                h = host(c.get("url") or c.get("link") or c.get("href") or c.get("domain"))
            else:
                h = host(c)
            if h:
                out.append(h)
    for s in r.get("sources") or []:           # gemini_api: title carries the domain
        if isinstance(s, dict):
            h = host(s.get("domain") or s.get("title"))
            if h:
                out.append(h)
    return out


def searched(r):
    e = r["engine"]
    if e == "chatgpt":
        return bool(r.get("web_search_triggered")) or bool(r.get("citations"))
    if e == "chatgpt_api":
        return bool(r.get("web_search_calls"))
    if e == "gemini_api":
        return bool(r.get("web_search_queries")) or bool(r.get("sources"))
    return bool(r.get("citations") or r.get("links_attached"))


def jac(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return None
    return len(a & b) / len(a | b)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def boot(vals, n=2000, seed=7):
    vals = [v for v in vals if v is not None]
    if len(vals) < 5:
        return (None, None)
    rng = random.Random(seed)
    ms = sorted(sum(rng.choice(vals) for _ in vals) / len(vals) for _ in range(n))
    return (ms[int(0.025 * n)], ms[int(0.975 * n)])


def load():
    rows = {}
    for name in ("bd_chatgpt", "bd_gemini", "bd_google", "api_chatgpt", "api_gemini"):
        for r in read_jsonl_gz(os.path.join(DATA, f"{name}.jsonl.gz")):
            if not r.get("usable"):
                continue
            key = (r["engine"], int(r["n"]), int(r.get("run") or 1))
            rows.setdefault(key, r)
    ents = {(e["engine"], int(e["n"]), int(e["run"])): e["brands"] for e in read_jsonl_gz(os.path.join(DATA, "entities.jsonl.gz"))}
    return rows, ents


def main():
    prompts = {p["n"]: p for p in load_prompts()}
    rows, ents = load()
    by = defaultdict(dict)  # (engine, n) -> run -> row
    for (e, n, run), r in rows.items():
        by[(e, n)][run] = r

    def brands(e, n, run):
        b = ents.get((e, n, run))
        if b is None:
            return None
        out, seen = [], set()
        for x in b:
            k = norm_brand(x)
            if k and k not in seen:
                seen.add(k)
                out.append(k)
        return out

    # ---- per-channel descriptive stats
    chan = []
    for e in ("chatgpt", "gemini", "google", "chatgpt_api", "gemini_api"):
        rs = [r for (ee, _, _), r in rows.items() if ee == e]
        if not rs:
            continue
        lens = [len(r.get("answer_text_markdown") or r.get("answer_text") or "") for r in rs]
        doms = [len(set(domains_of(r))) for r in rs]
        bl = [len(brands(e, int(r["n"]), int(r.get("run") or 1)) or []) for r in rs
              if (e, int(r["n"]), int(r.get("run") or 1)) in ents]
        chan.append({"channel": e, "answers": len(rs), "prompts": len({int(r["n"]) for r in rs}),
                     "searched_pct": round(100 * sum(searched(r) for r in rs) / len(rs), 1),
                     "median_chars": int(statistics.median(lens)), "mean_cited_domains": round(sum(doms) / len(doms), 2),
                     "mean_brands": round(sum(bl) / len(bl), 2) if bl else None})

    # ---- paired comparison
    per_prompt = []
    for label, (ui, api) in PAIRS.items():
        for n, p in prompts.items():
            uiruns = by.get((ui, n), {})
            if label.endswith("_searched_only"):
                uiruns = {k: v for k, v in uiruns.items() if searched(v)}
            apir = by.get((api, n), {}).get(1)
            if len(uiruns) < 2:
                continue
            runs = sorted(uiruns)
            ub = {r: brands(ui, n, r) for r in runs}
            ud = {r: domains_of(uiruns[r]) for r in runs}
            rec = {"comparison": label, "n": n, "pid": p["pid"], "set": p["set"], "kind": p["kind"], "lang": p["lang"],
                   "ui_runs": len(runs)}
            okb = [r for r in runs if ub[r] is not None]
            rec["noise_brand_j"] = mean(jac(ub[a], ub[b]) for a, b in itertools.combinations(okb, 2))
            rec["noise_domain_j"] = mean(jac(ud[a], ud[b]) for a, b in itertools.combinations(runs, 2))
            rec["noise_top1"] = mean((ub[a][0] == ub[b][0]) if ub[a] and ub[b] else None
                                     for a, b in itertools.combinations(okb, 2))
            if apir is not None:
                ab = brands(api, n, 1)
                ad = domains_of(apir)
                if ab is not None:
                    rec["gap_brand_j"] = mean(jac(ab, ub[r]) for r in okb)
                    rec["gap_top1"] = mean((ab[0] == ub[r][0]) if ab and ub[r] else None for r in okb)
                    union = set().union(*[set(ub[r]) for r in okb]) if okb else set()
                    rec["api_brands"] = len(ab)
                    rec["api_brands_never_on_site_pct"] = (100 * len([x for x in ab if x not in union]) / len(ab)) if ab else None
                rec["gap_domain_j"] = mean(jac(ad, ud[r]) for r in runs)
            per_prompt.append(rec)

    fields = sorted({k for r in per_prompt for k in r}, key=lambda k: (k not in ("comparison", "n", "pid", "set", "kind", "lang"), k))
    with open(os.path.join(RES, "per_prompt.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(per_prompt)

    def summarize(sub):
        out = {"prompts": len(sub)}
        for m in ("noise_brand_j", "gap_brand_j", "noise_domain_j", "gap_domain_j", "noise_top1", "gap_top1",
                  "api_brands_never_on_site_pct"):
            vals = [r.get(m) for r in sub]
            mu = mean(vals)
            lo, hi = boot(vals)
            out[m] = None if mu is None else {"mean": round(mu, 3), "ci95": [None if lo is None else round(lo, 3),
                                                                             None if hi is None else round(hi, 3)],
                                              "n": len([v for v in vals if v is not None])}
        for kind in ("brand", "domain"):
            d = [r[f"noise_{kind}_j"] - r[f"gap_{kind}_j"] for r in sub
                 if r.get(f"noise_{kind}_j") is not None and r.get(f"gap_{kind}_j") is not None]
            lo, hi = boot(d)
            out[f"delta_{kind}_j"] = None if not d else {"mean": round(mean(d), 3), "ci95": [lo and round(lo, 3), hi and round(hi, 3)], "n": len(d)}
        return out

    summary = {"channels": chan, "comparisons": {}}
    for label in PAIRS:
        sub = [r for r in per_prompt if r["comparison"] == label]
        summary["comparisons"][label] = {"all": summarize(sub),
                                         "global": summarize([r for r in sub if r["set"] == "global"]),
                                         "baltic": summarize([r for r in sub if r["set"] == "baltic"]),
                                         "by_kind": {k: summarize([r for r in sub if r["kind"] == k])
                                                     for k in sorted({r["kind"] for r in sub})}}
    json.dump(summary, open(os.path.join(RES, "summary.json"), "w", encoding="utf-8"), indent=1)
    with open(os.path.join(RES, "channels.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(chan[0].keys()) if chan else ["channel"])
        w.writeheader()
        w.writerows(chan)

    # ---- markdown summary
    def fmt(x):
        if not x:
            return "n/a"
        return f"{x['mean']:.2f} ({x['ci95'][0]}–{x['ci95'][1]}, n={x['n']})"
    L = ["# Results summary (auto-generated by scripts/analyze.py)", "",
         "Jaccard = shared items / all items (1 = identical lists, 0 = nothing in common).",
         "Noise = website vs website (same prompt, different runs). Gap = API vs website. Delta = noise minus gap.", "",
         "## Channels", "", "| Channel | Answers | Prompts | Searched % | Median chars | Cited domains | Brands |", "|---|---|---|---|---|---|---|"]
    for c in chan:
        L.append(f"| {c['channel']} | {c['answers']} | {c['prompts']} | {c['searched_pct']} | {c['median_chars']} | {c['mean_cited_domains']} | {c['mean_brands']} |")
    for label, s in summary["comparisons"].items():
        a = s["all"]
        note = {"google_ai_mode": " (website vs Gemini API stand-in: AI Mode has no API)",
                "chatgpt_searched_only": " (only website runs that searched the web)",
                "gemini_searched_only": " (only website runs that searched the web)"}.get(label, "")
        L += ["", f"## {label}{note}", "",
              "| Metric | Website vs website (noise) | API vs website (gap) |", "|---|---|---|",
              f"| Brand overlap (Jaccard) | {fmt(a['noise_brand_j'])} | {fmt(a['gap_brand_j'])} |",
              f"| Cited-domain overlap (Jaccard) | {fmt(a['noise_domain_j'])} | {fmt(a['gap_domain_j'])} |",
              f"| Same first brand | {fmt(a['noise_top1'])} | {fmt(a['gap_top1'])} |", "",
              f"- Brand delta (noise − gap): {fmt(a['delta_brand_j'])}",
              f"- Domain delta (noise − gap): {fmt(a['delta_domain_j'])}",
              f"- Share of API-named brands that never appear in any of the 5 website runs: {fmt(a['api_brands_never_on_site_pct'])} %",
              f"- Global prompts: brand noise {fmt(s['global']['noise_brand_j'])}, gap {fmt(s['global']['gap_brand_j'])}",
              f"- Baltic prompts: brand noise {fmt(s['baltic']['noise_brand_j'])}, gap {fmt(s['baltic']['gap_brand_j'])}"]
    open(os.path.join(RES, "summary.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

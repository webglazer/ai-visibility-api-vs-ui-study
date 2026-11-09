"""Truth check: does each "What does <company> in <country> do?" answer describe the company correctly?

Ground truth = raw/baltic-study/derived/ground-truth.csv (register-based activity summary per company).
Judge = Gemini 3.5 Flash (Vertex), JSON verdict: correct | partly_correct | wrong | no_knowledge.
Calibration: a blind 300-answer sample (60 per channel) is written to results/truth_calibration_sheet.csv
for a human to label; the judge's verdicts for those rows are kept separately in
results/truth_calibration_key.csv so the human label can be compared later (Rogan-Gladen correction).
Resumable: data/truth_verdicts.jsonl.gz."""
import csv, json, os, random, sys, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
from common import DATA, STUDY, append_jsonl_gz, gemini, gemini_text, load_prompts, log, read_jsonl_gz

DER = os.path.join(STUDY, "..", "..", "raw", "baltic-study", "derived")
RES = os.path.join(STUDY, "results")
OUT = os.path.join(DATA, "truth_verdicts.jsonl.gz")
MODEL = "gemini-3.5-flash"
SCHEMA = {"type": "OBJECT", "properties": {
    "verdict": {"type": "STRING", "enum": ["correct", "partly_correct", "wrong", "no_knowledge"]},
    "reason": {"type": "STRING"}}, "required": ["verdict", "reason"]}
PROMPT = """You check whether an AI answer correctly describes a specific company.
COMPANY: {brand} ({country})
WHAT IT REALLY DOES (from the official business register / company website): {truth}

AI ANSWER:
{answer}

Verdicts:
- correct: describes this company's main business correctly (minor missing details are fine).
- partly_correct: right company, but the main business is incomplete or mixed with wrong claims.
- wrong: describes a different company or a wrong main business.
- no_knowledge: says it cannot find / does not know the company, or gives no description.
Return JSON {{"verdict", "reason"}} with a one-sentence reason."""


def main():
    prompts = {p["n"]: p for p in load_prompts() if p["kind"] == "identity_en"}
    gt = {r["company_id"]: r for r in csv.DictReader(open(os.path.join(DER, "ground-truth.csv"), encoding="utf-8"))}
    items, seen = [], set()
    for name in ("bd_chatgpt", "bd_gemini", "bd_google", "api_chatgpt", "api_gemini"):
        for r in read_jsonl_gz(os.path.join(DATA, f"{name}.jsonl.gz")):
            n = int(r["n"])
            k = (r["engine"], n, int(r.get("run") or 1))
            if not r.get("usable") or n not in prompts or k in seen:
                continue
            seen.add(k)
            items.append((k, prompts[n], (r.get("answer_text_markdown") or r.get("answer_text") or "")[:6000]))
    done = {(v["engine"], v["n"], v["run"]) for v in read_jsonl_gz(OUT)}
    todo = [x for x in items if x[0] not in done]
    log(f"truth: {len(done)} judged, {len(todo)} to judge with {MODEL}")

    def one(x):
        k, p, answer = x
        g = gt.get(p["company_id"], {})
        q = PROMPT.format(brand=g.get("brand_name", p["company_id"]), country=g.get("hq_country", p["country"]),
                          truth=g.get("activity_summary", ""), answer=answer)
        for attempt in range(10):
            st, resp = gemini(q, model=MODEL, json_schema=SCHEMA, temperature=0)
            if st == 200:
                try:
                    v = json.loads(gemini_text(resp))
                    append_jsonl_gz(OUT, [{"engine": k[0], "n": k[1], "run": k[2], "company_id": p["company_id"], **v}])
                    return True
                except Exception:
                    pass
            time.sleep(min(90, 3 * 2 ** attempt))
        return False

    with ThreadPoolExecutor(4) as ex:
        ok = sum(ex.map(one, todo))
    log(f"truth: judged {ok}/{len(todo)} this pass")

    verdicts = read_jsonl_gz(OUT)
    by = defaultdict(Counter)
    for v in verdicts:
        by[v["engine"]][v["verdict"]] += 1
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, "truth_by_channel.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["channel", "answers", "correct", "partly_correct", "wrong", "no_knowledge", "correct_pct"])
        for ch, c in sorted(by.items()):
            t = sum(c.values())
            w.writerow([ch, t, c["correct"], c["partly_correct"], c["wrong"], c["no_knowledge"], round(100 * c["correct"] / t, 1)])
    # blind calibration sample, 60 per channel
    rng = random.Random(20261105)
    vmap = {(v["engine"], v["n"], v["run"]): v for v in verdicts}
    text = {x[0]: x for x in items}
    sheet, key = [], []
    for ch in ("chatgpt", "gemini", "google", "chatgpt_api", "gemini_api"):
        ks = sorted(k for k in vmap if k[0] == ch and k in text)
        for i, k in enumerate(rng.sample(ks, min(60, len(ks)))):
            _, p, answer = text[k]
            g = gt.get(p["company_id"], {})
            rid = f"{ch}-{k[1]}-{k[2]}"
            sheet.append({"row_id": rid, "company": g.get("brand_name", ""), "country": g.get("hq_country", ""),
                          "what_it_really_does": g.get("activity_summary", ""), "evidence_url": g.get("evidence_url", ""),
                          "ai_answer": answer[:3000], "your_label (correct/partly_correct/wrong/no_knowledge)": ""})
            key.append({"row_id": rid, "judge_verdict": vmap[k]["verdict"], "judge_reason": vmap[k]["reason"]})
    rng.shuffle(sheet)
    for name, rows in (("truth_calibration_sheet.csv", sheet), ("truth_calibration_key.csv", key)):
        with open(os.path.join(RES, name), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    log(f"truth: wrote results/truth_by_channel.csv and a {len(sheet)}-row blind calibration sheet")


if __name__ == "__main__":
    main()

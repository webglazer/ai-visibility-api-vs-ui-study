"""Build the 1,250-prompt set: 610 Baltic (exact August 2026 wording) + 640 global (Search Arena).

Baltic 610 = buyer question #01 in English + #01 in the home language for each of the 126
country+niche groups (252) + "What does <company> in <country> do?" for all 358 companies.
Global 640 = real English first-turn prompts from lmarena-ai/search-arena-24k (CC-BY-4.0),
screened by Gemini for being a standalone question an AI search engine would answer with
brands or sources, then sampled 256 recommendation / 128 local / 128 informational / 128 branded.
"""
import json, os, random, re, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
from common import DATA, gemini, gemini_text, log

SEED = 20261105
ARENA = os.environ.get("ARENA_PARQUET", "search-arena-24k.parquet")
LOCAL_LANG = {"LV": "lv", "LT": "lt", "EE": "et"}
TARGET = {"recommendation": 256, "local": 128, "informational": 128, "branded": 128}


def baltic():
    P = json.load(open(os.path.join(DATA, "august-prompts.json"), encoding="utf-8"))
    out = []
    groups = sorted({(p["country"], p["niche"]) for p in P.values() if p["kind"] == "buyer"})
    for country, niche in groups:
        cc = country.lower()
        for lang in ("en", LOCAL_LANG[country]):
            pid = f"{cc}-{niche}-buyer-{lang}-01"
            p = P[pid]  # KeyError = a group without question 01: fail loudly
            out.append({"pid": pid, "set": "baltic", "kind": "buyer", "lang": lang, "country": country,
                        "niche": niche, "company_id": "", "text": p["text"]})
    for p in sorted((p for p in P.values() if p["kind"] == "identity_en"), key=lambda x: x["prompt_id"]):
        out.append({"pid": p["prompt_id"], "set": "baltic", "kind": "identity_en", "lang": "en",
                    "country": p["country"], "niche": p["niche"], "company_id": p["company_id"], "text": p["text"]})
    log(f"build: baltic {len(out)} prompts ({len(groups)} country+niche groups)")
    return out


CLASSIFY_SCHEMA = {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
    "id": {"type": "INTEGER"}, "keep": {"type": "BOOLEAN"},
    "category": {"type": "STRING", "enum": ["recommendation", "local", "informational", "branded", "none"]}},
    "required": ["id", "keep", "category"]}}

CLASSIFY_PROMPT = """You screen real user prompts for a study of AI search engines (ChatGPT, Gemini, Google AI Mode).
KEEP a prompt only if ALL are true:
- It is a standalone question or request someone could type into an AI search engine today (Nov 2026) and get a sensible answer.
- A good answer would name companies, products, services, places or websites, or cite web sources.
- It is NOT tied to a specific past news moment or "today/latest match/this week's" event, NOT a coding/debugging task, NOT a request to write, translate, roleplay or summarise given text, NOT about the AI itself, NOT sexual, hateful or personal data.
CATEGORY (only if keep):
- recommendation: asks which/best/top products, tools, services or companies to choose or buy.
- local: asks about businesses, places or services in a specific city or region.
- branded: names a specific brand, company or product and asks about it (reviews, comparison, pricing, what it is).
- informational: other how/what/why questions on a topic where sources would be cited (health, money, travel, tech, science...).
If several fit, prefer local > branded > recommendation > informational.
Return a JSON array with one object per prompt: {"id", "keep", "category"} (category "none" when keep is false).

Prompts:
"""


def arena_candidates():
    import pandas as pd
    df = pd.read_parquet(ARENA)
    df = df[(df.turn == 1)]
    df = df[df.languages.apply(lambda l: list(l) == ["English"])]
    df = df[df.primary_intent.isin(["Recommendation", "Factual Lookup", "Info Synthesis", "Analysis",
                                    "Explanation", "Guidance"])]
    df["text"] = df.messages_a.apply(lambda m: re.sub(r"\s+", " ", m[0]["content"]).strip())
    df = df[df.text.str.len().between(20, 300)]
    df = df[~df.text.str.contains(r"```|def |import |\{|\}|<[a-z]+>|http", regex=True)]
    df["norm"] = df.text.str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True).str.strip()
    df = df.drop_duplicates("norm")
    rows = df[["text", "primary_intent"]].to_dict("records")
    random.Random(SEED).shuffle(rows)
    log(f"build: search arena candidates after rule filters: {len(rows)}")
    return rows


def classify(rows, batch=40):
    batches = [rows[i:i + batch] for i in range(0, len(rows), batch)]

    def run(b):
        listing = "\n".join(f"{i}. {r['text']}" for i, r in enumerate(b))
        for _ in range(4):
            st, resp = gemini(CLASSIFY_PROMPT + listing, json_schema=CLASSIFY_SCHEMA, temperature=0)
            if st == 200:
                try:
                    res = json.loads(gemini_text(resp))
                    return [(b[x["id"]], x) for x in res if 0 <= x.get("id", -1) < len(b)]
                except Exception:
                    pass
        return []

    out = []
    with ThreadPoolExecutor(8) as ex:
        for res in ex.map(run, batches):
            out.extend(res)
    return out


def global_set():
    rows = arena_candidates()
    pool = {k: [] for k in TARGET}
    done = 0
    for start in range(0, len(rows), 1200):  # classify in chunks until every category is full
        chunk = rows[start:start + 1200]
        for r, x in classify(chunk):
            if x["keep"] and x["category"] in pool:
                pool[x["category"]].append(r["text"])
        done += len(chunk)
        log(f"build: classified {done}: " + ", ".join(f"{k} {len(v)}" for k, v in pool.items()))
        if all(len(pool[k]) >= TARGET[k] * 1.2 for k in TARGET):
            break
    rng = random.Random(SEED)
    out, short = [], 0
    for k, n in TARGET.items():
        pick = rng.sample(pool[k], min(n, len(pool[k])))
        short += n - len(pick)
        out += [{"set": "global", "kind": k, "text": t} for t in pick]
    if short:  # fill any shortfall from unused recommendation prompts
        used = {o["text"] for o in out}
        extra = [t for t in pool["recommendation"] if t not in used]
        out += [{"set": "global", "kind": "recommendation", "text": t} for t in rng.sample(extra, short)]
        log(f"build: filled {short} short slots from recommendation")
    for i, o in enumerate(out, 1):
        o.update(pid=f"g-{o['kind'][:4]}-{i:04d}", lang="en", country="US", niche="", company_id="")
    return out


if __name__ == "__main__":
    b = baltic()
    g = global_set()
    allp = b + g
    assert len({p["pid"] for p in allp}) == len(allp)
    for i, p in enumerate(allp):
        p["n"] = i  # stable integer id used as the Bright Data `index` base
    with open(os.path.join(DATA, "prompts.jsonl"), "w", encoding="utf-8") as f:
        for p in allp:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    log(f"build: wrote {len(allp)} prompts (baltic {len(b)}, global {len(g)}) to data/prompts.jsonl")

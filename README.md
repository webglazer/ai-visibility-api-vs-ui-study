# Can you track AI visibility with an API? (WebGlazer study, November 2026)

We asked ChatGPT, Gemini and Google AI Mode the same 1,250 prompts through their websites (5 times each) and through their APIs (once each), all with web search on. That gives 21,250 answers. This repo has every answer, the code that collected and analysed them, and the results.

Data on Hugging Face: https://huggingface.co/datasets/webglazer/ai-visibility-api-vs-ui-2026-11

## What we found

Jaccard = how much two lists overlap (1 = identical, 0 = nothing shared). "Website vs website" is the same prompt asked again on the same website, which is the normal run-to-run noise. If the API were a perfect stand-in, "API vs website" would match it.

| Brands named (Jaccard) | Website vs website | API vs website |
|---|---|---|
| ChatGPT | 0.50 | 0.31 |
| Gemini | 0.39 | 0.36 |
| Google AI Mode (vs Gemini API, AI Mode has no API) | 0.38 | 0.30 |

| Sites cited (Jaccard) | Website vs website | API vs website |
|---|---|---|
| ChatGPT | 0.35 | 0.23 |
| Gemini | 0.22 | 0.14 |
| Google AI Mode (vs Gemini API) | 0.47 | 0.14 |

- **ChatGPT:** the API is a poor stand-in. It shares far fewer brands with the website than the website shares with itself. On recommendation prompts its answers are about half as long (median 169 vs 331 words) and name fewer brands (median 5 vs 7). 29% of the brands the API names never appear in any of the 5 website answers.
- **Gemini:** the API is close to the app for brands (0.36 vs 0.39), though it cites different sites.
- **Google AI Mode:** there is no API, and the Gemini API cites very different sources (0.14 vs 0.47).
- **The ChatGPT website searched the web in only 46% of answers** even with web search requested. The Gemini app searched in 70%, AI Mode in 99%.

All numbers with 95% confidence intervals: `results/summary.md`. Answer length and brand counts: `results/answer_shape.md`.

## How it was collected

| Channel | How | Model | Answers | Date (UTC) |
|---|---|---|---|---|
| ChatGPT website | Bright Data ChatGPT scraper, `web_search=true`, logged out | whatever chatgpt.com served | 6,250 | 2026-11-05 |
| Gemini app | Bright Data Gemini scraper, prompt + "Search the web for current information and cite your sources." | app reported "Flash-Lite" | 6,250 | 2026-11-05 |
| Google AI Mode | Bright Data Google AI Mode scraper | n/a | 6,250 | 2026-11-05 |
| ChatGPT API | OpenAI Responses API, `web_search` tool forced. 21 answers direct from OpenAI, 1,229 through OpenRouter (provider pinned to OpenAI) after the OpenAI account hit a daily request cap | `gpt-5.4-mini` | 1,250 | 2026-11-05 and 06 |
| Gemini API | Vertex AI `generateContent` with Google Search grounding, same nudge text as the app | `gemini-3.5-flash-lite` | 1,250 | 2026-11-05 |

Prompts (`data/prompts.jsonl`):

- **640 global prompts**: real user prompts sampled from [lmarena-ai/search-arena-24k](https://huggingface.co/datasets/lmarena-ai/search-arena-24k) (CC BY 4.0): 256 recommendation, 128 local, 128 informational, 128 branded.
- **610 Baltic prompts**: 252 buyer questions ("Who are the best grain trading companies in Estonia?") in English and the local language, and 358 identity questions ("What does X in Latvia do?") about real companies in Estonia, Latvia and Lithuania. Each prompt was sent with the matching country set.

Brands in every answer were extracted with `gemini-3.6-flash` (temperature 0, JSON schema, URLs stripped first): `data/entities.jsonl.gz`.

## Files

| File | What |
|---|---|
| `data/bd_chatgpt.jsonl.gz`, `bd_gemini`, `bd_google` | Website answers (text, markdown, citations, links, whether search ran) |
| `data/api_chatgpt.jsonl.gz`, `api_gemini` | API answers (text, citations or grounding sources, search queries, token usage) |
| `data/entities.jsonl.gz` | Brands named in each answer |
| `data/truth_verdicts.jsonl.gz` | AI-judge verdicts on whether identity answers describe the company correctly |
| `data/prompts.jsonl` | The 1,250 prompts |
| `results/` | Summary tables |
| `scripts/` | Collection and analysis code (Python 3, standard library; `build_prompts.py` also needs pandas and pyarrow) |

Join answers, brands and prompts on `engine`, `n` (prompt number) and `run` (1 to 5 for websites, 1 for APIs).

## Limits

- One collection window (5 to 6 November 2026). Answers change over time.
- Websites were scraped logged out. Logged-in users with memory, location history or paid plans can see different answers.
- One API model per engine. A different model or settings would give different numbers; `chat-latest`, the model behind ChatGPT, was not tested.
- Google AI Overviews were not tested.
- The truth check uses an AI judge that has not yet been checked against human labels, and the ground truth it compares against is not part of this release. Treat `truth_verdicts` as unverified.
- Brand extraction is done by an LLM and will miss or merge some names.

## Reproduce

```
export STUDY_SECRETS=~/.study-secrets   # files: brightdata.key, openai.key, openrouter.key
export STUDY_GCP_PROJECT=your-project    # Google Cloud project for the Gemini API arm (gcloud auth)
python scripts/analyze.py                # rebuilds results/ from data/
python scripts/answer_shape.py
```

## License

Data: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Credit "WebGlazer, Can you track AI visibility with an API? (2026)". The global prompts come from Search Arena (CC BY 4.0, lmarena-ai). Code: MIT (`LICENSE`).

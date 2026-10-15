# FreightTiger SWE Intern Case Study

Solution for the 24-hour take-home. It groups weekly shipments by route, tracks cost per tonne-km, flags anomalies against each route's own history and against peer routes, and checks the flags against the context notes to decide whether a spike is justified or unexplained.

- Live Streamlit demo: https://freighttiger-shipping-assistant.streamlit.app
- Cloudflare LLM proxy: https://freighttiger-llm-proxy.vatxzz.workers.dev

## TL;DR

- Core pipeline is deterministic by default: no LLM calls in default mode, $0.00, byte-identical output across runs.
- Matches all 3 reference rows in `sample_output_format_v2.csv` exactly.
- Eval harness: 12/12 verdicts, 12/12 note attributions, 6/6 planted traps rejected. 20/20 unit tests pass.
- An opt-in LLM mode is also available (`--mode llm`), pre-gated and response-cached. The LLM is also used in the interactive query layer (CLI + Streamlit).

## How to run

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Generate submission_output.csv (default deterministic mode)
python3 src/cli.py run --mode deterministic

# Or run with opt-in LLM evaluation (pre-gated and response-cached)
python3 src/cli.py run --mode llm

# Check 3-run reproducibility (matching SHA-256)
python3 src/cli.py verify-reproducibility

# Eval harness (12 labeled cases, supports --mode deterministic and --mode llm)
python3 src/eval_harness.py

# Unit tests (20)
pytest

# Ask a question in plain English (stretch goal)
python3 src/cli.py query "why did Chennai-Bangalore get pricier in March 2025?"

# Streamlit dashboard
streamlit run app.py
```

## Approach

### Cost and baselines

- **Weekly cost per tonne-km:** grouped by `origin-destination`, `route_type` and Monday date (`week_of`). Computed as `sum(freight_cost_inr) / sum(quantity_tonnes * distance_km)`. I pooled cost and tonne-km instead of averaging per-shipment rates so small loads don't skew the corridor number. On this data the two methods differ by under 1% per week, but pooling is the right definition for spend.
- **vs own history:** mean of the prior 8 weeks, strictly before the current week (no lookahead). With fewer than 8 weeks it averages what exists, with no padding. Week 1 returns `N/A (first week on this route)`.
- **vs similar routes:** mean of the other routes with the same `route_type` in the same week. The route itself is excluded.

### Anomaly rules

A route-week is flagged if any of these hold:

1. `vs_hist_pct >= 15%` (jump against own history).
2. `vs_peer_pct >= 20%` **and** `vs_hist_pct > 0%` (peer outlier). The `> 0` condition stops a route being flagged when its own rates were flat and a peer simply dropped.
3. Network-wide median `vs_hist_pct >= 5%`, which catches macro shocks like a nationwide diesel hike where every peer rises together and the peer gap stays near zero.

Moving the historical cutoff anywhere between 12% and 20% gives the same flagged rows on this dataset, so I didn't tune it further.

**Mumbai-Pune rate creep:** rates climbed about 20% in late 2025. Because it happened gradually, the rolling 8-week history kept adapting and week-on-week jumps stayed under 10%. The 20% peer threshold against Delhi-Jaipur is what caught it. A multi-week persistence rule would isolate creep more cleanly in production.

### Context notes and traps

`context_notes.csv` has 10 notes, and most are traps:

| Note | Scope | Outcome |
| :-- | :-- | :-- |
| N001 | Chennai-Bangalore, floods | Valid for Feb 24 - Mar 8 only. Later weeks are flagged `Yes` (roads reopened Mar 9) |
| N002 | Ahmedabad-Mumbai, festival surcharge | Valid for week of 2025-01-20 |
| N003 | All routes, diesel hike | Valid for the macro-surge week of 2025-05-05 |
| N004 | Routes outside the dataset | Rejected |
| N005 | Mumbai-Delhi maintenance | Rejected, costs were unaffected |
| N006 | Quarterly report, stable demand | Rejected, can't explain Mumbai-Pune |
| N007 | Delhi-Jaipur resurfacing done | Rejected |
| N008 | Kolkata-Bhubaneswar normal movement | Rejected |
| N009 | Chennai-Bangalore back to normal | Rejected |
| N010 | Tracking mandate, costs absorbed | Rejected |

`src/justification.py` uses two stages:

1. TF-IDF retrieval (`TfidfVectorizer` + cosine similarity) finds candidate notes for the corridor and week.
2. A deterministic gate then rejects notes with negation or normalcy language (`costs were not affected`, `costs absorbed`, `returned to normal`, `outside this dataset`) and checks that the week falls inside the note's active date window.

If no note passes, the row is `Yes` with a blank `matched_note_id`.

With only 10 notes, TF-IDF is effectively the same as a filtered scan. I kept it as the retrieval step so the design carries over to a larger notes corpus, but it isn't doing much work here.

### Modes

The pipeline provides two execution modes:
- **`deterministic` (default):** 0 LLM calls, $0.00 cost, byte-identical output across runs (`57bfdedcc27726a31f51c1257372d39377ddd47d15a294efdd978dcc7a75dede`).
- **`llm` (opt-in):** Evaluates candidate notes using Google Gemini 3.5 Flash-Lite. Candidates are pre-gated by corridor, active date window, and normalcy rejection before reaching the model (if no note survives, 0 LLM calls are made). The model must return an exact `evidence_quote` from the note text or it falls back to deterministic. All responses are cached in `output/llm_cache.json` (keyed by SHA-256 of model, prompt, and candidate texts), so repeat runs make 0 API calls (`070c6212337bf7a573b15136461a9e4edab9408c997ef8ef3a68ed57b4c823f7`).

> **Note on LLM mode vs Deterministic mode:**
> LLM mode is an opt-in layer. On the provided notes, the deterministic gate resolves every hard case before the model is called (disqualifying 16 of the 24 candidate anomalies, including all 6 distractor notes). The 8 rows that reached the model are straightforward matches (N001, N002, N003), so both modes agree on all 24 verdicts and note attributions by construction. The 6/6 distractors rejected in LLM mode reflects the gate's work, not the model's. The LLM's true value is robustness to unseen, nuanced note phrasing, which this curated dataset does not stress.

### Why deterministic is the default

The brief asked for reproducibility and trust. Handing records unconstrained to an LLM to decide flags and attribution brings run-to-run drift in flags and wording, a risk of pinning a cost spike on a distractor note, and API cost and latency for a batch job. Rules plus TF-IDF give 0 LLM calls in default mode and identical output every run.

I kept LLM calls for the opt-in evaluator mode and the interactive assistant (`src/cli.py query` and the Streamlit chat), where free-form questions benefit from generation. It goes through a Cloudflare Worker proxy (`cloudflare_worker/`) to Google Gemini 3.5 Flash-Lite, with a local template fallback when offline.

## Verification

**Sample parity.** Matches the 3 reference rows in `sample_output_format_v2.csv` character for character:

- `Delhi-Jaipur` (2024-11-11): 4.17 cptk, +35.5% vs history, +21.0% vs peers, `Yes` (unexplained).
- `Ahmedabad-Mumbai` (2025-01-20): 3.29 cptk, +29.5% vs history, +22.5% vs peers, `No (justified)`, note `N002`.
- `Mumbai-Pune` (2025-09-15): 3.98 cptk, +9.2% vs history, +23.6% vs peers, `Yes` (distractor N006 rejected).

**Reproducibility.** Three clean runs over `shipment_records.csv` gave the same SHA-256 (also in `output/reproducibility_hashes.txt`):

`57bfdedcc27726a31f51c1257372d39377ddd47d15a294efdd978dcc7a75dede`

**Eval harness** (`src/eval_harness.py`, 12 labeled cases in `tests/labeled_evaluation_set.csv`):

Scores 100% in both default deterministic and LLM modes (`--mode deterministic` and `--mode llm`):
- Verdict accuracy: 12/12 (100.0%)
- Note attribution accuracy: 12/12 (100.0%)
- Planted trap rejection: 6/6 (100.0% distractors rejected — pre-gate rejected before LLM)

The set is small and I wrote it myself, so treat it as a regression check, not a benchmark.

## Cost log

- Core pipeline (default mode): 0 LLM calls, 0 tokens, $0.00.
- Core pipeline (LLM mode): 8 API calls on initial run ($0.00053); 0 API calls and 8 cache hits on repeat run ($0.00000).
- Interactive query: about 220 input / 130 output tokens, roughly $0.000056 per query at Gemini Flash rates.

## Repo structure

- `src/core_analysis.py`: weekly aggregation, rolling history, peer averages, anomaly rules
- `src/retrieval.py`: TF-IDF search over context notes
- `src/justification.py`: date windows, negation regexes, pre-gating, trap rejection
- `src/pipeline.py`: loads data, scores anomalies, justifies, exports CSV
- `src/llm_client.py`: Cloudflare proxy client with offline fallback
- `src/eval_harness.py`: scores the pipeline against the labeled set
- `src/cli.py`: `run`, `verify-reproducibility`, `cost-log`, `query`
- `app.py`: Streamlit dashboard
- `output/llm_cache.json`: pre-populated LLM response cache for zero-drift reproducibility
- `submission_output.csv`: output in the required format
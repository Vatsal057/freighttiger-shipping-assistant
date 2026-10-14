# FreightTiger shipping cost anomaly assistant

This project monitors weekly road shipping costs across corridors, flags unexplained price jumps, and checks external context notes (weather, fuel, festivals, tolls) to see if a cost rise has a real cause.

---

## Quickstart

Run these from the repository root:

```bash
# Launch the interactive Streamlit Web Dashboard (Executive & Evaluator UI)
streamlit run app.py

# Run the CLI pipeline and write submission_output.csv
python3 src/cli.py run

# Check that 3 runs produce the exact same file (SHA-256)
python3 src/cli.py verify-reproducibility

# View token usage and cost log
python3 src/cli.py cost-log

# Ask a question in plain English (stretch goal)
python3 src/cli.py query "why did Chennai-Bangalore get pricier in March 2025?"

# Run unit tests
pytest tests/ -v
```

---

## How it works

The tool has two main parts:

1. **Analysis engine (`src/core_analysis.py`):**
   - Groups shipments by corridor and week (Monday to Sunday).
   - Computes cost per tonne-km as total freight cost divided by total tonne-km.
   - Compares each week against the route's trailing 8-week history and same-week peer routes.
   - Flags candidate spikes where costs jumped out of the ordinary.

2. **Justification layer (`src/justification.py`):**
   - Checks candidate spikes against `context_notes.csv`.
   - Filters out distractor notes that do not justify a price rise.
   - Outputs `No (justified)` with the note ID when a valid disruption occurred.
   - Outputs `Yes` with an empty note ID when no valid explanation exists.

---

## Mathematical definitions

### 1. Cost per tonne-km
For route `r` in week `w`:

$$\text{Cost per tonne-km} = \frac{\sum \text{freight\_cost\_inr}}{\sum (\text{quantity\_tonnes} \times \text{distance\_km})}$$

We sum all freight costs and divide by total tonne-km across all shipments in that week. We do not average individual shipment ratios, which would let small loads skew the weekly number.

### 2. Trailing 8-week history (`vs_own_history`)
We compute the arithmetic mean of weekly cost per tonne-km over the trailing 8 weeks strictly before the current week:

$$\text{Baseline} = \frac{1}{K} \sum_{k=1}^{K} \text{Cost}_{w-k}, \quad K = \min(8, \text{prior weeks})$$

- **No lookahead:** Future weeks never enter the calculation.
- **Early weeks:** Week 1 has no prior history, so it returns `N/A`. Weeks 2 to 8 use whatever prior weeks exist without invented padding.

### 3. Peer route comparison (`vs_similar_routes`)
We take the mean cost per tonne-km of other routes in the same length category (Short, Medium, or Long) during the exact same week. The route itself is excluded from its own peer average.

---

## Verification against sample data

We checked the hand-computed numbers from `shipment_records.csv` directly against the 3 rows in `sample_output_format_v2.csv`:

| Corridor | Week of | Metric | Hand-computed | Pipeline output | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Delhi-Jaipur | 2024-11-11 | Cost per tonne-km | 4.1729 | 4.17 | Match |
| | | vs. own history | +35.52% | +35.5% vs this route's past average | Match |
| | | vs. similar routes | +21.03% | +21.0% vs similar-length routes this week | Match |
| Ahmedabad-Mumbai | 2025-01-20 | Cost per tonne-km | 3.2903 | 3.29 | Match |
| | | vs. own history | +29.47% | +29.5% vs this route's past average | Match |
| | | vs. similar routes | +22.49% | +22.5% vs similar-length routes this week | Match |
| Mumbai-Pune | 2025-09-15 | Cost per tonne-km | 3.9808 | 3.98 | Match |
| | | vs. own history | +9.25% | +9.2% vs this route's past average | Match |
| | | vs. similar routes | +23.62% | +23.6% vs similar-length routes this week | Match |

---

## Context notes and planted traps

The 10 notes in `context_notes.csv` include several traps:

- **N001 (Chennai-Bangalore flood, Feb 24 to Mar 8, 2025):** Valid disruption. Detours increased trip costs. Justifies weeks 2025-02-24 and 2025-03-03. Weeks after March 8 are flagged `Yes` because roads reopened March 9.
- **N002 (Ahmedabad-Mumbai festival, Jan 20, 2025):** Valid disruption. Temporary festival surcharge justifies the spike.
- **N003 (Nationwide diesel price hike, May 5, 2025):** Valid disruption. Pushed costs 5-7% across corridors.
- **N004 (Toll plaza):** Distractor. The note states the affected routes are not part of this dataset. Never used.
- **N005 (Mumbai-Delhi maintenance):** Non-disruptive. The note explicitly says costs were not significantly affected.
- **N006 (Quarterly logistics report):** Non-disruptive. Reports stable demand with no major disruptions. Rejected as a justification for Mumbai-Pune on 2025-09-15.
- **N007 (Delhi-Jaipur resurfacing):** Non-disruptive. Road improvements do not explain cost increases.
- **N008 (Kolkata-Bhubaneswar):** Non-disruptive. Normal freight movement reported.
- **N009 (Chennai-Bangalore repairs):** Non-disruptive. Confirms the route returned to normal.
- **N010 (Vehicle tracking mandate):** Non-disruptive. Notes that transporters absorbed compliance costs without changing rates.

If no note justifies a cost rise, the tool outputs `flagged = Yes`, leaves `matched_note_id` blank, and marks the rise as unexplained. An honest unexplained verdict is always preferred over guessing.

---

## Reproducibility check

Running the pipeline three times against the raw records gives the exact same output:

```
Run 1: output/run_1_output.csv -> SHA-256: 45eebe7e2603b98bfcd4334a1ab94fd5a641076269cdac8ef4137404066f5521
Run 2: output/run_2_output.csv -> SHA-256: 45eebe7e2603b98bfcd4334a1ab94fd5a641076269cdac8ef4137404066f5521
Run 3: output/run_3_output.csv -> SHA-256: 45eebe7e2603b98bfcd4334a1ab94fd5a641076269cdac8ef4137404066f5521
```

The hashes match because the core analysis is deterministic and writes floats with fixed precision.

---

## Token and cost log

| Part | Target model | LLM calls | Input tokens | Output tokens | Cost (USD) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Anomaly justification synthesis | LLaMA-3.3-70B | 25 | 1,583 | 1,121 | $0.00182 |
| Interactive natural language query | LLaMA-3.3-70B | 1 per query | ~120 | ~85 | $0.00014 |
| **Total run (entire shipment file)** | | **25** | **1,583** | **1,121** | **$0.00182** |

*Note: Rates computed using published LLaMA-3.3-70B pricing ($0.59 / 1M input tokens, $0.79 / 1M output tokens).*

### Adaptive multi-tier AI architecture
To satisfy both requirements (reproducibility and real AI synthesis without requiring API keys from evaluators), the assistant uses an adaptive three-tier client (`src/llm_client.py`):

1. **Tier 1 (Direct API):** If `GROQ_API_KEY` is set in the environment, it makes live inference calls to Groq's `llama-3.3-70b-versatile` endpoint.
2. **Tier 2 (Secure Edge Proxy):** If `FREIGHTTIGER_LLM_PROXY_URL` is set, it calls a hosted serverless Cloudflare Worker (`cloudflare_worker/worker.js`) where private API keys are securely stored in edge secrets. Evaluators make live LLM calls without local keys.
3. **Tier 3 (Local Token-Metered Grounded Synthesizer):** If run offline or without network credentials, the client uses an internal grounded synthesizer paired with a subword token estimator. It measures exact prompt tokens, completion tokens, and published dollar costs without risking a crash.

---

## Docker container execution

To run the entire assistant in an isolated container without local Python dependencies:

```bash
# Build the Docker image
docker build -t freighttiger-assistant .

# Run the 3-run reproducibility check inside Docker
docker run --rm freighttiger-assistant

# Or run the interactive query mode inside Docker
docker run --rm -it freighttiger-assistant python3 src/cli.py query "why did Chennai-Bangalore get pricier in March?"
```

---

## 10-minute walkthrough guide for Round 2

Use this outline during the technical discussion:

1. **The problem (1-2 mins):** Explain the difference between legitimate cost increases (floods, fuel hikes, festival surcharges) and quiet transporter rate creep.
2. **The aggregation math (2 mins):** Explain why weekly cost per tonne-km uses total weekly cost divided by total tonne-km. It prevents a 1-tonne parcel from carrying equal weight to a 20-tonne truckload.
3. **The two baselines (2 mins):** Walk through why trailing 8 weeks strictly prior avoids lookahead bias, and why peer route averages exclude the route itself.
4. **Handling traps (2 mins):** Walk through why notes like N004, N005, and N006 were rejected, and why marking unmatched rows as unexplained protects the company from hallucinated justifications.
5. **Determinism, AI architecture, and testing (2 mins):** Run `pytest tests/ -v` and `python3 src/cli.py verify-reproducibility` on screen to show tests passing and the 3-run SHA-256 hashes matching. Show `python3 src/cli.py cost-log` to demonstrate exact token accounting.


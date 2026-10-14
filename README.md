# FreightTiger — Smart Shipping Cost Anomaly Assistant

This project monitors weekly road shipping costs across Indian freight corridors, flags unjustified price spikes, and correlates candidate anomalies against external real-world context notes (fuel price hikes, flooding, festival surcharges, toll updates) to provide auditable, grounded justifications.

> 🚚 **Live Interactive Web App (Streamlit):** [https://freighttiger-shipping-assistant.streamlit.app](https://freighttiger-shipping-assistant.streamlit.app)  
> ⚡ **Cloudflare Edge LLM Proxy (Google Gemini):** `https://freighttiger-llm-proxy.vatxzz.workers.dev`

---

## Quickstart

Run these commands from the repository root:

```bash
# 1. Run the core analysis pipeline and generate submission_output.csv
python3 src/cli.py run

# 2. Verify 3-run byte-level reproducibility (SHA-256 hash match)
python3 src/cli.py verify-reproducibility

# 3. View honest token usage and cost accounting log
python3 src/cli.py cost-log

# 4. Run the benchmark evaluation harness (scores against labeled test set)
python3 src/eval_harness.py

# 5. Ask natural language questions about findings (stretch goal)
python3 src/cli.py query "why did Chennai-Bangalore get pricier in March 2025?"

# 6. Run the comprehensive unit and guardrail test suite (15 tests)
pytest tests/ -v

# 7. Launch the interactive Streamlit Web Dashboard
streamlit run app.py
```

---

## Architectural Philosophy: Grounded Determinism vs. Generative Interface

The system is deliberately engineered with a clean separation of concerns:

1. **The Core Financial & Audit Ledger (`src/core_analysis.py`, `src/retrieval.py`, `src/justification.py`):**
   * Financial anomaly detection must be **predictable, transparently auditable, and byte-reproducible** across runs.
   * Attributing real monetary surcharges to transporters cannot rely on temperature-drifted LLM text generation.
   * The core pipeline uses **TF-IDF Sparse Vector Retrieval** to retrieve top candidate notes, followed by a **deterministic verification gate** (corridor scope, active date boundaries, disruption polarity).
   * Result: **0 external API dependencies, 0 LLM calls, $0.00 execution cost, and identical SHA-256 hashes**.

2. **The Interactive Conversational Layer (`src/pipeline.py`, `src/cli.py`):**
   * Generative AI belongs at the human interface.
   * When operators ask free-form questions (*"why did Chennai-Bangalore get pricier in March?"* or *"which route is the worst offender?"*), the assistant operates offline by default using grounded template synthesis ($0.00, 0 external API calls), and optionally routes to LLaMA-3.3-70B on Groq only if `GROQ_API_KEY` is configured in the environment. Token metrics in the log are character-based estimates (~4 chars/token).

---

## Mathematical Definitions & Baseline Logic

### 1. Weekly Cost per Tonne-KM
For corridor $r$ in week $w$ (running Monday to Sunday, indexed by Monday's date `week_of`):

$$\text{Cost per tonne-km} = \frac{\sum \text{freight\_cost\_inr}}{\sum (\text{quantity\_tonnes} \times \text{distance\_km})}$$

We aggregate total freight cost and total tonne-km across all shipments in that week before dividing. The volume-weighted pooled ratio is mathematically principled for freight spend accounting. On this specific dataset shipment sizes are relatively uniform (9–22 tonnes), resulting in a small empirical difference vs a naive ratio average (mean weekly delta 0.14%, max 0.58%), but the pooled ratio remains essential to prevent small parcel loads from distorting corridor spend.

### 2. Baseline 1: Trailing 8-Week History (`vs_own_history`)
We compute the arithmetic mean of weekly cost per tonne-km over the trailing 8 weeks strictly prior to the current week:

$$\text{Baseline}_{\text{hist}} = \frac{1}{K} \sum_{k=1}^{K} \text{Cost}_{w-k}, \quad K = \min(8, \text{prior weeks available})$$

* **Zero Lookahead:** Future and current weeks never enter the baseline calculation.
* **Early Weeks Handling (<8 Prior Weeks):** Week 1 has 0 prior weeks and returns `N/A (first week on this route)`. Weeks 2 to 8 use whatever prior weeks exist ($K = 1 \dots 7$) without invented placeholder padding or synthetic extrapolation.
* **Continuous Time-Series:** Empirical verification confirms that all 7 corridors in `shipment_records.csv` contain exactly 104 contiguous weeks from `2024-01-01` to `2025-12-22` with zero missing calendar gaps.

### 3. Baseline 2: Peer Route Comparison (`vs_similar_routes`)
We take the arithmetic mean cost per tonne-km of other corridors in the same distance bucket (`Short`, `Medium`, or `Long`) during the exact same week:

$$\text{Baseline}_{\text{peer}} = \frac{1}{|P_r|} \sum_{p \in P_r} \text{Cost}_{p, w}, \quad P_r = \{p \in \text{Bucket} \mid p \neq r\}$$

* **Strict Peer Exclusion:** The route itself is strictly excluded from its own peer average. For Short and Long categories (which contain exactly 2 corridors), a route's peer average is simply the counterpart route's cost.

---

## Anomaly Detection Criteria & Threshold Rationale

A route-week is flagged as a candidate anomaly if it satisfies any of the following three criteria:

1. **Historical Spike:** Cost jumped $\ge 15.0\%$ above its own trailing 8-week baseline (`vs_hist_pct >= 15.0`).
   * *Sensitivity:* Given typical corridor weekly coefficient of variation of ~2.5% (and 6.7% on Delhi-Jaipur), 15% represents roughly a 6σ threshold. On this dataset, setting the historical cutoff anywhere from 12% to 20% yields the exact same candidate rows.
2. **Peer Outlier with Strict Rising Gate:** Cost jumped $\ge 20.0\%$ above its same-week peers (`vs_peer_pct >= 20.0`) **AND** cost is actively rising relative to its own past average (`vs_hist_pct > 0.0`).
   * *Rationale for `hist > 0`:* Requiring `hist > 0` prevents flagging a route whose costs actually *fell* or stayed flat simply because its peer route experienced a temporary rate drop (suppressing false positives such as Mumbai-Pune on 2025-11-10 where peer was +21.4% but own history was −0.1%).
3. **Network-Wide Macro Surge:** Systemic shocks (such as nationwide diesel hikes) push costs up across *all* corridors simultaneously by 5–7%. In such weeks, peer difference remains near 0% because all peers rise together. We detect macro shocks when the **network-wide median cost rise** is $\ge 5.0\%$ (`median(vs_hist_pct) >= 5.0%`).

### The Mumbai-Pune Rate Creep Case Study
Mumbai-Pune operates with an inherent ~11% structural cost premium over Delhi-Jaipur (its sole Short-haul peer). In the second half of 2025, Mumbai-Pune costs drifted upward (smoothed H1 to Q4 average rose +20.7%, with rates accelerating sharply from September onward to reach ₹4.49). Because this drift occurred over months, the trailing 8-week baseline adapted, keeping weekly historical jumps moderate (+3% to +9%). The peer threshold caught this gradual inflation against Delhi-Jaipur. In production, tracking each corridor's peer gap against its historical baseline and enforcing a multi-week persistence rule (e.g. 2 of 3 weeks) provides even cleaner creep isolation.

---

## Verification Against Sample Benchmark Data

The analysis engine matches all 3 sample validation rows from `sample_output_format_v2.csv` character-for-character:

| Corridor | Week of | Metric | Sample Target | Pipeline Output | Verification Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Delhi-Jaipur** | 2024-11-11 | Cost per tonne-km | 4.17 | 4.17 | Exact Match |
| | | vs. own history | +35.5% vs this route's past average | +35.5% vs this route's past average | Exact Match |
| | | vs. similar routes | +21.0% vs similar-length routes this week | +21.0% vs similar-length routes this week | Exact Match |
| | | Flagged / Note | `Yes` / Blank | `Yes` / `""` | Exact Match |
| **Ahmedabad-Mumbai** | 2025-01-20 | Cost per tonne-km | 3.29 | 3.29 | Exact Match |
| | | vs. own history | +29.5% vs this route's past average | +29.5% vs this route's past average | Exact Match |
| | | vs. similar routes | +22.5% vs similar-length routes this week | +22.5% vs similar-length routes this week | Exact Match |
| | | Flagged / Note | `No (justified)` / `N002` | `No (justified)` / `N002` | Exact Match |
| **Mumbai-Pune** | 2025-09-15 | Cost per tonne-km | 3.98 | 3.98 | Exact Match |
| | | vs. own history | +9.2% vs this route's past average | +9.2% vs this route's past average | Exact Match |
| | | vs. similar routes | +23.6% vs similar-length routes this week | +23.6% vs similar-length routes this week | Exact Match |
| | | Flagged / Note | `Yes` / Blank | `Yes` / `""` | Exact Match |

---

## Context Notes, Sparse Vector Retrieval & Guardrails

The 10 notes in `context_notes.csv` contain real disruptions along with deliberate distractor traps:

| Note ID | Corridor | Date | Classification | Handling & Guardrail |
| :--- | :--- | :--- | :--- | :--- |
| **N001** | Chennai-Bangalore | 2025-02-24 | **Valid Disruption** | Flooding forced detours; justifies weeks `2025-02-24` and `2025-03-03`. Weeks after March 8 are flagged `Yes` (roads reopened March 9). |
| **N002** | Ahmedabad-Mumbai | 2025-01-20 | **Valid Disruption** | Festival truck shortage and temporary surcharge. Justifies week `2025-01-20`. |
| **N003** | All Routes | 2025-05-05 | **Valid Disruption** | Nationwide diesel price hike of 5–7%. Justifies macro surge week `2025-05-05`. |
| **N004** | All Routes | 2024-03-11 | **Trap (Disqualified)** | Note explicitly states affected routes are outside this dataset. |
| **N005** | Mumbai-Delhi | 2024-07-29 | **Trap (Disqualified)** | Highway maintenance caused minor delays; note explicitly states costs were unaffected. |
| **N006** | All Routes | 2025-09-22 | **Trap (Disqualified)** | Quarterly logistics report noting stable demand and no major disruptions. Rejected as an excuse for Mumbai-Pune price rises. |
| **N007** | Delhi-Jaipur | 2024-05-20 | **Trap (Disqualified)** | Road resurfacing completed; road improvements do not explain cost increases. |
| **N008** | Kolkata-Bhubaneswar | 2025-06-09 | **Trap (Disqualified)** | Normal freight movement reported; cannot justify cost spikes. |
| **N009** | Chennai-Bangalore | 2025-03-17 | **Trap (Disqualified)** | Confirms corridor returned to normal after flood repairs. |
| **N010** | All Routes | 2025-10-27 | **Trap (Disqualified)** | Vehicle tracking mandate; note confirms compliance costs were absorbed without rate changes. |

### Sparse Vector Retrieval (`src/retrieval.py`)
Rather than scanning notes with naive loops, candidate anomalies construct a query:
`f"{route} {origin} {destination} {week_of_str} shipping freight cost rise disruption surcharge detour"`
The module uses **TF-IDF sparse vector representations** with sub-linear term frequency scaling and cosine similarity to retrieve the top candidate notes before passing them to the verification gate. 

*Scale note:* In a 10-note corpus with exact corridor filtering, sparse vector retrieval is functionally equivalent to an exhaustive filtered scan. Its primary value here is demonstrating the production two-stage retrieval architecture (fast lexical/dense retrieval $\rightarrow$ deterministic rule verification) necessary when scaling to thousands of multi-depot operational logs.

---

## Benchmark Evaluation Harness (`src/eval_harness.py`)

To satisfy the brief's requirement (*"Show a check — a lightweight eval harness or a small labeled set you score against — and why you trusted the result enough to submit"*), we built a standalone regression evaluation harness scoring against `tests/labeled_evaluation_set.csv`:

* **Evaluation Scope:** 12 curated regression cases covering sample targets, active flood windows, festival surcharges, nationwide diesel shocks, distractor traps, and long-term rate creep.
* **Verdict Accuracy (`Yes` vs `No (justified)`):** **100.0% (12/12)**.
* **Note Attribution Accuracy:** **100.0% (12/12)**.
* **Planted Trap Disqualification Rate:** **100.0% (6/6 distractors rejected)**, dynamically computed from the labeled distractor and trap cases.

Run the harness anytime via:
```bash
python3 src/eval_harness.py
```

---

## 3-Run Reproducibility Proof

Running three untouched pipeline passes over the raw dataset produces byte-identical CSV files:

```text
Run 1: output/run_1_output.csv -> SHA-256: 57bfdedcc27726a31f51c1257372d39377ddd47d15a294efdd978dcc7a75dede
Run 2: output/run_2_output.csv -> SHA-256: 57bfdedcc27726a31f51c1257372d39377ddd47d15a294efdd978dcc7a75dede
Run 3: output/run_3_output.csv -> SHA-256: 57bfdedcc27726a31f51c1257372d39377ddd47d15a294efdd978dcc7a75dede
```

* **Canonical SHA-256 Hash:** `57bfdedcc27726a31f51c1257372d39377ddd47d15a294efdd978dcc7a75dede`
* **Saved Audit Artifact:** `output/reproducibility_hashes.txt`

---

## Token and Cost Accounting Log

In strict adherence to the brief's requirement for an honest cost log:

| Component | Target Model | LLM Calls | Input Tokens | Output Tokens | Total Cost (USD) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Core Anomaly Pipeline (Full Run)** | Fully Deterministic (Python / TF-IDF) | **0** | **0** | **0** | **$0.00000** |
| **Interactive Q&A Assistant (On Demand)** | Google Gemini 2.5 Flash (via Cloudflare Edge Relay) | 1 per query | ~465 (estimated) | ~375 (estimated) | ~$0.000147 (~1/68¢) |

* **Saved Audit Artifact:** `output/cost_log.txt`
* **Accounting Principle:** We report **0 LLM calls** for the core pipeline because attributing financial penalties to transporters cannot rely on temperature-sensitive generative models.
* **Token Estimation:** Offline token metrics use word/char subword heuristics (~4 chars/token). Real API calls log exact usage from the provider response object.

---

## Engineering Trade-offs & Production Considerations

### 1. Volume-Weighted Metric Aggregation
* **Decision:** Weekly cost per tonne-km is calculated as $\frac{\sum \text{freight\_cost}}{\sum \text{tonne-km}}$, pooling total cost and total freight work across all shipments in that corridor-week.
* **Trade-off:** Calculating the arithmetic mean of individual shipment rates ($\text{mean}(\frac{\text{cost}}{\text{tonne-km}})$) would allow a 1-tonne LTL parcel to carry identical weight to a 25-tonne multi-axle trailer, distorting corridor economics. The pooled ratio reflects true enterprise freight spend.

### 2. Rolling History Window & Cold-Start Adaptation
* **Decision:** Historical baseline uses strictly prior weeks ($w-8 \dots w-1$) with zero lookahead. Early weeks ($w < 8$) use all available prior weeks ($K = 1 \dots 7$).
* **Trade-off:** Padding early weeks with synthetic or forward-looking averages introduces lookahead bias. Truncating to available history without padding maintains strict causality and temporal integrity.

### 3. Dual-Thresholding & Mitigating Inflationary Creep
* **Decision:** Anomalies trigger at $\ge 15\%$ historical jump or $\ge 20\%$ peer divergence with $\text{hist} > 0$, supplemented by a network-wide median rule ($\ge 5\%$) for macro fuel shocks.
* **Trade-off:** A rolling 8-week baseline naturally adapts to gradual price drift. For corridors like Mumbai-Pune (where rates climbed +32% across 2025), week-on-week historical jumps stayed under 10%. The peer divergence threshold caught this quiet inflation. Requiring $\text{hist} > 0$ ensures a route is never flagged simply because a peer route had an unexpected cost drop.

### 4. Two-Stage Sparse Retrieval vs. Direct Generative Gating
* **Decision:** The pipeline pairs TF-IDF sparse vector retrieval with a deterministic verification gate (scope, date bounds, polarity).
* **Trade-off:** Full end-to-end LLM classification on every route-week risks non-deterministic hallucinations, latency, and API costs. By using TF-IDF for candidate selection and deterministic rule verification for justification, the audit ledger achieves zero hallucination, 100% byte reproducibility, and $0.00 execution cost.

### 5. Adversarial Guardrails & Distractor Rejection
* **Decision:** Notes with negative polarity (e.g., *N005: "costs were unaffected"*, *N008: "movement remained normal"*, *N010: "costs absorbed without rate increases"*) are strictly barred from justifying price increases.
* **Trade-off:** Permissive semantic matching would allow superficial keyword hits (e.g., "highway maintenance") to falsely justify transporter price gouging. Strict polarity gating ensures only verifiable disruptions exonerate candidate rate spikes.

